"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.

``gh`` runs as the card's assignee profile (``profile_home``), not the ambient
login: :func:`_gh_env` resolves that profile's own credentials/config for the
subprocess — a multi-profile host's default ``gh`` login cannot read another
org's private repos (#122689).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


class _GateAuthError(RuntimeError):
    """The assignee login could not resolve the repository — private to another
    login, missing grant, or wrong name. An ambiguity to verify, not proof the
    credentials are wrong, and not a transient failure to retry."""


class _GatePolicyError(RuntimeError):
    """Required-check policy was refused after the repository resolved. Never
    evidence that no checks are required: completion stays blocked (fail-closed)."""


class _GateRetryError(RuntimeError):
    """Transient refusal (rate limit): wait and retry; not a credential problem."""


def _json_or_none(text: str | None):
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _graphql_refusal(payload: dict, name: str) -> None:
    """Raise on any structured GraphQL refusal; return only on complete evidence.

    A visibility refusal rides HTTP 200: ``data.repository`` null with a
    ``NOT_FOUND``/``FORBIDDEN`` error (or no error at all), at any exit status —
    gh need not put ``HTTP 40x`` in stderr, so the body is the only signal.
    """
    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    errors = payload.get("errors") or []
    unresolved = "repository" not in data or data["repository"] is None
    if not unresolved and not errors:
        return
    if unresolved and (not errors or any(e.get("type") in {"NOT_FOUND", "FORBIDDEN"} for e in errors)):
        raise _GateAuthError(f"repository unresolved on {name}")
    raise ValueError("GitHub returned incomplete GraphQL evidence")


# (operation, HTTP status) → gate raised at the subprocess boundary. Unlisted
# combinations (5xx, timeouts, odd 40x after visibility was proven) re-raise the
# original failure → generic infra. Only the repository read turns a 40x into an
# identity/visibility ambiguity; a policy 403 is a fail-closed capability gap
# (#122009), never a wrong-credential diagnosis.
_REFUSALS = {
    ("repository", "401"): _GateAuthError,
    ("repository", "403"): _GateAuthError,
    ("repository", "404"): _GateAuthError,
    ("policy", "403"): _GatePolicyError,
    ("evidence", "401"): _GateAuthError,
}


def _gate_transport(failure: subprocess.CalledProcessError, name: str,
                    repository_read: bool) -> None:
    """Classify a nonzero-exit refusal, or re-raise it (→ generic infra)."""
    stderr = failure.stderr or ""
    if re.search(r"rate limit", stderr, re.I):
        raise _GateRetryError(f"rate-limited on {name}")
    status = re.search(r"HTTP (\d{3})", stderr)
    code = status[1] if status else ""
    op = ("repository" if repository_read
          else "policy" if "/rules/branches/" in name else "evidence")
    gate = _REFUSALS.get((op, code))
    if gate is not None:
        raise gate(f"HTTP {code} on {name}")
    raise failure


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False,
         profile_home: str | None = None, label: str | None = None):
    name = label or endpoint.split("?")[0]
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=30, check=True, env=_gh_env(profile_home))
    except subprocess.CalledProcessError as caught:
        stdout, failure = caught.stdout, caught
    else:
        stdout, failure = result.stdout, None
    # Gate errors persist only status code + endpoint, never gh's stderr
    # (credentials/host details). The body is parsed even on failure: GraphQL
    # refusals travel on HTTP 200 regardless of the exit status.
    payload = _json_or_none(stdout)
    if query is not None and isinstance(payload, dict):
        _graphql_refusal(payload, name)
    if failure is not None:
        _gate_transport(failure, name, query is not None)
    if payload is None:
        payload = json.loads(stdout)  # non-JSON body → ValueError → infra
    return payload


def _gh_env(profile_home: str | None) -> dict[str, str]:
    """Child env for ``gh``: the card's profile identity when one is resolvable.

    The completion boundary runs in the worker (assignee), the CLI, or a
    reviewer/dispatcher turn, so an ambient ``gh`` login is whichever process
    happened to call it. ``served_profile_child_env(inherit_credentials=True)``
    is the blessed seam for "this child acts for that profile": it scrubs the
    launch profile's credential residue and overlays the target profile's own
    ``GH_TOKEN``/``GH_CONFIG_DIR`` (its ``.env`` + external secret sources).
    ``None`` keeps the ambient env — unassigned cards and single-profile hosts
    behave exactly as before.
    """
    if not profile_home:
        return os.environ
    from tools.environments.local import _is_routed_home, served_profile_child_env
    base = dict(os.environ)
    if _is_routed_home(profile_home):
        # gh's config dir decides which login `gh api` uses, yet it is a path,
        # not a credential, so no scrub list sees it: drop launch residue (a
        # unit-file export never shows in the launch `.env` the strip reads).
        # The target's own GH_CONFIG_DIR is overlaid from its `.env` below; a
        # same-home target keeps operator exports, like every other child.
        base.pop("GH_CONFIG_DIR", None)
    return served_profile_child_env(base=base, target_home=profile_home,
                                    inherit_credentials=True)


def _assignee_profile_home(conn, task_id: str) -> "str | None":
    """Home whose ``gh`` login must read the contract repo, or None.

    The assignee owns the PR and its repo; the reviewer fallback matches
    dispatch, which re-spawns a review-lane task under the assignee's profile.
    Name resolution goes through ``get_profile_dir`` (rooted at the DEFAULT
    profile root, not this process's launch home): the boundary may run inside
    a routed turn, and profile operations are HOME-anchored by design.
    Unresolvable names (unassigned, "worker", uninstalled) return None so the
    ambient login is used rather than guessing an identity.
    """
    try:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    except Exception:
        return None
    assignee = row["assignee"] if row else None
    if not assignee or assignee == "worker":
        return None
    try:
        from hermes_cli.profiles import get_profile_dir, profile_exists
        if not profile_exists(assignee):
            return None
        return str(get_profile_dir(assignee))
    except Exception:
        return None


def collect_acceptance(contract: str, published_pr: str | None,
                       profile_home: str | None = None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    try:
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        repo, number = match[1], int(match[2])
        receipt["pr_url"] = url
        owner, name = repo.split("/")
        query = '''{repository(owner:%s,name:%s){pullRequest(number:%d){headRefOid baseRefName state
            baseRef{branchProtectionRule{requiredStatusChecks{context app{databaseId}}}}}}}''' % (
                json.dumps(owner), json.dumps(name), number)
        repository = _api("graphql", query=query, profile_home=profile_home,
                          label=f"graphql {repo}")["data"]["repository"]
        pr = repository["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        required = {(r["context"], (r.get("app") or {}).get("databaseId")) for r in protection.get("requiredStatusChecks", [])}
        rules = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100",
                     paginate=True, profile_home=profile_home)
        for page in rules:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        if not required:
            receipt["detail"] = "No repository-required checks are configured; explicitly use a local-only contract for non-CI tasks."
            return receipt
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest",
                     paginate=True, profile_home=profile_home)
        runs = [run for page in pages for run in page["check_runs"]]
        if len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        statuses = [{**s, "sha": sha} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100",
                                                       paginate=True, profile_home=profile_home) for s in page]
        outcomes = []
        for context, app_id in sorted(required, key=str):
            matching = [r for r in runs if r["name"] == context and
                        (app_id in (None, -1) or r["app"]["id"] == app_id)]
            # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
            legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
            selected = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
            if not selected:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
            for check in selected:
                is_run = "conclusion" in check
                outcome = check.get("conclusion") if is_run else check["state"]
                classification = _classify(check, sha, outcome, is_run)
                outcomes.append(classification)
                receipt["checks"].append({"name": context, "id": check["id"],
                    "url": check.get("html_url") or check.get("target_url"),
                    "head_sha": check.get("head_sha", check.get("sha")),
                    "classification": classification, "conclusion": outcome})
        # Re-read after all pages: old-head successes are never transferable.
        current = _api(f"repos/{repo}/pulls/{number}", profile_home=profile_home)
        if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        receipt["classification"] = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
        receipt["ok"] = receipt["classification"] == "success"
        return receipt
    except _GateAuthError as exc:
        receipt.update(classification="auth",
                       detail=f"GitHub refused the acceptance read ({exc}) as the assignee profile's gh login; "
                              "the repository is invisible to that login (private, missing grant, or wrong name) — "
                              "check that the profile's GitHub credentials can access the repository, then retry completion.")
        return receipt
    except _GatePolicyError as exc:
        receipt.update(classification="policy",
                       detail=f"Required-check policy read refused ({exc}) after the repository resolved; "
                              "the login may lack rulesets/branch-policy read capability. This is NOT evidence "
                              "that no checks are required — completion stays blocked until the policy is readable "
                              "(grant the rulesets read or configure checks another way), then retry.")
        return receipt
    except _GateRetryError as exc:
        receipt.update(classification="retry",
                       detail=f"GitHub API rate limit ({exc}); wait for the window to reset, then retry completion.")
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.")
        return receipt


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
