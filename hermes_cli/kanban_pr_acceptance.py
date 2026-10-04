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
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_BILLING_PAYWALL_MESSAGE = (
    "The job was not started because recent account payments have failed "
    "or your spending limit needs to be increased."
)


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False,
         profile_home: str | None = None):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30,
                                check=True, env=_gh_env(profile_home))
    except subprocess.CalledProcessError as exc:
        # 401/403/404 = the login cannot see this repository (wrong profile identity
        # or missing grant), not a transient API failure. Persist only the status
        # code + endpoint, never gh's stderr (credentials/host details).
        denied = re.search(r"HTTP (40[134])", exc.stderr or "")
        if denied:
            raise _GateAuthError(f"HTTP {denied[1]} on {endpoint.split('?')[0]}") from None
        if exc.returncode == 4:  # gh's authentication-required exit: this profile has no login
            raise _GateAuthError(f"gh has no login for {endpoint.split('?')[0]}") from None
        raise
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


class _GateAuthError(RuntimeError):
    """gh was refused at HTTP 401/403/404 (or GraphQL returned no repository):
    this profile's login cannot see the repo — an identity problem to fix, not
    an infrastructure blip to retry."""


def _gh_env(profile_home: str | None) -> dict[str, str] | None:
    """Child env for ``gh``: the card's profile identity when one is resolvable.

    The completion boundary runs in the worker (assignee), the CLI, or a
    reviewer/dispatcher turn, so an ambient ``gh`` login is whichever process
    happened to call it (#122689). ``served_profile_child_env(inherit_credentials=True)``
    is the seam for "this child acts for that profile": it scrubs the launch
    profile's credential residue and overlays the target profile's own
    ``GH_TOKEN``/``GH_CONFIG_DIR`` (its ``.env`` + external secret sources).
    ``None`` keeps the ambient env — unassigned cards behave exactly as before.
    """
    if not profile_home:
        return None
    from tools.environments.local import _is_routed_home, hermes_subprocess_env, served_profile_child_env
    base = hermes_subprocess_env(inherit_credentials=True)
    routed = _is_routed_home(profile_home)
    if routed:
        # gh's config dir decides which login `gh api` uses, yet it is a path, not a
        # credential, so no scrub list sees it; the target's own value is overlaid from its .env.
        base.pop("GH_CONFIG_DIR", None)
    env = served_profile_child_env(base=base, target_home=profile_home, inherit_credentials=True)
    if routed and not (env.keys() & {"GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR"}):
        # HOME/XDG_CONFIG_HOME are still the launch process's: without a login of its own the
        # child would fall through to ~/.config/gh/hosts.yml — the ambient login. Pin gh's config
        # to a profile-owned dir so it fails "not logged in" (exit 4 -> auth) instead.
        env["GH_CONFIG_DIR"] = str(Path(profile_home) / "gh")
    return env


def _assignee_profile_home(assignee: str | None) -> str | None:
    """Home whose ``gh`` login must read the contract repo — the assignee's, resolved
    exactly as the dispatcher resolves the worker's home — or None (unassigned) so the
    ambient login is used. An assigned card whose profile cannot be resolved is an
    identity failure (``auth``), never a silent fall-through to the ambient login."""
    if not assignee:
        return None
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
    try:
        return resolve_profile_env(normalize_profile_name(assignee))
    except (FileNotFoundError, ValueError):
        raise _GateAuthError(f"assignee profile {assignee!r} cannot be resolved") from None


def _billing_exception_repositories(policy: dict | None) -> set[str]:
    if not isinstance(policy, dict):
        return set()
    exception = policy.get("github_actions_billing_exception")
    if not isinstance(exception, dict) or exception.get("enabled") is not True:
        return set()
    repositories = exception.get("repositories")
    if not isinstance(repositories, list):
        return set()
    return {repo for repo in repositories if isinstance(repo, str) and _REPO.fullmatch(repo)}


def _collect_billing_exception(repo: str, number: int, sha: str, branch: str,
                               receipt: dict, profile_home: str | None = None) -> dict | None:
    pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest",
                 paginate=True, profile_home=profile_home)
    runs = [run for page in pages for run in page["check_runs"]]
    if not pages or len({run["id"] for run in runs}) != pages[0]["total_count"]:
        raise ValueError("Incomplete check-run pagination")

    failed = [run for run in runs if run.get("status") == "completed" and run.get("conclusion") == "failure"]
    if any(run.get("status") != "completed" for run in runs):
        return None
    if not failed:
        if not runs:
            # No check-runs to verify; cannot accept a billing-exception reconciliation
            # without any CI evidence. Fall through to the regular path.
            return None
        if not all(run.get("conclusion") == "success" for run in runs):
            # Only all-success is acceptable; cancelled/timed_out/skipped/neutral
            # and missing conclusions are NOT equivalent to success per Kanban
            # completion-contract policy.
            return None
        # All check-runs completed with no failures. For billing-exception repos on a
        # free plan, the branch-protection rules API (rules/branches) returns HTTP 403,
        # so the required-checks set cannot be populated via that path. Accept all
        # non-failing check-runs on the exact head as a local-only reconciliation
        # (per board policy: github_actions_billing_exception).
        receipt["checks"] = [
            {
                "name": run["name"],
                "id": run["id"],
                "url": run.get("html_url") or run.get("details_url"),
                "head_sha": run.get("head_sha"),
                "classification": str(run.get("conclusion") or run.get("status") or "unknown"),
                "conclusion": run.get("conclusion"),
            }
            for run in runs
        ]
        current = _api(f"repos/{repo}/pulls/{number}", profile_home=profile_home)
        receipt["base_sha"] = (current.get("base") or {}).get("sha")
        if (
            current["head"]["sha"] != sha
            or current["base"]["ref"] != branch
            or (current["state"] == "closed" and not current.get("merged"))
        ):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        receipt.update(
            ok=True,
            classification="billing/paywall — not executed",
            detail=(
                "Repository is on a free plan; branch-protection rules API unavailable (HTTP 403). "
                "All check-runs verified non-failing on exact head via check-runs API; "
                "exact PR head/base, local verification, and independent review remain required."
            ),
            policy="github_actions_billing_exception",
        )
        return receipt

    waived_ids: set[int] = set()
    for run in failed:
        annotations = _api(f"repos/{repo}/check-runs/{run['id']}/annotations",
                           profile_home=profile_home)
        if not isinstance(annotations, list) or not any(
            isinstance(annotation, dict)
            and _BILLING_PAYWALL_MESSAGE in str(annotation.get("message") or "")
            for annotation in annotations
        ):
            return None
        waived_ids.add(run["id"])

    receipt["checks"] = [
        {
            "name": run["name"],
            "id": run["id"],
            "url": run.get("html_url") or run.get("details_url"),
            "head_sha": run.get("head_sha"),
            "classification": (
                "billing/paywall — not executed" if run["id"] in waived_ids
                else str(run.get("conclusion") or run.get("status") or "unknown")
            ),
            "conclusion": run.get("conclusion"),
        }
        for run in runs
    ]

    # Re-read after all pages and annotations: old-head billing evidence is not transferable.
    current = _api(f"repos/{repo}/pulls/{number}", profile_home=profile_home)
    receipt["base_sha"] = (current.get("base") or {}).get("sha")
    if (
        current["head"]["sha"] != sha
        or current["base"]["ref"] != branch
        or (current["state"] == "closed" and not current.get("merged"))
    ):
        receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
        return receipt

    receipt.update(
        ok=True,
        classification="billing/paywall — not executed",
        detail=(
            "Board policy treats only GitHub Actions jobs carrying the exact account-billing "
            "not-started annotation as non-gating; local verification and review remain external gates."
        ),
        policy="github_actions_billing_exception",
    )
    return receipt


def _board_acceptance_policy() -> dict | None:
    try:
        from hermes_cli.kanban_db import get_current_board, read_board_metadata
        return read_board_metadata(get_current_board()).get("pr_acceptance")
    except Exception:
        return None


def collect_acceptance(contract: str, published_pr: str | None,
                       assignee: str | None = None, *, policy: dict | None = None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    try:
        profile_home = _assignee_profile_home(assignee)
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
        repository = _api("graphql", query=query, profile_home=profile_home)["data"]["repository"]
        if repository is None:
            # A private repo the login cannot read resolves to null, not an error.
            raise _GateAuthError(f"HTTP 404 on graphql {repo}")
        pr = repository["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"] = sha
        receipt["base_branch"] = branch
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        required = {(r["context"], (r.get("app") or {}).get("databaseId")) for r in protection.get("requiredStatusChecks", [])}
        if repo in _billing_exception_repositories(policy):
            waived = _collect_billing_exception(repo, number, sha, branch, receipt, profile_home)
            if waived is not None:
                return waived
            # Billing exception did not apply (e.g., non-billing check failures).
            # The rules API also returns 403 on free plans, so skip it and let
            # required stay as GraphQL-only (typically empty on free plans),
            # which makes the "no required checks" path report the check-run
            # evidence already gathered by _collect_billing_exception.
            rules = []
        else:
            rules = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}"
                         f"?per_page=100", paginate=True, profile_home=profile_home)
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
        login = f"assignee profile {assignee!r}'s gh login" if assignee else "the ambient gh login"
        receipt.update(classification="auth",
                       detail=f"GitHub refused the acceptance read ({exc}) as {login}; "
                              "fix that profile's GitHub credentials/access to the repository, then retry completion.")
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
