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
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")


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


_PROBE_TTL_SECONDS = 300
_probe_cache: dict[tuple[str, str | None], tuple[float, bool, str]] = {}


def _required_checks(repo: str, profile_home: str | None) -> tuple[str, int]:
    """``(default_branch, required_check_count)`` for *repo*, counted exactly as
    :func:`collect_acceptance` counts them: the base ref's branch protection plus
    every ``required_status_checks`` rule on that branch. Raises on anything that
    is not a clean reading, so an unexpected shape never reads as "zero"."""
    owner, name = repo.split("/")
    query = (f'{{repository(owner:{json.dumps(owner)},name:{json.dumps(name)})'
             "{defaultBranchRef{name branchProtectionRule"
             "{requiredStatusChecks{context}}}}}}")
    repository = _api("graphql", query=query, profile_home=profile_home)["data"]["repository"]
    if not isinstance(repository, dict):
        raise _GateAuthError(f"repository {repo} is not visible to this login")
    ref = repository.get("defaultBranchRef")
    if not isinstance(ref, dict) or not ref.get("name"):
        raise ValueError(f"default branch of {repo} is unavailable")
    branch = ref["name"]
    protection = ref.get("branchProtectionRule") or {}
    required = len(protection.get("requiredStatusChecks") or [])
    pages = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100",
                 paginate=True, profile_home=profile_home)
    for page in pages:
        for rule in page:
            if rule.get("type") == "required_status_checks":
                required += len(rule["parameters"]["required_status_checks"])
    return branch, required


def _gate_path(profile_home: str | None) -> str:
    """The ``PATH`` the gate's own ``gh`` spawn resolves against, so a missing gh
    is judged in the environment completion would really use — never the ambient
    shell's."""
    env = _gh_env(profile_home)
    return (env or os.environ).get("PATH") or os.environ.get("PATH", "")


def _probe_contract(contract: str, assignee: str | None) -> tuple[bool, str]:
    """The uncached probe behind :func:`contract_is_usable`."""
    try:
        profile_home = _assignee_profile_home(assignee)
    except _GateAuthError as exc:
        return True, f"unverified: {exc}"
    if shutil.which("gh", path=_gate_path(profile_home)) is None:
        return False, ("PR acceptance evidence is read with the gh CLI, which the "
                       "gate's environment cannot find, so no PR-form contract can "
                       "ever pass here")
    declared = _PR.fullmatch(contract)
    repo = declared[1] if declared else contract
    try:
        branch, required = _required_checks(repo, profile_home)
    except _GateAuthError as exc:
        return True, f"unverified: {exc}"
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # A transient failure is not evidence of a broken contract.
        return True, "unverified: the required-check probe could not complete"
    if not required:
        return False, (f"repository {repo} configures no required status checks on its "
                       f"default branch {branch!r}, so PR acceptance has no evidence to "
                       "gate on; declare local-only and verify by hand, or add the checks")
    return True, ""


def settle_contract(contract: str | None, assignee: str | None = None) -> tuple[str, dict]:
    """``(contract, payload)`` for card creation. A PR form that can never pass
    acceptance is rewritten to ``local-only`` and ``payload`` carries the
    ``contract_downgrade`` note for the ``created`` event; a usable declaration
    returns an empty payload. A failed probe is never allowed to fail creation."""
    contract = validate_contract(contract)
    if contract == "local-only":
        return contract, {}
    try:
        usable, reason = contract_is_usable(contract, assignee)
    except Exception:  # health: allow BLE001 -- creation must survive a failed probe
        logger.exception("completion-contract probe failed; keeping %r", contract)
        return contract, {}
    if usable:
        return contract, {}
    note = {"from": contract, "to": "local-only", "reason": reason}
    return "local-only", {"contract_downgrade": note}


def contract_is_usable(contract: str, assignee: str | None = None) -> tuple[bool, str]:
    """Can a PR-form *contract* ever pass acceptance? Returns ``(usable, reason)``.

    Only a DEFINITIVE negative counts, because the caller rewrites the declaration
    on ``False``: this host has no ``gh`` to collect evidence with, or the repo
    configures no required status checks at all. Anything uncertain — a transient
    API failure, a login that cannot see the repo, an unexpected response shape —
    fails open (``True``), leaving the completion gate to report the real reason
    rather than silently discarding a creator's explicit contract.
    """
    if contract == "local-only":
        return True, ""
    key = (contract, assignee)
    cached = _probe_cache.get(key)
    now = time.monotonic()
    if cached and now - cached[0] < _PROBE_TTL_SECONDS:
        return cached[1], cached[2]
    usable, reason = _probe_contract(contract, assignee)
    _probe_cache[key] = (now, usable, reason)
    return usable, reason


def collect_acceptance(contract: str, published_pr: str | None,
                       assignee: str | None = None) -> dict:
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
        query = f'{{repository(owner:{json.dumps(owner)},name:{json.dumps(name)}){{pullRequest(number:{number:d}){{headRefOid baseRefName state\n            baseRef{{branchProtectionRule{{requiredStatusChecks{{context app{{databaseId}}}}}}}}}}}}}}'
        repository = _api("graphql", query=query, profile_home=profile_home)["data"]["repository"]
        if repository is None:
            # A private repo the login cannot read resolves to null, not an error.
            raise _GateAuthError(f"HTTP 404 on graphql {repo}")
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
