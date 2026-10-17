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
# GitHub's own Actions app. Unpinned and third-party requirements are not this.
_ACTIONS_APP_ID = 15368
# Events branch protection actually evaluates. workflow_dispatch, pull_request_target,
# schedule, and other triggers are not required-check provenance.
_TRUSTED_ACTIONS_EVENTS = frozenset({"merge_group", "pull_request", "push"})


def _permission_denial(stderr: str) -> str | None:
    """Name the missing read permission, or None when the 403 is not a grant denial.

    Rate-limit and SAML refusals must not look like a Checks permission gap.
    The returned name is a permission label, never a credential.
    """
    text = stderr or ""
    lowered = text.lower()
    if "rate limit" in lowered or "saml enforcement" in lowered:
        return None
    header = re.search(r"X-Accepted-GitHub-Permissions:\s*([A-Za-z0-9_.-]+)=read", text)
    if header:
        return header.group(1)
    if "Resource not accessible by integration" in text:
        return "integration"
    return None


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
        endpoint_path = endpoint.split("?")[0]
        denied = re.search(r"HTTP (40[134])", exc.stderr or "")
        if denied:
            permission = _permission_denial(exc.stderr or "") if denied[1] == "403" else None
            raise _GateAuthError(f"HTTP {denied[1]} on {endpoint_path}", status=denied[1],
                                 endpoint=endpoint_path, permission=permission) from None
        if exc.returncode == 4:  # gh's authentication-required exit: this profile has no login
            raise _GateAuthError(f"gh has no login for {endpoint_path}", endpoint=endpoint_path) from None
        raise
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


class _GateAuthError(RuntimeError):
    """gh was refused at HTTP 401/403/404 (or GraphQL returned no repository):
    this profile's login cannot see the repo — an identity problem to fix, not
    an infrastructure blip to retry.

    ``permission`` is a checks/statuses-specific denial signal, never a token.
    """

    def __init__(self, message: str, *, status: str | None = None, endpoint: str = "",
                 permission: str | None = None):
        super().__init__(message)
        self.status = status
        self.endpoint = endpoint
        self.permission = permission


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
        _apply_actions_evidence(receipt, repo, sha, number, branch, required, profile_home)
        return receipt
    except _GateAuthError as exc:
        login = f"assignee profile {assignee!r}'s gh login" if assignee else "the ambient gh login"
        receipt.update(classification="auth",
                       detail=f"GitHub refused the acceptance read ({exc}) as {login}; "
                              "fix that profile's GitHub credentials/access to the repository, then retry completion.")
        return receipt
    except ValueError as exc:
        # Our own messages name the failed evidence phase. Never persist gh stderr.
        receipt.update(classification="infra", detail=str(exc) or "GitHub acceptance evidence unavailable or incomplete")
        return receipt
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, IndexError):
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.")
        return receipt


def _slurp_rows(pages, key: str) -> list:
    if not isinstance(pages, list) or not pages or not isinstance(pages[0], dict):
        raise ValueError(f"Incomplete {key} pagination")
    total = pages[0].get("total_count")
    rows = [row for page in pages for row in page[key]]
    if not isinstance(total, int) or len({row["id"] for row in rows}) != total:
        raise ValueError(f"Incomplete {key} pagination")
    return rows


def _trusted_workflow_path(path) -> bool:
    return (isinstance(path, str) and path.startswith(".github/workflows/")
            and ".." not in path.split("/") and (path.endswith(".yml") or path.endswith(".yaml")))


def _trusted_actions_run(run: dict, repo: str) -> None:
    """Fail closed unless this required-check event is a protected-repo workflow.

    Untrusted events are not passed here. A workflow_dispatch or
    pull_request_target run is ignored by the caller so it cannot override
    a later pull_request, push, or merge_group run.
    """
    if not _trusted_workflow_path(run.get("path")):
        raise ValueError("Actions run on the PR head lacks trusted workflow provenance")
    head_repo = run.get("head_repository")
    full_name = head_repo.get("full_name") if isinstance(head_repo, dict) else None
    if full_name != repo:
        raise ValueError("Actions run head repository is not the protected repository; untrusted provenance")
    if not isinstance(run.get("run_attempt"), int) or run["run_attempt"] < 1:
        raise ValueError("Actions run attempt is missing")


def _require_base_workflow(repo: str, branch: str, path: str, profile_home) -> None:
    """A workflow that exists only on the PR head is not an approved base-branch workflow."""
    try:
        body = _api(f"repos/{repo}/contents/{quote(path, safe='/')}?ref={quote(branch, safe='')}",
                    profile_home=profile_home)
    except _GateAuthError as exc:
        if exc.status == "404":
            raise ValueError("Actions workflow is not on the protected base branch; unknown provenance") from None
        raise
    if not isinstance(body, dict) or body.get("type") != "file" or body.get("path") != path:
        raise ValueError("Actions workflow is not a file on the protected base branch; unknown provenance")


def _finish_receipt(receipt, outcomes, *, repo, number, sha, branch, profile_home):
    # Re-read after all pages: old-head successes are never transferable.
    current = _api(f"repos/{repo}/pulls/{number}", profile_home=profile_home)
    if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
        receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.", ok=False)
        return receipt
    receipt["classification"] = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
    receipt["ok"] = receipt["classification"] == "success"
    return receipt


def _apply_actions_evidence(receipt, repo, sha, number, branch, required, profile_home):
    """Prove only all-Actions-pinned requirements from exact-head workflow jobs.

    Fail closed unless every requirement is the GitHub Actions app and each
    required job's latest trusted run succeeded on that exact SHA. An older
    workflow_dispatch or other untrusted event is not evidence and cannot
    override that run. Within the selected run, every same-named job must pass.
    """
    foreign = [context if app_id in (None, -1) else f"{context} (app {app_id})"
               for context, app_id in sorted(required, key=str) if app_id != _ACTIONS_APP_ID]
    if foreign:
        # A repository-rule mismatch, not a transient failure: retrying cannot change it.
        receipt.update(classification="infra", evidence_source="actions", ok=False,
                       recovery="Change the base branch's required checks, then complete again; do not rerun CI for this.",
                       detail=(f"Actions-only acceptance cannot prove mixed or unpinned required checks. Pin each "
                               f"to GitHub Actions (app {_ACTIONS_APP_ID}) in the base branch ruleset or protection, "
                               "or use a local-only contract if another app must gate merges. Checks: "
                               + ", ".join(foreign[:3]) + (f" and {len(foreign) - 3} more." if len(foreign) > 3 else ".")))
        return receipt
    try:
        runs = _slurp_rows(_api(
            f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100",
            paginate=True, profile_home=profile_home), "workflow_runs")
        trusted = []
        skipped_untrusted = False
        for run in runs:
            if run.get("head_sha") != sha:
                continue
            if run.get("event") not in _TRUSTED_ACTIONS_EVENTS:
                skipped_untrusted = True
                continue
            _trusted_actions_run(run, repo)
            trusted.append(run)
        if not trusted and skipped_untrusted:
            receipt.update(classification="infra", evidence_source="actions", ok=False,
                           detail=("No trusted Actions run on the PR head. workflow_dispatch, "
                                   "pull_request_target, and other untrusted events are not "
                                   "required-check evidence."))
            return receipt
        for path in sorted({run["path"] for run in trusted}):
            _require_base_workflow(repo, branch, path, profile_home)
        jobs = []
        for run in trusted:
            run_jobs = _slurp_rows(_api(
                f"repos/{repo}/actions/runs/{int(run['id'])}/jobs?filter=latest&per_page=100",
                paginate=True, profile_home=profile_home), "jobs")
            for job in run_jobs:
                if job.get("run_attempt") != run["run_attempt"]:
                    continue
                jobs.append((run, job))
    except _GateAuthError as exc:
        receipt.update(classification="auth", evidence_source="actions", ok=False,
                       detail=(f"GitHub refused Actions evidence ({exc}); Actions cannot prove the required jobs."))
        return receipt
    outcomes = []
    receipt["evidence_source"] = "actions"
    by_name: dict[str, list] = {}
    for run, job in jobs:
        by_name.setdefault(job.get("name"), []).append((run, job))
    for context, _app_id in sorted(required, key=str):
        selected = by_name.get(context) or []
        if selected:
            latest = max(int(run["id"]) for run, _job in selected)
            selected = [(run, job) for run, job in selected if int(run["id"]) == latest]
        if not selected:
            outcomes.append("missing")
            receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha,
                                      "evidence_source": "actions"})
        for run, job in selected:
            outcome = job.get("conclusion")
            classification = _classify(job, sha, outcome, True)
            outcomes.append(classification)
            receipt["checks"].append({"name": context, "id": job.get("id"), "url": job.get("html_url"),
                                      "head_sha": job.get("head_sha"), "classification": classification,
                                      "conclusion": outcome, "evidence_source": "actions",
                                      "workflow": run.get("path"), "run_attempt": run.get("run_attempt")})
    return _finish_receipt(receipt, outcomes, repo=repo, number=number, sha=sha,
                           branch=branch, profile_home=profile_home)


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
