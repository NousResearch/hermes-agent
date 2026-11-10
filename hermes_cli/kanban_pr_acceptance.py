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
from typing import Any
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _json_documents(text: str) -> list:
    """Decode ``gh api --paginate`` stdout: one JSON document per page, concatenated.

    ``--slurp`` would be tidier but only exists from gh 2.57 (Ubuntu 24.04 ships
    2.46): on older installs the entire call dies with "unknown flag: --slurp",
    which no caller can classify as anything but a retryable infra failure.
    Decoding concatenated documents works on every gh version.
    """
    documents: list = []
    decoder = json.JSONDecoder()
    index = 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            return documents
        document, index = decoder.raw_decode(text, index)
        documents.append(document)


# A private repository on a plan without branch protection/rules answers this
# endpoint with 403 "Upgrade to GitHub Pro" (404 where rulesets are not
# available). "No rules exist" is an answer, not an auth failure to retry.
_NO_RULES = re.compile(r"Upgrade to GitHub Pro|HTTP (404|410)\b|not available", re.IGNORECASE)


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False,
         profile_home: str | None = None, missing_ok: bool = False) -> Any:
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30,
                                check=True, env=_gh_env(profile_home))
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if missing_ok and _NO_RULES.search(stderr):
            return []
        # 401/403/404 = the login cannot see this repository (wrong profile identity
        # or missing grant), not a transient API failure. Persist only the status
        # code + endpoint, never gh's stderr (credentials/host details).
        denied = re.search(r"HTTP (40[134])", stderr)
        if denied:
            raise _GateAuthError(f"HTTP {denied[1]} on {endpoint.split('?')[0]}") from None
        if exc.returncode == 4:  # gh's authentication-required exit: this profile has no login
            raise _GateAuthError(f"gh has no login for {endpoint.split('?')[0]}") from None
        raise
    if paginate:
        return _json_documents(result.stdout)
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


class _GateAuthError(RuntimeError):
    """gh was refused at HTTP 401/403/404 (or GraphQL returned no repository):
    this profile's login cannot see the repo — an identity problem to fix, not
    an infrastructure blip to retry."""


# Where `gh auth login` may have stored this profile's credentials: its default
# XDG location first, then the legacy profile-owned directory the gate pins.
_GH_CONFIG_SUBDIRS = ((".config", "gh"), ("gh",))


def _profile_gh_config_dir(profile_home: str) -> str:
    """The profile's gh config dir, preferring one that actually has a login.

    ``gh auth login`` writes ``~/.config/gh/hosts.yml``; pinning the child to
    ``<profile_home>/gh`` alone made every normally-onboarded profile look
    unauthenticated ("no login for graphql"). Fall back to the pinned dir when
    neither holds a login, so the child still fails closed instead of borrowing
    the launch user's ``~/.config/gh``.
    """
    home = Path(profile_home)
    for parts in _GH_CONFIG_SUBDIRS:
        candidate = home.joinpath(*parts)
        if (candidate / "hosts.yml").is_file():
            return str(candidate)
    return str(home / "gh")


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
        env["GH_CONFIG_DIR"] = _profile_gh_config_dir(profile_home)
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
    receipt = {"ok": False, "classification": "missing", "head_sha": None, "state": None,
               "merge_sha": None, "merged": False, "pr_url": published_pr, "checks": [],
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
        pull = ("headRefOid baseRefName state mergeCommit{oid} "
                "baseRef{branchProtectionRule{requiredStatusChecks{context app{databaseId}}}}")
        query = ("{repository(owner:" + json.dumps(owner) + ",name:" + json.dumps(name) + ")"
                 "{pullRequest(number:" + str(number) + "){" + pull + "}}}")
        repository = _api("graphql", query=query, profile_home=profile_home)["data"]["repository"]
        if repository is None:
            # A private repo the login cannot read resolves to null, not an error.
            raise _GateAuthError(f"HTTP 404 on graphql {repo}")
        pr = repository["pullRequest"]
        sha, branch, state = pr["headRefOid"], pr["baseRefName"], pr["state"]
        merged = state == "MERGED"
        merge_sha = ((pr.get("mergeCommit") or {}).get("oid") or None) if merged else None
        receipt["head_sha"] = sha
        receipt["state"] = state
        receipt["merge_sha"] = merge_sha
        receipt["merged"] = merged
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or state not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        required = {(r["context"], (r.get("app") or {}).get("databaseId")) for r in protection.get("requiredStatusChecks", [])}
        # missing_ok: a private repo on a plan without branch protection/rules
        # answers 403 "Upgrade to GitHub Pro" here — that means "no rules", not
        # "your login is wrong".
        rules = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100",
                     paginate=True, profile_home=profile_home, missing_ok=True)
        for page in rules:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        if not required:
            # No CI gate is configured on the target ref. This is the normal
            # fork-PR path: the fork has no branch protection, so there is no
            # required check-run to wait for. The PR is already verified OPEN or
            # MERGED with a resolvable head above, so accept it instead of
            # leaving ok=False (which would strand a perfectly valid fork PR).
            receipt["detail"] = ("No repository-required checks are configured on the "
                                 "target ref; accepting OPEN/MERGED PR without a CI gate.")
            receipt["classification"] = "success"
            receipt["ok"] = True
            return receipt

        def evidence(target_sha: str) -> tuple[list, list]:
            """(check runs, statuses) published for one commit, each paired with its sha."""
            pages = _api(f"repos/{repo}/commits/{target_sha}/check-runs?per_page=100&filter=latest",
                         paginate=True, profile_home=profile_home)
            runs = [run for page in pages for run in page["check_runs"]]
            if len({r["id"] for r in runs}) != pages[0]["total_count"]:
                raise ValueError("Incomplete check-run pagination")
            statuses = [{**s, "sha": target_sha}
                        for page in _api(f"repos/{repo}/commits/{target_sha}/statuses?per_page=100",
                                         paginate=True, profile_home=profile_home)
                        for s in page]
            return [(run, target_sha) for run in runs], [(status, target_sha) for status in statuses]

        evidence_runs, evidence_statuses = evidence(sha)
        if merge_sha and merge_sha != sha:
            # GitHub re-runs required checks on the merged commit; for an
            # already-merged PR that is the evidence that still answers "did
            # this land green".
            merged_runs, merged_statuses = evidence(merge_sha)
            evidence_runs += merged_runs
            evidence_statuses += merged_statuses
        outcomes = []
        for context, app_id in sorted(required, key=str):
            matching = [(run, esha) for run, esha in evidence_runs if run["name"] == context and
                        (app_id in (None, -1) or run["app"]["id"] == app_id)]
            # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
            legacy = [(status, esha) for status, esha in evidence_statuses
                      if status["context"] == context] if app_id in (None, -1) else []
            # Head-commit evidence wins; the merged commit is only a fallback.
            matching.sort(key=lambda item: item[1] != sha)
            if legacy:
                pool = [item for item in legacy if item[1] == sha] or legacy
                legacy = [max(pool, key=lambda item: item[0]["id"])]
            selected = matching + legacy
            if not selected:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
            for check, check_sha in selected:
                is_run = "conclusion" in check
                outcome = check.get("conclusion") if is_run else check["state"]
                classification = _classify(check, check_sha, outcome, is_run)
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
        classification = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
        if classification != "success" and merged and merge_sha:
            # The PR is already merged: its head evidence can no longer change
            # and nothing is left to fix or retry, so a missing or pending
            # required check must not strand the card (that defect forced
            # operator archives). Accept the merged state and keep every
            # observed outcome in the receipt for audit.
            receipt["classification"] = "success"
            receipt["ok"] = True
            receipt["merged_override"] = "evidence-unavailable"
            receipt["detail"] = (f"PR is already MERGED (merge commit {merge_sha}); accepting the merged state "
                                 f"instead of re-deriving head evidence. Observed required-check outcomes: "
                                 f"{sorted(set(x for x in outcomes if x)) or ['none']}.")
            return receipt
        receipt["classification"] = classification
        receipt["ok"] = classification == "success"
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
