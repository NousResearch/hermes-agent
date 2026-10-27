"""Exact-head forge acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.

``gh`` and Gitea API reads use the card's assignee profile (``profile_home``),
not ambient credentials: :func:`_profile_credential_env` resolves that profile's
own secrets/config so a multi-profile host cannot accept with another identity
(#122689).
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")


@dataclass(frozen=True)
class _GiteaPR:
    base_url: str
    repo: str
    number: int
    url: str


@dataclass(frozen=True)
class _GiteaRepo:
    base_url: str
    repo: str


def _http_host_is_lan(hostname: str) -> bool:
    if hostname.lower().rstrip(".") == "localhost" or "." not in hostname:
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _parse_gitea_url(value: str, *, require_pr: bool) -> _GiteaPR | _GiteaRepo | None:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or
            parsed.query or parsed.fragment):
        return None
    if parsed.hostname.lower().rstrip(".") == "github.com":
        return None
    if parsed.scheme == "http" and not _http_host_is_lan(parsed.hostname):
        return None
    if port is not None and not 1 <= port <= 65535:
        return None
    parts = parsed.path.split("/")
    if require_pr:
        if len(parts) != 5 or parts[0] or parts[3] != "pulls" or not parts[4].isdigit():
            return None
        owner, repo, number_text = parts[1], parts[2], parts[4]
        if not _REPO.fullmatch(f"{owner}/{repo}") or not re.fullmatch(r"[1-9][0-9]*", number_text):
            return None
        base_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        return _GiteaPR(base_url, f"{owner}/{repo}", int(number_text), value)
    if len(parts) != 3 or parts[0]:
        return None
    owner, repo = parts[1], parts[2]
    if not _REPO.fullmatch(f"{owner}/{repo}"):
        return None
    base_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return _GiteaRepo(base_url, f"{owner}/{repo}")


def _parse_gitea_pr(value: str | None) -> _GiteaPR | None:
    parsed = _parse_gitea_url(value, require_pr=True) if isinstance(value, str) else None
    return parsed if isinstance(parsed, _GiteaPR) else None


def _parse_gitea_repo(value: str | None) -> _GiteaRepo | None:
    if not isinstance(value, str) or not value.startswith("gitea:"):
        return None
    parsed = _parse_gitea_url(value.removeprefix("gitea:"), require_pr=False)
    return parsed if isinstance(parsed, _GiteaRepo) else None


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (
        _REPO.fullmatch(value) or _PR.fullmatch(value) or
        _parse_gitea_pr(value) or _parse_gitea_repo(value)
    ):
        raise ValueError(
            "completion_contract must be local-only, OWNER/REPO, an exact GitHub PR URL, "
            "an exact Gitea PR URL, or gitea:SCHEME://HOST/OWNER/REPO"
        )
    return value


def bind_contract(contract: str, published_pr: str | None) -> str | None:
    """Bind a repository declaration to one exact PR without crossing forges."""
    github = _PR.fullmatch(published_pr or "")
    if github and contract == github[1]:
        return published_pr
    declared = _parse_gitea_repo(contract)
    published = _parse_gitea_pr(published_pr)
    if declared and published and (declared.base_url, declared.repo) == (published.base_url, published.repo):
        return published_pr
    return None


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
    """A forge refused the read as an identity/access failure, not an API blip."""


def _profile_credential_env(profile_home: str | None) -> dict[str, str] | None:
    if not profile_home:
        return None
    from tools.environments.local import _is_routed_home, hermes_subprocess_env, served_profile_child_env
    base = hermes_subprocess_env(inherit_credentials=True)
    if _is_routed_home(profile_home):
        base.pop("GH_CONFIG_DIR", None)
    return served_profile_child_env(base=base, target_home=profile_home, inherit_credentials=True)


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
    from tools.environments.local import _is_routed_home
    routed = _is_routed_home(profile_home)
    env = _profile_credential_env(profile_home)
    assert env is not None
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


def _collect_github(contract: str, published_pr: str | None,
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


def _gitea_token(profile_home: str | None) -> str | None:
    env = _profile_credential_env(profile_home) if profile_home else os.environ
    value = (env or {}).get("GITEA_TOKEN", "").strip()
    return value or None


def _gitea_api(ref: _GiteaPR, endpoint: str, *, token: str | None,
                query: dict[str, int] | None = None):
    url = f"{ref.base_url}/api/v1/{endpoint.lstrip('/')}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    headers = {"Accept": "application/json", "User-Agent": "hermes-kanban-pr-acceptance"}
    if token:
        headers["Authorization"] = f"token {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        from hermes_cli.urllib_security import open_credentialed_url
        with open_credentialed_url(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403, 404}:
            raise _GateAuthError(f"HTTP {exc.code} on /api/v1/{endpoint.split('?')[0]}") from None
        raise ValueError(f"Gitea HTTP {exc.code}") from None


def _gitea_combined_status(ref: _GiteaPR, sha: str, token: str | None) -> dict:
    statuses = []
    total = None
    page = 1
    while total is None or len(statuses) < total:
        value = _gitea_api(ref, f"repos/{ref.repo}/commits/{sha}/status",
                            token=token, query={"limit": 100, "page": page})
        if value.get("sha") != sha:
            raise ValueError("Gitea returned status evidence for a different commit")
        batch = value.get("statuses")
        if not isinstance(batch, list) or not isinstance(value.get("total_count"), int):
            raise ValueError("Gitea returned incomplete status evidence")
        total = value["total_count"]
        statuses.extend(batch)
        if not batch and len(statuses) < total:
            raise ValueError("Incomplete Gitea status pagination")
        page += 1
    if len({item["id"] for item in statuses}) != total:
        raise ValueError("Incomplete Gitea status pagination")
    return {"sha": sha, "statuses": statuses}


def _classify_gitea_status(status: dict, sha: str) -> str:
    if status.get("sha", sha) != sha:
        return "stale"
    value = str(status.get("status", "")).lower()
    if value in {"pending", "running"}:
        return "pending"
    if value == "success":
        return "success"
    if value in {"failure", "failed", "warning", "cancelled", "canceled", "skipped"}:
        return "failure"
    return "infra"


def _collect_gitea(ref: _GiteaPR, assignee: str | None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": ref.url, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    token = None
    try:
        profile_home = _assignee_profile_home(assignee)
        token = _gitea_token(profile_home)
        pr_endpoint = f"repos/{ref.repo}/pulls/{ref.number}"
        pr = _gitea_api(ref, pr_endpoint, token=token)
        if pr.get("number", ref.number) != ref.number:
            raise ValueError("Gitea returned a different pull request")
        if pr.get("html_url"):
            identity = _parse_gitea_pr(pr["html_url"])
            if not identity or (identity.repo, identity.number) != (ref.repo, ref.number):
                raise ValueError("Gitea returned a different repository or pull request")
        sha = pr["head"]["sha"]
        branch = pr["base"]["ref"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", sha):
            raise ValueError("Gitea pull request head is unavailable")
        if pr.get("state") == "closed" and not pr.get("merged"):
            receipt.update(classification="failure", detail="Gitea PR is closed without merge.")
            return receipt
        if pr.get("state") != "open" and not pr.get("merged"):
            raise ValueError("Gitea pull request state is unavailable")
        branch_info = _gitea_api(ref, f"repos/{ref.repo}/branches/{quote(branch, safe='')}", token=token)
        required = set(branch_info.get("status_check_contexts") or []) if branch_info.get("enable_status_check", branch_info.get("required_status_checks", False)) else set()
        receipt["required"] = [{"context": context, "app_id": None} for context in sorted(required)]
        if not required:
            receipt["detail"] = "No repository-required checks are configured; explicitly use a local-only contract for non-CI tasks."
            return receipt
        combined = _gitea_combined_status(ref, sha, token)
        outcomes = []
        for context in sorted(required):
            matching = [item for item in combined["statuses"] if item.get("context") == context]
            if not matching:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
                continue
            status = max(matching, key=lambda item: item["id"])
            classification = _classify_gitea_status(status, sha)
            outcomes.append(classification)
            receipt["checks"].append({"name": context, "id": status["id"],
                "url": status.get("target_url"), "head_sha": status.get("sha", sha),
                "classification": classification, "conclusion": status.get("status")})
        current = _gitea_api(ref, pr_endpoint, token=token)
        if (current["head"]["sha"] != sha or current["base"]["ref"] != branch or
                (current.get("state") == "closed" and not current.get("merged"))):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        receipt["classification"] = next(
            (item for item in outcomes if item != "success"),
            "missing" if not outcomes else "success",
        )
        receipt["ok"] = receipt["classification"] == "success"
        return receipt
    except _GateAuthError as exc:
        login = f"assignee profile {assignee!r}'s Gitea token" if assignee else "the configured Gitea token"
        token_hint = " Configure GITEA_TOKEN in that profile's .env." if token is None else ""
        receipt.update(classification="auth",
                       detail=f"Gitea refused the acceptance read ({exc}) as {login}.{token_hint}")
        return receipt
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        receipt.update(classification="infra",
                       detail="Gitea acceptance evidence unavailable or incomplete; check API access and retry.")
        return receipt


def collect_acceptance(contract: str, published_pr: str | None,
                       assignee: str | None = None) -> dict:
    declared = _parse_gitea_pr(contract)
    if declared:
        if published_pr and published_pr != contract:
            return _missing_receipt(published_pr)
        return _collect_gitea(declared, assignee)
    repo_contract = _parse_gitea_repo(contract)
    if repo_contract:
        bound = bind_contract(contract, published_pr)
        ref = _parse_gitea_pr(bound)
        return _collect_gitea(ref, assignee) if ref else _missing_receipt(published_pr)
    return _collect_github(contract, published_pr, assignee)


def _missing_receipt(published_pr: str | None) -> dict:
    return {"ok": False, "classification": "missing", "head_sha": None,
            "pr_url": published_pr, "checks": [],
            "detail": "Supply metadata.published_pr matching the persisted completion contract.",
            "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                        "Use kanban_block if human input is needed; receipts remain on the task event log."}
