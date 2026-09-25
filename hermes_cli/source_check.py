"""Passive source update decisions for one exact installation and profile.

No fetch, lock repair, or Git writes. Desktop, CLI, and dashboard share this owner.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote
import urllib.error
import urllib.request

from hermes_constants import get_hermes_home
from hermes_cli.source_releases import OFFICIAL_REPOSITORY, _GITHUB_ORIGIN, resolve_source_target

logger = logging.getLogger(__name__)
UPDATE_AVAILABLE_NO_COUNT = -1
_UPDATE_CHECK_CACHE_SECONDS = 24 * 3600
_UPDATE_CHECK_FAILURE_CACHE_SECONDS = 3600


def _quiet(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def source_git_env() -> dict[str, str]:
    """Keep read-only Git probes in their explicit cwd, not an inherited worktree.

    No probe may lazy-fetch from a partial clone's promisor remote: asking about
    an upstream tip the clone never fetched would download its history, and the
    probe timeout kills only git itself, orphaning the fetch (see NO_LAZY_FETCH_ENV).
    """
    from hermes_cli._subprocess_compat import NO_LAZY_FETCH_ENV, noninteractive_git_env

    env = noninteractive_git_env()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_SHALLOW_FILE", "GIT_NAMESPACE"):
        env.pop(key, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.update(NO_LAZY_FETCH_ENV)
    return env


_GIT_TEXT_KW = {"text": True, "encoding": "utf-8", "errors": "replace"}


def _git_run(args: list[str], *, cwd: Optional[Path] = None, timeout: int = 5, text: bool = True,
             git: str = "git"):
    """Read Git state without prompts, optional index writes, or inherited targeting."""
    from hermes_cli._subprocess_compat import windows_hide_flags

    kwargs: dict = {"creationflags": windows_hide_flags(), "env": source_git_env(), "stdin": subprocess.DEVNULL}
    try:
        return subprocess.run(
            [git, *args], capture_output=True, timeout=timeout, cwd=str(cwd) if cwd is not None else None,
            **(_GIT_TEXT_KW if text else {}), **kwargs)
    except Exception:
        return None


def _git_stdout(args: list[str], *, cwd: Path, timeout: int = 5, git: str = "git") -> Optional[str]:
    result = _git_run(args, cwd=cwd, timeout=timeout, git=git)
    if result is None or result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _git_ok(args: list[str], **kw) -> bool:
    """True when ``git <args>`` ran and exited 0 (output discarded)."""
    result = _git_run(args, text=False, **kw)
    return result is not None and result.returncode == 0


def _git_count(args: list[str], *, cwd: Path) -> Optional[int]:
    """``int`` of a successful ``git rev-list --count``-style command, else None."""
    result = _git_run(args, cwd=cwd)
    if result is not None and result.returncode == 0:
        return _quiet(lambda: int(result.stdout.strip()))
    return None


def _is_full_sha(value: Optional[str]) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdefABCDEF" for c in value)


def _github_compare(current_rev: str, target_rev: str, repository: str = OFFICIAL_REPOSITORY) -> Optional[dict]:
    # Do not memoize this separately: force must bypass failed AND successful network results.
    if not (_is_full_sha(current_rev) and _is_full_sha(target_rev)):
        return None
    payload = _quiet(lambda: json.loads(_request(
        f"https://api.github.com/repos/{repository}/compare/{current_rev}...{target_rev}")))
    return payload if isinstance(payload, dict) else None


def _github_compare_behind(current_rev: str, target_rev: str, repository: str = OFFICIAL_REPOSITORY) -> Optional[int]:
    payload = _github_compare(current_rev, target_rev, repository)
    ahead = payload.get("ahead_by") if payload else None
    return ahead if isinstance(ahead, int) and not isinstance(ahead, bool) and ahead >= 0 else None


def _request(url: str, accept: str = "application/vnd.github+json") -> str:
    """GET an api.github.com resource with the credential ladder in hermes_cli.github_api.

    A token GitHub rejects (401) drops this request to anonymous rather than
    failing the check on a stale credential.
    """
    from hermes_cli.github_api import github_token

    token = github_token()
    try:
        return _request_with(url, accept, token)
    except urllib.error.HTTPError as exc:
        if token is None or exc.code != 401:
            raise
        logger.debug("GitHub rejected the configured token; retrying anonymously")
        return _request_with(url, accept, None)


def _request_with(url: str, accept: str, token: str | None) -> str:
    headers = {"Accept": accept, "User-Agent": "hermes-update-check"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read(2 * 1024 * 1024).decode("utf-8-sig").strip()


def _branch_tip(repository: str | None, branch: str, root: Path, git: str,
                remote: str = "origin") -> tuple[str | None, bool, str | None]:
    """``(sha, missing, failure)``: ``missing`` only on a confirmed empty advertisement;
    ``failure`` names why no tip could be read, for the user-facing message."""
    # A successful empty ref advertisement alone proves a branch was deleted.
    # GitHub 404 can also mean a private repository: it must not heal a branch.
    failure = None
    if repository:
        from hermes_cli.github_api import describe_github_failure, github_token
        try:
            sha = _request(f"https://api.github.com/repos/{repository}/commits/{quote(branch, safe='')}",
                           "application/vnd.github.sha")
        except Exception as exc:
            sha = None
            failure = describe_github_failure(exc, authenticated=github_token() is not None)
        if _is_full_sha(sha):
            return sha, False, None
        if failure is None:
            failure = "api.github.com returned no commit for the branch."
        if branch == "main" and remote == "origin":
            return None, False, failure
    result = _git_run(["ls-remote", "--exit-code", "--heads", remote, f"refs/heads/{branch}"],
                      cwd=root, git=git, timeout=10)
    if result is None:
        return None, False, failure or f"`git ls-remote {remote}` could not run."
    sha = result.stdout.split()[0] if result.returncode == 0 and result.stdout else None
    if _is_full_sha(sha):
        return sha, False, None
    if result.returncode == 2:
        return None, True, None
    detail = (result.stderr or "").strip().splitlines()
    return None, False, failure or (f"`git ls-remote {remote}` failed: {detail[-1]}" if detail
                                    else f"`git ls-remote {remote}` returned no tip.")


def _commits(payload: dict | None) -> list[dict]:
    from datetime import datetime
    rows = []
    entries = (payload or {}).get("commits", [])
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("sha"), str):
            continue
        commit = entry.get("commit") or {}
        when = (commit.get("committer") or {}).get("date") or ""
        at = _quiet(lambda: int(datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp() * 1000), 0)
        rows.append({"sha": entry["sha"], "summary": str(commit.get("message", "")).split("\n", 1)[0],
                     "author": str((commit.get("author") or {}).get("name", "")), "at": at})
    return rows[::-1]


@dataclass(frozen=True)
class _Checkout:
    """Read-only facts about the checkout under test, gathered once per check."""
    root: Path
    git: str
    embedded: Optional[str]
    head: Optional[str]
    current_branch: Optional[str]
    origin: str
    repository: Optional[str]
    dirty: bool


def _read_json(path: Path):
    return _quiet(lambda: json.loads(path.read_text(encoding="utf-8-sig")))


def _unsupported_reason(stamp: dict, root: Path, *, explicit_root: bool, embedded: Optional[str]) -> Optional[dict]:
    """Fields explaining why this install cannot self-update from Git, or None when it can."""
    from hermes_cli.config import detect_install_method
    from hermes_cli.update_contract import COMMIT_BUILD_UPDATE_MESSAGE

    if stamp.get("source") == "commit-build":
        return {"reason": "commit-build", "message": COMMIT_BUILD_UPDATE_MESSAGE}
    if stamp.get("payload") in {"bundled", "light", "runtime"} or (
            not explicit_root and detect_install_method(root) in {"docker", "apt"}):
        return {"reason": "not-a-git-checkout"}
    if not embedded and not (root / ".git").exists():
        return {"reason": "not-a-git-checkout", "message": "This install has no git checkout to update."}
    if stamp.get("updateMechanism") not in (None, "self") and not embedded:
        return {"reason": "update-root-steward-owned-git-tree",
                "message": "This installation is managed by its install method; use its updater.", "advice": "git pull"}
    return None


def _read_checkout(root: Path, git: str, embedded: Optional[str]) -> _Checkout:
    # An embedded revision has no checkout to ask: it always tracks the official repository.
    head = embedded or _git_stdout(["rev-parse", "HEAD"], cwd=root, git=git)
    current_branch = None if embedded else _git_stdout(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root, git=git)
    origin = "" if embedded else (_git_stdout(["remote", "get-url", "origin"], cwd=root, git=git) or "")
    match = _GITHUB_ORIGIN.fullmatch(origin)
    repository = OFFICIAL_REPOSITORY if embedded else (match[1] if match else None)
    dirty = False if embedded else bool(_git_stdout(["status", "--porcelain"], cwd=root, git=git))
    return _Checkout(root, git, embedded, head, current_branch, origin, repository, dirty)


def _configured_branch(desktop_config) -> Optional[str]:
    value = desktop_config.get("branch") if isinstance(desktop_config, dict) else None
    return (value.strip() or None) if isinstance(value, str) else None


def _checked_out_branch(current_branch: Optional[str], fallback: Optional[str]) -> Optional[str]:
    """The checkout's branch, or ``fallback`` when detached or unreadable."""
    return current_branch if current_branch and current_branch != "HEAD" else fallback


def _cached_status(cache_file: Path, identity: dict, now: float) -> Optional[dict]:
    """A still-fresh supported status cached for exactly this identity, else None.

    Failures expire sooner so a transient network error does not hide updates for a day.
    """
    cached = _read_json(cache_file)
    if not (isinstance(cached, dict) and cached.get("identity") == identity
            and isinstance(cached.get("status"), dict) and cached["status"].get("supported") is True):
        return None
    status = cached.get("status", {})
    ttl = _UPDATE_CHECK_FAILURE_CACHE_SECONDS if status.get("error") else _UPDATE_CHECK_CACHE_SECONDS
    ts = cached.get("ts")
    return status if isinstance(ts, (float, int)) and 0 <= now - ts < ttl else None


def _write_cache(cache_file: Path, identity: dict, now: float, result: dict) -> None:
    try:
        from utils import atomic_json_write
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(cache_file, {"identity": identity, "ts": now, "status": result})
    except OSError as exc:
        logger.debug("Could not cache source check: %s", exc)


def _resolve_channel(result: dict, channel: str, co: _Checkout):
    """Resolve a release channel's target into ``result``; the SourceTarget, or None on error.

    A target with a pinned commit is final; one without names a branch to follow instead.
    """
    try:
        source_target = resolve_source_target(channel, [co.git] if not co.embedded else None, co.root,
                                              repository=co.repository or OFFICIAL_REPOSITORY)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result.update(error="release-unavailable", message=f"Could not resolve the {channel} source channel: {exc}")
        return None
    if source_target.commit:
        target = source_target.commit
        result.pop("branch", None)
        result.update(channel=channel, targetSha=target, updateAvailable=co.head != target,
                      behind=0 if co.head == target else UPDATE_AVAILABLE_NO_COUNT,
                      sourceVersion=source_target.version, buildId=source_target.build_id)
        if source_target.retired:
            result["retirement"] = {"destination": source_target.channel, "sourceOnly": True}
    return source_target


def _branch_remote(co: _Checkout, selected_branch: str) -> str:
    official_ssh = (co.repository and co.repository.lower() == OFFICIAL_REPOSITORY.lower()
                    and co.origin.lower().startswith(("git@", "ssh://")))
    # The public official repo does not require the user's SSH credentials.
    # Forks must keep their own origin, including its authentication.
    return (f"https://github.com/{OFFICIAL_REPOSITORY}.git"
            if co.embedded or (official_ssh and selected_branch != "main") else "origin")


def _heal_deleted_branch(branch_config_path: Path, desktop_config: dict) -> None:
    """Point the Desktop branch setting back at main after its branch was deleted upstream."""
    # Do not overwrite a concurrent choice made while the network probe ran.
    if _read_json(branch_config_path) == desktop_config:
        from utils import atomic_json_write
        atomic_json_write(branch_config_path, {**desktop_config, "branch": "main"})


def _behind_count(co: _Checkout, target: str) -> tuple[int, list[dict]]:
    """``(behind, commits)`` for ``target``: local ancestry first, then the GitHub compare API."""
    if co.head == target or (not co.embedded and _git_ok(
            ["merge-base", "--is-ancestor", target, co.head], cwd=co.root, git=co.git)):
        return 0, []
    if co.repository:
        payload = _github_compare(co.head, target, co.repository)
        ahead = (payload or {}).get("ahead_by")
        if isinstance(ahead, int) and not isinstance(ahead, bool) and ahead >= 0:
            return ahead, (_quiet(lambda: _commits(payload), []) if ahead else [])
    return UPDATE_AVAILABLE_NO_COUNT, []


def _check_branch(result: dict, co: _Checkout, selected_branch: str, *,
                  heal: Optional[tuple[Path, dict]]) -> None:
    """Compare the checkout with ``selected_branch``'s remote tip, falling back to main if it was deleted."""
    result["branch"] = selected_branch
    remote = _branch_remote(co, selected_branch)
    target, missing, failure = _branch_tip(co.repository, selected_branch, co.root, co.git, remote)
    if missing and selected_branch != "main":
        result["branch"] = "main"
        if heal:
            _heal_deleted_branch(*heal)
        target, _, failure = _branch_tip(co.repository, "main", co.root, co.git, remote if co.embedded else "origin")
    if target is None:
        result.update(error="fetch-failed",
                      message=f"Could not resolve the remote branch tip: {failure}" if failure
                      else "Could not resolve the remote branch tip.")
        return
    behind, commits = _behind_count(co, target)
    result["commits"] = commits
    result.update(targetSha=target, behind=behind, updateAvailable=behind != 0)


def check_for_updates(*, install_root: Path | None = None, home: Path | None = None,
                      branch: str | None = None, channel: str | None = None,
                      cache_path: Path | None = None, branch_config_path: Path | None = None,
                      force: bool = False,
                      passive: bool = False, git: str = "git") -> dict:
    """Return a presentation-ready status. Omitted branch follows the current checkout.

    Only the default (running installation) may use HERMES_REVISION. An explicit
    target must never inherit the host process's embedded revision or stamp.
    """
    from hermes_cli.config import get_project_root, require_readable_config_before_write
    from hermes_cli.steward import read_install_stamp
    from hermes_cli.update_channel import install_id, resolve_update_channel
    from hermes_cli.release_channels import validate_name

    embedded = (os.environ.get("HERMES_REVISION") or None) if install_root is None else None
    root = Path(install_root if install_root is not None else get_project_root()).resolve()
    home = Path(home if home is not None else get_hermes_home()).resolve()
    result = {"supported": False, "hermesRoot": str(root), "behind": None, "commits": []}
    unsupported = _unsupported_reason(read_install_stamp(root), root,
                                      explicit_root=install_root is not None, embedded=embedded)
    if unsupported:
        return {**result, **unsupported}
    config = require_readable_config_before_write(home / "config.yaml")
    if passive and (config.get("updates") or {}).get("check") is False:
        return {**result, "reason": "disabled"}
    channel = resolve_update_channel(config, root) if channel is None else validate_name(channel)
    co = _read_checkout(root, git, embedded)
    desktop_config = _read_json(branch_config_path) if branch_config_path else None
    configured_branch = _configured_branch(desktop_config)
    updates = config.get("updates")
    # An opted-in maintained branch receives main's updates without leaving the
    # checkout. Probe that source, not a local-only branch absent from GitHub.
    in_place_main = (channel == "main" and isinstance(updates, dict)
                     and updates.get("parked_branch_strategy") == "update_in_place")
    check_branch = None if in_place_main else co.current_branch
    selected_branch = branch or configured_branch or _checked_out_branch(check_branch, "main")
    result.update(supported=True, currentSha=co.head, currentBranch=co.current_branch, dirty=co.dirty)
    if channel != "main":
        result["channel"] = channel
    else:
        result["branch"] = selected_branch
    identity = {"root": str(root), "home": str(home), "head": co.head, "origin": co.origin, "branch": selected_branch,
                "channel": channel, "embedded": embedded, "branchOverride": branch is not None, "channelProtocol": 1}
    cache_file = Path(cache_path) if cache_path is not None else home / "source-checks" / f"{install_id(root)}.json"
    now = time.time()
    cached = None if force else _cached_status(cache_file, identity, now)
    if cached is not None:
        return {**cached, "dirty": co.dirty, "currentBranch": co.current_branch}
    result["fetchedAt"] = int(now * 1000)
    source_target = None
    if not _is_full_sha(co.head):
        result.update(error="head-unavailable", message="Could not read the installed revision.")
    elif branch is None:
        source_target = _resolve_channel(result, channel, co)
        if source_target is not None and not source_target.commit:
            # Explicit pins still win; in-place main follows its update source.
            selected_branch = configured_branch or _checked_out_branch(check_branch, source_target.branch)
    if "error" not in result and (source_target is None or source_target.branch is not None):
        # Only a Desktop-configured branch the caller did not override is healed.
        heal = branch_config_path and not branch and configured_branch == selected_branch
        _check_branch(result, co, selected_branch,
                      heal=(branch_config_path, desktop_config) if heal else None)
    _write_cache(cache_file, identity, now, result)
    return result


def main() -> None:
    import argparse
    import contextlib
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--git", default="git")
    parser.add_argument("--branch")
    from hermes_cli.release_channels import validate_name
    parser.add_argument("--channel", type=validate_name)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--branch-config-path", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        result = check_for_updates(**vars(args))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
