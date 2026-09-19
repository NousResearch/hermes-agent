"""Passive source update decisions for one exact installation and profile.

No fetch, lock repair, or Git writes. Desktop, CLI, and dashboard share this owner.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote
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
    """Keep read-only Git probes in their explicit cwd, not an inherited worktree."""
    from hermes_cli._subprocess_compat import noninteractive_git_env

    env = noninteractive_git_env()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_SHALLOW_FILE", "GIT_NAMESPACE"):
        env.pop(key, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
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
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "hermes-update-check"})
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.read(2 * 1024 * 1024).decode("utf-8-sig").strip()


def _branch_tip(repository: str | None, branch: str, root: Path, git: str, remote: str = "origin") -> tuple[str | None, bool]:
    # A successful empty ref advertisement alone proves a branch was deleted.
    # GitHub 404 can also mean a private repository: it must not heal a branch.
    if repository:
        sha = _quiet(lambda: _request(
            f"https://api.github.com/repos/{repository}/commits/{quote(branch, safe='')}",
            "application/vnd.github.sha"))
        if _is_full_sha(sha):
            return sha, False
        if branch == "main" and remote == "origin":
            return None, False
    result = _git_run(["ls-remote", "--exit-code", "--heads", remote, f"refs/heads/{branch}"],
                      cwd=root, git=git, timeout=10)
    if result is None:
        return None, False
    sha = result.stdout.split()[0] if result.returncode == 0 and result.stdout else None
    return (sha if _is_full_sha(sha) else None), result.returncode == 2


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


def check_for_updates(*, install_root: Path | None = None, home: Path | None = None,
                      branch: str | None = None, channel: str | None = None,
                      cache_path: Path | None = None, branch_config_path: Path | None = None,
                      force: bool = False,
                      passive: bool = False, git: str = "git") -> dict:
    """Return a presentation-ready status. Omitted branch follows the current checkout.

    Only the default (running installation) may use HERMES_REVISION. An explicit
    target must never inherit the host process's embedded revision or stamp.
    """
    from hermes_cli.config import detect_install_method, get_project_root, require_readable_config_before_write
    from hermes_cli.steward import read_install_stamp
    from hermes_cli.update_channel import install_id, resolve_update_channel
    from hermes_cli.release_channels import validate_name
    from hermes_cli.update_contract import COMMIT_BUILD_UPDATE_MESSAGE

    embedded = (os.environ.get("HERMES_REVISION") or None) if install_root is None else None
    root = Path(install_root if install_root is not None else get_project_root()).resolve()
    home = Path(home if home is not None else get_hermes_home()).resolve()
    stamp = read_install_stamp(root)
    result = {"supported": False, "hermesRoot": str(root), "behind": None, "commits": []}
    if stamp.get("source") == "commit-build":
        return {**result, "reason": "commit-build", "message": COMMIT_BUILD_UPDATE_MESSAGE}
    if stamp.get("payload") in {"bundled", "light"} or (install_root is None and detect_install_method(root) in {"docker", "apt"}):
        return {**result, "reason": "not-a-git-checkout"}
    if not embedded and not (root / ".git").exists():
        return {**result, "reason": "not-a-git-checkout",
                "message": "This install has no git checkout to update."}
    if stamp.get("updateMechanism") not in (None, "self") and not embedded:
        return {**result, "reason": "update-root-steward-owned-git-tree",
                "message": "This installation is managed by its install method; use its updater.", "advice": "git pull"}
    config = require_readable_config_before_write(home / "config.yaml")
    if passive and (config.get("updates") or {}).get("check") is False:
        return {**result, "reason": "disabled"}
    channel = resolve_update_channel(config, root) if channel is None else validate_name(channel)
    head = embedded or _git_stdout(["rev-parse", "HEAD"], cwd=root, git=git)
    current_branch = None if embedded else _git_stdout(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root, git=git)
    desktop_config = _quiet(lambda: json.loads(branch_config_path.read_text(encoding="utf-8-sig"))) if branch_config_path else None
    configured_branch = desktop_config.get("branch") if isinstance(desktop_config, dict) else None
    if isinstance(configured_branch, str):
        configured_branch = configured_branch.strip() or None
    else:
        configured_branch = None
    selected_branch = branch or configured_branch or (current_branch if current_branch and current_branch != "HEAD" else "main")
    origin = "" if embedded else (_git_stdout(["remote", "get-url", "origin"], cwd=root, git=git) or "")
    match = _GITHUB_ORIGIN.fullmatch(origin)
    repository = OFFICIAL_REPOSITORY if embedded else (match[1] if match else None)
    dirty = False if embedded else bool(_git_stdout(["status", "--porcelain"], cwd=root, git=git))
    result.update(supported=True, currentSha=head, currentBranch=current_branch, dirty=dirty)
    if channel != "main":
        result["channel"] = channel
    else:
        result["branch"] = selected_branch
    identity = {"root": str(root), "home": str(home), "head": head, "origin": origin, "branch": selected_branch,
                "channel": channel, "embedded": embedded, "branchOverride": branch is not None, "channelProtocol": 1}
    cache_file = Path(cache_path) if cache_path is not None else home / "source-checks" / f"{install_id(root)}.json"
    cached = _quiet(lambda: json.loads(cache_file.read_text(encoding="utf-8-sig")))
    now = time.time()
    if (not force and isinstance(cached, dict) and cached.get("identity") == identity
            and isinstance(cached.get("status"), dict) and cached["status"].get("supported") is True):
        status = cached.get("status", {})
        ttl = _UPDATE_CHECK_FAILURE_CACHE_SECONDS if status.get("error") else _UPDATE_CHECK_CACHE_SECONDS
        ts = cached.get("ts")
        if isinstance(ts, (float, int)) and 0 <= now - ts < ttl:
            return {**status, "dirty": dirty, "currentBranch": current_branch}
    result["fetchedAt"] = int(now * 1000)
    source_target = None
    if not _is_full_sha(head):
        result.update(error="head-unavailable", message="Could not read the installed revision.")
    elif branch is None:
        try:
            source_target = resolve_source_target(channel, [git] if not embedded else None, root,
                                                  repository=repository or OFFICIAL_REPOSITORY)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result.update(error="release-unavailable", message=f"Could not resolve the {channel} source channel: {exc}")
        else:
            if source_target.commit:
                target = source_target.commit
                result.pop("branch", None)
                result.update(channel=channel, targetSha=target, updateAvailable=head != target,
                              behind=0 if head == target else UPDATE_AVAILABLE_NO_COUNT,
                              sourceVersion=source_target.version, buildId=source_target.build_id)
                if source_target.retired:
                    result["retirement"] = {"destination": source_target.channel, "sourceOnly": True}
            else:
                # The record supplies a default, not permission to leave the user's branch.
                selected_branch = configured_branch or (
                    current_branch if current_branch and current_branch != "HEAD" else source_target.branch)
    if "error" not in result and (source_target is None or source_target.branch is not None):
        result["branch"] = selected_branch
        official_ssh = (repository and repository.lower() == OFFICIAL_REPOSITORY.lower()
                        and origin.lower().startswith(("git@", "ssh://")))
        # The public official repo does not require the user's SSH credentials.
        # Forks must keep their own origin, including its authentication.
        remote = (f"https://github.com/{OFFICIAL_REPOSITORY}.git"
                  if embedded or (official_ssh and selected_branch != "main") else "origin")
        target, missing = _branch_tip(repository, selected_branch, root, git, remote)
        if missing and selected_branch != "main":
            result["branch"] = "main"
            if branch_config_path and not branch and configured_branch == selected_branch:
                # Do not overwrite a concurrent choice made while the network probe ran.
                current_config = _quiet(lambda: json.loads(branch_config_path.read_text(encoding="utf-8-sig")))
                if current_config == desktop_config:
                    from utils import atomic_json_write
                    atomic_json_write(branch_config_path, {**desktop_config, "branch": "main"})
            target, _ = _branch_tip(repository, "main", root, git, remote if embedded else "origin")
        if target is None:
            result.update(error="fetch-failed", message="Could not resolve the remote branch tip.")
        else:
            behind = UPDATE_AVAILABLE_NO_COUNT
            if head == target or (not embedded and _git_ok(
                    ["merge-base", "--is-ancestor", target, head], cwd=root, git=git)):
                behind = 0
            elif repository:
                payload = _github_compare(head, target, repository)
                ahead = (payload or {}).get("ahead_by")
                if isinstance(ahead, int) and not isinstance(ahead, bool) and ahead >= 0:
                    behind = ahead
                    result["commits"] = _quiet(lambda: _commits(payload), []) if behind else []
            result.update(targetSha=target, behind=behind, updateAvailable=behind != 0)
    try:
        from utils import atomic_json_write
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(cache_file, {"identity": identity, "ts": now, "status": result})
    except OSError as exc:
        logger.debug("Could not cache source check: %s", exc)
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
