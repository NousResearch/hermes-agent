#!/usr/bin/env python3
"""Resolve the workspace a fresh Hermes session should start in.

Ladder: RECALL the last active git-repo workspace from this profile's session
store → RANK candidate repos by recency-weighted signals → report NONE.

Reads $HERMES_HOME (fallback ~/.hermes) and $HERMES_PROFILE exactly like the
agent does, so each profile recalls its own sessions. Stdlib only; `git` is
optional (needed for the rank pass only). Never mutates the live session --
`--sync` writes config for FUTURE launches via `hermes config set`.

Output (default): a human-readable decision summary + a JSON decision object.
With --json: only the JSON object. Exit code is 0 in every decided case,
including mode=none; 2 on usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

CANDIDATE_ROOTS = [
    "~/Projects", "~/projects", "~/code", "~/dev", "~/repos", "~/src", "~/work",
]
# Score weights: last-commit recency dominates, dirty state and session count
# are tie-breakers (together worth less than one recency step).
W_COMMIT_RECENCY = 100
W_DIRTY = 10
W_SESSION_COUNT = 5
RECENCY_STEPS = [1, 3, 7, 30]  # days; score = first step the age fits in


def _hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".hermes"


def _profile_name() -> str:
    raw = (os.environ.get("HERMES_PROFILE") or "default").strip()
    return raw or "default"


def _profile_root() -> Path:
    home = _hermes_home()
    name = _profile_name()
    if name == "default":
        return home
    # Profiles live under ~/.hermes/profiles/<name>/ mirroring the default home.
    profile_dir = home / "profiles" / name
    return profile_dir if profile_dir.is_dir() else home


def _state_db_path() -> Path:
    return _profile_root() / "state.db"


def _is_dir(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).expanduser().is_dir()


def _is_repo(path: str | None) -> bool:
    if not path:
        return False
    return _is_dir(path) and (Path(path).expanduser() / ".git").exists()


def recall_last_workspace(db_path: Path) -> dict | None:
    """Most recent non-archived session in this profile whose cwd still exists."""
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        rows_all = con.execute(
            """
            SELECT id, cwd, git_branch, git_repo_root, last_activity_at, started_at, profile_name
            FROM sessions
            WHERE archived = 0
            ORDER BY COALESCE(last_activity_at, started_at) DESC
            LIMIT 200
            """
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    profile = _profile_name()
    rows = []
    for row in rows_all:
        sid, cwd, branch, repo_root, last_act, started, row_profile = row
        # This profile's profile-scoped view (skip other profiles in a shared store).
        if row_profile is not None and row_profile != profile:
            continue
        rows.append(row)

    for sid, cwd, branch, repo_root, last_act, started, _ in rows:
        # Recall must land in a genuine git worktree, not the plain home
        # directory this skill exists to escape. Accept the session as a
        # recall hit only when it carries git context (repo_root is an
        # existing repo, or the session was on a branch) and at least one of
        # the two recorded paths is still a directory.
        repo_hit = _is_repo(repo_root)
        ctx_hit = bool(repo_hit or branch)
        if not ctx_hit:
            continue
        target = repo_root if repo_hit else (cwd if _is_dir(cwd) else None)
        if not target:
            continue
        ts = last_act or started or 0
        age_h = max(0.0, (time.time() - ts) / 3600) if ts else None
        return {
            "mode": "recall",
            "target": str(Path(target).expanduser().resolve()),
            "session_id": sid,
            "git_branch": branch,
            "age_hours": round(age_h, 1) if age_h is not None else None,
        }
    return None


def _git(args: list[str], cwd: Path) -> tuple[int, str]:
    if not shutil.which("git"):
        return 1, ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return proc.returncode, (proc.stdout or "").strip()


def _commit_age_days(repo: Path) -> float | None:
    code, out = _git(["log", "-1", "--format=%ct"], repo)
    if code != 0 or not out:
        return None
    try:
        return max(0.0, (time.time() - float(out)) / 86400)
    except ValueError:
        return None


def _is_dirty(repo: Path) -> bool:
    code, out = _git(["status", "--porcelain"], repo)
    return code == 0 and bool(out)


def _session_counts(db_path: Path) -> dict[str, int]:
    """cwd -> non-archived session count, for the rank pass."""
    if not db_path.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        rows = con.execute(
            "SELECT cwd, COUNT(*) FROM sessions WHERE archived = 0 GROUP BY cwd"
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()
    return {cwd: count for cwd, count in rows if cwd}


def _candidate_repos(db_path: Path) -> list[Path]:
    # Never walk the home directory itself: ~/.hermes/.git and other dot-repo
    # caches live there, and the whole point of this skill is to ESCAPE the
    # home directory. Scan the known project roots only.
    roots: list[Path] = []
    for spec in CANDIDATE_ROOTS + list(
        filter(None, (os.environ.get("CWD_ROOTS") or "").split(os.pathsep))
    ):
        p = Path(spec).expanduser()
        if p.is_dir() and p.resolve() != Path.home().resolve() and p not in roots:
            roots.append(p)
    seen: set[Path] = set()
    repos: list[Path] = []
    for root in roots:
        try:
            for entry in root.iterdir():
                if entry.is_dir() and (entry / ".git").exists() and entry not in seen:
                    seen.add(entry)
                    repos.append(entry)
        except OSError:
            continue
    return repos


def _recency_score(age_days: float | None) -> float:
    if age_days is None:
        return 0.0
    for i, days in enumerate(RECENCY_STEPS):
        if age_days <= days:
            return float(len(RECENCY_STEPS) - i)
    return 0.0


def _recency_step(age_days: float | None) -> int:
    """The coarse bucket _recency_score uses; equal-age repos must tie exactly
    so dirty-state / session-count cleanly sub-rank them (float jitter on
    two repos with the same commit age otherwise leaks past the tie-breaker)."""
    if age_days is None:
        return -1
    for i, days in enumerate(RECENCY_STEPS):
        if age_days <= days:
            return len(RECENCY_STEPS) - i
    return -1


def rank_candidates(db_path: Path, limit: int = 5) -> list[dict]:
    counts = _session_counts(db_path)
    scored: list[dict] = []
    for repo in _candidate_repos(db_path):
        age = _commit_age_days(repo)
        dirty = _is_dirty(repo)
        count = counts.get(str(repo), 0)
        step = _recency_step(age)
        score = step * W_COMMIT_RECENCY + (W_DIRTY if dirty else 0) + min(count, 20) * W_SESSION_COUNT
        scored.append({
            "path": str(repo),
            "commit_age_days": round(age, 1) if age is not None else None,
            "recency_step": step,
            "dirty": dirty,
            "session_count": count,
            "score": round(score, 1),
        })
    # Tie-break inside an equal recency bucket: dirty first, then more sessions.
    scored.sort(key=lambda r: (r["recency_step"], r["dirty"], r["session_count"], r["score"]), reverse=True)
    return scored[:limit]


def sync_config(target: str) -> dict:
    """Write terminal.cwd for future launches via the hermes CLI (never hand-edit)."""
    hermes = shutil.which("hermes")
    if not hermes:
        return {"synced": False, "error": "hermes CLI not found on PATH"}
    try:
        proc = subprocess.run(
            [hermes, "config", "set", "terminal.cwd", target],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"synced": False, "error": str(exc)}
    return {"synced": proc.returncode == 0, "error": None if proc.returncode == 0 else (proc.stderr or proc.stdout or "hermes config set failed")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a single JSON object only")
    parser.add_argument("--sync", action="store_true", help="also set terminal.cwd in config.yaml for future launches")
    parser.add_argument("--candidates", action="store_true", help="print the ranked candidate list instead of acting")
    args = parser.parse_args(argv)

    db_path = _state_db_path()
    decision = recall_last_workspace(db_path)
    if decision is None:
        ranked = rank_candidates(db_path)
        if ranked:
            decision = {"mode": "rank", "target": ranked[0]["path"], "top": ranked[0]}
        else:
            decision = {"mode": "none", "target": None}

    decision["profile"] = _profile_name()
    decision["state_db"] = str(db_path)
    if args.sync and decision["target"]:
        decision["sync"] = sync_config(decision["target"])

    if args.candidates and decision["mode"] != "recall":
        decision["candidates"] = rank_candidates(db_path)

    if args.json:
        print(json.dumps(decision, indent=2))
        return 0
    mode = decision["mode"]
    if mode == "recall":
        print(f"[recall] {decision['target']}")
        if decision.get("git_branch"):
            print(f"  last branch: {decision['git_branch']}, session age: {decision.get('age_hours')}h")
    elif mode == "rank":
        print(f"[rank] {decision['target']}")
        top = decision.get("top") or {}
        print(f"  commit_age_days: {top.get('commit_age_days')}, dirty: {top.get('dirty')}, sessions: {top.get('session_count')}")
    else:
        print("[none] no prior workspace found; staying in the current directory")
    if decision.get("sync"):
        s = decision["sync"]
        print(f"  sync: {'ok' if s.get('synced') else 'failed: ' + str(s.get('error'))}")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())