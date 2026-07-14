#!/usr/bin/env python3
"""Git-based cross-machine menu work-locks (shared branch reservation).

Locks live on a single remote branch (default: work-locks) in
``.work/menu-queue.md``.  Reserving = append your row + push; first push wins.

**Clock skew:** Expiry is judged with each machine's local clock.  All
machines that share a work-locks remote MUST run synchronized clocks
(NTP or equivalent).  Optional ``--skew-seconds N`` adds a conservative
grace when deciding whether *another* owner's lock is expired — the lock
is not taken over until ``expires_at + skew_seconds`` is in the past.

stdlib only · Python 3.9+
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

QUEUE_REL = Path(".work") / "menu-queue.md"
HEADER = "| menu_id | owner | project | acquired_at | expires_at |"
SEPARATOR = "| --- | --- | --- | --- | --- |"
DEFAULT_BRANCH = "work-locks"
DEFAULT_TTL_HOURS = 48
MAX_TTL_HOURS = 720  # 30 days

EXIT_OK = 0
EXIT_HELD = 1
EXIT_ERR = 2

_ROW_RE = re.compile(
    r"^\|\s*(?P<menu_id>[^|]+?)\s*"
    r"\|\s*(?P<owner>[^|]+?)\s*"
    r"\|\s*(?P<project>[^|]+?)\s*"
    r"\|\s*(?P<acquired_at>[^|]+?)\s*"
    r"\|\s*(?P<expires_at>[^|]+?)\s*"
    r"\|\s*$"
)


# ---------------------------------------------------------------------------
# time / json helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC time, honouring test overrides."""
    raw = os.environ.get("WORK_LOCKS_NOW")
    if raw:
        return _parse_iso(raw)
    return datetime.now(timezone.utc)


def _parse_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# field / ttl validation (prevents queue-row injection + false acquisition)
# ---------------------------------------------------------------------------

def _validate_field(
    name: str, value: str, *, allow_empty: bool = False
) -> Optional[str]:
    """Validate a value destined for a markdown queue cell.

    Rejects empty (unless allow_empty), pipe ``|``, newlines, CR, and other
    ASCII control characters (ord < 32 or DEL).  Returns an error detail
    string on failure, or None when the value is clean.
    """
    if value is None:
        value = ""
    if value == "":
        if allow_empty:
            return None
        return "empty"
    for ch in value:
        o = ord(ch)
        if ch == "|" or o < 32 or o == 127:
            return "contains forbidden character"
    return None


def _fail_invalid(name: str, detail: str) -> int:
    msg = f"invalid {name}: {detail}"
    _err(msg)
    _emit({"ok": False, "error": msg})
    return EXIT_ERR


def _fail_corruption(menu: str) -> int:
    msg = f"queue corruption: menu {menu} has invalid timestamp"
    _err(msg)
    _emit({"ok": False, "error": msg})
    return EXIT_ERR


def _validate_ttl(ttl: int) -> Optional[str]:
    """TTL must be an int in 1..MAX_TTL_HOURS inclusive."""
    if not isinstance(ttl, int) or isinstance(ttl, bool):
        return "ttl-hours must be 1..720"
    if ttl < 1 or ttl > MAX_TTL_HOURS:
        return "ttl-hours must be 1..720"
    return None


def _fail_ttl() -> int:
    msg = "ttl-hours must be 1..720"
    _err(msg)
    _emit({"ok": False, "error": msg})
    return EXIT_ERR


def _validate_skew(skew: int) -> Optional[str]:
    """Skew grace must be non-negative (negative enables premature takeover)."""
    if not isinstance(skew, int) or isinstance(skew, bool):
        return "skew-seconds must be >= 0"
    if skew < 0:
        return "skew-seconds must be >= 0"
    return None


def _fail_skew() -> int:
    msg = "skew-seconds must be >= 0"
    _err(msg)
    _emit({"ok": False, "error": msg})
    return EXIT_ERR


def _resolve_owner(args: argparse.Namespace) -> str:
    if args.owner is not None:
        return args.owner
    return _default_owner()


def _validate_cmd_fields(
    args: argparse.Namespace,
    *,
    need_menu: bool = True,
    need_ttl: bool = False,
) -> Optional[int]:
    """Validate menu/owner/project[/ttl] before any git work.

    On success, writes resolved ``args.owner`` (and normalises project).
    Returns an exit code on failure, else None.
    """
    if need_menu:
        detail = _validate_field("menu", args.menu or "")
        if detail is not None:
            return _fail_invalid("menu", detail)

    owner = _resolve_owner(args)
    detail = _validate_field("owner", owner)
    if detail is not None:
        return _fail_invalid("owner", detail)
    args.owner = owner

    project = args.project if args.project is not None else ""
    detail = _validate_field("project", project, allow_empty=True)
    if detail is not None:
        return _fail_invalid("project", detail)
    args.project = project

    if need_ttl:
        detail = _validate_ttl(int(args.ttl_hours))
        if detail is not None:
            return _fail_ttl()

    detail = _validate_skew(int(getattr(args, "skew_seconds", 0) or 0))
    if detail is not None:
        return _fail_skew()

    return None


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------

def _git(
    args: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Keep commits hermetic / non-interactive
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=capture,
        text=True,
        env=env,
    )


def _git_ok(args: List[str], cwd: Optional[Path] = None) -> bool:
    r = _git(args, cwd=cwd, check=False)
    return r.returncode == 0


def _git_out(args: List[str], cwd: Optional[Path] = None) -> str:
    r = _git(args, cwd=cwd, check=True)
    return (r.stdout or "").strip()


def _configure_identity(repo: Path) -> None:
    """Set local identity if missing (so commits never fail)."""
    name = _git(["config", "user.name"], cwd=repo, check=False)
    email = _git(["config", "user.email"], cwd=repo, check=False)
    if name.returncode != 0 or not (name.stdout or "").strip():
        _git(["config", "user.name", "work-locks"], cwd=repo)
    if email.returncode != 0 or not (email.stdout or "").strip():
        _git(["config", "user.email", "work-locks@local"], cwd=repo)


def _hostname() -> str:
    try:
        host = socket.gethostname()
        if host:
            return host
    except OSError:
        pass
    try:
        host = os.uname().nodename
        if host:
            return host
    except (AttributeError, OSError):
        pass
    return "unknown-host"


def _default_owner() -> str:
    """Return ``<name>@<hostname>`` so two machines never silently collide.

    Name preference: git user.name → WORK_LOCKS_OWNER → $USER → $USERNAME → unknown.
    Hostname is always appended.
    """
    name = ""
    for cmd in (["config", "user.name"], ["config", "--global", "user.name"]):
        r = _git(cmd, check=False)
        if r.returncode == 0 and (r.stdout or "").strip():
            name = (r.stdout or "").strip()
            break
    if not name:
        name = (
            os.environ.get("WORK_LOCKS_OWNER")
            or os.environ.get("USER")
            or os.environ.get("USERNAME")
            or "unknown"
        )
    return f"{name}@{_hostname()}"


# ---------------------------------------------------------------------------
# queue file parse / render
# ---------------------------------------------------------------------------

def _is_sep_or_header(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|"):
        return True
    # header
    if "menu_id" in s and "owner" in s:
        return True
    # markdown separator |---|---|
    cells = [c.strip() for c in s.strip("|").split("|")]
    if cells and all(re.fullmatch(r":?-+:?", c or "-") for c in cells):
        return True
    return False


def parse_queue(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip() or _is_sep_or_header(line):
            continue
        m = _ROW_RE.match(line.strip())
        if not m:
            continue
        rows.append(
            {
                "menu_id": m.group("menu_id").strip(),
                "owner": m.group("owner").strip(),
                "project": m.group("project").strip(),
                "acquired_at": m.group("acquired_at").strip(),
                "expires_at": m.group("expires_at").strip(),
            }
        )
    return rows


def render_queue(rows: List[Dict[str, str]]) -> str:
    lines = [HEADER, SEPARATOR]
    for r in rows:
        lines.append(
            f"| {r['menu_id']} | {r['owner']} | {r['project']} "
            f"| {r['acquired_at']} | {r['expires_at']} |"
        )
    return "\n".join(lines) + "\n"


def empty_queue_text() -> str:
    return render_queue([])


def find_lock(
    rows: List[Dict[str, str]], menu_id: str
) -> Optional[Dict[str, str]]:
    for r in rows:
        if r["menu_id"] == menu_id:
            return r
    return None


def is_corrupt(row: Dict[str, str]) -> bool:
    """True if acquired_at or expires_at cannot be parsed as ISO timestamps."""
    for field in ("acquired_at", "expires_at"):
        try:
            _parse_iso(row[field])
        except (ValueError, KeyError, TypeError):
            return True
    return False


def is_expired(
    row: Dict[str, str],
    now: datetime,
    *,
    skew_seconds: int = 0,
) -> bool:
    """Return True if the lock is expired under *now* (+ optional skew).

    Raises ValueError when timestamps are unparseable — callers MUST treat
    that as queue corruption (fail closed), never as free/expired.
    """
    _parse_iso(row["acquired_at"])  # validate; raises on corrupt
    exp = _parse_iso(row["expires_at"])
    # Conservative: require expires_at + skew to be in the past
    return exp + timedelta(seconds=int(skew_seconds or 0)) <= now


def _other_owner_skew(row: Dict[str, str], owner: str, skew_seconds: int) -> int:
    """Skew only when judging *another* owner's lock; own lock uses 0."""
    if row.get("owner") == owner:
        return 0
    return int(skew_seconds or 0)


# ---------------------------------------------------------------------------
# worktree lifecycle: clone/fetch work-branch, mutate, push
# ---------------------------------------------------------------------------

class WorkLockRepo:
    """Temporary working clone of the central remote for one CLI invocation."""

    def __init__(
        self,
        remote: str,
        work_branch: str = DEFAULT_BRANCH,
        cache_dir: Optional[str] = None,
    ):
        self.remote = remote
        self.work_branch = work_branch
        self._tmp: Optional[Path] = None
        self.repo: Optional[Path] = None
        self._owned_tmp = False
        self.cache_dir = cache_dir

    def __enter__(self) -> "WorkLockRepo":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def open(self) -> Path:
        if self.cache_dir:
            root = Path(self.cache_dir)
            root.mkdir(parents=True, exist_ok=True)
            self.repo = root / "clone"
            self._owned_tmp = False
            if not (self.repo / ".git").exists():
                self.repo.mkdir(parents=True, exist_ok=True)
                self._init_and_fetch()
            else:
                self._fetch_rebase()
        else:
            self._tmp = Path(tempfile.mkdtemp(prefix="work-locks-"))
            self.repo = self._tmp / "clone"
            self.repo.mkdir(parents=True, exist_ok=True)
            self._owned_tmp = True
            self._init_and_fetch()
        assert self.repo is not None
        return self.repo

    def close(self) -> None:
        if self._owned_tmp and self._tmp and self._tmp.exists():
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = None
        self.repo = None

    def _init_and_fetch(self) -> None:
        assert self.repo is not None
        _git(["init", "-b", self.work_branch], cwd=self.repo)
        _configure_identity(self.repo)
        _git(["remote", "add", "origin", self.remote], cwd=self.repo)
        self._fetch_rebase()

    def _remote_has_branch(self) -> bool:
        assert self.repo is not None
        r = _git(
            ["ls-remote", "--heads", "origin", self.work_branch],
            cwd=self.repo,
            check=False,
        )
        if r.returncode != 0:
            return False
        return bool((r.stdout or "").strip())

    def _fetch_rebase(self) -> None:
        """Always read the LATEST remote state for work-branch."""
        assert self.repo is not None
        _configure_identity(self.repo)

        if not self._remote_has_branch():
            # Orphan empty branch with just the queue header.
            # Detach any previous history by resetting to an empty orphan tree.
            _git(["checkout", "--orphan", self.work_branch], cwd=self.repo, check=False)
            # Clear index / working tree junk if any
            _git(["rm", "-rf", "--cached", "."], cwd=self.repo, check=False)
            for p in self.repo.iterdir():
                if p.name == ".git":
                    continue
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            self._ensure_queue_file()
            _git(["add", str(QUEUE_REL)], cwd=self.repo)
            # Only commit if nothing yet (first init)
            has_commit = _git_ok(["rev-parse", "HEAD"], cwd=self.repo)
            if not has_commit:
                _git(
                    ["commit", "-m", "init work-locks queue"],
                    cwd=self.repo,
                )
            return

        # Fetch remote branch and hard-reset to it (authoritative remote state)
        r = _git(
            ["fetch", "origin", f"+{self.work_branch}:refs/remotes/origin/{self.work_branch}"],
            cwd=self.repo,
            check=False,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"git fetch failed: {(r.stderr or r.stdout or '').strip()}"
            )

        # Checkout local tracking branch at remote tip
        if _git_ok(["show-ref", "--verify", f"refs/heads/{self.work_branch}"], cwd=self.repo):
            _git(["checkout", self.work_branch], cwd=self.repo)
            _git(
                ["reset", "--hard", f"refs/remotes/origin/{self.work_branch}"],
                cwd=self.repo,
            )
        else:
            _git(
                [
                    "checkout",
                    "-B",
                    self.work_branch,
                    f"refs/remotes/origin/{self.work_branch}",
                ],
                cwd=self.repo,
            )

        self._ensure_queue_file()

    def _ensure_queue_file(self) -> None:
        assert self.repo is not None
        path = self.repo / QUEUE_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(empty_queue_text(), encoding="utf-8")

    def read_rows(self) -> List[Dict[str, str]]:
        assert self.repo is not None
        path = self.repo / QUEUE_REL
        if not path.exists():
            return []
        return parse_queue(path.read_text(encoding="utf-8"))

    def write_rows(self, rows: List[Dict[str, str]]) -> None:
        assert self.repo is not None
        path = self.repo / QUEUE_REL
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_queue(rows), encoding="utf-8")

    def commit_push(self, message: str) -> Tuple[bool, str]:
        """Stage queue, commit if dirty, push. Returns (ok, detail)."""
        assert self.repo is not None
        _git(["add", str(QUEUE_REL)], cwd=self.repo)
        # Skip empty commit
        st = _git(["status", "--porcelain", str(QUEUE_REL)], cwd=self.repo)
        if not (st.stdout or "").strip():
            # Nothing to commit — still try push in case local is ahead
            pass
        else:
            _git(["commit", "-m", message], cwd=self.repo)

        push = _git(
            ["push", "origin", f"HEAD:refs/heads/{self.work_branch}"],
            cwd=self.repo,
            check=False,
        )
        if push.returncode != 0:
            detail = (push.stderr or push.stdout or "push rejected").strip()
            return False, detail
        return True, "pushed"

    def refresh_from_remote(self) -> None:
        """Re-fetch after a rejected push and re-read remote truth."""
        self._fetch_rebase()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_acquire(args: argparse.Namespace) -> int:
    bad = _validate_cmd_fields(args, need_menu=True, need_ttl=True)
    if bad is not None:
        return bad

    owner = args.owner
    project = args.project or ""
    ttl = int(args.ttl_hours)
    skew = int(getattr(args, "skew_seconds", 0) or 0)
    now = _parse_iso(args.now_override) if args.now_override else _utcnow()
    menu = args.menu

    try:
        with WorkLockRepo(args.remote, args.work_branch, args.cache_dir) as wr:
            # Always start from latest remote
            rows = wr.read_rows()
            existing = find_lock(rows, menu)

            if existing:
                if is_corrupt(existing):
                    return _fail_corruption(menu)
                skew_for = _other_owner_skew(existing, owner, skew)
                try:
                    expired = is_expired(existing, now, skew_seconds=skew_for)
                except ValueError:
                    return _fail_corruption(menu)

                if not expired:
                    if existing["owner"] == owner:
                        # Already mine — treat as success (idempotent acquire)
                        _emit(
                            {
                                "ok": True,
                                "menu": menu,
                                "owner": owner,
                                "project": existing["project"],
                                "acquired_at": existing["acquired_at"],
                                "expires_at": existing["expires_at"],
                                "already_held": True,
                            }
                        )
                        return EXIT_OK
                    # Live lock held by someone else
                    _emit(
                        {
                            "ok": False,
                            "held_by": existing["owner"],
                            "expires_at": existing["expires_at"],
                            "menu": menu,
                        }
                    )
                    return EXIT_HELD

            # Free, expired, or missing — claim it
            acquired_at = _fmt_iso(now)
            expires_at = _fmt_iso(now + timedelta(hours=ttl))
            new_row = {
                "menu_id": menu,
                "owner": owner,
                "project": project,
                "acquired_at": acquired_at,
                "expires_at": expires_at,
            }
            # Replace expired row or append
            new_rows = [r for r in rows if r["menu_id"] != menu]
            new_rows.append(new_row)
            wr.write_rows(new_rows)

            ok, detail = wr.commit_push(f"acquire {menu} by {owner}")
            if ok:
                _emit(
                    {
                        "ok": True,
                        "menu": menu,
                        "owner": owner,
                        "project": project,
                        "acquired_at": acquired_at,
                        "expires_at": expires_at,
                    }
                )
                return EXIT_OK

            # Push rejected — re-fetch and judge from remote
            wr.refresh_from_remote()
            rows2 = wr.read_rows()
            remote_lock = find_lock(rows2, menu)
            if remote_lock:
                if is_corrupt(remote_lock):
                    return _fail_corruption(menu)
                skew_for = _other_owner_skew(remote_lock, owner, skew)
                try:
                    remote_expired = is_expired(
                        remote_lock, now, skew_seconds=skew_for
                    )
                except ValueError:
                    return _fail_corruption(menu)
                if not remote_expired and remote_lock["owner"] != owner:
                    _emit(
                        {
                            "ok": False,
                            "held_by": remote_lock["owner"],
                            "expires_at": remote_lock["expires_at"],
                            "menu": menu,
                        }
                    )
                    return EXIT_HELD

            # Remote free/expired/ours after race — one retry
            new_rows2 = [r for r in rows2 if r["menu_id"] != menu]
            new_rows2.append(new_row)
            wr.write_rows(new_rows2)
            ok2, detail2 = wr.commit_push(f"acquire {menu} by {owner} (retry)")
            if ok2:
                _emit(
                    {
                        "ok": True,
                        "menu": menu,
                        "owner": owner,
                        "project": project,
                        "acquired_at": acquired_at,
                        "expires_at": expires_at,
                    }
                )
                return EXIT_OK

            # Still lost — re-check who holds it
            wr.refresh_from_remote()
            rows3 = wr.read_rows()
            rl = find_lock(rows3, menu)
            if rl:
                if is_corrupt(rl):
                    return _fail_corruption(menu)
                skew_for = _other_owner_skew(rl, owner, skew)
                try:
                    rl_expired = is_expired(rl, now, skew_seconds=skew_for)
                except ValueError:
                    return _fail_corruption(menu)
                if not rl_expired and rl["owner"] != owner:
                    _emit(
                        {
                            "ok": False,
                            "held_by": rl["owner"],
                            "expires_at": rl["expires_at"],
                            "menu": menu,
                        }
                    )
                    return EXIT_HELD

            _err(f"acquire push failed: {detail2 or detail}")
            _emit({"ok": False, "error": detail2 or detail})
            return EXIT_ERR
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        _err(str(exc))
        _emit({"ok": False, "error": str(exc)})
        return EXIT_ERR


def cmd_check(args: argparse.Namespace) -> int:
    bad = _validate_cmd_fields(args, need_menu=True, need_ttl=False)
    if bad is not None:
        return bad

    owner = args.owner
    skew = int(getattr(args, "skew_seconds", 0) or 0)
    now = _parse_iso(args.now_override) if args.now_override else _utcnow()
    menu = args.menu

    try:
        with WorkLockRepo(args.remote, args.work_branch, args.cache_dir) as wr:
            rows = wr.read_rows()
            lock = find_lock(rows, menu)
            if not lock:
                _emit(
                    {
                        "menu": menu,
                        "held_by": None,
                        "expired": False,
                        "mine": False,
                    }
                )
                return EXIT_OK

            if is_corrupt(lock):
                # Report corrupt but never claim the menu is free
                _emit(
                    {
                        "menu": menu,
                        "held_by": lock["owner"],
                        "expired": False,
                        "mine": False,
                        "corrupt": True,
                        "expires_at": lock["expires_at"],
                        "owner": lock["owner"],
                    }
                )
                return EXIT_OK

            skew_for = _other_owner_skew(lock, owner, skew)
            try:
                expired = is_expired(lock, now, skew_seconds=skew_for)
            except ValueError:
                _emit(
                    {
                        "menu": menu,
                        "held_by": lock["owner"],
                        "expired": False,
                        "mine": False,
                        "corrupt": True,
                        "expires_at": lock["expires_at"],
                        "owner": lock["owner"],
                    }
                )
                return EXIT_OK

            held_by = None if expired else lock["owner"]
            mine = (not expired) and lock["owner"] == owner
            _emit(
                {
                    "menu": menu,
                    "held_by": held_by,
                    "expired": expired,
                    "mine": mine,
                    "expires_at": lock["expires_at"],
                    "owner": lock["owner"],
                }
            )
            return EXIT_OK
    except Exception as exc:  # noqa: BLE001
        _err(str(exc))
        _emit({"ok": False, "error": str(exc)})
        return EXIT_ERR


def cmd_release(args: argparse.Namespace) -> int:
    bad = _validate_cmd_fields(args, need_menu=True, need_ttl=False)
    if bad is not None:
        return bad

    owner = args.owner
    now = _parse_iso(args.now_override) if args.now_override else _utcnow()
    menu = args.menu

    try:
        with WorkLockRepo(args.remote, args.work_branch, args.cache_dir) as wr:
            rows = wr.read_rows()
            lock = find_lock(rows, menu)
            if not lock:
                # Nothing to release — treat as success (idempotent)
                _emit({"ok": True, "menu": menu, "released": False, "reason": "absent"})
                return EXIT_OK

            if is_corrupt(lock):
                return _fail_corruption(menu)

            try:
                expired = is_expired(lock, now, skew_seconds=0)
            except ValueError:
                return _fail_corruption(menu)

            if expired:
                # Expired lock: only owner may clean it up
                # Spec: release/renew ONLY by current owner. Expired still has an owner field.
                if lock["owner"] != owner:
                    _emit(
                        {
                            "ok": False,
                            "held_by": lock["owner"],
                            "menu": menu,
                            "reason": "not_owner",
                        }
                    )
                    return EXIT_HELD
            elif lock["owner"] != owner:
                _emit(
                    {
                        "ok": False,
                        "held_by": lock["owner"],
                        "menu": menu,
                        "reason": "not_owner",
                    }
                )
                return EXIT_HELD

            new_rows = [r for r in rows if r["menu_id"] != menu]
            wr.write_rows(new_rows)
            ok, detail = wr.commit_push(f"release {menu} by {owner}")
            if not ok:
                wr.refresh_from_remote()
                rows2 = wr.read_rows()
                lock2 = find_lock(rows2, menu)
                if lock2:
                    if is_corrupt(lock2):
                        return _fail_corruption(menu)
                    try:
                        lock2_expired = is_expired(lock2, now, skew_seconds=0)
                    except ValueError:
                        return _fail_corruption(menu)
                    if lock2["owner"] != owner and not lock2_expired:
                        _emit(
                            {
                                "ok": False,
                                "held_by": lock2["owner"],
                                "menu": menu,
                                "reason": "not_owner",
                            }
                        )
                        return EXIT_HELD
                _err(f"release push failed: {detail}")
                _emit({"ok": False, "error": detail})
                return EXIT_ERR

            _emit({"ok": True, "menu": menu, "released": True, "owner": owner})
            return EXIT_OK
    except Exception as exc:  # noqa: BLE001
        _err(str(exc))
        _emit({"ok": False, "error": str(exc)})
        return EXIT_ERR


def cmd_renew(args: argparse.Namespace) -> int:
    bad = _validate_cmd_fields(args, need_menu=True, need_ttl=True)
    if bad is not None:
        return bad

    owner = args.owner
    ttl = int(args.ttl_hours)
    now = _parse_iso(args.now_override) if args.now_override else _utcnow()
    menu = args.menu

    try:
        with WorkLockRepo(args.remote, args.work_branch, args.cache_dir) as wr:
            rows = wr.read_rows()
            lock = find_lock(rows, menu)
            if not lock:
                _emit(
                    {
                        "ok": False,
                        "menu": menu,
                        "held_by": None,
                        "reason": "absent",
                    }
                )
                return EXIT_HELD

            if is_corrupt(lock):
                return _fail_corruption(menu)

            if lock["owner"] != owner:
                _emit(
                    {
                        "ok": False,
                        "held_by": lock["owner"],
                        "menu": menu,
                        "reason": "not_owner",
                    }
                )
                return EXIT_HELD

            try:
                expired = is_expired(lock, now, skew_seconds=0)
            except ValueError:
                return _fail_corruption(menu)

            if expired:
                # Expired: no longer renewable as owner of a live lock
                _emit(
                    {
                        "ok": False,
                        "held_by": lock["owner"],
                        "menu": menu,
                        "expired": True,
                        "reason": "expired",
                    }
                )
                return EXIT_HELD

            new_exp = _fmt_iso(now + timedelta(hours=ttl))
            new_rows = []
            for r in rows:
                if r["menu_id"] == menu:
                    updated = dict(r)
                    updated["expires_at"] = new_exp
                    new_rows.append(updated)
                else:
                    new_rows.append(r)
            wr.write_rows(new_rows)
            ok, detail = wr.commit_push(f"renew {menu} by {owner}")
            if not ok:
                wr.refresh_from_remote()
                rows2 = wr.read_rows()
                lock2 = find_lock(rows2, menu)
                if lock2 and is_corrupt(lock2):
                    return _fail_corruption(menu)
                try:
                    lock2_bad = (
                        not lock2
                        or lock2["owner"] != owner
                        or is_expired(lock2, now, skew_seconds=0)
                    )
                except ValueError:
                    return _fail_corruption(menu)
                if lock2_bad:
                    held = lock2["owner"] if lock2 else None
                    _emit(
                        {
                            "ok": False,
                            "held_by": held,
                            "menu": menu,
                            "reason": "not_owner",
                        }
                    )
                    return EXIT_HELD
                _err(f"renew push failed: {detail}")
                _emit({"ok": False, "error": detail})
                return EXIT_ERR

            _emit(
                {
                    "ok": True,
                    "menu": menu,
                    "owner": owner,
                    "expires_at": new_exp,
                }
            )
            return EXIT_OK
    except Exception as exc:  # noqa: BLE001
        _err(str(exc))
        _emit({"ok": False, "error": str(exc)})
        return EXIT_ERR


def cmd_status(args: argparse.Namespace) -> int:
    skew = int(getattr(args, "skew_seconds", 0) or 0)
    if _validate_skew(skew) is not None:
        return _fail_skew()

    now = _parse_iso(args.now_override) if args.now_override else _utcnow()

    try:
        with WorkLockRepo(args.remote, args.work_branch, args.cache_dir) as wr:
            rows = wr.read_rows()
            active = []
            for r in rows:
                if is_corrupt(r):
                    # Do not list as active free; surface separately if needed.
                    # Corrupt rows are never treated as free elsewhere.
                    continue
                try:
                    expired = is_expired(r, now, skew_seconds=skew)
                except ValueError:
                    continue
                if not expired:
                    active.append(
                        {
                            "menu_id": r["menu_id"],
                            "owner": r["owner"],
                            "project": r["project"],
                            "acquired_at": r["acquired_at"],
                            "expires_at": r["expires_at"],
                        }
                    )
            _emit(active)
            return EXIT_OK
    except Exception as exc:  # noqa: BLE001
        _err(str(exc))
        _emit({"ok": False, "error": str(exc)})
        return EXIT_ERR


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="work_locks.py",
        description="Git-based cross-machine menu work-locks",
    )
    p.add_argument(
        "command",
        choices=["acquire", "check", "release", "renew", "status"],
        help="Action to perform",
    )
    p.add_argument(
        "--remote",
        required=True,
        help="Path or URL of the central git remote",
    )
    p.add_argument(
        "--menu",
        default=None,
        help="Menu id to lock/check/release/renew (required except status)",
    )
    p.add_argument(
        "--owner",
        default=None,
        help="Lock owner (default: <git user.name|$USER>@<hostname>)",
    )
    p.add_argument(
        "--project",
        default="",
        help="Optional project label stored on the lock row",
    )
    p.add_argument(
        "--ttl-hours",
        type=int,
        default=DEFAULT_TTL_HOURS,
        help=f"Lock lifetime in hours (default {DEFAULT_TTL_HOURS}, range 1..{MAX_TTL_HOURS})",
    )
    p.add_argument(
        "--skew-seconds",
        type=int,
        default=0,
        help=(
            "Conservative clock-skew grace (seconds) when judging another "
            "owner's lock as expired (default 0). Requires NTP-synced clocks."
        ),
    )
    p.add_argument(
        "--work-branch",
        default=DEFAULT_BRANCH,
        help=f"Shared locks branch (default {DEFAULT_BRANCH})",
    )
    p.add_argument(
        "--now-override",
        default=None,
        help="ISO-8601 UTC 'now' for tests (or set WORK_LOCKS_NOW)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Optional persistent clone directory (else fresh tempfile)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "status" and not args.menu:
        _err("--menu is required for this command")
        _emit({"ok": False, "error": "--menu is required"})
        return EXIT_ERR

    # Env fallback for now if flag not set
    if not args.now_override and os.environ.get("WORK_LOCKS_NOW"):
        args.now_override = os.environ["WORK_LOCKS_NOW"]

    handlers = {
        "acquire": cmd_acquire,
        "check": cmd_check,
        "release": cmd_release,
        "renew": cmd_renew,
        "status": cmd_status,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
