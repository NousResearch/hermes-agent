"""Durable record of the gateways and services ``hermes update`` paused on Windows.

The pause token used to live only in the updater's memory (plus an ``atexit`` resume), so a
killed updater (console closed, ``taskkill``, power loss) left every paused gateway and SCM
service stopped with nothing on disk saying so. The record is written under the ROOT Hermes
home BEFORE anything is stopped, names its owner (pid + creation time, the update-marker
identity format), is rewritten as the token changes, and is removed only after a verified
resume. A later launch whose owner check finds the updater dead (and no update live) resumes it.

Resuming is gated on a whole tree: no interrupted-pull marker, no tracked change beyond the ones
present at pause time, and HEAD still at the pre-update commit or dependencies current for the
moved HEAD. Otherwise the record stays for the next launch, which runs after the interrupted-pull
restore and the dependency sync.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from contextlib import suppress
from pathlib import Path

RECORD_NAME = ".hermes-update-paused-gateways.json"
# Same tolerance as the update marker's creation-time identity (C1 rule 3).
_CT_TOLERANCE_S = 2.0


def record_path() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / RECORD_NAME


def install_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _create_time(pid: int) -> float | None:
    from hermes_cli.process_identity import _process_create_time
    return _process_create_time(pid)


def identity(pid: int | None = None) -> dict:
    """``{"pid", "ct"}`` with ``ct`` spelled like the update marker's line 3 (``ct:<s.3f>``)."""
    pid = os.getpid() if pid is None else int(pid)
    created = _create_time(pid)
    return {"pid": pid, "ct": None if created is None else f"ct:{created:.3f}"}


def identity_is_live(ident: dict | None) -> bool:
    """Alive (not a zombie) and, when a creation time was recorded, the same incarnation."""
    from hermes_cli._early_recovery import _pid_is_running
    try:
        pid = int((ident or {}).get("pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0 or not _pid_is_running(pid):
        return False
    recorded = str((ident or {}).get("ct") or "")
    if not recorded.startswith("ct:"):
        return True
    actual = _create_time(pid)
    if actual is None:
        return True
    try:
        return abs(float(recorded[3:]) - actual) <= _CT_TOLERANCE_S
    except ValueError:
        return True


def _git(root: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
            errors="replace", stdin=subprocess.DEVNULL, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def head_sha(root: Path) -> str | None:
    result = _git(root, "rev-parse", "--verify", "HEAD")
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip() or None


def tracked_changes(root: Path) -> list[str] | None:
    """Tracked paths that differ from HEAD (index or worktree); ``None`` when git cannot say."""
    result = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no")
    if result is None or result.returncode != 0:
        return None
    paths = set()
    entries = result.stdout.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        paths.add(entry[3:])
        if entry[0] in "RC":  # rename/copy: the source path follows as its own entry
            index += 1
    return sorted(paths)


def stamp_tree(token: dict, root: Path | None = None) -> dict:
    """Record what "the tree before this update" was, for the whole-tree gate."""
    root = root or install_root()
    if (root / ".git").exists():
        token.setdefault("pre_sha", head_sha(root))
        token.setdefault("dirty_at_pause", tracked_changes(root))
    token.setdefault("pause_id", uuid.uuid4().hex)
    return token


def _venv_is_current(root: Path) -> bool:
    try:
        import pm
        return bool(pm.venv_is_current(project_root=root))
    except Exception:
        return False


def tree_is_whole(token: dict, root: Path | None = None) -> tuple[bool, str]:
    """May paused gateways start on this checkout now? ``(verdict, reason when not)``."""
    from hermes_cli._early_recovery import interrupted_pull_marker
    root = root or install_root()
    if (root / ".git").exists():
        if interrupted_pull_marker(root).exists():
            return False, "the checkout is mid-pull (interrupted-pull marker present)"
        changes = tracked_changes(root)
        if changes is None:
            return False, "git cannot read the checkout state"
        unexpected = sorted(set(changes) - set(token.get("dirty_at_pause") or []))
        if unexpected:
            return False, f"the checkout has {len(unexpected)} file(s) git left half-written (e.g. {unexpected[0]})"
        head = head_sha(root)
        if head is None:
            return False, "the checkout HEAD is unreadable"
        if head == token.get("pre_sha"):
            return True, ""
    if not _venv_is_current(root):
        return False, "dependencies are not current for the updated code yet"
    return True, ""


def _atomic_write(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read(path: Path | None = None) -> dict | None:
    path = path or record_path()
    try:
        body = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) and isinstance(body.get("token"), dict) else None


def write(token: dict, *, owner: dict | None = None, path: Path | None = None) -> None:
    """Persist *token*; the owner of an existing record for the same pause is kept."""
    path = path or record_path()
    existing = read(path)
    if owner is None:
        if existing is not None and existing["token"].get("pause_id") == token.get("pause_id"):
            owner = existing["owner"]
        else:
            owner = identity()
    _atomic_write(path, {"schema": 1, "owner": owner, "install_root": str(install_root()), "token": token})


def discharge(token: dict, path: Path | None = None) -> None:
    """Compare-and-delete: remove the record only while it still names this pause."""
    path = path or record_path()
    existing = read(path)
    if existing is not None and existing["token"].get("pause_id") == token.get("pause_id"):
        with suppress(OSError):
            path.unlink()


def sync(token: dict) -> None:
    """Mirror the token after a resume attempt: done → delete; anything still owed → rewrite."""
    if not token.get("pause_id"):
        return
    try:
        if token.get("resume_needed") or token.get("resume_deferred"):
            owed = {**token, "resume_needed": True}
            owed.pop("resume_deferred", None)
            write(owed)
        else:
            discharge(token)
    except OSError as exc:
        print(f"  ⚠ Could not update the paused-gateway record {record_path()}: {exc}")


def _live_update_elsewhere() -> bool:
    # TODO(LP-LOCK): switch to update_lock.update_in_progress(install_root) once it lands; the
    # marker read below is the existing path and misses an orphaned completion child.
    from hermes_cli.update_lock import read_live_update
    holder = read_live_update()
    return holder is not None and holder.pid != os.getpid()


def orphaned_record(path: Path | None = None) -> dict | None:
    """The record when its owner is dead, it belongs to this install and no other update is live."""
    body = read(path)
    if body is None or body.get("install_root") != str(install_root()):
        return None
    if identity_is_live(body.get("owner")) or _live_update_elsewhere():
        return None
    return body


def drop_never_stopped(token: dict) -> dict:
    """Entries whose recorded process is still the same live incarnation were never stopped."""
    alive = {pid for pid, ct in (token.get("identities") or {}).items()
             if identity_is_live({"pid": pid, "ct": ct})}
    token["profiles"] = {p: pid for p, pid in (token.get("profiles") or {}).items() if str(pid) not in alive}
    token["unmapped"] = [u for u in (token.get("unmapped") or []) if str(u.get("pid")) not in alive]
    return token


def merge_into(token: dict | None, adopted: dict) -> dict:
    """Fold an orphaned pause into this update's token so its resume brings both sets back."""
    token = token if token is not None else {"resume_needed": True, "profiles": {}, "unmapped_pids": [], "unmapped": []}
    token["resume_needed"] = True
    profiles = token.setdefault("profiles", {})
    for name, pid in (adopted.get("profiles") or {}).items():
        profiles.setdefault(name, pid)
    token.setdefault("unmapped", []).extend(adopted.get("unmapped") or [])
    services = token.setdefault("services", [])
    services.extend(s for s in adopted.get("services") or [] if s not in services)
    if services:
        token.setdefault("expected_services", []).extend(s for s in services if s not in token["expected_services"])
        token.setdefault("restarted_services", [])
        token.setdefault("service_profiles", {}).update(adopted.get("service_profiles") or {})
    return token


def recover(argv: list[str] | None = None) -> None:
    """Start-of-run recovery: resume an orphaned pause. Never raises."""
    from hermes_cli._parser import command_argv
    command = command_argv(list(sys.argv[1:] if argv is None else argv))
    if command[:1] == ["update"] or command[:2] == ["gateway", "run"]:
        return  # the update adopts it itself; a booting gateway must not block on its siblings
    try:
        body = orphaned_record()
        if body is None:
            return
        path = record_path()
        claim = path.with_name(f"{path.name}.{os.getpid()}.claim")
        try:
            os.rename(path, claim)  # exactly one concurrent launch wins the record
        except OSError:
            return
        token = drop_never_stopped(dict(body["token"]))
        token["resume_needed"] = True
        print("→ Restarting gateway(s) paused by an interrupted `hermes update`...", file=sys.stderr)
        try:
            # Its own finally mirrors the outcome: whatever is still owed is re-recorded
            # under this process, so the next launch retries once we have exited.
            from hermes_cli.update_cmd_windows import _resume_windows_gateways_after_update
            _resume_windows_gateways_after_update(token)
        except Exception as exc:
            print(f"  ⚠ Could not restart every paused gateway: {exc}. Run `hermes update` or "
                  "`hermes gateway start`.", file=sys.stderr)
        finally:
            with suppress(OSError):
                claim.unlink()
    except Exception as exc:  # never brick a launch on recovery
        print(f"  ⚠ Paused-gateway recovery skipped: {exc}", file=sys.stderr)
