"""One isolated native HTTP Run and its verified process control."""

from __future__ import annotations

import json
import ctypes
import os
import signal
import sqlite3
import sys
import threading
import time
from contextlib import suppress

FRAME_PREFIX = "\x1ehermes-run:"


def process_fingerprint(pid: int) -> int:
    """An absent fingerprint never authorizes a signal."""
    try:
        if sys.platform == "linux":
            with open(f"/proc/{pid}/stat", encoding="ascii") as source:
                # comm may contain spaces or closing parentheses. Field 22 is
                # index 19 after the final ') ' (field 3 is index 0).
                return int(source.read().rsplit(") ", 1)[1].split()[19])
        from gateway.status import get_process_start_time
        return int(get_process_start_time(pid) or 0)
    except (IndexError, OSError, TypeError, ValueError):
        return 0


def verified_process_alive(pid: int, started: int) -> bool:
    if pid <= 0 or started <= 0 or process_fingerprint(pid) != started:
        return False
    if os.name == "posix":
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as source:
                if source.read().rsplit(") ", 1)[1][0] == "Z":
                    return False
        except OSError:
            return False
    return True


def _stopped_or_exited(pid: int, started: int, deadline: float) -> bool:
    """A sent SIGSTOP is insufficient until the exact task stops scheduling."""
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat", encoding="ascii") as source:
                fields = source.read().rsplit(") ", 1)[1].split()
            if int(fields[19]) != started or fields[0] in {"T", "t", "Z"}:
                return True
        except (OSError, ValueError, IndexError):
            return True
        time.sleep(0.01)
    return False


def process_group_exited(pid: int) -> bool:
    """Absence of the owned group proves every process in that boundary exited."""
    if os.name != "posix" or pid <= 0:
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            with open(entry.path + "/stat", encoding="ascii") as source:
                fields = source.read().rsplit(") ", 1)[1].split()
            if int(fields[2]) == pid and fields[0] != "Z":
                return False
    except (OSError, ValueError, IndexError):
        return False
    return True


def terminate_verified_process(pid: int, started: int, *, grace_seconds: float = 5.0) -> bool:
    """TERM, bounded grace, KILL, and exit check inside the run's process group."""
    if sys.platform != "linux":
        return False
    try:
        pidfd = os.pidfd_open(pid, 0)
    except OSError:
        return False
    try:
        return _terminate_verified_process_held(pid, started, grace_seconds=grace_seconds)
    finally:
        os.close(pidfd)


def _terminate_verified_process_held(pid: int, started: int, *, grace_seconds: float) -> bool:
    if not verified_process_alive(pid, started):
        return False
    if os.name == "posix":
        try:
            if os.getpgid(pid) != pid:
                return False
        except ProcessLookupError:
            return False
        send = lambda sig: os.killpg(pid, sig)
    else:
        send = lambda sig: os.kill(pid, sig)
    try:
        send(signal.SIGTERM)
    except OSError:
        return process_group_exited(pid)
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if not verified_process_alive(pid, started):
            return process_group_exited(pid)
        time.sleep(0.05)
    if not verified_process_alive(pid, started):
        return process_group_exited(pid)
    if os.name == "posix" and os.getpgid(pid) != pid:
        return False
    try:
        send(getattr(signal, "SIGKILL", signal.SIGTERM))
    except OSError:
        return process_group_exited(pid)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not verified_process_alive(pid, started):
            return process_group_exited(pid)
        time.sleep(0.05)
    return process_group_exited(pid)


def signal_verified_group(pid: int, started: int, sig: int) -> bool:
    """Control only the same live run leader and its owned process group."""
    if sys.platform != "linux":
        return False
    try:
        pidfd = os.pidfd_open(pid, 0)
    except OSError:
        return False
    try:
        if not verified_process_alive(pid, started) or os.getpgid(pid) != pid:
            return False
        os.killpg(pid, sig)
    except (OSError, ProcessLookupError):
        return False
    finally:
        os.close(pidfd)
    return True


def signal_verified_pid(pid: int, started: int, sig: int) -> bool:
    """Hold a pidfd across fingerprint validation and exact-process signal."""
    if sys.platform != "linux":
        return False
    try:
        pidfd = os.pidfd_open(pid, 0)
    except OSError:
        return False
    try:
        if not verified_process_alive(pid, started):
            return False
        signal.pidfd_send_signal(pidfd, sig, None, 0)
    except (OSError, ProcessLookupError):
        return False
    finally:
        os.close(pidfd)
    return True


def become_child_subreaper() -> None:
    """Keep daemonized tool descendants attached until this run exits."""
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "run subreaper setup failed")


def _trusted_family(parents: dict[int, tuple[int, int]], root: int,
                    started: int) -> dict[int, int] | None:
    """Follow parent edges only while each parent generation is still verified."""
    family = {root: started}
    while True:
        additions: dict[int, int] = {}
        for child, (parent, child_started) in parents.items():
            if parent not in family or child in family:
                continue
            if not verified_process_alive(parent, family[parent]):
                return None
            if child_started and verified_process_alive(child, child_started):
                additions[child] = child_started
        if not additions:
            return family
        family.update(additions)


def owned_descendant_snapshot(pid: int, started: int) -> list[dict[str, int]] | None:
    """Freeze and fingerprint the entire still-parented process tree."""
    if not signal_verified_group(pid, started, signal.SIGSTOP):
        return None
    frozen: dict[int, int] = {}
    deadline = time.monotonic() + 2.0
    for _ in range(8):
        if not verified_process_alive(pid, started) or not _stopped_or_exited(pid, started, deadline):
            break
        parents: dict[int, tuple[int, int]] = {}
        try:
            entries = os.scandir("/proc")
        except OSError:
            break
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                with open(entry.path + "/stat", encoding="ascii") as source:
                    fields = source.read().rsplit(") ", 1)[1].split()
                # Read PPID (field 4) and starttime (field 22) from one stat
                # snapshot. Never combine an old parent edge with a reused PID.
                parents[int(entry.name)] = (int(fields[1]), int(fields[19]))
            except (OSError, ValueError, IndexError):
                continue
        family = _trusted_family(parents, pid, started)
        if family is None:
            break
        found = {child: fingerprint for child, fingerprint in family.items() if child != pid}
        freeze_failed = False
        for child, fingerprint in found.items():
            if child not in frozen and verified_process_alive(child, fingerprint):
                if not signal_verified_pid(child, fingerprint, signal.SIGSTOP):
                    freeze_failed = True
        if freeze_failed or not all(_stopped_or_exited(child, fingerprint, deadline)
                                    for child, fingerprint in found.items()):
            frozen.update(found)
            break
        if found == frozen:
            return [{"pid": child, "started": fingerprint}
                    for child, fingerprint in sorted(found.items())]
        frozen = found
    for child, fingerprint in frozen.items():
        if verified_process_alive(child, fingerprint):
            signal_verified_pid(child, fingerprint, signal.SIGCONT)
    signal_verified_group(pid, started, signal.SIGCONT)
    return None


def terminate_snapshot_processes(snapshot: list[dict[str, int]]) -> bool:
    """TERM/grace/KILL only fingerprinted descendants in a durable snapshot."""
    members = [(int(row["pid"]), int(row["started"])) for row in snapshot]
    for pid, started in members:
        if verified_process_alive(pid, started):
            signal_verified_pid(pid, started, signal.SIGTERM)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and any(verified_process_alive(pid, started) for pid, started in members):
        time.sleep(0.05)
    for pid, started in members:
        if verified_process_alive(pid, started):
            signal_verified_pid(pid, started, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and any(verified_process_alive(pid, started) for pid, started in members):
        time.sleep(0.05)
    return not any(verified_process_alive(pid, started) for pid, started in members)


def resume_snapshot_processes(snapshot: list[dict[str, int]]) -> None:
    for row in snapshot:
        signal_verified_pid(int(row["pid"]), int(row["started"]), signal.SIGCONT)


def _owned_group_members() -> dict[int, int]:
    """Snapshot verified members while this process still leads the group."""
    if os.name != "posix":
        return {}
    group = os.getpid()
    members = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == group:
            continue
        try:
            with open(entry.path + "/stat", encoding="ascii") as source:
                fields = source.read().rsplit(") ", 1)[1].split()
            if int(fields[2]) == group:
                fingerprint = int(fields[19])
                if fingerprint and verified_process_alive(pid, fingerprint):
                    members[pid] = fingerprint
        except (OSError, ValueError, IndexError):
            continue
    return members


def _owned_descendants() -> dict[int, int] | None:
    """Find this run's descendants even when a tool created a new session."""
    if os.name != "posix":
        return {}
    parents: dict[int, tuple[int, int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(entry.path + "/stat", encoding="ascii") as source:
                fields = source.read().rsplit(") ", 1)[1].split()
            parents[int(entry.name)] = (int(fields[1]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            continue
    root = os.getpid()
    family = _trusted_family(parents, root, process_fingerprint(root))
    if family is None:
        return None
    return {pid: started for pid, started in family.items() if pid != root}


def _freeze_owned_members() -> tuple[dict[int, int], bool]:
    """Stop and stably enumerate this child subreaper's whole family."""
    members: dict[int, int] = {}
    snapshot_verified = False
    deadline = time.monotonic() + 2.0
    for _ in range(8):
        descendants = _owned_descendants()
        if descendants is None:
            break
        found = {**_owned_group_members(), **descendants}
        freeze_failed = False
        for pid, started in found.items():
            if pid not in members and verified_process_alive(pid, started):
                if not signal_verified_pid(pid, started, signal.SIGSTOP):
                    freeze_failed = True
        if freeze_failed or not all(_stopped_or_exited(pid, started, deadline)
                                    for pid, started in found.items()):
            members.update(found)
            break
        if found == members:
            snapshot_verified = True
            break
        members = found
    return members, snapshot_verified


def _cleanup_owned_processes(task_id: str) -> tuple[bool, list[dict[str, int]], bool]:
    """Reap descendants, then freeze exact remaining retry targets before exit."""
    with suppress(Exception):
        from tools.process_registry import process_registry
        process_registry.kill_all(task_id, source="isolated_run_exit", consume_output=True)
    members, initial_verified = _freeze_owned_members()
    for pid, started in members.items():
        if verified_process_alive(pid, started):
            signal_verified_pid(pid, started, signal.SIGCONT)
            signal_verified_pid(pid, started, signal.SIGTERM)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and any(
        verified_process_alive(pid, started) for pid, started in members.items()):
        time.sleep(0.05)
    for pid, started in members.items():
        if verified_process_alive(pid, started):
            signal_verified_pid(pid, started, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and any(
        verified_process_alive(pid, started) for pid, started in members.items()):
        time.sleep(0.05)
    remaining, final_verified = _freeze_owned_members()
    snapshot = [{"pid": pid, "started": started} for pid, started in sorted(remaining.items())]
    snapshot_verified = initial_verified and final_verified
    return snapshot_verified and not remaining, snapshot, snapshot_verified


def _persist_cleanup_proof(path: str | None, run_id: str, pid: int, started: int,
                           verified: bool, descendants: list[dict[str, int]],
                           snapshot_verified: bool) -> None:
    """Retain cleanup targets and proof across gateway loss."""
    if not path:
        return
    with sqlite3.connect(path, timeout=5) as conn:
        conn.execute(
            "UPDATE run_idempotency SET status_json=json_set(status_json,"
            "'$.detached_cleanup_verified',json(?),'$.execution_descendants',json(?),"
            "'$.descendant_snapshot_verified',json(?)), "
            "updated_at=? WHERE run_id=? AND json_extract(status_json,'$.execution_pid')=? "
            "AND json_extract(status_json,'$.execution_started')=?",
            (json.dumps(verified), json.dumps(descendants), json.dumps(snapshot_verified),
             time.time(), run_id, pid, started))


def _real_activity(label: str) -> str | None:
    """Timer ticks, transport waits and watchdog messages are not execution."""
    if label.startswith(("executing tool:", "executing ")):
        return "tool_running"
    if label.startswith("tool completed:"):
        return "active"
    if label == "receiving stream response":
        return "model_wait"
    if label.startswith("starting API call #"):
        return "model_wait"
    if label.startswith("API call #") and label.endswith(" completed"):
        return "active"
    if label == "starting new turn":
        return "starting"
    return None


def main() -> int:
    """Run one agent in a fresh interpreter; emit private JSONL observations."""
    control_fd = int(os.environ.pop("HERMES_RUN_CONTROL_FD"))
    os.set_inheritable(control_fd, False)
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import clear_session_vars
    from tools.approval import register_gateway_notify, resolve_gateway_approval, unregister_gateway_notify

    launch = json.loads(sys.stdin.readline())
    become_child_subreaper()
    if os.name == "posix":
        def _term(_signal, _frame):
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, _term)
    run_id, session_id = launch["run_id"], launch["session_id"]
    output_lock = threading.Lock()

    def emit(kind: str, **fields) -> None:
        with output_lock:
            pending = memoryview((FRAME_PREFIX + json.dumps(
                {"kind": kind, **fields}, default=str) + "\n").encode())
            while pending:
                pending = pending[os.write(control_fd, pending):]

    def commands() -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
                if command.get("kind") == "approval":
                    resolved = resolve_gateway_approval(
                        run_id, command["choice"], resolve_all=bool(command.get("resolve_all")),
                        request_id=command.get("request_id"))
                    emit("approval_response", resolved=resolved, command_id=command.get("command_id"))
            except Exception:
                continue

    threading.Thread(target=commands, name="run-control", daemon=True).start()
    adapter = APIServerAdapter(PlatformConfig())
    tokens = []
    try:
        with adapter._profile_scope(launch.get("request_profile")):
            tokens = adapter._bind_api_server_session(
                chat_id=session_id, session_key=run_id, session_id=session_id,
                profile=launch.get("request_profile") or "",
                browser_control_principal=launch.get("browser_control_principal") or "",
                browser_control_transport_family=launch.get("browser_control_transport_family") or "",
                session_history_delivery="1" if launch.get("session_history_delivery") else "")
            register_gateway_notify(run_id, lambda event: emit("approval", event=event))
            def tool_event(event_type, tool_name=None, preview=None, args=None, **kw):
                emit("tool_event", event_type=event_type,
                     fields={"tool_name": tool_name, "preview": preview, "args": args, **kw})
            agent = adapter._create_agent(
                stream_delta_callback=lambda delta: emit("delta", delta=delta) if delta else None,
                tool_progress_callback=tool_event,
                interim_assistant_callback=lambda text, **kw: emit("interim", text=text, fields=kw),
                **launch["agent_kwargs"])
            agent._run_activity_callback = lambda label: emit("activity", phase=phase) if (
                phase := _real_activity(label)) else None
            result = agent.run_conversation(
                user_message=launch["user_message"],
                conversation_history=launch["conversation_history"], task_id=session_id,
                **({"turn_author": launch["turn_author"]} if launch.get("turn_author") else {}))
            if not isinstance(result, dict):
                result = {}
            from gateway.platforms.api_server_runs import _run_usage, _served_runtime
            public_result = {
                key: result.get(key) for key in (
                    "completed", "partial", "interrupted", "failed", "turn_exit_reason",
                    "pending_steer", "final_response", "error") if key in result}
            emit("result", result=public_result, usage=_run_usage(agent), runtime=_served_runtime(agent))
            return 0
    except BaseException as exc:
        emit("error", error=type(exc).__name__)
        return 1
    finally:
        cleanup_verified = False
        descendants: list[dict[str, int]] = []
        snapshot_verified = False
        with suppress(Exception):
            cleanup_verified, descendants, snapshot_verified = _cleanup_owned_processes(session_id)
        with suppress(Exception):
            _persist_cleanup_proof(launch.get("run_store_path"), run_id, os.getpid(),
                                   process_fingerprint(os.getpid()), cleanup_verified,
                                   descendants, snapshot_verified)
        with suppress(Exception):
            emit("cleanup", verified=cleanup_verified, descendants=descendants,
                 snapshot_verified=snapshot_verified)
        with suppress(Exception):
            unregister_gateway_notify(run_id)
        if tokens:
            with suppress(Exception):
                clear_session_vars(tokens)
        with suppress(Exception):
            adapter._close_run_state()


if __name__ == "__main__":
    raise SystemExit(main())
