"""Subprocess lifecycle manager for the google_meet bot.

One active meeting at a time is recorded under the active profile's
``$HERMES_HOME/workspace/meetings``. The bot is a detached subprocess reached
through its state, transcript, and queue files so the agent loop never blocks.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from plugins.google_meet._jsonfile import read_json
from plugins.google_meet.queue_io import append_jsonl
from utils import atomic_json_write


_NO_ACTIVE = {"ok": False, "reason": "no active meeting"}

# These are behavioral bridge values. They must not be inherited from the
# gateway's ambient environment; the active bot-host profile owns them.
_MEET_CONFIG_ENV_VARS = (
    "HERMES_MEET_DEBUG_STATUS",
    "HERMES_MEET_PROXY_SERVER",
    "HERMES_MEET_PROXY_BYPASS",
    "HERMES_MEET_REALTIME_READY_TIMEOUT",
    "HERMES_MEET_STALL_AFTER",
    "HERMES_MEET_XVFB",
)

# Start inputs are request-scoped, not persistent process settings. Clearing
# them prevents a previous shell launch from supplying an implicit duration or
# authentication state to a new bot.
_MEET_START_ENV_VARS = (
    "HERMES_MEET_URL",
    "HERMES_MEET_OUT_DIR",
    "HERMES_MEET_GUEST_NAME",
    "HERMES_MEET_HEADED",
    "HERMES_MEET_AUTH_STATE",
    "HERMES_MEET_DURATION",
    "HERMES_MEET_MODE",
    "HERMES_MEET_REALTIME_MODEL",
    "HERMES_MEET_REALTIME_VOICE",
    "HERMES_MEET_REALTIME_INSTRUCTIONS",
)


def _root() -> Path:
    return Path(get_hermes_home()) / "workspace" / "meetings"


def _active_file() -> Path:
    return _root() / ".active.json"


def _last_file() -> Path:
    return _root() / ".last.json"


def _read_active() -> Optional[Dict[str, Any]]:
    return read_json(_active_file())


def _read_last() -> Optional[Dict[str, Any]]:
    return read_json(_last_file())


def _clean_session_id(value: Any) -> Optional[str]:
    session_id = str(value or "").strip()
    return session_id or None


def _read_status(out_dir: Path) -> Dict[str, Any]:
    status = read_json(out_dir / "status.json")
    return status if isinstance(status, dict) else {}


def _write_last(data: Dict[str, Any]) -> None:
    atomic_json_write(_last_file(), data)


def _write_active(data: Dict[str, Any]) -> None:
    atomic_json_write(_active_file(), data)
    _write_last(data)


def _clear_active() -> None:
    with contextlib.suppress(FileNotFoundError):
        _active_file().unlink()


def _pid_alive(pid: int) -> bool:
    # Not ``os.kill(pid, 0)``: on Windows that can kill the target (bpo-14484).
    from gateway.status import _pid_exists
    return bool(pid) and _pid_exists(pid)


def _kill(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, sig)


def _record_stop_reason(active: Dict[str, Any], reason: str) -> None:
    out_dir = active.get("out_dir")
    if not out_dir:
        return
    status_path = Path(out_dir) / "status.json"
    status = _read_status(Path(out_dir))
    status.update(
        {
            "meetingId": active.get("meeting_id"),
            "url": active.get("url"),
            "exited": True,
            "leaveReason": (reason or "requested").strip() or "requested",
        }
    )
    atomic_json_write(status_path, status)


def _positive_seconds(value: Any, default: int) -> str:
    try:
        if float(value) > 0:
            return str(value)
    except (TypeError, ValueError):
        pass
    return str(default)


def _resolve_meet_config() -> Dict[str, Any]:
    """Read effective config for the currently bound bot-host profile."""
    from hermes_cli.config_effective import load_user_config_effective

    root = load_user_config_effective()
    config = root.get("google_meet", {}) if isinstance(root, dict) else {}
    if not isinstance(config, dict):
        config = {}
    proxy = config.get("proxy", {})
    if not isinstance(proxy, dict):
        proxy = {}
    return {
        "debug_status": config.get("debug_status") is True,
        "xvfb": config.get("xvfb", "auto"),
        "proxy_server": str(proxy.get("server") or "").strip(),
        "proxy_bypass": proxy.get("bypass"),
        "realtime_ready_timeout": _positive_seconds(
            config.get("realtime_ready_timeout"), 15
        ),
        "stall_after": _positive_seconds(config.get("stall_after"), 90),
    }


def _apply_meet_config_to_env(env: Dict[str, str], config: Dict[str, Any]) -> None:
    """Bridge only profile-owned behavioral config into the child process."""
    for name in _MEET_CONFIG_ENV_VARS:
        env.pop(name, None)
    if config["debug_status"]:
        env["HERMES_MEET_DEBUG_STATUS"] = "1"
    if config["proxy_server"]:
        env["HERMES_MEET_PROXY_SERVER"] = config["proxy_server"]
        # ``None`` preserves the bot's WebRTC-safe bypass default; an empty
        # string is an explicit config choice to disable that bypass.
        if config["proxy_bypass"] is not None:
            env["HERMES_MEET_PROXY_BYPASS"] = str(config["proxy_bypass"])
    env["HERMES_MEET_REALTIME_READY_TIMEOUT"] = config["realtime_ready_timeout"]
    env["HERMES_MEET_STALL_AFTER"] = config["stall_after"]


def _apply_start_env(
    env: Dict[str, str],
    *,
    url: str,
    out_dir: Path,
    guest_name: str,
    headed: bool,
    auth_state: Optional[str],
    duration: Optional[str],
    mode: str,
    realtime_model: Optional[str],
    realtime_voice: Optional[str],
    realtime_instructions: Optional[str],
) -> None:
    for name in _MEET_START_ENV_VARS:
        env.pop(name, None)
    env.update(
        {
            "HERMES_MEET_URL": url,
            "HERMES_MEET_OUT_DIR": str(out_dir),
            "HERMES_MEET_GUEST_NAME": guest_name,
            "HERMES_MEET_MODE": mode,
        }
    )
    if headed:
        env["HERMES_MEET_HEADED"] = "1"
    if auth_state:
        env["HERMES_MEET_AUTH_STATE"] = auth_state
    if duration:
        env["HERMES_MEET_DURATION"] = duration
    if realtime_model:
        env["HERMES_MEET_REALTIME_MODEL"] = realtime_model
    if realtime_voice:
        env["HERMES_MEET_REALTIME_VOICE"] = realtime_voice
    if realtime_instructions:
        env["HERMES_MEET_REALTIME_INSTRUCTIONS"] = realtime_instructions


def _apply_realtime_credential(
    env: Dict[str, str],
    *,
    mode: str,
    realtime_api_key: Optional[str],
) -> None:
    """Pass just the scoped realtime credential, never ambient provider secrets."""
    env.pop("HERMES_MEET_REALTIME_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    if mode != "realtime":
        return
    if not realtime_api_key:
        from agent.secret_scope import get_secret

        realtime_api_key = get_secret("HERMES_MEET_REALTIME_KEY") or get_secret(
            "OPENAI_API_KEY"
        )
    if realtime_api_key:
        env["HERMES_MEET_REALTIME_KEY"] = realtime_api_key


def _headed_launch_prefix(
    policy: Any, env: Dict[str, str]
) -> tuple[list[str], bool, Optional[str]]:
    """Return a Linux-only Xvfb prefix for an explicitly headed launch."""
    if not sys.platform.startswith("linux"):
        return [], False, None

    normalized = str(policy if policy is not None else "auto").strip().lower()
    display = env.get("DISPLAY", "").strip()
    disabled = {"0", "false", "no", "off", "disable", "disabled"}
    forced = {"1", "true", "yes", "on", "force", "forced"}

    if normalized in disabled:
        if display:
            return [], False, None
        return (
            [],
            False,
            (
                "headed Meet launch requested, but DISPLAY is unset and google_meet.xvfb disables xvfb-run"
            ),
        )
    if display and normalized not in forced:
        return [], False, None

    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run:
        return [xvfb_run, "-a"], True, None
    if display:
        return [], False, None
    return (
        [],
        False,
        (
            "headed Meet launch requested, but DISPLAY is unset and xvfb-run is unavailable; "
            "set headed=false or install xvfb-run"
        ),
    )


def start(
    url: str,
    *,
    out_dir: Optional[Path] = None,
    headed: bool = False,
    auth_state: Optional[str] = None,
    guest_name: str = "Hermes Agent",
    duration: Optional[str] = None,
    persist_after_session: bool = False,
    session_id: Optional[str] = None,
    mode: str = "transcribe",
    realtime_model: Optional[str] = None,
    realtime_voice: Optional[str] = None,
    realtime_instructions: Optional[str] = None,
    realtime_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Spawn the Meet bot, replacing a live prior bot for this profile."""
    from plugins.google_meet.meet_bot import _is_safe_meet_url, _meeting_id_from_url

    if not _is_safe_meet_url(url):
        return {
            "ok": False,
            "error": "refusing: only https://meet.google.com/ URLs are allowed. got: "
            + repr(url),
        }

    existing = _read_active()
    if existing:
        if _pid_alive(int(existing.get("pid", 0) or 0)):
            stop(reason="replaced by new meet_join")
        else:
            _clear_active()

    meeting_id = _meeting_id_from_url(url)
    out = Path(out_dir) if out_dir is not None else _root() / meeting_id
    out.mkdir(parents=True, exist_ok=True)
    for name in ("transcript.txt", "status.json"):
        with contextlib.suppress(OSError):
            (out / name).unlink()

    from tools.environments.local import served_profile_child_env

    env = served_profile_child_env(
        target_home=get_hermes_home(), inherit_credentials=False
    )
    meet_config = _resolve_meet_config()
    _apply_meet_config_to_env(env, meet_config)
    _apply_start_env(
        env,
        url=url,
        out_dir=out,
        guest_name=guest_name,
        headed=headed,
        auth_state=auth_state,
        duration=duration,
        mode=mode,
        realtime_model=realtime_model,
        realtime_voice=realtime_voice,
        realtime_instructions=realtime_instructions,
    )
    _apply_realtime_credential(env, mode=mode, realtime_api_key=realtime_api_key)

    cmd = [sys.executable, "-m", "plugins.google_meet.meet_bot"]
    xvfb = False
    if headed:
        prefix, xvfb, error = _headed_launch_prefix(meet_config["xvfb"], env)
        if error:
            return {"ok": False, "error": error}
        cmd = [*prefix, *cmd]

    log_path = out / "bot.log"
    with open(log_path, "ab", buffering=0) as log_file:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )

    record = {
        "pid": proc.pid,
        "meeting_id": meeting_id,
        "out_dir": str(out),
        "url": url,
        "started_at": time.time(),
        "duration": duration,
        "persist_after_session": bool(persist_after_session),
        "session_id": _clean_session_id(session_id),
        "log_path": str(log_path),
        "mode": mode,
        "headed": bool(headed),
        "xvfb": xvfb,
    }
    _write_active(record)
    return {"ok": True, **record}


def status() -> Dict[str, Any]:
    """Return current process and bot status without exposing a finished bot as active."""
    active = _read_active()
    if not active:
        return dict(_NO_ACTIVE)

    pid = int(active.get("pid", 0) or 0)
    alive = _pid_alive(pid) if pid else False
    bot_status = _read_status(Path(active.get("out_dir", "")))
    if pid and not alive:
        _clear_active()
        return {
            "ok": False,
            "reason": "no active meeting",
            "lastStatus": bot_status,
            "meetingId": active.get("meeting_id"),
            "url": active.get("url"),
            "outDir": active.get("out_dir"),
            "sessionId": active.get("session_id"),
        }

    return {
        "ok": True,
        "alive": alive,
        "pid": pid,
        "meetingId": active.get("meeting_id"),
        "url": active.get("url"),
        "startedAt": active.get("started_at"),
        "duration": active.get("duration"),
        "persistAfterSession": bool(active.get("persist_after_session")),
        "outDir": active.get("out_dir"),
        "sessionId": active.get("session_id"),
        **bot_status,
    }


def transcript(
    last: Optional[int] = None,
    *,
    include_finished: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Read an active transcript, or an explicitly requested matching finished one."""
    active = _read_active()
    from_last = False
    if active:
        pid = int(active.get("pid", 0) or 0)
        if not pid or not _pid_alive(pid):
            _clear_active()
            active = None

    if not active and include_finished:
        active = _read_last()
        from_last = bool(active)
        if active:
            requested_session_id = _clean_session_id(session_id)
            active_session_id = _clean_session_id(active.get("session_id"))
            if not requested_session_id:
                return {
                    "ok": False,
                    "reason": "finished transcript requires session id",
                }
            if active_session_id != requested_session_id:
                return {"ok": False, "reason": "no finished meeting for this session"}

    if not active:
        return dict(_NO_ACTIVE)

    out_dir = Path(active.get("out_dir", ""))
    transcript_path = out_dir / "transcript.txt"
    lines = []
    if transcript_path.is_file():
        text = transcript_path.read_text(encoding="utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
    selected = lines[-last:] if last else lines
    bot_status = _read_status(out_dir)
    return {
        "ok": True,
        "meetingId": active.get("meeting_id"),
        "sessionId": active.get("session_id"),
        "lines": selected,
        "total": len(lines),
        "path": str(transcript_path),
        "active": not from_last,
        "fromLast": from_last,
        "stale": from_last,
        "leaveReason": bot_status.get("leaveReason"),
        "error": bot_status.get("error"),
    }


def enqueue_say(text: str) -> Dict[str, Any]:
    """Queue speech only when the live realtime bot can receive microphone audio."""
    text = (text or "").strip()
    if not text:
        return {"ok": False, "reason": "text is required"}

    active = _read_active()
    if not active:
        return dict(_NO_ACTIVE)
    if active.get("mode") != "realtime":
        return {
            "ok": False,
            "reason": (
                "active meeting is in transcribe mode — pass mode='realtime' to meet_join "
                "to enable agent speech"
            ),
        }

    pid = int(active.get("pid", 0) or 0)
    if not pid or not _pid_alive(pid):
        _clear_active()
        return dict(_NO_ACTIVE)

    out_dir = Path(active.get("out_dir", ""))
    if not out_dir.is_dir():
        return {"ok": False, "reason": f"out_dir missing: {out_dir}"}

    bot_status = _read_status(out_dir)
    if bot_status.get("exited"):
        return {"ok": False, "reason": "active realtime meeting has exited"}
    if bot_status.get("error") or bot_status.get("leaveReason"):
        detail = bot_status.get("error") or bot_status.get("leaveReason")
        return {
            "ok": False,
            "reason": f"active realtime meeting is not usable: {detail}",
        }
    if not bot_status.get("inCall"):
        return {"ok": False, "reason": "active realtime meeting is not in call yet"}
    if not (bot_status.get("realtime") and bot_status.get("realtimeReady")):
        return {"ok": False, "reason": "realtime is not ready"}
    pump_pid = int(bot_status.get("realtimeAudioPumpPid", 0) or 0)
    if (
        bot_status.get("realtimeAudioPumpStatus") != "ready"
        or not pump_pid
        or not _pid_alive(pump_pid)
    ):
        return {"ok": False, "reason": "realtime audio pump is not ready"}
    if bot_status.get("localMicrophoneOn") is not True:
        return {"ok": False, "reason": "realtime microphone is not enabled"}

    queue_path = out_dir / "say_queue.jsonl"
    entry = {"id": uuid.uuid4().hex[:12], "text": text}
    append_jsonl(queue_path, entry)
    return {
        "ok": True,
        "meetingId": active.get("meeting_id"),
        "enqueued_id": entry["id"],
        "queue_path": str(queue_path),
    }


def stop(*, reason: str = "requested") -> Dict[str, Any]:
    """Terminate the active bot, record its leave reason, and clear the active pointer."""
    active = _read_active()
    if not active:
        return dict(_NO_ACTIVE)

    pid = int(active.get("pid", 0) or 0)
    out_dir = active.get("out_dir")
    transcript_path = Path(out_dir) / "transcript.txt" if out_dir else None
    if pid and _pid_alive(pid):
        _kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.5)
        if _pid_alive(pid):
            _kill(pid, signal.SIGKILL)

    _record_stop_reason(active, reason)
    _clear_active()
    return {
        "ok": True,
        "reason": reason,
        "meetingId": active.get("meeting_id"),
        "transcriptPath": str(transcript_path) if transcript_path else None,
    }
