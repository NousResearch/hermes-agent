"""Deterministic, LLM-free handlers for local speaker commands."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import subprocess
from pathlib import Path
from typing import Any
from hermes_constants import get_hermes_home

_SPEAKER_USAGE = """*Speaker commands*
• `/speaker status` — show the speaker state
• `/speaker start` — resume the current media without interrupting it
• `/speaker stop` — stop playback
• `/speaker volume up|down|mute|unmute|0-100`
• `/speaker fm 30m|1h|1.5h` — idle-safe FM
• `/speaker schedule` — show the schedule setup
• `/speaker roulette [random|longest-title|shortest-title|newest|oldest]`
• `/speaker npc <text>` or `/speaker haunted <text>`
• `/speaker hn` or `/speaker podcast` — idle-safe briefs
• `/speaker oracle <event> [details]`
• `/speaker sleep-timer <1h|2h|3h|4h|6h|8h|12h|off>`

Roulette rules: `random` · `longest-title` · `shortest-title` · `newest` · `oldest`"""

_LONG_RUNNING_ACTIONS = frozenset({"start", "fm", "npc", "haunted", "roulette", "hn", "podcast", "oracle", "clip", "podcast-readout"})


def _speaker_args(text: str) -> tuple[str, dict[str, Any]]:
    parts = shlex.split(text)
    args = parts[1:] if parts and parts[0].split("@", 1)[0].lower() == "/speaker" else parts
    if not args:
        raise ValueError(_SPEAKER_USAGE)
    action = args[0].lower().replace("_", "-")
    rest = args[1:]
    if action == "volume":
        if not rest:
            return action, {"volume_action": ""}
        return action, {"volume_action": rest[0].lower() if rest[0].lower() in {"up", "down", "mute", "unmute"} else "",
                        "volume_level": "" if rest[0].lower() in {"up", "down", "mute", "unmute"} else rest[0]}
    if action == "fm":
        if len(rest) != 1:
            raise ValueError("Usage: /speaker fm <30m|1h|1.5h>.")
        return action, {"duration": rest[0]}
    if action in {"npc", "haunted"}:
        return action, {"text": " ".join(rest)}
    if action == "roulette":
        return action, {"rule": rest[0] if rest else "random"}
    if action == "oracle":
        if not rest:
            raise ValueError("Usage: /speaker oracle <event> [details].")
        return action, {"oracle_event": rest[0], "oracle_detail": " ".join(rest[1:])}
    if action == "sleep-timer":
        if len(rest) != 1:
            raise ValueError("Usage: /speaker sleep-timer <1h|2h|3h|4h|6h|8h|12h|off>.")
        return action, {"sleep_timer_hours": rest[0]}
    if action == "clip":
        if len(rest) != 1:
            raise ValueError("Usage: /speaker clip <local-audio-path>.")
        return action, {"clip_path": rest[0]}
    if action == "podcast-readout":
        if not rest:
            raise ValueError("Usage: /speaker podcast-readout <summary-id>.")
        return action, {"text": rest[0]}
    return action, {}


def _speaker_root() -> Path:
    return get_hermes_home() / "sonos-brain"


class GatewayHardwareCommandsMixin:
    """Gateway command handlers; adapter panels use the same strict argv builder."""

    async def _handle_speaker_command(self, event) -> str:
        from integrations.sonos_commands import _speaker_command_argv

        try:
            action, options = _speaker_args(getattr(event, "text", "") or "")
            if action == "schedule":
                return ("FM schedule: use the Telegram speaker panel’s Schedule FM button. "
                        "Example schedule: `08:00 1h weekdays`.")
            argv = _speaker_command_argv(action, **options)
        except (ValueError, TypeError) as exc:
            return str(exc) or _SPEAKER_USAGE

        try:
            is_long_running = action in _LONG_RUNNING_ACTIONS
            # health: allow HX006 -- selected background actions intentionally persist; every short action below has a deadline and kill/reap path
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(_speaker_root()),
                stdout=asyncio.subprocess.DEVNULL if is_long_running else asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL if is_long_running else asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"Speaker command could not start: {type(exc).__name__}."

        if is_long_running and getattr(process, "returncode", None) is None:
            if action == "start":
                return "Speaker command `start` started. It will not interrupt current playback."
            return f"Speaker command `{action}` started; playback remains idle-safe unless you selected an explicit switch."

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=12)
        except TimeoutError:
            process.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=2)
            return f"Speaker command `{action}` timed out and was stopped."
        except (OSError, asyncio.SubprocessError) as exc:
            return f"Speaker command `{action}` failed: {type(exc).__name__}."
        output = self._safe_speaker_output(stdout or stderr or b"")
        if getattr(process, "returncode", 0):
            return f"Speaker command `{action}` failed (exit {process.returncode})." + (f"\n{output}" if output else "")
        return output or f"Speaker command `{action}` completed."

    @staticmethod
    def _safe_speaker_output(raw: bytes | str) -> str:
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        try:
            from agent.redact import redact_sensitive_text
            text = redact_sensitive_text(text, force=True)
        except (ImportError, TypeError, ValueError):
            text = "Speaker output redacted."
        safe_lines = [line for line in text.splitlines()
                      if not any(marker in line.lower() for marker in ("http://", "https://", "plex-token", "x-plex-token", "uri="))]
        return "\n".join(safe_lines)[:2500]
