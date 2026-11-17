"""Strict argv builders for the allowlisted Sonos command surface."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from hermes_constants import get_hermes_home


def _script() -> Path:
    return get_hermes_home() / "sonos-brain" / "sonos_brain.py"


def _clean_text(value: str, *, limit: int = 280) -> str:
    text = " ".join(str(value).split())
    if not text or len(text) > limit or any(ord(char) < 32 for char in text):
        raise ValueError("Text must be 1–280 printable characters.")
    return text


def _speaker_command_argv(
    action: str,
    *,
    text: str = "",
    duration: str = "",
    volume_action: str = "",
    volume_level: str = "",
    rule: str = "random",
    oracle_event: str = "",
    oracle_detail: str = "",
    sleep_timer_hours: str = "",
    clip_path: str = "",
    query: str = "",
    track_id: str = "",
    limit: int = 10,
    switch: bool = False,
    regenerate: bool = False,
) -> list[str]:
    """Build a validated command without a shell; chat text never becomes shell syntax."""
    action = str(action).strip().lower()
    values = locals()
    if action in {"status", "start", "stop", "next"}:
        args = [action]
    else:
        builder = _ACTION_BUILDERS.get(action)
        if builder is None:
            raise ValueError("Unknown speaker action.")
        args = builder(values)
    return [sys.executable, str(_script()), *args]


def _volume_args(values: dict[str, object]) -> list[str]:
    operation = str(values["volume_action"]).strip().lower()
    level = str(values["volume_level"]).strip()
    if not operation and not level:
        return ["volume", "status"]
    if operation in {"status", "up", "down", "mute", "unmute"} and not level:
        return ["volume", operation]
    if not operation and level.isdigit() and 0 <= int(level) <= 100:
        return ["volume", str(int(level))]
    if not operation and level.isdigit():
        raise ValueError("Volume must be between 0 and 100.")
    raise ValueError("Volume must be up, down, mute, unmute, or an integer from 0 to 100.")


def _fm_args(values: dict[str, object]) -> list[str]:
    value = str(values["duration"]).strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(m|h)", value)
    if not match:
        raise ValueError("FM duration must look like 30m or 1.5h.")
    minutes = float(match.group(1)) * (60 if match.group(2) == "h" else 1)
    if not 15 <= minutes <= 480:
        raise ValueError("FM duration must be between 15 minutes and 8 hours.")
    switch = bool(values["switch"])
    return ["fm", "--duration", value, "--switch" if switch else "--only-if-idle", "--no-wait"]


def _overlay_args(action: str, values: dict[str, object]) -> list[str]:
    args = [action, "--overlay", "--duck-by", "18", "--only-if-idle", "--no-wait"]
    text = str(values["text"])
    return [*args, "--text", _clean_text(text)] if text.strip() else args


def _roulette_args(values: dict[str, object]) -> list[str]:
    allowed = {"random", "longest-title", "shortest-title", "newest", "oldest"}
    selected = str(values["rule"]).strip().lower()
    if selected not in allowed:
        raise ValueError("Unknown roulette rule.")
    return ["roulette", "--rule", selected, "--only-if-idle", "--no-wait"]


def _brief_args(action: str) -> list[str]:
    brief = "hn-brief" if action == "hn" else "podcast-brief"
    return [brief, "--only-if-idle", "--no-wait"]


def _oracle_args(values: dict[str, object]) -> list[str]:
    args = ["oracle", _clean_text(str(values["oracle_event"]), limit=80)]
    detail = str(values["oracle_detail"])
    if detail.strip():
        args.append(_clean_text(detail))
    return [*args, "--only-if-idle", "--no-wait"]


def _sleep_timer_args(values: dict[str, object]) -> list[str]:
    hours = str(values["sleep_timer_hours"]).strip().lower().removesuffix("h")
    if hours == "off":
        return ["sleep-timer", "--off"]
    if hours in {"1", "2", "3", "4", "6", "8", "12"}:
        return ["sleep-timer", "--hours", hours, "--no-wait"]
    raise ValueError("Sleep timer must be off or 1h, 2h, 3h, 4h, 6h, 8h, or 12h.")


def _plex_search_args(values: dict[str, object]) -> list[str]:
    query = _clean_text(str(values["query"]), limit=120)
    limit = values["limit"]
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Plex search result limit must be between 1 and 20.")
    return ["plex-search", "--query", query, "--limit", str(limit)]


def _plex_play_args(values: dict[str, object]) -> list[str]:
    track_id = str(values["track_id"]).strip()
    if not re.fullmatch(r"\d{1,20}", track_id):
        raise ValueError("Invalid Plex track id.")
    return ["plex-play", track_id, "--no-wait"]


def _clip_args(values: dict[str, object]) -> list[str]:
    clip_path = str(values["clip_path"])
    path = Path(clip_path)
    if not clip_path or "\x00" in clip_path or path.suffix.lower() not in {".ogg", ".mp3", ".wav", ".m4a"}:
        raise ValueError("Choose a supported local audio clip.")
    return ["clip", str(path), "--duck-by", "18"]


def _podcast_readout_args(values: dict[str, object]) -> list[str]:
    podcast_id = str(values["text"]).strip()
    if not re.fullmatch(r"[A-Fa-f0-9]{8,32}", podcast_id):
        raise ValueError("Invalid podcast id.")
    args = ["podcast-readout", podcast_id,
            "--switch" if values["switch"] else "--only-if-idle", "--no-wait"]
    return [*args, "--regenerate"] if values["regenerate"] else args


def _npc_args(values: dict[str, object]) -> list[str]:
    return _overlay_args("npc", values)


def _haunted_args(values: dict[str, object]) -> list[str]:
    return _overlay_args("haunted", values)


_ACTION_BUILDERS: dict[str, Callable[[dict[str, object]], list[str]]] = {
    "volume": _volume_args,
    "fm": _fm_args,
    "npc": _npc_args,
    "haunted": _haunted_args,
    "roulette": _roulette_args,
    "hn": lambda _: _brief_args("hn"),
    "podcast": lambda _: _brief_args("podcast"),
    "oracle": _oracle_args,
    "sleep-timer": _sleep_timer_args,
    "sleep_timer": _sleep_timer_args,
    "plex-search": _plex_search_args,
    "plex-play": _plex_play_args,
    "clip": _clip_args,
    "podcast-list": lambda _: ["podcast-list", "--limit", "20"],
    "podcast-readout": _podcast_readout_args,
}


def _podcast_reader_command_argv(
    action: str,
    *,
    summary_id: str = "",
    switch: bool = False,
    regenerate: bool = False,
) -> list[str]:
    """Build allowlisted commands for saved podcast summaries."""
    if action == "list":
        return _speaker_command_argv("podcast-list")
    if action in {"play", "switch"}:
        return _speaker_command_argv(
            "podcast-readout", text=summary_id, switch=(switch or action == "switch"), regenerate=regenerate)
    raise ValueError("Unknown podcast reader action.")
