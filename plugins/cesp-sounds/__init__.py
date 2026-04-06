"""
Hermes CESP Sounds Plugin
==========================

Wires Hermes lifecycle events to the OpenPeon CESP (Coding Event Sound Pack)
v1.0 standard, playing audio notifications automatically — no MCP or LLM
decision-making needed.

Event mapping:
  on_session_start          → session.start
  on_session_end            → session.end
  pre_tool_call(clarify)    → input.required
  post_tool_call(error)     → task.error

NOTE: task.complete is DISABLED by default because post_tool_call fires on
every single tool invocation in Hermes, which is way too noisy. Re-enable
by setting CESP_ENABLE_TASK_COMPLETE=1.

Only fires from the interactive CLI session (platform == 'cli').
Gateway/cron sessions stay silent by default — configurable via env var.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _find_default_packs_dir() -> Optional[Path]:
    """Search common peon-ping install locations in priority order."""
    candidates = [
        Path.home() / ".claude" / "hooks" / "peon-ping" / "packs",      # Homebrew install
        Path.home() / ".local" / "share" / "openpeon" / "packs",        # XDG
        Path.home() / ".config" / "peon-ping" / "packs",                # Config
        Path.home() / ".openpeon" / "packs",                             # Spec default
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


_PACKS_DIR = _find_default_packs_dir()
PACKS_DIR = Path(os.environ.get("CESP_PACKS_DIR", _PACKS_DIR or Path.home() / ".claude" / "hooks" / "peon-ping" / "packs"))

# Master volume 0.0 - 1.0
VOLUME = float(os.environ.get("CESP_VOLUME", "0.5"))

# Minimum seconds between sounds in the same category (debounce)
DEBOUNCE_SEC = float(os.environ.get("CESP_DEBOUNCE", "2.0"))

# task.complete was too noisy — disabled by default. Set CESP_ENABLE_TASK_COMPLETE=1 to re-enable.
ENABLE_TASK_COMPLETE = os.environ.get("CESP_ENABLE_TASK_COMPLETE", "0") == "1"

# Which Hermes platforms should emit sounds (CLI + optionally gateway)
ENABLED_PLATFORMS = set(
    p.strip()
    for p in os.environ.get("CESP_PLATFORMS", "cli").split(",")
    if p.strip()
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_last_sound: Dict[str, float] = {}  # category -> timestamp
_last_sound_name: Dict[str, str] = {}  # category -> filename (no-repeat)
_mute = False
_state_path = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "plugins" / "cesp-sounds" / "cesp-state.json"


def _load_state() -> None:
    global _last_sound, _last_sound_name, _mute
    if _state_path.exists():
        try:
            data = json.loads(_state_path.read_text())
            _last_sound = data.get("last_sound", {})
            _last_sound_name = data.get("last_sound_name", {})
            _mute = data.get("mute", False)
        except Exception:
            pass


def _save_state() -> None:
    try:
        _state_path.parent.mkdir(parents=True, exist_ok=True)
        _state_path.write_text(json.dumps({
            "last_sound": _last_sound,
            "last_sound_name": _last_sound_name,
            "mute": _mute,
        }))
    except Exception:
        pass


def _get_active_pack() -> Optional[Path]:
    """Find the active sound pack.

    Priority:
    1. CESP_ACTIVE_PACK env var (pack name)
    2. Most recently modified pack in PACKS_DIR
    3. Fall back to 'peon' if it exists
    """
    env_pack = os.environ.get("CESP_ACTIVE_PACK", "").strip()
    if env_pack:
        candidate = PACKS_DIR / env_pack
        if candidate.is_dir():
            return candidate

    if not PACKS_DIR.is_dir():
        return None

    packs = sorted(PACKS_DIR.iterdir(),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in packs:
        if (p / "openpeon.json").exists():
            return p

    # Fallback: peon pack
    peon = PACKS_DIR / "peon"
    if peon.is_dir() and (peon / "openpeon.json").exists():
        return peon

    return None


def _resolve_sound(pack_dir: Path, category: str) -> Optional[Path]:
    """Load manifest and pick a sound for the category.

    Uses no-repeat logic (excludes last-played sound if alternatives exist).
    """
    manifest_path = pack_dir / "openpeon.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None

    categories = manifest.get("categories", {})
    aliases = manifest.get("category_aliases", {})

    # Resolve category (try direct, then alias)
    cat_data = categories.get(category)
    if cat_data is None and category in aliases:
        cat_data = categories.get(aliases[category])

    if not cat_data:
        return None

    sounds = cat_data.get("sounds", [])
    if not sounds:
        return None

    # No-repeat: exclude last played sound if there are alternatives
    last = _last_sound_name.get(category)
    if len(sounds) > 1 and last:
        sounds = [s for s in sounds if s.get("file") != last]
        if not sounds:
            # All excluded (edge case with 2 sounds), pick from original
            sounds = cat_data.get("sounds", [])

    sound = random.choice(sounds)
    file_path = sound.get("file", "")

    # Resolve path (manifest paths may or may not include sounds/ prefix)
    if "/" not in file_path:
        file_path = f"sounds/{file_path}"

    sound_path = pack_dir / file_path
    if not sound_path.exists():
        return None

    _last_sound_name[category] = sound.get("file", file_path)
    return sound_path


# ---------------------------------------------------------------------------
# Audio playback (cross-platform, async)
# ---------------------------------------------------------------------------

_LINUX_BACKENDS = [
    # (binary, volume_arg_template)
    ("pw-play", lambda v: ["--volume", str(v)]),
    ("paplay", lambda v: ["--volume", str(int(v * 32768))]),
    ("ffplay", lambda v: ["-nodisp", "-autoexit", "-volume", str(int(v * 100))]),
    ("mpv", lambda v: ["--no-terminal", "--volume", str(int(v * 100))]),
    ("play", lambda v: ["-v", str(v)]),
    ("aplay", lambda v: []),
]


def _play(sound_path: Path, volume: float = 0.5) -> None:
    """Play a sound file asynchronously. Never blocks the caller."""
    try:
        system = platform.system()

        if system == "Darwin":
            # macOS — afplay
            cmd = ["afplay", "-v", str(volume), str(sound_path)]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        if system == "Linux":
            for binary, vol_args in _LINUX_BACKENDS:
                try:
                    import shutil
                    if shutil.which(binary):
                        cmd = [binary] + vol_args(volume) + [str(sound_path)]
                        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return
                except Exception:
                    continue
            return

        if system == "Windows":
            # PowerShell MediaPlayer
            ps = f"""
            $player = New-Object System.Windows.Media.MediaPlayer
            $player.Open([Uri]::new("{sound_path}"))
            $player.Volume = {volume}
            $player.Play()
            """
            subprocess.Popen(
                ["powershell", "-Command", ps],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return

    except Exception as exc:
        logger.warning("cesp-sounds: playback failed: %s", exc)


# ---------------------------------------------------------------------------
# Public emit function
# ---------------------------------------------------------------------------

def _emit(category: str, platform: str = "cli") -> None:
    """Fire a CESP sound event. Thread-safe, debounced, non-blocking."""
    global _mute

    if _mute:
        return

    if platform not in ENABLED_PLATFORMS:
        return

    # Debounce
    now = time.time()
    if now - _last_sound.get(category, 0) < DEBOUNCE_SEC:
        return

    pack_dir = _get_active_pack()
    if pack_dir is None:
        return

    sound_path = _resolve_sound(pack_dir, category)
    if sound_path is None:
        return

    _last_sound[category] = now
    _save_state()

    # Play in background thread
    t = threading.Thread(target=_play, args=(sound_path, VOLUME), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Hook handlers  — registered with Hermes plugin system
# ---------------------------------------------------------------------------

def _handle_session_start(**kwargs: Any) -> None:
    _emit("session.start", platform=kwargs.get("platform", "cli"))


def _handle_pre_tool_call(**kwargs: Any) -> None:
    """Detect when the agent is about to block for user input."""
    tool_name = kwargs.get("tool_name", "")
    platform = kwargs.get("platform", "cli")

    # clarify and approval callbacks mean "waiting on user"
    if tool_name in ("clarify",):
        _emit("input.required", platform=platform)


def _handle_post_tool_call(**kwargs: Any) -> None:
    """Emit task.error on failures; task.complete only if explicitly enabled."""
    platform = kwargs.get("platform", "cli")
    result = kwargs.get("result", "")

    # Skip clarifications — they're user interactions, not tasks
    tool_name = kwargs.get("tool_name", "")
    if tool_name == "clarify":
        return

    # Check if the tool call failed
    # Hermes returns JSON with "error" key on failure
    try:
        data = json.loads(result) if result else {}
        if "error" in data:
            _emit("task.error", platform=platform)
        elif ENABLE_TASK_COMPLETE:
            _emit("task.complete", platform=platform)
    except (json.JSONDecodeError, TypeError):
        # Non-JSON results (e.g. plain text) are treated as success
        if ENABLE_TASK_COMPLETE:
            _emit("task.complete", platform=platform)


def _handle_session_end(**kwargs: Any) -> None:
    _emit("session.end", platform=kwargs.get("platform", "cli"))


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register CESP sounds hooks on the Hermes plugin system."""
    _load_state()
    logger.info("cesp-sounds: loaded (packs_dir=%s, volume=%.1f)", PACKS_DIR, VOLUME)

    ctx.register_hook("on_session_start", _handle_session_start)
    ctx.register_hook("pre_tool_call", _handle_pre_tool_call)
    ctx.register_hook("post_tool_call", _handle_post_tool_call)
    ctx.register_hook("on_session_end", _handle_session_end)
