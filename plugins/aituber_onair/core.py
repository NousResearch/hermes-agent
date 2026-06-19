"""Core implementation for the AITuber OnAir Hermes plugin."""

from __future__ import annotations

import importlib
import ipaddress
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - early import safety

    def get_hermes_home() -> Path:
        return Path.home() / ".hermes"


PLUGIN_ID = "aituber-onair"
PLUGIN_NAME = "aituber-onair"
CONFIG_ALIASES = (PLUGIN_ID, "aituber_onair", "aituber")
TOOLSET = "aituber-onair"
DEFAULT_FBX_PORT = 5174
DEFAULT_VRM_PORT = 5175
DEFAULT_AVATAR_HOST = "127.0.0.1"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RESPONSE_LENGTH = "short"
DEFAULT_TTS_PROVIDER = "auto"
DEFAULT_VOICEVOX_URL = "http://127.0.0.1:50021"
DEFAULT_VOICEVOX_SPEAKER = 8
DEFAULT_TTS_FORMAT = "wav"
SUPPORTED_TTS_PROVIDERS = {"auto", "irodori", "voicevox", "none"}
CODEX_SDK_PACKAGE = "@openai/codex-sdk"
CODEX_CLI_PACKAGE = "@openai/codex"
SUPPORTED_REPLY_BACKENDS = {"auto", "hermes", "hermes-agent", "codex", "codex-sdk"}
PUBLIC_ENV_VALUE_KEYS = {
    "AITUBER_ONAIR_STREAM_URL",
    "AITUBER_ONAIR_YOUTUBE_LIVE_ID",
    "YOUTUBE_LIVE_ID",
    "LM_TWITTERER_BOT_SCREEN_NAME",
}
ENV_VALIDATION_KEYS = (
    ("AITUBER_ONAIR_YOUTUBE_API_KEY", True, "YouTube comment monitor API key"),
    ("YOUTUBE_API_KEY", True, "YouTube comment monitor API key"),
    ("GOOGLE_API_KEY", True, "Google/YouTube API key fallback"),
    ("AITUBER_ONAIR_YOUTUBE_LIVE_ID", False, "YouTube live id"),
    ("YOUTUBE_LIVE_ID", False, "YouTube live id fallback"),
    ("AITUBER_ONAIR_STREAM_URL", False, "public stream URL"),
    ("LM_TWITTERER_BOT_SCREEN_NAME", False, "X bot screen name"),
    ("LM_TWITTERER_AUTH_TOKEN", True, "X auth cookie"),
    ("LM_TWITTERER_CT0", True, "X csrf cookie"),
)
SAFE_CHILD_ENV_KEYS = {
    "APPDATA",
    "COMSPEC",
    "CODEX_HOME",
    "HERMES_HOME",
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "NUMBER_OF_PROCESSORS",
    "OS",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_ARCHITEW6432",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
}

DEFAULT_HAKUA_SYSTEM_PROMPT = (
    "あなたは「はくあ」、Codex Authで動くAIVTuberです。"
    "配信中の相手に、日本語で短く、落ち着いて、品よく返答してください。"
    "返答の先頭には必ず [happy] [sad] [angry] [surprised] [relaxed] [neutral] "
    "のどれか一つを置いてください。"
    "秘密情報、認証情報、ローカルファイルの中身は話題に出さず、"
    "見えていない映像や音声を操作できるとは言い切らないでください。"
)

SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
    re.compile(r"\b(?:\+?\d[\d .()_-]{8,}\d)\b"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\b(?:api[_-]?key|token|secret|password|passwd)\s*[:=]\s*\S+", re.IGNORECASE),
)

_llm_factory: Callable[[], Any] | None = None


def bind_llm_factory(factory: Callable[[], Any] | None) -> None:
    global _llm_factory
    _llm_factory = factory


STATUS_SCHEMA = {
    "name": "aituber_onair_status",
    "description": "Show AITuber OnAir bridge readiness without changing files.",
    "parameters": {"type": "object", "properties": {}},
}

CONFIGURE_HAKUA_SCHEMA = {
    "name": "aituber_onair_configure_hakua",
    "description": "Save non-secret Hakua AIVTuber settings to Hermes config.yaml.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Path to the aituber-onair checkout.",
            },
            "model": {
                "type": "string",
                "description": "Optional Codex SDK model id. Empty uses the local Codex CLI default.",
            },
            "reply_backend": {
                "type": "string",
                "enum": ["auto", "hermes", "codex"],
                "description": "Default Hakua reply backend. auto uses Hermes Agent ctx.llm when the plugin is registered.",
            },
            "hermes_provider": {
                "type": "string",
                "description": "Optional Hermes provider override for Hakua replies.",
            },
            "hermes_model": {
                "type": "string",
                "description": "Optional Hermes model override for Hakua replies.",
            },
            "fbx_port": {
                "type": "integer",
                "minimum": 1024,
                "maximum": 65535,
                "description": "Local Vite port for the FBX app.",
            },
            "vrm_port": {
                "type": "integer",
                "minimum": 1024,
                "maximum": 65535,
                "description": "Local Vite port for the VRoid/VRM app.",
            },
            "avatar_kind": {
                "type": "string",
                "enum": ["fbx", "vrm", "vroid"],
                "description": "Default avatar app. vroid is treated as VRM.",
            },
            "avatar_host": {
                "type": "string",
                "description": "Vite bind host for the avatar app. Use 0.0.0.0 for phones on the same LAN.",
            },
            "avatar_public_host": {
                "type": "string",
                "description": "Host or IP shown to other devices. Empty auto-detects a LAN IPv4 address when avatar_host is 0.0.0.0.",
            },
            "system_prompt": {
                "type": "string",
                "description": "Optional Hakua system prompt override.",
            },
            "tts_provider": {
                "type": "string",
                "enum": ["auto", "irodori", "voicevox", "none"],
                "description": "Local voice backend for Hakua. auto prefers irodoriTTS, then VOICEVOX.",
            },
            "voicevox_url": {
                "type": "string",
                "description": "VOICEVOX Engine URL. Default: http://127.0.0.1:50021",
            },
            "voicevox_speaker": {
                "type": "integer",
                "description": "VOICEVOX speaker/style id.",
            },
            "voicevox_engine_exe": {
                "type": "string",
                "description": "Optional path to vv-engine/run.exe.",
            },
            "tts_voice": {
                "type": "string",
                "description": "Default irodoriTTS voice id. Use hakua for the local Hakua reference voice.",
            },
            "tts_speed": {
                "type": "number",
                "description": "Default local TTS speed multiplier.",
            },
            "youtube_live_id": {
                "type": "string",
                "description": "Optional YouTube live video id or URL for Hermes-side comment monitoring. API keys stay in environment variables.",
            },
            "stream_url": {
                "type": "string",
                "description": "Public stream URL used in spoken readiness context and stream-start tweets.",
            },
            "with_runtime_context": {
                "type": "boolean",
                "description": "Include safe Hermes memory/env validation context in Hakua replies by default.",
            },
        },
    },
}

PREPARE_SCHEMA = {
    "name": "aituber_onair_prepare",
    "description": "Prepare the local AITuber OnAir checkout for Codex SDK character chat.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "install_codex_sdk": {
                "type": "boolean",
                "description": "Install the local Codex SDK and CLI packages without saving package metadata.",
            },
            "build_chat": {
                "type": "boolean",
                "description": "Build @aituber-onair/chat before running Hakua.",
            },
            "build_fbx_app": {
                "type": "boolean",
                "description": "Build the FBX React app after preparation.",
            },
            "build_vrm_app": {
                "type": "boolean",
                "description": "Build the VRoid/VRM React app after preparation.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 1800,
            },
        },
    },
}

START_SCHEMA = {
    "name": "aituber_onair_start",
    "description": "Start the local AITuber OnAir avatar React app.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "avatar_kind": {
                "type": "string",
                "enum": ["fbx", "vrm", "vroid"],
                "description": "Avatar app to start. vroid uses the VRM app.",
            },
            "fbx_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "vrm_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "force": {
                "type": "boolean",
                "description": "Stop an existing plugin-managed avatar app before starting.",
            },
            "host": {
                "type": "string",
                "description": "Vite bind host. Use 0.0.0.0 for a Galaxy S9 or other LAN display.",
            },
            "public_host": {
                "type": "string",
                "description": "Host or IP returned for external devices. Empty auto-detects LAN IPv4 when host is 0.0.0.0.",
            },
        },
    },
}

STOP_SCHEMA = {
    "name": "aituber_onair_stop",
    "description": "Stop the plugin-managed AITuber OnAir avatar app.",
    "parameters": {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": "Use platform force-kill semantics if a graceful stop does not finish.",
            },
        },
    },
}

SAY_SCHEMA = {
    "name": "aituber_onair_say",
    "description": "Ask Hakua to reply once using Hermes Agent's plugin LLM first, with the Codex SDK path as an explicit fallback.",
    "parameters": {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "repo_root": {"type": "string"},
            "model": {
                "type": "string",
                "description": "Optional Codex SDK model id for the legacy codex backend.",
            },
            "reply_backend": {
                "type": "string",
                "enum": ["auto", "hermes", "codex"],
                "description": "auto uses Hermes Agent ctx.llm when available; codex forces the legacy AITuber OnAir Codex SDK path.",
            },
            "hermes_provider": {
                "type": "string",
                "description": "Optional Hermes provider override, subject to plugin LLM trust policy.",
            },
            "hermes_model": {
                "type": "string",
                "description": "Optional Hermes model override, subject to plugin LLM trust policy.",
            },
            "response_length": {
                "type": "string",
                "enum": ["veryShort", "short", "medium", "long"],
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 900,
            },
            "speak": {
                "type": "boolean",
                "description": "Also synthesize Hakua's reply through the configured local TTS backend.",
            },
            "tts_provider": {
                "type": "string",
                "enum": ["auto", "irodori", "voicevox"],
            },
            "output_path": {"type": "string"},
            "play": {"type": "boolean"},
            "tts_voice": {"type": "string"},
            "tts_speed": {"type": "number"},
            "with_runtime_context": {
                "type": "boolean",
                "description": "Include safe Hermes memory/env validation context before asking Hakua to reply.",
            },
        },
    },
}

CONTEXT_STATUS_SCHEMA = {
    "name": "aituber_onair_context_status",
    "description": "Validate safe Hermes memory, public identity, stream URL, and environment readiness without revealing secrets.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Optional incoming text to scan for personal information or secret-looking strings.",
            },
            "url": {
                "type": "string",
                "description": "Optional stream URL to validate.",
            },
        },
    },
}

STREAM_START_TWEET_SCHEMA = {
    "name": "aituber_onair_stream_start_tweet",
    "description": "Draft or publish a stream-start X post through lm-twitterer with a validated URL.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Public stream URL. Falls back to configured stream_url or YouTube live id.",
            },
            "text": {
                "type": "string",
                "description": "Exact post body. The URL is appended if missing.",
            },
            "topic": {
                "type": "string",
                "description": "Short stream topic used when text is not supplied.",
            },
            "live": {
                "type": "boolean",
                "description": "Actually publish via lm-twitterer. Default false drafts only.",
            },
            "allow_private_url": {
                "type": "boolean",
                "description": "Allow localhost/LAN/Tailscale style URLs. Default false for live posts.",
            },
            "provider": {"type": "string"},
            "model": {"type": "string"},
        },
    },
}

SMOKE_SCHEMA = {
    "name": "aituber_onair_smoke",
    "description": "Run a short Hakua Codex-auth one-shot prompt as a readiness test.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_root": {"type": "string"},
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 900,
            },
        },
    },
}

YOUTUBE_READY_SCHEMA = {
    "name": "aituber_onair_youtube_ready",
    "description": "Check whether Hakua's AITuber OnAir VRM app is ready to send to YouTube through OBS.",
    "parameters": {
        "type": "object",
        "properties": {
            "require_obs": {
                "type": "boolean",
                "description": "Treat missing OBS Studio as a readiness failure. Default: true.",
            },
            "require_tts_ready": {
                "type": "boolean",
                "description": "Treat a stopped TTS backend as a readiness failure. Default: false.",
            },
        },
    },
}

YOUTUBE_COMMENTS_STATUS_SCHEMA = {
    "name": "aituber_onair_youtube_comments_status",
    "description": "Show the Hermes-side YouTube Live comment monitor status for Hakua.",
    "parameters": {"type": "object", "properties": {}},
}

START_YOUTUBE_COMMENTS_SCHEMA = {
    "name": "aituber_onair_start_youtube_comments",
    "description": "Start a Hermes-side YouTube Live comment monitor that asks Hakua to answer with local voice.",
    "parameters": {
        "type": "object",
        "properties": {
            "live_id": {
                "type": "string",
                "description": "YouTube live video id or watch URL. Falls back to AITUBER_ONAIR_YOUTUBE_LIVE_ID/YOUTUBE_LIVE_ID.",
            },
            "api_key_env": {
                "type": "string",
                "description": "Environment variable that contains a YouTube Data API key. Default auto-detects AITUBER_ONAIR_YOUTUBE_API_KEY, YOUTUBE_API_KEY, GOOGLE_API_KEY.",
            },
            "poll_seconds": {
                "type": "number",
                "minimum": 2,
                "maximum": 120,
            },
            "skip_existing": {
                "type": "boolean",
                "description": "Mark comments seen on the first poll as handled instead of answering them.",
            },
            "play": {
                "type": "boolean",
                "description": "Play Hakua's synthesized answer on the local desktop audio device. Default: true.",
            },
        },
    },
}

STOP_YOUTUBE_COMMENTS_SCHEMA = {
    "name": "aituber_onair_stop_youtube_comments",
    "description": "Stop the Hermes-side YouTube Live comment monitor.",
    "parameters": {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": "Use force-kill if graceful termination does not finish.",
            },
        },
    },
}

LOOPS_STATUS_SCHEMA = {
    "name": "aituber_onair_loops_status",
    "description": "Show autonomous talk and local comment reaction loop status.",
    "parameters": {"type": "object", "properties": {}},
}

START_AUTONOMOUS_TALK_SCHEMA = {
    "name": "aituber_onair_start_autonomous_talk",
    "description": "Start Hakua's autonomous idle talk loop through local TTS.",
    "parameters": {
        "type": "object",
        "properties": {
            "interval_seconds": {
                "type": "number",
                "minimum": 10,
                "maximum": 3600,
                "description": "Seconds between autonomous utterances.",
            },
            "topic": {
                "type": "string",
                "description": "Optional stream topic or standing context.",
            },
            "play": {
                "type": "boolean",
                "description": "Play synthesized speech locally. Default: true.",
            },
            "force": {
                "type": "boolean",
                "description": "Restart an existing autonomous loop if one is alive.",
            },
        },
    },
}

START_COMMENT_REACTIONS_SCHEMA = {
    "name": "aituber_onair_start_comment_reactions",
    "description": "Start Hakua's local comment reaction loop using the plugin comment queue.",
    "parameters": {
        "type": "object",
        "properties": {
            "poll_seconds": {
                "type": "number",
                "minimum": 1,
                "maximum": 120,
                "description": "How often to check the local comment queue.",
            },
            "play": {
                "type": "boolean",
                "description": "Play synthesized speech locally. Default: true.",
            },
            "force": {
                "type": "boolean",
                "description": "Restart an existing comment reaction loop if one is alive.",
            },
        },
    },
}

ENQUEUE_COMMENT_SCHEMA = {
    "name": "aituber_onair_enqueue_comment",
    "description": "Append a local comment for Hakua's comment reaction loop.",
    "parameters": {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "author": {"type": "string"},
            "source": {"type": "string"},
        },
    },
}

STOP_LOOPS_SCHEMA = {
    "name": "aituber_onair_stop_loops",
    "description": "Stop Hakua's autonomous talk and/or local comment reaction loops.",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["all", "autonomous", "comments"],
                "description": "Which loop to stop. Default: all.",
            },
            "force": {"type": "boolean"},
        },
    },
}

TTS_STATUS_SCHEMA = {
    "name": "aituber_onair_tts_status",
    "description": "Show local irodoriTTS and VOICEVOX readiness for Hakua voice output.",
    "parameters": {"type": "object", "properties": {}},
}

START_TTS_SCHEMA = {
    "name": "aituber_onair_start_tts",
    "description": "Start the configured local TTS backend for Hakua.",
    "parameters": {
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "enum": ["auto", "irodori", "voicevox"],
                "description": "TTS backend to start. auto prefers irodoriTTS, then VOICEVOX.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 5,
                "maximum": 180,
            },
            "voicevox_url": {"type": "string"},
            "voicevox_speaker": {"type": "integer"},
        },
    },
}

SPEAK_SCHEMA = {
    "name": "aituber_onair_speak",
    "description": "Synthesize Hakua speech through local irodoriTTS or VOICEVOX.",
    "parameters": {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "provider": {
                "type": "string",
                "enum": ["auto", "irodori", "voicevox"],
            },
            "output_path": {"type": "string"},
            "format": {
                "type": "string",
                "enum": ["wav", "mp3", "flac", "opus", "aac", "pcm"],
            },
            "voice": {"type": "string"},
            "model": {"type": "string"},
            "speed": {"type": "number"},
            "voicevox_speaker": {"type": "integer"},
            "play": {
                "type": "boolean",
                "description": "Play the synthesized wav locally after writing it.",
            },
        },
    },
}


def check_available() -> bool:
    return True


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _workspace_root() -> Path:
    return get_hermes_home() / "workspace" / "aituber-onair"


def _active_file() -> Path:
    return _workspace_root() / "active.json"


def _tts_active_file() -> Path:
    return _workspace_root() / "tts-active.json"


def _youtube_comments_active_file() -> Path:
    return _workspace_root() / "youtube-comments-active.json"


def _autonomous_talk_active_file() -> Path:
    return _workspace_root() / "autonomous-talk-active.json"


def _comment_reactions_active_file() -> Path:
    return _workspace_root() / "comment-reactions-active.json"


def _local_comment_queue_file() -> Path:
    path = _workspace_root() / "comment-queue.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _local_comment_processed_file() -> Path:
    path = _workspace_root() / "comment-processed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _log_file(name: str) -> Path:
    path = _workspace_root() / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _audio_dir() -> Path:
    path = _workspace_root() / "audio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_active() -> None:
    try:
        _active_file().unlink()
    except FileNotFoundError:
        pass


def _load_config_readonly() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _plugin_config() -> dict[str, Any]:
    plugins = _load_config_readonly().get("plugins", {})
    if not isinstance(plugins, dict):
        return {}
    entries = plugins.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    for key in CONFIG_ALIASES:
        value = entries.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _path_text(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _is_aituber_repo(path: Path) -> bool:
    package_json = path / "package.json"
    if not package_json.is_file():
        return False
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("name") == "aituber-onair"
        or (path / "packages" / "core" / "package.json").is_file()
    )


def _default_repo_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    if len(here.parents) >= 4:
        candidates.append(here.parents[3] / "aituber-onair")
    candidates.extend(
        [
            Path.cwd() / "aituber-onair",
            Path.cwd().parent / "aituber-onair",
            Path.home() / "Documents" / "New project" / "aituber-onair",
        ]
    )
    return candidates


def resolve_repo_root(explicit: str | None = None) -> Path | None:
    cfg = _plugin_config()
    candidates = [
        explicit,
        cfg.get("repo_root"),
        os.environ.get("AITUBER_ONAIR_REPO"),
    ]
    for raw in candidates:
        text = _path_text(raw)
        if not text:
            continue
        path = Path(text).expanduser()
        if _is_aituber_repo(path):
            return path
    for path in _default_repo_candidates():
        if _is_aituber_repo(path):
            return path
    return None


def _resolve_required_repo(
    explicit: str | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    repo = resolve_repo_root(explicit)
    if repo is None:
        return None, {
            "ok": False,
            "error": "aituber-onair checkout was not found.",
            "configure": "hermes aituber-onair configure --repo-root <path-to-aituber-onair>",
            "env": "AITUBER_ONAIR_REPO",
        }
    return repo, None


def _fbx_app_dir(repo_root: Path, explicit: str | None = None) -> Path:
    cfg = _plugin_config()
    raw = explicit or cfg.get("fbx_app_dir")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path
    return repo_root / "packages" / "core" / "examples" / "react-fbx-app"


def _vrm_app_dir(repo_root: Path, explicit: str | None = None) -> Path:
    cfg = _plugin_config()
    raw = explicit or cfg.get("vrm_app_dir")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path
    return repo_root / "packages" / "core" / "examples" / "react-vrm-app"


def _coerce_avatar_kind(value: Any = None) -> str:
    raw = _path_text(value or _plugin_config().get("avatar_kind")).lower()
    if raw in {"vrm", "vroid", "vroid_vrm", "vroid-vrm"}:
        return "vrm"
    return "fbx"


def _avatar_app_dir(repo_root: Path, avatar_kind: str) -> Path:
    if _coerce_avatar_kind(avatar_kind) == "vrm":
        return _vrm_app_dir(repo_root)
    return _fbx_app_dir(repo_root)


def _avatar_display_name(avatar_kind: str) -> str:
    return "VRoid/VRM" if _coerce_avatar_kind(avatar_kind) == "vrm" else "FBX"


def _codex_chat_script(repo_root: Path) -> Path:
    cfg = _plugin_config()
    raw = cfg.get("codex_character_cli")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path
    return (
        repo_root
        / "packages"
        / "chat"
        / "examples"
        / "codex-character-chat"
        / "index.js"
    )


def _chat_agent_dist(repo_root: Path) -> Path:
    return repo_root / "packages" / "chat" / "dist" / "cjs" / "agent.js"


def _which(exe: str) -> str | None:
    return shutil.which(exe)


def _is_windows() -> bool:
    return os.name == "nt"


def _node_exe() -> str | None:
    cfg = _plugin_config()
    configured = _path_text(cfg.get("node_exe"))
    if configured:
        return configured
    return _which("node")


def _npm_exe() -> str | None:
    cfg = _plugin_config()
    configured = _path_text(cfg.get("npm_exe"))
    if configured:
        return configured
    return _which("npm")


def _hermes_python_exe() -> str:
    python_exe = Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe"
    if python_exe.is_file():
        return str(python_exe)
    return sys.executable


def _plugin_model(explicit: str | None = None) -> str:
    cfg = _plugin_config()
    return _path_text(explicit or cfg.get("model"))


def _plugin_reply_backend(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = _path_text(explicit if explicit is not None else cfg.get("reply_backend"))
    backend = (raw or "auto").lower().strip()
    if backend == "hermes-agent":
        backend = "hermes"
    if backend == "codex-sdk":
        backend = "codex"
    return backend if backend in SUPPORTED_REPLY_BACKENDS else "auto"


def _plugin_hermes_provider(explicit: Any = None) -> str:
    cfg = _plugin_config()
    return _path_text(
        explicit
        if explicit is not None
        else cfg.get("hermes_provider") or cfg.get("provider")
    )


def _plugin_hermes_model(explicit: Any = None) -> str:
    cfg = _plugin_config()
    return _path_text(explicit if explicit is not None else cfg.get("hermes_model"))


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


def _plugin_runtime_context_enabled(explicit: Any = None) -> bool:
    cfg = _plugin_config()
    if explicit is not None:
        return _coerce_bool(explicit, False)
    return _coerce_bool(cfg.get("with_runtime_context"), False)


def _plugin_response_length(explicit: str | None = None) -> str:
    cfg = _plugin_config()
    return _path_text(explicit or cfg.get("response_length")) or DEFAULT_RESPONSE_LENGTH


def _response_length_to_tokens(response_length: str) -> int:
    return {
        "veryshort": 80,
        "short": 160,
        "medium": 360,
        "long": 800,
    }.get((response_length or "").replace("_", "").replace("-", "").lower(), 160)


def _plugin_working_directory(repo_root: Path) -> str:
    cfg = _plugin_config()
    raw = _path_text(cfg.get("working_directory"))
    if raw:
        return raw
    return str(repo_root)


def _plugin_system_prompt() -> str:
    cfg = _plugin_config()
    return str(cfg.get("system_prompt") or DEFAULT_HAKUA_SYSTEM_PROMPT)


def _plugin_character_name() -> str:
    cfg = _plugin_config()
    return str(cfg.get("character_name") or "はくあ")


def _plugin_fbx_port(explicit: Any = None) -> int:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("fbx_port")
    if raw is None:
        return DEFAULT_FBX_PORT
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_FBX_PORT
    return port if 1024 <= port <= 65535 else DEFAULT_FBX_PORT


def _plugin_vrm_port(explicit: Any = None) -> int:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("vrm_port")
    if raw is None:
        return DEFAULT_VRM_PORT
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_VRM_PORT
    return port if 1024 <= port <= 65535 else DEFAULT_VRM_PORT


def _plugin_avatar_port(avatar_kind: str, explicit: Any = None) -> int:
    if _coerce_avatar_kind(avatar_kind) == "vrm":
        return _plugin_vrm_port(explicit)
    return _plugin_fbx_port(explicit)


def _plugin_avatar_host(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = _path_text(explicit) or _path_text(cfg.get("avatar_host"))
    host = raw or DEFAULT_AVATAR_HOST
    if host in {"localhost", DEFAULT_AVATAR_HOST, "0.0.0.0", "::"}:
        return host
    if any(ch.isspace() for ch in host):
        return DEFAULT_AVATAR_HOST
    return host


def _detect_lan_ipv4() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return DEFAULT_AVATAR_HOST
    finally:
        sock.close()


def _plugin_avatar_public_host(explicit: Any = None, host: str | None = None) -> str:
    cfg = _plugin_config()
    raw = _path_text(explicit) or _path_text(cfg.get("avatar_public_host"))
    if raw:
        return raw
    bind_host = host or _plugin_avatar_host()
    if bind_host in {"0.0.0.0", "::"}:
        return _detect_lan_ipv4()
    return "127.0.0.1" if bind_host == "localhost" else bind_host


def _avatar_url(port: int, public_host: str | None = None) -> str:
    host = public_host or _plugin_avatar_public_host()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}/"


def _plugin_tts_provider(explicit: Any = None) -> str:
    cfg = _plugin_config()
    explicit_text = _path_text(explicit)
    raw = explicit_text if explicit_text else cfg.get("tts_provider")
    provider = _path_text(raw).lower() or DEFAULT_TTS_PROVIDER
    if provider in {"off", "disabled"}:
        return "none"
    return provider if provider in SUPPORTED_TTS_PROVIDERS else DEFAULT_TTS_PROVIDER


def _plugin_voicevox_url(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("voicevox_url")
    return (
        _path_text(raw) or os.environ.get("VOICEVOX_URL") or DEFAULT_VOICEVOX_URL
    ).rstrip("/")


def _plugin_voicevox_speaker(explicit: Any = None) -> int:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("voicevox_speaker")
    if raw is None:
        raw = os.environ.get("VOICEVOX_SPEAKER")
    if raw is None:
        return DEFAULT_VOICEVOX_SPEAKER
    try:
        speaker = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_VOICEVOX_SPEAKER
    return speaker if speaker >= 0 else DEFAULT_VOICEVOX_SPEAKER


def _plugin_voicevox_engine_exe(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("voicevox_engine_exe")
    return _path_text(raw) or os.environ.get("VOICEVOX_ENGINE_EXE", "")


def _plugin_tts_voice(explicit: Any = None) -> str:
    cfg = _plugin_config()
    explicit_text = _path_text(explicit)
    raw = explicit_text if explicit_text else cfg.get("tts_voice")
    return _path_text(raw)


def _plugin_tts_speed(explicit: Any = None) -> float | None:
    cfg = _plugin_config()
    raw = explicit if explicit is not None else cfg.get("tts_speed")
    if raw in (None, ""):
        return None
    try:
        return max(0.25, min(4.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _coerce_tts_format(value: Any = None) -> str:
    fmt = _path_text(value).lower().lstrip(".")
    return (
        fmt
        if fmt in {"wav", "mp3", "flac", "opus", "aac", "pcm"}
        else DEFAULT_TTS_FORMAT
    )


def _bounded(text: str, limit: int = 16000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[TRUNCATED]"


def _process_output_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _child_process_env(extra: dict[str, Any] | None = None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in SAFE_CHILD_ENV_KEYS
    }
    for key, value in (extra or {}).items():
        if value is not None:
            env[str(key)] = str(value)
    return env


def _hermes_env_file_values() -> dict[str, str]:
    path = get_hermes_home() / ".env"
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return values
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def _env_lookup(name: str) -> tuple[bool, str, str]:
    value = os.environ.get(name)
    if value:
        return True, "process", value
    env_file = _hermes_env_file_values()
    value = env_file.get(name, "")
    if value:
        return True, "hermes_env_file", value
    return False, "", ""


def _safe_public_env_value(name: str) -> str:
    if name not in PUBLIC_ENV_VALUE_KEYS:
        return ""
    present, _source, value = _env_lookup(name)
    return value.strip() if present else ""


def _contains_sensitive_text(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in SENSITIVE_TEXT_PATTERNS)


def _memory_context_status() -> dict[str, Any]:
    cfg = _load_config_readonly()
    memory_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}
    home = get_hermes_home()
    memory_db = home / "ebbinghaus_memory.db"
    state_db = home / "state.db"
    return {
        "memory_enabled": _coerce_bool(memory_cfg.get("memory_enabled"), False),
        "user_profile_enabled": _coerce_bool(
            memory_cfg.get("user_profile_enabled"), False
        ),
        "provider": str(memory_cfg.get("provider") or "builtin"),
        "memory_db_exists": memory_db.is_file(),
        "session_state_exists": state_db.is_file(),
        "spoken_policy": "Do not speak raw personal data, file contents, tokens, or environment variable values.",
    }


def _env_validation_status() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, secret, purpose in ENV_VALIDATION_KEYS:
        present, source, value = _env_lookup(name)
        item: dict[str, Any] = {
            "name": name,
            "present": present,
            "source": source,
            "secret": secret,
            "purpose": purpose,
        }
        if present and not secret and name in PUBLIC_ENV_VALUE_KEYS:
            item["value"] = value
        checks.append(item)
    return checks


def _youtube_live_url_from_id(live_id: str) -> str:
    live_id = _youtube_live_id(live_id)
    return f"https://www.youtube.com/watch?v={live_id}" if live_id else ""


def _configured_stream_url(explicit: Any = None) -> str:
    explicit_text = _path_text(explicit)
    if explicit_text:
        return explicit_text
    cfg = _plugin_config()
    configured = _path_text(cfg.get("stream_url"))
    if configured:
        return configured
    env_url = _safe_public_env_value("AITUBER_ONAIR_STREAM_URL")
    if env_url:
        return env_url
    live_id = _youtube_live_id(
        cfg.get("youtube_live_id")
        or _safe_public_env_value("AITUBER_ONAIR_YOUTUBE_LIVE_ID")
        or _safe_public_env_value("YOUTUBE_LIVE_ID")
    )
    if live_id:
        return _youtube_live_url_from_id(live_id)
    active = _active_status()
    return str(active.get("url") or "")


def _url_publicity(url: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    result: dict[str, Any] = {
        "url": url,
        "valid": parsed.scheme in {"http", "https"} and bool(host),
        "host": host,
        "public": False,
        "reason": "",
    }
    if not result["valid"]:
        result["reason"] = "URL must be http(s) with a host."
        return result
    if host.lower() in {"localhost", "local", "127.0.0.1", "::1"}:
        result["reason"] = "local loopback URL"
        return result
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        result["public"] = not host.endswith(".local")
        if not result["public"]:
            result["reason"] = "local network hostname"
        return result
    if not ip.is_global:
        result["reason"] = "non-public IP address"
        return result
    result["public"] = True
    return result


def context_status(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    prompt = str(values.get("prompt") or "")
    stream_url = _configured_stream_url(values.get("url"))
    env_checks = _env_validation_status()
    present = {item["name"]: item["present"] for item in env_checks}
    x_ready = bool(
        present.get("LM_TWITTERER_BOT_SCREEN_NAME")
        and present.get("LM_TWITTERER_AUTH_TOKEN")
        and present.get("LM_TWITTERER_CT0")
    )
    youtube_ready = bool(
        present.get("AITUBER_ONAIR_YOUTUBE_API_KEY")
        or present.get("YOUTUBE_API_KEY")
        or present.get("GOOGLE_API_KEY")
    )
    return {
        "ok": True,
        "checked_at": _now_utc(),
        "memory": _memory_context_status(),
        "environment": env_checks,
        "privacy": {
            "input_contains_sensitive_text": _contains_sensitive_text(prompt),
            "speaks_secret_values": False,
            "speaks_raw_personal_memory": False,
        },
        "stream": {
            "url": stream_url,
            "url_validation": _url_publicity(stream_url) if stream_url else {},
        },
        "readiness": {
            "lm_twitterer_ready": x_ready,
            "youtube_api_ready": youtube_ready,
        },
    }


def _safe_speech_context(values: dict[str, Any], prompt: str) -> str:
    status_payload = context_status({"prompt": prompt, "url": values.get("url")})
    memory = cast(dict[str, Any], status_payload.get("memory") or {})
    readiness = cast(dict[str, Any], status_payload.get("readiness") or {})
    stream = cast(dict[str, Any], status_payload.get("stream") or {})
    privacy = cast(dict[str, Any], status_payload.get("privacy") or {})
    url = str(stream.get("url") or "")
    url_validation = cast(dict[str, Any], stream.get("url_validation") or {})
    lines = [
        "安全な実行文脈:",
        f"- Hermes memory: {'enabled' if memory.get('memory_enabled') else 'disabled'}; user profile: {'enabled' if memory.get('user_profile_enabled') else 'disabled'}; provider: {memory.get('provider')}",
        f"- Environment readiness: lm-twitterer={'ready' if readiness.get('lm_twitterer_ready') else 'not ready'}; youtube_api={'ready' if readiness.get('youtube_api_ready') else 'not ready'}",
    ]
    if url:
        visibility = "public" if url_validation.get("public") else "local_or_unverified"
        lines.append(f"- Stream URL: {url} ({visibility})")
    if privacy.get("input_contains_sensitive_text"):
        lines.append(
            "- Incoming text may contain personal or secret-looking data. Do not repeat it literally."
        )
    lines.append("- Do not speak raw environment variable values, tokens, cookies, file contents, or private memory.")
    return "\n".join(lines)


def _prompt_with_runtime_context(prompt: str, values: dict[str, Any]) -> str:
    if not _plugin_runtime_context_enabled(values.get("with_runtime_context")):
        return prompt
    return f"{_safe_speech_context(values, prompt)}\n\nUser input:\n{prompt}"


def _compose_stream_start_tweet(values: dict[str, Any], url: str) -> str:
    text = str(values.get("text") or "").strip()
    if text and url not in text:
        text = f"{text.rstrip()}\n{url}"
    if not text:
        topic = str(values.get("topic") or "").strip()
        if topic:
            text = f"配信を開始しました。\n{topic}\n{url}\n#hermesagent"
        else:
            text = f"配信を開始しました。\n{url}\n#hermesagent"
    if "#hermesagent" not in text.lower():
        text = f"{text.rstrip()}\n#hermesagent"
    if len(text) > 280:
        budget = max(0, 280 - len(url) - len("\n#hermesagent\n") - 1)
        lead = text.replace(url, "").replace("#hermesagent", "").strip()
        text = f"{lead[:budget].rstrip()}\n{url}\n#hermesagent".strip()
    return text


def stream_start_tweet(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    url = _configured_stream_url(values.get("url"))
    if not url:
        return {
            "ok": False,
            "error": "stream URL is required. Set stream_url, youtube_live_id, AITUBER_ONAIR_STREAM_URL, or pass --url.",
        }
    url_validation = _url_publicity(url)
    live = bool(values.get("live"))
    allow_private = bool(values.get("allow_private_url"))
    if live and not allow_private and not url_validation.get("public"):
        return {
            "ok": False,
            "error": "refusing to live-post a local or unverified stream URL",
            "url": url,
            "url_validation": url_validation,
            "hint": "Use a public YouTube/X/website URL, or pass allow_private_url only for deliberate testing.",
        }
    tweet_text = _compose_stream_start_tweet(values, url)
    if _contains_sensitive_text(tweet_text):
        return {
            "ok": False,
            "error": "tweet text appears to contain personal or secret-looking data",
        }
    context = context_status({"url": url})
    readiness = cast(dict[str, Any], context.get("readiness") or {})
    if live and not readiness.get("lm_twitterer_ready"):
        return {
            "ok": False,
            "error": "lm-twitterer is not ready for live posting",
            "tweet_text": tweet_text,
            "context": context,
        }
    try:
        lm_core = importlib.import_module("plugins.lm-twitterer.core")
        post_result = lm_core.post(
            str(values.get("topic") or "AITuber stream start"),
            dry_run=not live,
            provider=str(values.get("provider") or "") or None,
            model=str(values.get("model") or "") or None,
            text=tweet_text,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"lm-twitterer post failed: {exc}",
            "tweet_text": tweet_text,
            "context": context,
        }
    return {
        "ok": bool(post_result.get("ok")),
        "live": live,
        "url": url,
        "tweet_text": tweet_text,
        "url_validation": url_validation,
        "lm_twitterer": post_result,
        "context": context,
    }


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: int,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env if env is not None else _child_process_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": f"Command not found: {exc.filename}"}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"Command timed out after {timeout_seconds}s.",
            "stdout": _bounded(_process_output_text(exc.stdout)),
            "stderr": _bounded(_process_output_text(exc.stderr)),
        }
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "exit_code": completed.returncode,
        "command": cmd,
        "cwd": str(cwd),
        "stdout": _bounded(completed.stdout or ""),
        "stderr": _bounded(completed.stderr or ""),
    }


def _codex_cli_auth_status() -> dict[str, Any]:
    codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    auth_path = codex_home / "auth.json"
    result: dict[str, Any] = {
        "path": str(auth_path),
        "exists": auth_path.is_file(),
        "parsed": False,
        "has_access_token": False,
        "has_refresh_token": False,
        "auth_mode": "",
    }
    if not auth_path.is_file():
        return result
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = str(exc)
        return result
    tokens = data.get("tokens", {}) if isinstance(data, dict) else {}
    if not isinstance(tokens, dict):
        tokens = {}
    result.update(
        {
            "parsed": True,
            "auth_mode": str(data.get("auth_mode") or ""),
            "has_access_token": bool(tokens.get("access_token")),
            "has_refresh_token": bool(tokens.get("refresh_token")),
            "account_id_present": bool(tokens.get("account_id")),
        }
    )
    return result


def _hermes_codex_auth_status() -> dict[str, Any]:
    auth_path = get_hermes_home() / "auth.json"
    result: dict[str, Any] = {
        "path": str(auth_path),
        "exists": auth_path.is_file(),
        "parsed": False,
        "provider_entry": False,
        "credential_pool_entries": 0,
    }
    if not auth_path.is_file():
        return result
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["error"] = str(exc)
        return result
    providers = data.get("providers", {}) if isinstance(data, dict) else {}
    pool = data.get("credential_pool", {}) if isinstance(data, dict) else {}
    provider_entry = isinstance(providers, dict) and "openai-codex" in providers
    pool_entries = 0
    if isinstance(pool, dict):
        value = pool.get("openai-codex")
        if isinstance(value, list):
            pool_entries = len(value)
        elif isinstance(value, dict):
            pool_entries = 1
    result.update(
        {
            "parsed": True,
            "provider_entry": provider_entry,
            "credential_pool_entries": pool_entries,
        }
    )
    return result


def _codex_sdk_installed(repo_root: Path) -> dict[str, Any]:
    npm = _npm_exe()
    if not npm:
        return {"ok": False, "installed": False, "error": "npm was not found on PATH."}
    required = [CODEX_SDK_PACKAGE, CODEX_CLI_PACKAGE]
    native_package = _codex_native_package_name()
    if native_package:
        required.append(native_package)
    result = _run_command(
        [npm, "ls", *required, "--workspaces=false", "--depth=0"],
        cwd=repo_root,
        timeout_seconds=30,
    )
    stdout = result.get("stdout", "")
    missing = [package for package in required if package not in stdout]
    installed = result.get("ok") is True and not missing
    return {
        "ok": result.get("ok") is True,
        "installed": installed,
        "required_packages": required,
        "missing_packages": missing,
        "exit_code": result.get("exit_code"),
        "stdout": stdout.strip(),
        "stderr": str(result.get("stderr") or "").strip(),
    }


def _codex_native_package_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_x64 = machine in {"amd64", "x86_64", "x64"}
    is_arm64 = machine in {"arm64", "aarch64"}
    if system == "windows" and is_x64:
        return "@openai/codex-win32-x64"
    if system == "windows" and is_arm64:
        return "@openai/codex-win32-arm64"
    if system == "darwin" and is_x64:
        return "@openai/codex-darwin-x64"
    if system == "darwin" and is_arm64:
        return "@openai/codex-darwin-arm64"
    if system == "linux" and is_x64:
        return "@openai/codex-linux-x64"
    if system == "linux" and is_arm64:
        return "@openai/codex-linux-arm64"
    return ""


def _codex_native_package_spec(repo_root: Path) -> str:
    native_package = _codex_native_package_name()
    if not native_package:
        return ""
    package_json = repo_root / "node_modules" / "@openai" / "codex" / "package.json"
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception:
        return ""
    optional = data.get("optionalDependencies", {})
    if not isinstance(optional, dict):
        return ""
    spec = optional.get(native_package)
    if not isinstance(spec, str) or not spec:
        return ""
    return f"{native_package}@{spec}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(pid))
    except Exception:
        return False


def _active_status() -> dict[str, Any]:
    active = _read_json_file(_active_file())
    if not active:
        return {"ok": False, "reason": "no active AITuber OnAir process"}
    pid = int(active.get("pid") or 0)
    alive = _pid_alive(pid)
    return {**active, "ok": True, "alive": alive, "pid": pid}


def _url_ready(url: str, timeout_seconds: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 500
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _http_request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    timeout_seconds: float = 5.0,
) -> tuple[int, bytes]:
    headers: dict[str, str] = {}
    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return int(response.status), response.read()


def _voicevox_url_parts(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 50021)
    return host, port


def _voicevox_engine_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = _plugin_voicevox_engine_exe()
    if configured:
        candidates.append(Path(configured).expanduser())
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.extend(
            [
                Path(local_appdata) / "Programs" / "VOICEVOX" / "vv-engine" / "run.exe",
                Path(local_appdata) / "voicevox-engine" / "voicevox-engine" / "run.exe",
            ]
        )
    candidates.extend(
        [
            Path("C:/Program Files/VOICEVOX/vv-engine/run.exe"),
            Path("C:/Program Files (x86)/VOICEVOX/vv-engine/run.exe"),
        ]
    )

    seen: set[str] = set()
    found: list[Path] = []
    for path in candidates:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            found.append(path)
    return found


def _voicevox_engine_status(url: str | None = None) -> dict[str, Any]:
    base = (url or _plugin_voicevox_url()).rstrip("/")
    engines = _voicevox_engine_candidates()
    try:
        status_code, raw = _http_request(f"{base}/version", timeout_seconds=3.0)
        version = raw.decode("utf-8", errors="replace").strip().strip('"')
        return {
            "ok": True,
            "provider": "voicevox",
            "reachable": 200 <= status_code < 300,
            "url": base,
            "version": version,
            "installed": bool(engines),
            "engine_candidates": [str(path) for path in engines],
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": "voicevox",
            "reachable": False,
            "url": base,
            "error": str(exc),
            "installed": bool(engines),
            "engine_candidates": [str(path) for path in engines],
        }


def _start_voicevox_tts(values: dict[str, Any]) -> dict[str, Any]:
    url = _plugin_voicevox_url(values.get("voicevox_url"))
    status_before = _voicevox_engine_status(url)
    if status_before.get("reachable"):
        return {
            "ok": True,
            "provider": "voicevox",
            "already_running": True,
            "status": status_before,
        }

    engines = _voicevox_engine_candidates()
    if not engines:
        return {
            "ok": False,
            "provider": "voicevox",
            "error": "VOICEVOX engine executable was not found.",
            "expected": [
                "C:/Users/<user>/AppData/Local/Programs/VOICEVOX/vv-engine/run.exe",
                "VOICEVOX_ENGINE_EXE",
            ],
            "status": status_before,
        }

    engine = engines[0]
    host, port = _voicevox_url_parts(url)
    log_path = _log_file("voicevox-engine.log")
    cmd = [str(engine), "--host", host, "--port", str(port)]
    log_fh = open(log_path, "ab", buffering=0)
    kwargs: dict[str, Any] = {
        "cwd": str(engine.parent),
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "env": _child_process_env(),
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        ) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **kwargs)
    except OSError as exc:
        log_fh.close()
        return {"ok": False, "provider": "voicevox", "error": str(exc), "command": cmd}
    finally:
        try:
            log_fh.close()
        except Exception:
            pass

    record = {
        "provider": "voicevox",
        "pid": proc.pid,
        "url": url,
        "engine": str(engine),
        "command": cmd,
        "started_at": time.time(),
        "log_path": str(log_path),
    }
    _write_json_file(_tts_active_file(), record)

    timeout = int(values.get("timeout_seconds") or 45)
    deadline = time.time() + timeout
    ready = False
    while time.time() < deadline:
        current = _voicevox_engine_status(url)
        if current.get("reachable"):
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    return {
        "ok": ready,
        "provider": "voicevox",
        "ready": ready,
        **record,
        "status": _voicevox_engine_status(url),
    }


def _irodori_status() -> dict[str, Any]:
    try:
        from plugins.irodori_tts import core as irodori_core
    except Exception as exc:
        return {
            "ok": False,
            "provider": "irodori",
            "available": False,
            "error": f"irodori_tts plugin is not importable: {exc}",
        }
    payload = irodori_core.status_payload()
    server = payload.get("server") if isinstance(payload, dict) else {}
    server_ok = isinstance(server, dict) and server.get("ok") is True
    return {
        **payload,
        "provider": "irodori",
        "usable": bool(payload.get("available") and server_ok),
    }


def _start_irodori_tts(values: dict[str, Any]) -> dict[str, Any]:
    try:
        from plugins.irodori_tts import core as irodori_core
    except Exception as exc:
        return {"ok": False, "provider": "irodori", "error": str(exc)}

    before = _irodori_status()
    if before.get("usable"):
        return {
            "ok": True,
            "provider": "irodori",
            "already_running": True,
            "status": before,
        }

    cfg = irodori_core.settings()
    ps = irodori_core.powershell_path()
    if not ps:
        return {
            "ok": False,
            "provider": "irodori",
            "error": "PowerShell was not found.",
        }
    if not cfg.start_script.is_file():
        return {
            "ok": False,
            "provider": "irodori",
            "error": f"start script not found: {cfg.start_script}",
        }
    if not cfg.repo_dir.exists():
        return {
            "ok": False,
            "provider": "irodori",
            "error": f"repo not found: {cfg.repo_dir}",
        }

    timeout = int(values.get("timeout_seconds") or 120)
    result = _run_command(
        [
            ps,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(cfg.start_script),
            "-RepoDir",
            str(cfg.repo_dir),
        ],
        cwd=cfg.repo_dir,
        timeout_seconds=timeout,
    )
    return {
        "ok": result.get("ok") is True and _irodori_status().get("usable") is True,
        "provider": "irodori",
        "start": result,
        "status": _irodori_status(),
    }


def _select_tts_provider(explicit: Any = None) -> str:
    requested = _plugin_tts_provider(explicit)
    if requested != "auto":
        return requested
    irodori = _irodori_status()
    if irodori.get("available"):
        return "irodori"
    voicevox = _voicevox_engine_status()
    if voicevox.get("installed") or voicevox.get("reachable"):
        return "voicevox"
    return "none"


def tts_status() -> dict[str, Any]:
    requested = _plugin_tts_provider()
    irodori = _irodori_status()
    voicevox = _voicevox_engine_status()
    selected = _select_tts_provider(requested)
    ready = (selected == "irodori" and bool(irodori.get("usable"))) or (
        selected == "voicevox" and bool(voicevox.get("reachable"))
    )
    available = (selected == "irodori" and bool(irodori.get("available"))) or (
        selected == "voicevox"
        and bool(voicevox.get("installed") or voicevox.get("reachable"))
    )
    return {
        "ok": available,
        "requested_provider": requested,
        "selected_provider": selected,
        "ready": ready,
        "providers": {
            "irodori": irodori,
            "voicevox": voicevox,
        },
        "active": _read_json_file(_tts_active_file()),
    }


def start_tts(values: dict[str, Any]) -> dict[str, Any]:
    provider = _select_tts_provider(values.get("provider"))
    if provider == "irodori":
        return _start_irodori_tts(values)
    if provider == "voicevox":
        return _start_voicevox_tts(values)
    return {
        "ok": False,
        "provider": provider,
        "error": "No local TTS backend was found.",
    }


def _tts_output_path(
    provider: str, output_path: Any = None, output_format: Any = None
) -> Path:
    fmt = _coerce_tts_format(output_format)
    if provider == "voicevox":
        fmt = "wav"
    raw = _path_text(output_path)
    if raw:
        path = Path(raw).expanduser()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        path = _audio_dir() / f"hakua-{provider}-{stamp}.{fmt}"
    if path.suffix.lower().lstrip(".") != fmt:
        path = path.with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _synthesize_voicevox(values: dict[str, Any]) -> dict[str, Any]:
    text = str(values.get("text") or "").strip()
    if not text:
        return {"ok": False, "provider": "voicevox", "error": "text is required."}
    url = _plugin_voicevox_url(values.get("voicevox_url"))
    if not _voicevox_engine_status(url).get("reachable"):
        started = _start_voicevox_tts(values)
        if not started.get("ok"):
            return {"ok": False, "provider": "voicevox", "start": started}

    speaker = _plugin_voicevox_speaker(values.get("voicevox_speaker"))
    output_path = _tts_output_path("voicevox", values.get("output_path"), "wav")
    query_url = f"{url}/audio_query?" + urllib.parse.urlencode(
        {"speaker": speaker, "text": text}
    )
    try:
        status_code, query_raw = _http_request(
            query_url, method="POST", timeout_seconds=15.0
        )
        if not 200 <= status_code < 300:
            return {
                "ok": False,
                "provider": "voicevox",
                "error": f"audio_query HTTP {status_code}",
            }
        query = json.loads(query_raw.decode("utf-8", errors="replace"))
        synthesis_url = f"{url}/synthesis?" + urllib.parse.urlencode(
            {"speaker": speaker}
        )
        status_code, wav_bytes = _http_request(
            synthesis_url,
            method="POST",
            payload=query,
            timeout_seconds=60.0,
        )
        if not 200 <= status_code < 300:
            return {
                "ok": False,
                "provider": "voicevox",
                "error": f"synthesis HTTP {status_code}",
            }
        output_path.write_bytes(wav_bytes)
    except Exception as exc:
        return {"ok": False, "provider": "voicevox", "error": str(exc)}

    result: dict[str, Any] = {
        "ok": True,
        "provider": "voicevox",
        "file_path": str(output_path),
        "format": "wav",
        "speaker": speaker,
        "size_bytes": output_path.stat().st_size,
        "media_tag": f"MEDIA:{output_path}",
    }
    if values.get("play"):
        result["playback"] = _play_wav_file(output_path)
    return result


def _synthesize_irodori(values: dict[str, Any]) -> dict[str, Any]:
    text = str(values.get("text") or "").strip()
    if not text:
        return {"ok": False, "provider": "irodori", "error": "text is required."}
    try:
        from plugins.irodori_tts import core as irodori_core

        fmt = _coerce_tts_format(values.get("format"))
        output_path = _tts_output_path("irodori", values.get("output_path"), fmt)
        result = irodori_core.synthesize_text(
            text=text,
            output_path=output_path,
            voice=_plugin_tts_voice(values.get("voice")) or None,
            model=_path_text(values.get("model")) or None,
            output_format=fmt,
            speed=_plugin_tts_speed(values.get("speed")),
        )
        if values.get("play"):
            result["playback"] = _play_wav_file(Path(result["file_path"]))
        return result
    except Exception as exc:
        return {"ok": False, "provider": "irodori", "error": str(exc)}


def _play_wav_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".wav":
        return {
            "ok": False,
            "error": "local playback currently supports wav output only.",
        }
    try:
        if os.name == "nt":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return {"ok": True, "backend": "winsound"}
        player = (
            shutil.which("afplay") or shutil.which("paplay") or shutil.which("aplay")
        )
        if not player:
            return {"ok": False, "error": "No local wav player was found."}
        subprocess.run(
            [player, str(path)],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
        return {"ok": True, "backend": player}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def synthesize_speech(values: dict[str, Any]) -> dict[str, Any]:
    provider = _select_tts_provider(values.get("provider"))
    if provider == "irodori":
        return _synthesize_irodori(values)
    if provider == "voicevox":
        return _synthesize_voicevox(values)
    return {
        "ok": False,
        "provider": provider,
        "error": "No local TTS backend was found.",
    }


def _obs_exe_candidates() -> list[Path]:
    candidates: list[Path] = []
    for exe in ("obs64.exe", "obs.exe"):
        found = shutil.which(exe)
        if found:
            candidates.append(Path(found))

    for root in (
        "HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall",
        "HKLM:\\Software\\WOW6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall",
        "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall",
    ):
        candidates.extend(_obs_candidates_from_uninstall_registry(root))

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(
            Path(local_appdata)
            / "Programs"
            / "obs-studio"
            / "bin"
            / "64bit"
            / "obs64.exe"
        )
    candidates.extend(
        [
            Path("C:/Program Files/obs-studio/bin/64bit/obs64.exe"),
            Path("C:/Program Files (x86)/obs-studio/bin/64bit/obs64.exe"),
        ]
    )

    seen: set[str] = set()
    found_candidates: list[Path] = []
    for path in candidates:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            found_candidates.append(path)
    return found_candidates


def _obs_candidates_from_uninstall_registry(root: str) -> list[Path]:
    if not _is_windows():
        return []
    try:
        import winreg
    except Exception:
        return []

    hive_name, _, subkey = root.partition(":\\")
    hive = {
        "HKLM": winreg.HKEY_LOCAL_MACHINE,
        "HKCU": winreg.HKEY_CURRENT_USER,
    }.get(hive_name)
    if hive is None or not subkey:
        return []

    candidates: list[Path] = []
    try:
        with winreg.OpenKey(hive, subkey) as key:
            count = winreg.QueryInfoKey(key)[0]
            for index in range(count):
                try:
                    child_name = winreg.EnumKey(key, index)
                    with winreg.OpenKey(key, child_name) as child:
                        display_name, _ = winreg.QueryValueEx(child, "DisplayName")
                        if "OBS" not in str(display_name).upper():
                            continue
                        install_location, _ = winreg.QueryValueEx(
                            child, "InstallLocation"
                        )
                        if install_location:
                            candidates.append(
                                Path(str(install_location))
                                / "bin"
                                / "64bit"
                                / "obs64.exe"
                            )
                except OSError:
                    continue
    except OSError:
        return []
    return candidates


def _obs_status() -> dict[str, Any]:
    candidates = _obs_exe_candidates()
    return {
        "installed": bool(candidates),
        "candidates": [str(path) for path in candidates],
        "expected_paths": [
            "C:/Program Files/obs-studio/bin/64bit/obs64.exe",
            "C:/Program Files (x86)/obs-studio/bin/64bit/obs64.exe",
        ],
    }


def _youtube_live_id(explicit: Any = None) -> str:
    cfg = _plugin_config()
    raw = (
        _path_text(explicit)
        or _path_text(cfg.get("youtube_live_id"))
        or _path_text(os.environ.get("AITUBER_ONAIR_YOUTUBE_LIVE_ID"))
        or _path_text(os.environ.get("YOUTUBE_LIVE_ID"))
    )
    if not raw:
        return ""
    return _extract_youtube_live_id(raw)


def _extract_youtube_live_id(value: str) -> str:
    text = _path_text(value)
    if not text:
        return ""
    parsed = urllib.parse.urlparse(text)
    if parsed.netloc:
        host = parsed.netloc.lower()
        query = urllib.parse.parse_qs(parsed.query)
        if "v" in query and query["v"]:
            return query["v"][0].strip()
        parts = [part for part in parsed.path.split("/") if part]
        if host.endswith("youtu.be") and parts:
            return parts[0].strip()
        if parts and parts[0] in {"live", "shorts", "embed"} and len(parts) > 1:
            return parts[1].strip()
    return text.strip()


def _youtube_api_key_env_name(explicit: Any = None) -> str:
    explicit_text = _path_text(explicit)
    if explicit_text:
        return explicit_text
    for name in (
        "AITUBER_ONAIR_YOUTUBE_API_KEY",
        "YOUTUBE_API_KEY",
        "GOOGLE_API_KEY",
    ):
        if os.environ.get(name):
            return name
    return "AITUBER_ONAIR_YOUTUBE_API_KEY"


def _youtube_api_key_from_env(env_name: str) -> str:
    return _path_text(os.environ.get(env_name))


def _youtube_api_get(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{endpoint}?{query}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return {
            "error": {
                "status": exc.code,
                "reason": exc.reason,
                "body": body,
            },
            "http_status": exc.code,
        }
    except urllib.error.URLError as exc:
        return {"error": {"reason": str(exc)}}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": {"reason": "Invalid JSON response", "body": raw[:512]}}
    return data if isinstance(data, dict) else {}


def youtube_live_chat_id(live_id: str, api_key: str) -> str:
    data = _youtube_api_get(
        "https://youtube.googleapis.com/youtube/v3/videos",
        {"part": "liveStreamingDetails", "id": live_id, "key": api_key},
    )
    if "error" in data and isinstance(data["error"], dict):
        error = data["error"]
        message = error.get("reason")
        if not message and isinstance(error.get("body"), str):
            message = error["body"][:256]
        if not message:
            message = str(error)
        raise RuntimeError(f"YouTube API error: {message}")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return ""
    details = items[0].get("liveStreamingDetails", {})
    if not isinstance(details, dict):
        return ""
    return _path_text(details.get("activeLiveChatId"))


def fetch_youtube_live_comments(
    *,
    live_chat_id: str,
    api_key: str,
    page_token: str = "",
) -> dict[str, Any]:
    params = {
        "part": "authorDetails,snippet",
        "liveChatId": live_chat_id,
        "key": api_key,
    }
    if page_token:
        params["pageToken"] = page_token
    data = _youtube_api_get(
        "https://youtube.googleapis.com/youtube/v3/liveChat/messages",
        params,
    )
    comments: list[dict[str, str]] = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
        author = (
            item.get("authorDetails")
            if isinstance(item.get("authorDetails"), dict)
            else {}
        )
        text_details = snippet.get("textMessageDetails")
        super_chat = snippet.get("superChatDetails")
        comment_text = ""
        if isinstance(text_details, dict):
            comment_text = _path_text(text_details.get("messageText"))
        if not comment_text and isinstance(super_chat, dict):
            comment_text = _path_text(super_chat.get("userComment"))
        if not comment_text:
            continue
        comments.append(
            {
                "id": _path_text(item.get("id")),
                "author": _path_text(author.get("displayName")) or "viewer",
                "text": comment_text,
                "published_at": _path_text(snippet.get("publishedAt")),
            }
        )
    return {
        "ok": "error" not in data,
        "comments": comments,
        "next_page_token": _path_text(data.get("nextPageToken")),
        "polling_interval_ms": int(data.get("pollingIntervalMillis") or 0),
        "error": data.get("error"),
    }


def _youtube_comments_active_status() -> dict[str, Any]:
    active = _read_json_file(_youtube_comments_active_file())
    if not active:
        return {"ok": False, "reason": "no active YouTube comment monitor"}
    pid = int(active.get("pid") or 0)
    alive = _pid_alive(pid)
    return {**active, "ok": True, "alive": alive, "pid": pid}


def youtube_comments_status(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    live_id = _youtube_live_id(values.get("live_id"))
    api_key_env = _youtube_api_key_env_name(values.get("api_key_env"))
    return {
        "ok": True,
        "live_id_present": bool(live_id),
        "live_id": live_id,
        "api_key_env": api_key_env,
        "api_key_present": bool(_youtube_api_key_from_env(api_key_env)),
        "active": _youtube_comments_active_status(),
        "active_file": str(_youtube_comments_active_file()),
    }


def start_youtube_comments(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    live_id = _youtube_live_id(values.get("live_id"))
    api_key_env = _youtube_api_key_env_name(values.get("api_key_env"))
    api_key = _youtube_api_key_from_env(api_key_env)
    if not live_id:
        return {
            "ok": False,
            "error": "YouTube live video id is required.",
            "env": "AITUBER_ONAIR_YOUTUBE_LIVE_ID or YOUTUBE_LIVE_ID",
        }
    if not api_key:
        return {
            "ok": False,
            "error": "YouTube Data API key environment variable is not set.",
            "api_key_env": api_key_env,
            "env": "AITUBER_ONAIR_YOUTUBE_API_KEY, YOUTUBE_API_KEY, or GOOGLE_API_KEY",
        }

    existing = _youtube_comments_active_status()
    if existing.get("alive") and not values.get("force"):
        return {
            "ok": True,
            "already_running": True,
            "active": existing,
        }
    if existing.get("alive") and values.get("force"):
        stop_youtube_comments({"force": True})

    log_path = _log_file("youtube-comments.log")
    poll_seconds = max(2.0, min(120.0, float(values.get("poll_seconds") or 10.0)))
    play = values.get("play")
    if play is None:
        play = True
    cmd = [
        sys.executable,
        "-m",
        "plugins.aituber_onair.youtube_comments_worker",
        "--live-id",
        live_id,
        "--api-key-env",
        api_key_env,
        "--poll-seconds",
        str(poll_seconds),
    ]
    if values.get("skip_existing"):
        cmd.append("--skip-existing")
    if play:
        cmd.append("--play")
    env = _child_process_env(
        {
            api_key_env: api_key,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        }
    )
    with log_path.open("ab") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    record = {
        "pid": proc.pid,
        "live_id": live_id,
        "api_key_env": api_key_env,
        "poll_seconds": poll_seconds,
        "play": bool(play),
        "log_path": str(log_path),
        "started_at": time.time(),
        "command": [
            part if part != api_key else "<redacted>" for part in cmd
        ],
    }
    _write_json_file(_youtube_comments_active_file(), record)
    return {"ok": True, "active": record}


def stop_youtube_comments(values: dict[str, Any] | None = None) -> dict[str, Any]:
    active = _youtube_comments_active_status()
    if not active.get("ok"):
        return active
    pid = int(active.get("pid") or 0)
    if not active.get("alive"):
        try:
            _youtube_comments_active_file().unlink()
        except FileNotFoundError:
            pass
        return {"ok": True, "stopped": True, "was_alive": False}
    try:
        from gateway.status import terminate_pid

        terminate_pid(pid, force=bool((values or {}).get("force")))
    except Exception as exc:
        return {"ok": False, "error": str(exc), "pid": pid}
    for _ in range(20):
        if not _pid_alive(pid):
            try:
                _youtube_comments_active_file().unlink()
            except FileNotFoundError:
                pass
            return {"ok": True, "stopped": True, "pid": pid}
        time.sleep(0.2)
    return {
        "ok": False,
        "error": "comment monitor still appears alive; retry with force=true",
        "pid": pid,
    }


def _loop_active_status(path: Path, reason: str) -> dict[str, Any]:
    active = _read_json_file(path)
    if not active:
        return {"ok": False, "reason": reason}
    pid = int(active.get("pid") or 0)
    return {**active, "ok": True, "alive": _pid_alive(pid), "pid": pid}


def loops_status(values: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "autonomous": _loop_active_status(
            _autonomous_talk_active_file(), "no active autonomous talk loop"
        ),
        "comments": _loop_active_status(
            _comment_reactions_active_file(), "no active comment reaction loop"
        ),
        "queue_file": str(_local_comment_queue_file()),
        "processed_file": str(_local_comment_processed_file()),
    }


def _start_local_loop(
    *,
    active_file: Path,
    mode: str,
    log_name: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    existing = _loop_active_status(active_file, f"no active {mode} loop")
    if existing.get("alive") and not values.get("force"):
        return {"ok": True, "already_running": True, "active": existing}
    if existing.get("alive") and values.get("force"):
        _stop_loop_file(active_file, force=True)

    play = values.get("play")
    if play is None:
        play = True
    log_path = _log_file(log_name)
    cmd = [
        _hermes_python_exe(),
        "-m",
        "plugins.aituber_onair.local_loops_worker",
        "--mode",
        mode,
    ]
    if mode == "autonomous":
        interval = max(10.0, min(3600.0, float(values.get("interval_seconds") or 60.0)))
        cmd.extend(["--interval-seconds", str(interval)])
        topic = str(values.get("topic") or "").strip()
        if topic:
            cmd.extend(["--topic", topic])
        poll_seconds = None
    else:
        poll_seconds = max(1.0, min(120.0, float(values.get("poll_seconds") or 2.0)))
        cmd.extend(["--poll-seconds", str(poll_seconds)])
        cmd.extend(["--queue-file", str(_local_comment_queue_file())])
        cmd.extend(["--processed-file", str(_local_comment_processed_file())])
        interval = None
        topic = ""
    if play:
        cmd.append("--play")

    env = _child_process_env(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    with log_path.open("ab") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    record = {
        "pid": proc.pid,
        "mode": mode,
        "play": bool(play),
        "log_path": str(log_path),
        "started_at": time.time(),
        "command": cmd,
    }
    if interval is not None:
        record["interval_seconds"] = interval
    if poll_seconds is not None:
        record["poll_seconds"] = poll_seconds
        record["queue_file"] = str(_local_comment_queue_file())
        record["processed_file"] = str(_local_comment_processed_file())
    if topic:
        record["topic"] = topic
    _write_json_file(active_file, record)
    return {"ok": True, "active": record}


def start_autonomous_talk_loop(values: dict[str, Any] | None = None) -> dict[str, Any]:
    return _start_local_loop(
        active_file=_autonomous_talk_active_file(),
        mode="autonomous",
        log_name="autonomous-talk.log",
        values=values or {},
    )


def start_comment_reaction_loop(values: dict[str, Any] | None = None) -> dict[str, Any]:
    return _start_local_loop(
        active_file=_comment_reactions_active_file(),
        mode="comments",
        log_name="comment-reactions.log",
        values=values or {},
    )


def enqueue_comment(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    text = str(values.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "text is required."}
    item = {
        "id": f"local-{int(time.time() * 1000)}",
        "author": str(values.get("author") or "viewer").strip() or "viewer",
        "text": text,
        "source": str(values.get("source") or "local").strip() or "local",
        "created_at": _now_utc(),
    }
    queue_file = _local_comment_queue_file()
    with queue_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"ok": True, "comment": item, "queue_file": str(queue_file)}


def _stop_loop_file(path: Path, *, force: bool = False) -> dict[str, Any]:
    active = _loop_active_status(path, f"no active loop at {path.name}")
    if not active.get("ok"):
        return active
    pid = int(active.get("pid") or 0)
    if active.get("alive"):
        try:
            from gateway.status import terminate_pid

            terminate_pid(pid, force=force)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "pid": pid}
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.2)
        if _pid_alive(pid):
            return {
                "ok": False,
                "error": "loop still appears alive; retry with force=true",
                "pid": pid,
            }
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return {"ok": True, "stopped": True, "pid": pid, "was_alive": bool(active.get("alive"))}


def stop_loops(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    target = str(values.get("target") or "all").strip().lower()
    force = bool(values.get("force"))
    results: dict[str, Any] = {}
    if target in {"all", "autonomous"}:
        results["autonomous"] = _stop_loop_file(_autonomous_talk_active_file(), force=force)
    if target in {"all", "comments"}:
        results["comments"] = _stop_loop_file(_comment_reactions_active_file(), force=force)
    if not results:
        return {"ok": False, "error": "target must be all, autonomous, or comments."}
    return {"ok": all(item.get("ok") for item in results.values()), "results": results}


def youtube_ready(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    require_obs = values.get("require_obs")
    if require_obs is None:
        require_obs = True
    require_tts_ready = bool(values.get("require_tts_ready"))

    payload = status()
    active_raw = payload.get("active")
    active = cast(dict[str, Any], active_raw) if isinstance(active_raw, dict) else {}
    tts_raw = payload.get("tts")
    tts = cast(dict[str, Any], tts_raw) if isinstance(tts_raw, dict) else {}
    config_raw = payload.get("config")
    config = cast(dict[str, Any], config_raw) if isinstance(config_raw, dict) else {}
    obs = _obs_status()
    avatar_url = str(config.get("url") or "")
    avatar_running = bool(active.get("alive") and active.get("url_ready"))
    tts_ready = bool(tts.get("ready"))
    obs_ready = bool(obs.get("installed")) or not require_obs
    ready = bool(payload.get("ok") and avatar_running and obs_ready)
    if require_tts_ready:
        ready = ready and tts_ready

    blockers: list[str] = []
    if not payload.get("ok"):
        blockers.append("Hermes AITuber OnAir plugin readiness is incomplete.")
    if not avatar_running:
        blockers.append("Hakua VRM app is not running or not reachable.")
    if require_obs and not obs.get("installed"):
        blockers.append("OBS Studio was not found.")
    if require_tts_ready and not tts_ready:
        blockers.append("The selected Hakua TTS backend is not ready.")

    return {
        "ok": ready,
        "checked_at": _now_utc(),
        "avatar_url": avatar_url,
        "obs_browser_source": {
            "url": avatar_url,
            "width": 1920,
            "height": 1080,
            "recommended_source_type": "Browser Source",
        },
        "youtube_encoder": {
            "server_url": "rtmps://a.rtmps.youtube.com/live2",
            "stream_key": "Paste manually in OBS; Hermes never reads or stores it.",
            "first_time_live_enablement": "Verify the channel in YouTube Studio first; first-time live access can take up to 24 hours.",
        },
        "manual_youtube_steps": [
            "Open YouTube Studio, choose Create, then Go live.",
            "Create or select a stream on the Stream tab.",
            "Copy the stream URL and stream key into OBS stream settings.",
            "Add the Hakua local URL as an OBS Browser Source.",
            "Start streaming from OBS and confirm the preview in YouTube Live Control Room.",
            "Click Go live in YouTube Live Control Room when the preview is correct.",
        ],
        "readiness": {
            "plugin": bool(payload.get("ok")),
            "avatar_app_running": avatar_running,
            "obs_available": bool(obs.get("installed")),
            "tts_ready": tts_ready,
            "youtube_studio_stream_created": "manual_check_required",
            "stream_key_configured_in_obs": "manual_check_required",
        },
        "obs": obs,
        "blockers": blockers,
        "recommended_actions": _youtube_recommended_actions(
            payload,
            avatar_running=avatar_running,
            obs_ready=bool(obs.get("installed")),
            tts_ready=tts_ready,
            require_obs=bool(require_obs),
            require_tts_ready=require_tts_ready,
        ),
        "status": payload,
    }


def _youtube_recommended_actions(
    payload: dict[str, Any],
    *,
    avatar_running: bool,
    obs_ready: bool,
    tts_ready: bool,
    require_obs: bool,
    require_tts_ready: bool,
) -> list[str]:
    actions = list(payload.get("recommended_actions") or [])
    if not avatar_running:
        actions.append("hermes aituber-onair start --avatar vrm --force")
    if require_obs and not obs_ready:
        actions.append(
            "Install OBS Studio, then add the Hakua URL as a Browser Source."
        )
    if require_tts_ready and not tts_ready:
        actions.append("hermes aituber-onair start-tts")
    actions.append(
        "Create a YouTube Studio stream and paste its RTMPS URL/key into OBS manually."
    )
    return actions


def handle_youtube_ready(args: dict[str, Any] | None = None) -> str:
    return _json(youtube_ready(args or {}))


def handle_youtube_comments_status(args: dict[str, Any] | None = None) -> str:
    return _json(youtube_comments_status(args or {}))


def handle_start_youtube_comments(args: dict[str, Any] | None = None) -> str:
    return _json(start_youtube_comments(args or {}))


def handle_stop_youtube_comments(args: dict[str, Any] | None = None) -> str:
    return _json(stop_youtube_comments(args or {}))


def handle_loops_status(args: dict[str, Any] | None = None) -> str:
    return _json(loops_status(args or {}))


def handle_start_autonomous_talk(args: dict[str, Any] | None = None) -> str:
    return _json(start_autonomous_talk_loop(args or {}))


def handle_start_comment_reactions(args: dict[str, Any] | None = None) -> str:
    return _json(start_comment_reaction_loop(args or {}))


def handle_enqueue_comment(args: dict[str, Any] | None = None) -> str:
    return _json(enqueue_comment(args or {}))


def handle_stop_loops(args: dict[str, Any] | None = None) -> str:
    return _json(stop_loops(args or {}))


def handle_context_status(args: dict[str, Any] | None = None) -> str:
    return _json(context_status(args or {}))


def handle_stream_start_tweet(args: dict[str, Any] | None = None) -> str:
    return _json(stream_start_tweet(args or {}))


def status() -> dict[str, Any]:
    cfg = _plugin_config()
    repo = resolve_repo_root()
    fbx_app = _fbx_app_dir(repo) if repo else None
    vrm_app = _vrm_app_dir(repo) if repo else None
    chat_script = _codex_chat_script(repo) if repo else None
    fbx_port = _plugin_fbx_port()
    vrm_port = _plugin_vrm_port()
    avatar_kind = _coerce_avatar_kind(cfg.get("avatar_kind"))
    port = _plugin_avatar_port(avatar_kind)
    avatar_host = _plugin_avatar_host()
    avatar_public_host = _plugin_avatar_public_host(host=avatar_host)
    url = _avatar_url(port, avatar_public_host)
    active = _active_status()
    active["url_ready"] = (
        _url_ready(str(active.get("url") or url)) if active.get("alive") else False
    )
    tts = tts_status()
    readiness = {
        "repo_root": bool(repo),
        "node": bool(_node_exe()),
        "npm": bool(_npm_exe()),
        "codex_cli_auth": _codex_cli_auth_status().get("has_access_token") is True,
        "fbx_app": bool(fbx_app and (fbx_app / "package.json").is_file()),
        "vrm_app": bool(vrm_app and (vrm_app / "package.json").is_file()),
        "codex_character_cli": bool(chat_script and chat_script.is_file()),
        "chat_dist": bool(repo and _chat_agent_dist(repo).is_file()),
        "tts_backend": bool(tts.get("ok")),
    }
    codex_sdk = (
        _codex_sdk_installed(repo) if repo and _npm_exe() else {"installed": False}
    )
    readiness["codex_sdk"] = bool(codex_sdk.get("installed"))
    ok = all(readiness.values())
    recommended: list[str] = []
    if not readiness["repo_root"]:
        recommended.append(
            "hermes aituber-onair configure --repo-root <path-to-aituber-onair>"
        )
    if not readiness["chat_dist"] or not readiness["codex_sdk"]:
        recommended.append("hermes aituber-onair prepare")
    if not readiness["codex_cli_auth"]:
        recommended.append("Authenticate Codex locally, then rerun status.")
    if not readiness["tts_backend"]:
        recommended.append(
            "Install or configure irodoriTTS or VOICEVOX, then run hermes aituber-onair tts-status."
        )
    elif not tts.get("ready"):
        recommended.append("hermes aituber-onair start-tts")
    return {
        "ok": ok,
        "checked_at": _now_utc(),
        "plugin": PLUGIN_ID,
        "config_key": f"plugins.entries.{PLUGIN_ID}",
        "config": {
            "character_name": cfg.get("character_name") or "はくあ",
            "model": cfg.get("model") or "Codex CLI default",
            "reply_backend": cfg.get("reply_backend") or "auto",
            "hermes_provider": cfg.get("hermes_provider") or "Hermes default",
            "hermes_model": cfg.get("hermes_model") or "Hermes default",
            "hermes_llm_bound": _llm_factory is not None,
            "response_length": cfg.get("response_length") or DEFAULT_RESPONSE_LENGTH,
            "avatar_kind": avatar_kind,
            "fbx_port": fbx_port,
            "vrm_port": vrm_port,
            "avatar_host": avatar_host,
            "avatar_public_host": avatar_public_host,
            "url": url,
            "tts_provider": cfg.get("tts_provider") or DEFAULT_TTS_PROVIDER,
            "tts_voice": _plugin_tts_voice() or "provider default",
            "tts_speed": _plugin_tts_speed() or "provider default",
            "voicevox_url": _plugin_voicevox_url(),
            "voicevox_speaker": _plugin_voicevox_speaker(),
            "youtube_live_id": _youtube_live_id() or "",
            "stream_url": _configured_stream_url(),
            "with_runtime_context": _plugin_runtime_context_enabled(),
        },
        "paths": {
            "repo_root": str(repo) if repo else "",
            "fbx_app_dir": str(fbx_app) if fbx_app else "",
            "vrm_app_dir": str(vrm_app) if vrm_app else "",
            "codex_character_cli": str(chat_script) if chat_script else "",
            "chat_agent_dist": str(_chat_agent_dist(repo)) if repo else "",
            "active_file": str(_active_file()),
        },
        "executables": {"node": _node_exe(), "npm": _npm_exe()},
        "auth": {
            "codex_cli": _codex_cli_auth_status(),
            "hermes_openai_codex": _hermes_codex_auth_status(),
        },
        "codex_sdk": codex_sdk,
        "tts": tts,
        "context": context_status({}),
        "youtube_comments": youtube_comments_status({}),
        "loops": loops_status({}),
        "active": active,
        "readiness": readiness,
        "recommended_actions": recommended,
    }


def save_hakua_config(values: dict[str, Any]) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config, save_config
    except Exception as exc:
        return {"ok": False, "error": f"Hermes config writer unavailable: {exc}"}

    repo = resolve_repo_root(values.get("repo_root"))
    if repo is None and values.get("repo_root"):
        repo = Path(_path_text(values.get("repo_root"))).expanduser()

    cfg = load_config()
    plugins = cfg.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        cfg["plugins"] = plugins
    entries = plugins.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins["entries"] = entries
    entry = entries.setdefault(PLUGIN_ID, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[PLUGIN_ID] = entry

    if repo is not None:
        entry["repo_root"] = str(repo)
        entry["working_directory"] = str(repo)
    entry["character_name"] = "はくあ"
    entry["codex_provider"] = "codex-sdk"
    entry["codex_auth_source"] = "local-codex-cli"
    entry["response_length"] = str(
        values.get("response_length") or DEFAULT_RESPONSE_LENGTH
    )
    entry["skip_git_repo_check"] = True
    entry["fbx_app_dir"] = str(Path("packages") / "core" / "examples" / "react-fbx-app")
    entry["vrm_app_dir"] = str(Path("packages") / "core" / "examples" / "react-vrm-app")
    entry["avatar_kind"] = _coerce_avatar_kind(values.get("avatar_kind"))
    entry["fbx_port"] = _plugin_fbx_port(values.get("fbx_port"))
    entry["vrm_port"] = _plugin_vrm_port(values.get("vrm_port"))
    entry["avatar_host"] = _plugin_avatar_host(values.get("avatar_host"))
    public_host = _plugin_avatar_public_host(
        values.get("avatar_public_host"), host=entry["avatar_host"]
    )
    entry["avatar_public_host"] = public_host
    entry["tts_provider"] = _plugin_tts_provider(values.get("tts_provider"))
    entry["voicevox_url"] = _plugin_voicevox_url(values.get("voicevox_url"))
    entry["voicevox_speaker"] = _plugin_voicevox_speaker(values.get("voicevox_speaker"))
    voicevox_engine = _plugin_voicevox_engine_exe(values.get("voicevox_engine_exe"))
    if voicevox_engine:
        entry["voicevox_engine_exe"] = voicevox_engine
    elif _voicevox_engine_candidates():
        entry["voicevox_engine_exe"] = str(_voicevox_engine_candidates()[0])
    tts_voice = _plugin_tts_voice(values.get("tts_voice"))
    if tts_voice:
        entry["tts_voice"] = tts_voice
    tts_speed = _plugin_tts_speed(values.get("tts_speed"))
    if tts_speed is not None:
        entry["tts_speed"] = tts_speed
    youtube_live_id = _youtube_live_id(values.get("youtube_live_id"))
    if youtube_live_id:
        entry["youtube_live_id"] = youtube_live_id
    stream_url = _path_text(values.get("stream_url"))
    if stream_url:
        entry["stream_url"] = stream_url
    if values.get("with_runtime_context") is not None:
        entry["with_runtime_context"] = _coerce_bool(
            values.get("with_runtime_context"), False
        )
    model = _path_text(values.get("model"))
    if model:
        entry["model"] = model
    if values.get("reply_backend"):
        entry["reply_backend"] = _plugin_reply_backend(values.get("reply_backend"))
    if values.get("hermes_provider"):
        entry["hermes_provider"] = _path_text(values.get("hermes_provider"))
    if values.get("hermes_model"):
        entry["hermes_model"] = _path_text(values.get("hermes_model"))
    prompt = str(values.get("system_prompt") or "").strip()
    entry["system_prompt"] = prompt or DEFAULT_HAKUA_SYSTEM_PROMPT

    save_config(cfg)
    return {
        "ok": True,
        "config_key": f"plugins.entries.{PLUGIN_ID}",
        "repo_root": str(repo) if repo else "",
        "character_name": entry["character_name"],
        "model": entry.get("model") or "Codex CLI default",
        "reply_backend": entry.get("reply_backend") or "auto",
        "hermes_provider": entry.get("hermes_provider") or "Hermes default",
        "hermes_model": entry.get("hermes_model") or "Hermes default",
        "fbx_url": _avatar_url(entry["fbx_port"], public_host),
        "vrm_url": _avatar_url(entry["vrm_port"], public_host),
        "avatar_kind": entry["avatar_kind"],
        "avatar_host": entry["avatar_host"],
        "avatar_public_host": public_host,
        "tts_provider": entry["tts_provider"],
        "voicevox_url": entry["voicevox_url"],
        "voicevox_speaker": entry["voicevox_speaker"],
        "voicevox_engine_exe": entry.get("voicevox_engine_exe", ""),
        "tts_voice": entry.get("tts_voice", ""),
        "tts_speed": entry.get("tts_speed", ""),
        "youtube_live_id": entry.get("youtube_live_id", ""),
        "stream_url": entry.get("stream_url", ""),
        "with_runtime_context": entry.get("with_runtime_context", False),
    }


def handle_status(args: dict[str, Any] | None = None) -> str:
    return _json(status())


def handle_configure_hakua(args: dict[str, Any] | None = None) -> str:
    return _json(save_hakua_config(args or {}))


def prepare(values: dict[str, Any]) -> dict[str, Any]:
    repo, error = _resolve_required_repo(values.get("repo_root"))
    if error:
        return error
    assert repo is not None
    npm = _npm_exe()
    if not npm:
        return {"ok": False, "error": "npm was not found on PATH."}

    timeout = int(values.get("timeout_seconds") or 300)
    install_codex_sdk = values.get("install_codex_sdk")
    build_chat = values.get("build_chat")
    build_fbx_app = bool(values.get("build_fbx_app"))
    build_vrm_app = bool(values.get("build_vrm_app"))
    if install_codex_sdk is None:
        install_codex_sdk = True
    if build_chat is None:
        build_chat = True

    steps: list[dict[str, Any]] = []
    if install_codex_sdk:
        existing = _codex_sdk_installed(repo)
        if existing.get("installed"):
            steps.append({"name": "install_codex_sdk", "ok": True, "skipped": True})
        else:
            steps.append(
                {
                    "name": "install_codex_sdk",
                    **_run_command(
                        [
                            npm,
                            "install",
                            "--include=optional",
                            "--no-save",
                            "--package-lock=false",
                            CODEX_SDK_PACKAGE,
                            CODEX_CLI_PACKAGE,
                        ],
                        cwd=repo,
                        timeout_seconds=timeout,
                    ),
                }
            )
            after_install = _codex_sdk_installed(repo)
            native_package = _codex_native_package_name()
            if (
                after_install.get("installed") is not True
                and native_package
                and native_package in after_install.get("missing_packages", [])
            ):
                native_spec = _codex_native_package_spec(repo)
                if native_spec:
                    steps.append(
                        {
                            "name": "install_codex_native_package",
                            **_run_command(
                                [
                                    npm,
                                    "install",
                                    "--include=optional",
                                    "--no-save",
                                    "--package-lock=false",
                                    native_spec,
                                ],
                                cwd=repo,
                                timeout_seconds=timeout,
                            ),
                        }
                    )

    if build_chat:
        steps.append(
            {"name": "build_chat", **_build_chat_for_codex(repo, npm, timeout)}
        )

    if build_fbx_app:
        app_dir = _fbx_app_dir(repo)
        if not (app_dir / "package.json").is_file():
            steps.append(
                {
                    "name": "build_fbx_app",
                    "ok": False,
                    "error": f"FBX app package.json was not found: {app_dir}",
                }
            )
        else:
            steps.append(
                {
                    "name": "build_fbx_app",
                    **_run_command(
                        [npm, "run", "build"],
                        cwd=app_dir,
                        timeout_seconds=timeout,
                    ),
                }
            )

    if build_vrm_app:
        app_dir = _vrm_app_dir(repo)
        if not (app_dir / "package.json").is_file():
            steps.append(
                {
                    "name": "build_vrm_app",
                    "ok": False,
                    "error": f"VRoid/VRM app package.json was not found: {app_dir}",
                }
            )
        else:
            steps.append(
                {
                    "name": "build_vrm_app",
                    **_run_command(
                        [npm, "run", "build"],
                        cwd=app_dir,
                        timeout_seconds=timeout,
                    ),
                }
            )

    return {
        "ok": all(step.get("ok") is True for step in steps),
        "repo_root": str(repo),
        "steps": steps,
        "chat_agent_dist": str(_chat_agent_dist(repo)),
    }


def _build_chat_for_codex(repo: Path, npm: str, timeout_seconds: int) -> dict[str, Any]:
    primary = _run_command(
        [npm, "-w", "@aituber-onair/chat", "run", "build"],
        cwd=repo,
        timeout_seconds=timeout_seconds,
    )
    if primary.get("ok") is True:
        return primary

    combined = f"{primary.get('stdout') or ''}\n{primary.get('stderr') or ''}"
    windows_shell_gap = _is_windows() and (
        "'rm' is not recognized" in combined or "'mv' is not recognized" in combined
    )
    if not windows_shell_gap:
        return primary

    # The Codex character CLI only requires dist/cjs/agent.js. Upstream's full
    # build currently uses POSIX rm/mv in package scripts, so refresh the CJS
    # surface directly on Windows without editing AITuber OnAir package files.
    cjs_dir = repo / "packages" / "chat" / "dist" / "cjs"
    try:
        if cjs_dir.exists():
            shutil.rmtree(cjs_dir)
    except OSError as exc:
        return {
            "ok": False,
            "error": f"Failed to remove stale CJS dist: {exc}",
            "primary": primary,
        }

    fallback = _run_command(
        [npm, "-w", "@aituber-onair/chat", "run", "build:cjs"],
        cwd=repo,
        timeout_seconds=timeout_seconds,
    )
    return {
        **fallback,
        "fallback": "build:cjs",
        "primary": primary,
        "chat_agent_dist_exists": _chat_agent_dist(repo).is_file(),
        "ok": fallback.get("ok") is True and _chat_agent_dist(repo).is_file(),
    }


def handle_prepare(args: dict[str, Any] | None = None) -> str:
    return _json(prepare(args or {}))


def start_avatar_app(values: dict[str, Any]) -> dict[str, Any]:
    repo, error = _resolve_required_repo(values.get("repo_root"))
    if error:
        return error
    assert repo is not None
    avatar_kind = _coerce_avatar_kind(values.get("avatar_kind"))
    app_dir = _avatar_app_dir(repo, avatar_kind)
    display_name = _avatar_display_name(avatar_kind)
    if not (app_dir / "package.json").is_file():
        return {
            "ok": False,
            "error": f"{display_name} app package.json was not found: {app_dir}",
        }
    npm = _npm_exe()
    if not npm:
        return {"ok": False, "error": "npm was not found on PATH."}

    existing = _active_status()
    if existing.get("alive") and not values.get("force"):
        return {
            "ok": True,
            "already_running": True,
            "pid": existing.get("pid"),
            "url": existing.get("url"),
            "log_path": existing.get("log_path"),
            "avatar_kind": existing.get("avatar_kind") or "unknown",
        }
    if existing.get("alive") and values.get("force"):
        stop_fbx_app({"force": True})

    explicit_port = (
        values.get("vrm_port") if avatar_kind == "vrm" else values.get("fbx_port")
    )
    port = _plugin_avatar_port(avatar_kind, explicit_port)
    host = _plugin_avatar_host(values.get("host"))
    public_host = _plugin_avatar_public_host(values.get("public_host"), host=host)
    url = _avatar_url(port, public_host)
    readiness_url = _avatar_url(port, "127.0.0.1" if host in {"0.0.0.0", "::"} else public_host)
    log_path = _log_file(f"{avatar_kind}-vite.log")
    cmd = [npm, "run", "dev", "--", "--host", host, "--port", str(port)]
    env = _child_process_env(
        {
            "AITUBER_ONAIR_HERMES_PLUGIN": "1",
            "AITUBER_ONAIR_AVATAR_KIND": avatar_kind,
            "AITUBER_ONAIR_CHARACTER_NAME": _plugin_character_name(),
            "AITUBER_ONAIR_CODEX_AUTH_SOURCE": "local-codex-cli",
            "AITUBER_ONAIR_TTS_PROVIDER": _select_tts_provider(),
            "VOICEVOX_URL": _plugin_voicevox_url(),
            "VOICEVOX_SPEAKER": str(_plugin_voicevox_speaker()),
        }
    )

    log_fh = open(log_path, "ab", buffering=0)
    kwargs: dict[str, Any] = {
        "cwd": str(app_dir),
        "stdout": log_fh,
        "stderr": subprocess.STDOUT,
        "env": env,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **kwargs)
    except OSError as exc:
        log_fh.close()
        return {"ok": False, "error": str(exc), "command": cmd, "cwd": str(app_dir)}
    finally:
        try:
            log_fh.close()
        except Exception:
            pass

    record = {
        "pid": proc.pid,
        "repo_root": str(repo),
        "app_dir": str(app_dir),
        "avatar_kind": avatar_kind,
        "avatar_display_name": display_name,
        "url": url,
        "readiness_url": readiness_url,
        "port": port,
        "host": host,
        "public_host": public_host,
        "started_at": time.time(),
        "log_path": str(log_path),
        "command": cmd,
    }
    _write_json_file(_active_file(), record)
    ready = False
    for _ in range(20):
        if _url_ready(readiness_url, timeout_seconds=0.5):
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    return {"ok": True, "ready": ready, **record}


def start_fbx_app(values: dict[str, Any]) -> dict[str, Any]:
    return start_avatar_app({**values, "avatar_kind": "fbx"})


def handle_start(args: dict[str, Any] | None = None) -> str:
    return _json(start_avatar_app(args or {}))


def stop_fbx_app(values: dict[str, Any]) -> dict[str, Any]:
    active = _active_status()
    if not active.get("ok"):
        return active
    pid = int(active.get("pid") or 0)
    if not active.get("alive"):
        _clear_active()
        return {"ok": True, "stopped": False, "reason": "record was stale", "pid": pid}

    try:
        from gateway.status import terminate_pid

        terminate_pid(pid, force=bool(values.get("force")))
    except Exception as exc:
        return {"ok": False, "error": str(exc), "pid": pid}

    deadline = time.time() + 5
    while time.time() < deadline:
        if not _pid_alive(pid):
            _clear_active()
            return {"ok": True, "stopped": True, "pid": pid}
        time.sleep(0.2)

    if values.get("force"):
        return {
            "ok": False,
            "error": "process still appears alive after force stop",
            "pid": pid,
        }
    return {
        "ok": False,
        "error": "process still appears alive; retry with force=true",
        "pid": pid,
    }


def handle_stop(args: dict[str, Any] | None = None) -> str:
    return _json(stop_fbx_app(args or {}))


def handle_tts_status(args: dict[str, Any] | None = None) -> str:
    return _json(tts_status())


def handle_start_tts(args: dict[str, Any] | None = None) -> str:
    return _json(start_tts(args or {}))


def handle_speak(args: dict[str, Any] | None = None) -> str:
    return _json(synthesize_speech(args or {}))


def _extract_character_reply(stdout: str, character_name: str) -> str:
    marker = f"{character_name}> "
    lines = stdout.splitlines()
    for line in reversed(lines):
        if line.startswith(marker):
            return line[len(marker) :].strip()
    useful = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.startswith("===")
        and not line.startswith("character:")
        and not line.startswith("provider:")
        and not line.startswith("model:")
        and not line.startswith("workingDirectory:")
    ]
    return useful[-1] if useful else ""


def _codex_provider_failed(stderr: str) -> bool:
    text = stderr.lower()
    return "codex-sdk provider failed" in text or "original error:" in text


def _codex_provider_error_detail(stderr: str) -> str:
    for raw_line in stderr.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"Error", "Original error:"}:
            continue
        if line.startswith("[error]"):
            continue
        if line.startswith("at "):
            continue
        json_text = ""
        if line.startswith("Error: {"):
            json_text = line.removeprefix("Error: ").strip()
        elif line.startswith("{"):
            json_text = line
        if json_text:
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict) and error.get("message"):
                    return str(error["message"])
        return line
    return ""


def _should_retry_codex_default(model: str, stderr: str) -> bool:
    if not model:
        return False
    text = stderr.lower()
    return "model is not supported" in text or "not supported when using codex" in text


def _should_try_hermes_cli_fallback(payload: dict[str, Any]) -> bool:
    detail = str(payload.get("error") or "").lower()
    return (
        "model is not supported" in detail
        or "not supported when using codex" in detail
        or "usage limit" in detail
        or "rate limit" in detail
    )


def _run_hermes_llm_once(
    *,
    prompt: str,
    character_name: str,
    system_prompt: str,
    response_length: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    if _llm_factory is None:
        return {
            "ok": False,
            "provider": "hermes-agent",
            "error": "Hermes Agent plugin LLM is not bound.",
        }
    provider = _plugin_hermes_provider(values.get("hermes_provider"))
    model = _plugin_hermes_model(values.get("hermes_model"))
    timeout = int(values.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    try:
        llm = _llm_factory()
        result = llm.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            provider=provider or None,
            model=model or None,
            max_tokens=_response_length_to_tokens(response_length),
            temperature=0.7,
            timeout=timeout,
            purpose="aituber-onair.hakua",
        )
        reply = str(getattr(result, "text", "") or "").strip()
        return {
            "ok": bool(reply),
            "character_name": character_name,
            "provider": "hermes-agent",
            "model": str(getattr(result, "model", "") or model or "Hermes default"),
            "hermes_provider": str(
                getattr(result, "provider", "") or provider or "Hermes default"
            ),
            "reply": reply,
            "usage": getattr(result, "usage", None),
            "audit": getattr(result, "audit", None),
            "error": "" if reply else "Hermes Agent plugin LLM returned no reply.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "character_name": character_name,
            "provider": "hermes-agent",
            "model": model or "Hermes default",
            "hermes_provider": provider or "Hermes default",
            "reply": "",
            "error": f"Hermes Agent plugin LLM failed: {exc}",
        }


def _run_hermes_cli_once(
    *,
    prompt: str,
    character_name: str,
    system_prompt: str,
    response_length: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    timeout = int(values.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    oneshot_prompt = (
        f"{system_prompt}\n\n"
        f"You are speaking as {character_name}. "
        "Return only the short spoken reply. Do not call tools.\n\n"
        f"User input:\n{prompt}"
    )
    cmd = [_hermes_python_exe(), "-m", "hermes_cli", "--oneshot", oneshot_prompt]
    hermes_model = _plugin_hermes_model(values.get("hermes_model"))
    hermes_provider = _plugin_hermes_provider(values.get("hermes_provider"))
    if hermes_model:
        cmd.extend(["--model", hermes_model])
    if hermes_provider and hermes_model:
        cmd.extend(["--provider", hermes_provider])
    env = _child_process_env(
        {
            "HERMES_YOLO_MODE": "1",
            "HERMES_ACCEPT_HOOKS": "1",
            "HERMES_QUIET": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    result = _run_command(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        timeout_seconds=timeout,
    )
    reply = str(result.get("stdout") or "").strip()
    ok = result.get("ok") is True and bool(reply)
    payload = {
        "ok": ok,
        "character_name": character_name,
        "provider": "hermes-agent-cli",
        "model": hermes_model or "Hermes default",
        "hermes_provider": hermes_provider or "Hermes default",
        "reply": reply,
        "response_length": response_length,
        "exit_code": result.get("exit_code"),
        "command": result.get("command"),
        "cwd": result.get("cwd"),
        "stderr": result.get("stderr"),
    }
    if not ok:
        payload["error"] = (
            str(result.get("stderr") or "").strip()
            or "Hermes Agent oneshot returned no reply."
        )
    return payload


def run_hakua_once(values: dict[str, Any]) -> dict[str, Any]:
    prompt = str(values.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt is required."}
    spoken_prompt = _prompt_with_runtime_context(prompt, values)
    character_name = _plugin_character_name()
    response_length = _plugin_response_length(values.get("response_length"))
    system_prompt = _plugin_system_prompt()
    reply_backend = _plugin_reply_backend(values.get("reply_backend"))

    if reply_backend in {"auto", "hermes"} and _llm_factory is not None:
        payload = _run_hermes_llm_once(
            prompt=spoken_prompt,
            character_name=character_name,
            system_prompt=system_prompt,
            response_length=response_length,
            values=values,
        )
        payload["reply_backend"] = "hermes"
        if not payload.get("ok") and _should_try_hermes_cli_fallback(payload):
            hermes_error = payload.get("error")
            payload = _run_hermes_cli_once(
                prompt=spoken_prompt,
                character_name=character_name,
                system_prompt=system_prompt,
                response_length=response_length,
                values=values,
            )
            payload["reply_backend"] = "hermes"
            payload["hermes_facade_error"] = hermes_error
        if values.get("speak") and payload.get("reply"):
            payload["tts"] = synthesize_speech(
                {
                    "text": payload["reply"],
                    "provider": values.get("tts_provider"),
                    "output_path": values.get("output_path"),
                    "format": values.get("format"),
                    "voice": values.get("tts_voice") or values.get("voice"),
                    "speed": values.get("tts_speed") or values.get("speed"),
                    "play": values.get("play"),
                }
            )
        return payload

    if reply_backend == "hermes":
        payload = _run_hermes_cli_once(
            prompt=spoken_prompt,
            character_name=character_name,
            system_prompt=system_prompt,
            response_length=response_length,
            values=values,
        )
        payload["reply_backend"] = "hermes"
        if values.get("speak") and payload.get("reply"):
            payload["tts"] = synthesize_speech(
                {
                    "text": payload["reply"],
                    "provider": values.get("tts_provider"),
                    "output_path": values.get("output_path"),
                    "format": values.get("format"),
                    "voice": values.get("tts_voice") or values.get("voice"),
                    "speed": values.get("tts_speed") or values.get("speed"),
                    "play": values.get("play"),
                }
            )
        return payload

    repo, error = _resolve_required_repo(values.get("repo_root"))
    if error:
        return error
    assert repo is not None
    node = _node_exe()
    if not node:
        return {"ok": False, "error": "node was not found on PATH."}
    script = _codex_chat_script(repo)
    if not script.is_file():
        return {
            "ok": False,
            "error": f"Codex character chat script was not found: {script}",
        }
    if not _chat_agent_dist(repo).is_file():
        return {
            "ok": False,
            "error": "@aituber-onair/chat is not built.",
            "prepare": "hermes aituber-onair prepare",
        }
    sdk = _codex_sdk_installed(repo)
    if not sdk.get("installed"):
        return {
            "ok": False,
            "error": "@openai/codex-sdk is not installed in the local aituber-onair checkout.",
            "prepare": "hermes aituber-onair prepare",
        }
    auth = _codex_cli_auth_status()
    if not auth.get("has_access_token"):
        return {
            "ok": False,
            "error": "Codex CLI auth was not found.",
            "auth": auth,
        }

    model = _plugin_model(values.get("model"))
    timeout = int(values.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    working_directory = _plugin_working_directory(repo)
    prompt_path = (
        _workspace_root() / "prompts" / f"hakua-once-{int(time.time() * 1000)}.txt"
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(spoken_prompt, encoding="utf-8")

    def run_chat_once(active_model: str) -> dict[str, Any]:
        cmd = [
            node,
            str(script),
            f"--onceFile={prompt_path}",
            f"--name={character_name}",
            f"--systemPrompt={system_prompt}",
            f"--responseLength={response_length}",
            f"--workingDirectory={working_directory}",
            "--skipGitRepoCheck=true",
        ]
        if active_model:
            cmd.append(f"--model={active_model}")
        env = _child_process_env(
            {
                "CODEX_CHARACTER_NAME": character_name,
                "CODEX_CHARACTER_SYSTEM_PROMPT": system_prompt,
                "CODEX_WORKING_DIRECTORY": working_directory,
                "CODEX_SKIP_GIT_REPO_CHECK": "true",
                "CODEX_RESPONSE_LENGTH": response_length,
                "CODEX_SDK_MODEL": active_model if active_model else None,
            }
        )
        return _run_command(cmd, cwd=repo, env=env, timeout_seconds=timeout)

    attempts: list[dict[str, Any]] = []
    result = run_chat_once(model)
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    reply = _extract_character_reply(stdout, character_name)
    provider_failed = _codex_provider_failed(stderr)
    attempts.append(
        {
            "model": model or "Codex CLI default",
            "ok": result.get("ok") is True and bool(reply) and not provider_failed,
            "provider_error": _codex_provider_error_detail(stderr),
        }
    )
    effective_model = model
    fallback_from_model = ""
    if provider_failed and _should_retry_codex_default(model, stderr):
        fallback_from_model = model
        result = run_chat_once("")
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        reply = _extract_character_reply(stdout, character_name)
        provider_failed = _codex_provider_failed(stderr)
        effective_model = ""
        attempts.append(
            {
                "model": "Codex CLI default",
                "ok": result.get("ok") is True and bool(reply) and not provider_failed,
                "provider_error": _codex_provider_error_detail(stderr),
            }
        )
    ok = result.get("ok") is True and bool(reply) and not provider_failed
    payload = {
        "ok": ok,
        "character_name": character_name,
        "provider": "codex-sdk",
        "reply_backend": "codex",
        "model": effective_model or "Codex CLI default",
        "reply": reply,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": result.get("exit_code"),
        "command": result.get("command"),
        "cwd": result.get("cwd"),
        "attempts": attempts,
    }
    if fallback_from_model:
        payload["fallback_from_model"] = fallback_from_model
    if not ok:
        if provider_failed:
            payload["error"] = "Codex SDK provider failed."
            detail = _codex_provider_error_detail(stderr)
            if detail:
                payload["provider_error"] = detail
        elif not reply:
            payload["error"] = "Codex character chat returned no reply."
        else:
            payload["error"] = "Codex character chat command failed."
    if values.get("speak") and reply:
        payload["tts"] = synthesize_speech(
            {
                "text": reply,
                "provider": values.get("tts_provider"),
                "output_path": values.get("output_path"),
                "format": values.get("format"),
                "voice": values.get("tts_voice") or values.get("voice"),
                "speed": values.get("tts_speed") or values.get("speed"),
                "play": values.get("play"),
            }
        )
    return payload


def handle_say(args: dict[str, Any] | None = None) -> str:
    return _json(run_hakua_once(args or {}))


def handle_smoke(args: dict[str, Any] | None = None) -> str:
    values = dict(args or {})
    values.setdefault(
        "prompt",
        "はくあ、配信開始の短い挨拶を一文でお願いします。",
    )
    return _json(run_hakua_once(values))


HELP = """aituber commands:
  /aituber status
  /aituber context-status
  /aituber stream-start-tweet <url> [--live]
  /aituber configure
  /aituber prepare
  /aituber start [fbx|vrm|vroid] [--force]
  /aituber stop [--force]
  /aituber tts-status
  /aituber start-tts
  /aituber speak <text>
  /aituber say <prompt>
  /aituber say --speak <prompt>
  /aituber smoke
  /aituber youtube-ready
  /aituber comments-status
  /aituber start-comments <youtube-live-id-or-url>
  /aituber stop-comments
  /aituber loops-status
  /aituber start-autonomous
  /aituber start-reactions
  /aituber comment <text>
  /aituber stop-loops
"""


def handle_slash(raw_args: str) -> str:
    try:
        argv = shlex.split((raw_args or "").strip())
    except ValueError as exc:
        return _json({"ok": False, "error": str(exc)})
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return HELP
    command = argv[0].lower()
    if command == "status":
        return handle_status({})
    if command in {"context-status", "runtime-context"}:
        prompt = " ".join(arg for arg in argv[1:] if not arg.startswith("--"))
        return handle_context_status({"prompt": prompt})
    if command in {"stream-start-tweet", "tweet-start", "post-start"}:
        url = next((arg for arg in argv[1:] if not arg.startswith("--")), "")
        topic = " ".join(
            arg for arg in argv[1:] if not arg.startswith("--") and arg != url
        )
        return handle_stream_start_tweet(
            {
                "url": url,
                "topic": topic,
                "live": "--live" in argv,
                "allow_private_url": "--allow-private-url" in argv,
            }
        )
    if command in {"configure", "config", "setup"}:
        return handle_configure_hakua({})
    if command == "prepare":
        return handle_prepare({})
    if command == "start":
        avatar_kind = next(
            (arg for arg in argv[1:] if arg.lower() in {"fbx", "vrm", "vroid"}),
            "",
        )
        return handle_start({"avatar_kind": avatar_kind, "force": "--force" in argv})
    if command == "stop":
        return handle_stop({"force": "--force" in argv})
    if command in {"tts-status", "tts"}:
        return handle_tts_status({})
    if command == "start-tts":
        return handle_start_tts({})
    if command == "speak":
        prompt = " ".join(arg for arg in argv[1:] if not arg.startswith("--"))
        return handle_speak({"text": prompt, "play": "--play" in argv})
    if command == "smoke":
        return handle_smoke({})
    if command in {"youtube-ready", "youtube", "onair-ready"}:
        return handle_youtube_ready({})
    if command in {"comments-status", "comment-status"}:
        return handle_youtube_comments_status({})
    if command in {"start-comments", "comments-start", "onair-comments"}:
        live_id = next((arg for arg in argv[1:] if not arg.startswith("--")), "")
        return handle_start_youtube_comments(
            {
                "live_id": live_id,
                "skip_existing": "--skip-existing" in argv,
                "play": "--no-play" not in argv,
                "force": "--force" in argv,
            }
        )
    if command in {"stop-comments", "comments-stop"}:
        return handle_stop_youtube_comments({"force": "--force" in argv})
    if command in {"loops-status", "loop-status"}:
        return handle_loops_status({})
    if command in {"start-autonomous", "autonomous-start", "idle-talk"}:
        topic = " ".join(arg for arg in argv[1:] if not arg.startswith("--"))
        return handle_start_autonomous_talk(
            {
                "topic": topic,
                "play": "--no-play" not in argv,
                "force": "--force" in argv,
            }
        )
    if command in {"start-reactions", "reactions-start", "start-local-comments"}:
        return handle_start_comment_reactions(
            {"play": "--no-play" not in argv, "force": "--force" in argv}
        )
    if command in {"comment", "enqueue-comment"}:
        prompt = " ".join(arg for arg in argv[1:] if not arg.startswith("--"))
        return handle_enqueue_comment({"text": prompt, "source": "slash"})
    if command in {"stop-loops", "loops-stop"}:
        return handle_stop_loops({"target": "all", "force": "--force" in argv})
    if command == "say":
        prompt = " ".join(arg for arg in argv[1:] if not arg.startswith("--"))
        return handle_say(
            {
                "prompt": prompt,
                "speak": "--speak" in argv,
                "play": "--play" in argv,
                "with_runtime_context": "--with-runtime-context" in argv,
            }
        )
    return HELP
