"""Tool capability detection: which optional tool backends are available and active."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set

from rabbit_cli.config import get_env_value, load_config
from tools.tool_backend_helpers import (
    fal_key_is_configured, has_direct_modal_credentials, normalize_browser_cloud_provider,
    resolve_openai_audio_api_key,
)

_DEFAULT_PLATFORM_TOOLSETS = {"cli": "rabbit-cli"}
_LABELS = {"web": "Web tools", "image_gen": "Image generation", "video_gen": "Video generation",
           "tts": "Text-to-speech", "stt": "Speech-to-text", "browser": "Browser automation", "modal": "Modal execution"}
_FEATURE_ORDER = tuple(_LABELS)


@dataclass(frozen=True)
class FeatureState:
    key: str
    label: str
    available: bool
    active: bool
    direct_override: bool
    toolset_enabled: bool
    current_provider: str = ""
    explicit_configured: bool = False


@dataclass(frozen=True)
class ToolFeatures:
    features: Dict[str, FeatureState]

    def __getattr__(self, name: str) -> FeatureState:  # ``features.web`` -> per-key state
        if name in _FEATURE_ORDER:
            return self.features[name]
        raise AttributeError(name)

    def items(self) -> Iterable[FeatureState]:
        return (self.features[key] for key in _FEATURE_ORDER)


def _selected_provider(section: object, name_key: str = "provider") -> Optional[str]:
    """Stored provider name for a section, or None when never configured."""
    if not isinstance(section, dict):
        return None
    value = section.get(name_key)
    return None if value is None else (str(value).strip().lower() or None)


def _section(config: Dict[str, object], key: str) -> Dict[str, object]:
    """``config[key]`` when it is a dict, else ``{}`` (read-only view)."""
    value = config.get(key)
    return value if isinstance(value, dict) else {}




def _norm(value: object, default: str = "") -> str:
    return str(value or default).strip().lower()



def _toolset_enabled(config: Dict[str, object], toolset_key: str) -> bool:
    """True when some platform's configured toolsets cover every tool of ``toolset_key``."""
    from toolsets import resolve_toolset

    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        platform_toolsets = {"cli": [_DEFAULT_PLATFORM_TOOLSETS["cli"]]}
    target_tools = set(resolve_toolset(toolset_key))
    if not target_tools:
        return False
    from rabbit_cli.toolset_validation import parse_platform_toolsets_value
    for platform, raw_toolsets in platform_toolsets.items():
        toolset_names = list(parse_platform_toolsets_value(raw_toolsets) or [])
        if not toolset_names:
            toolset_names = [t for t in (_DEFAULT_PLATFORM_TOOLSETS.get(platform),) if t]
        available_tools: Set[str] = set()
        for toolset_name in toolset_names:
            if isinstance(toolset_name, str) and toolset_name:
                try:
                    available_tools.update(resolve_toolset(toolset_name))
                except Exception:
                    continue
        if target_tools.issubset(available_tools):
            return True
    return False


def _has_agent_browser() -> bool:
    # Read the runtime's choice; a broken resolver is not permission to
    # advertise an unchecked binary through a second discovery ladder.
    try:
        from tools.browser_tool_install import _find_agent_browser
        _find_agent_browser(validate=False)
    except (ImportError, OSError):
        return False
    return True


def _local_browser_runnable() -> bool:
    """True when the *local* browser backend would actually start: the CLI must be present AND a
    Chromium build on disk (else agent-browser hangs until the command timeout) unless the Lightpanda
    engine is selected. Mirrors the local-mode tail of tools.browser_tool_install.check_browser_requirements."""
    if not _has_agent_browser():
        return False
    try:
        from tools.browser_tool_install import _chromium_installed
        from tools.browser_tool_lightpanda_fallback import _using_lightpanda_engine
    except Exception:
        return True  # runtime probe unavailable: fall back to binary presence rather than crashing
    return _using_lightpanda_engine() or _chromium_installed()


# kind -> (default provider, provider -> display label)
_PROVIDER_LABELS = {
    "browser": ("local", {
        "browserbase": "Browserbase", "browser-use": "Browser Use", "firecrawl": "Firecrawl",
        "camofox": "Camofox", "local": "Local browser",
    }),
    "tts": ("edge", {
        "openai": "OpenAI TTS", "elevenlabs": "ElevenLabs", "edge": "Edge TTS", "xai": "xAI TTS",
        "mistral": "Mistral Voxtral TTS", "neutts": "NeuTTS",
    }),
    "stt": ("local", {
        "openai": "OpenAI Whisper", "groq": "Groq Whisper", "mistral": "Mistral Voxtral Transcribe",
        "local": "Local faster-whisper",
    }),
}


def _provider_label(kind: str, current_provider: str) -> str:
    default, mapping = _PROVIDER_LABELS[kind]
    return mapping.get(current_provider or default, current_provider or mapping[default])


def _local_stt_backend_available() -> bool:
    """True when faster-whisper imports or a custom local STT command is configured."""
    if get_env_value("RABBIT_LOCAL_STT_COMMAND"):
        return True
    try:
        from tools.transcription_tools import _HAS_FASTER_WHISPER

        return bool(_HAS_FASTER_WHISPER)
    except Exception:
        return False


def _any_env(*names: str) -> bool:
    """True when any of the named env vars (via get_env_value) is set."""
    return any(get_env_value(name) for name in names)


def _state(key: str, **fields) -> FeatureState:
    fields.setdefault("direct_override", fields["active"])
    return FeatureState(key, _LABELS[key], **fields)


def _web_feature(web_cfg: Dict[str, object], tool_enabled: bool) -> FeatureState:
    # Per-capability overrides decide the active search/extract backend independently of web.backend.
    backend, search_backend, extract_backend = (_norm(web_cfg.get(k)) for k in ("backend", "search_backend", "extract_backend"))
    direct = {
        "exa": _any_env("EXA_API_KEY"),
        "firecrawl": _any_env("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"),
        "parallel": _any_env("PARALLEL_API_KEY"),
        "tavily": _any_env("TAVILY_API_KEY") or "tavily" in {backend, search_backend, extract_backend},
        "perplexity": _any_env("PERPLEXITY_API_KEY"),
        "searxng": _any_env("SEARXNG_URL"),
    }
    active = direct.get(backend) or direct.get(search_backend) or (extract_backend in ("tavily", "perplexity") and direct[extract_backend])
    return _state(
        "web", available=any(direct.values()), active=bool(tool_enabled and active), toolset_enabled=tool_enabled,
        current_provider=backend or search_backend or extract_backend or "",
        explicit_configured=bool(backend or search_backend or extract_backend),
    )


def _fal_feature(key: str, tool_enabled: bool, direct: bool, selected: Optional[str]) -> FeatureState:
    label = "FAL" if (selected is not None or direct) else ""
    return _state(
        key, available=direct, active=bool(tool_enabled and direct), toolset_enabled=tool_enabled,
        current_provider=label, explicit_configured=selected is not None or direct,
    )


def _audio_features(
    tts_cfg: Dict[str, object], stt_cfg: Dict[str, object], tts_tool_enabled: bool, selected: Dict[str, Optional[str]],
) -> tuple[FeatureState, FeatureState]:
    tts_current = _norm(tts_cfg.get("provider"), "edge") or "edge"
    stt_current = _norm(stt_cfg.get("provider"), "local") or "local"
    # Whisper reuses the TTS audio key (VOICE_TOOLS_OPENAI_KEY, falling back to OPENAI_API_KEY).
    audio_key = bool(resolve_openai_audio_api_key())
    tts_available = bool({
        "edge": True, "neutts": True, "openai": audio_key,
        "elevenlabs": _any_env("ELEVENLABS_API_KEY"), "mistral": _any_env("MISTRAL_API_KEY"),
    }.get(tts_current, False))
    tts = _state(
        "tts", available=tts_available, active=bool(tts_tool_enabled and tts_available),
        toolset_enabled=tts_tool_enabled, current_provider=_provider_label("tts", tts_current),
        explicit_configured=selected["tts"] is not None and selected["tts"] != "edge",
    )
    # STT isn't a model-callable tool (the gateway voice middleware calls it on every inbound voice message).
    stt_available = bool({
        "local": _local_stt_backend_available(), "openai": audio_key,
        "groq": _any_env("GROQ_API_KEY"), "mistral": _any_env("MISTRAL_API_KEY"),
    }.get(stt_current, False))
    stt = _state(
        "stt", available=stt_available, active=stt_available, toolset_enabled=True,
        current_provider=_provider_label("stt", stt_current), explicit_configured=selected["stt"] is not None,
    )
    return tts, stt


def _browser_feature(browser_cfg: Dict[str, object], tool_enabled: bool, selected: Optional[str]) -> FeatureState:
    """Resolve browser availability using the same precedence as runtime."""
    explicit = "cloud_provider" in browser_cfg
    provider = normalize_browser_cloud_provider(browser_cfg.get("cloud_provider") if explicit else None)
    direct_firecrawl = _any_env("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL")
    # CAMOFOX_URL is the server address, not a selection: an explicit different choice wins over it.
    direct_camofox = _any_env("CAMOFOX_URL") and (selected is None or selected == "camofox")
    direct_browserbase = bool(get_env_value("BROWSERBASE_API_KEY") and get_env_value("BROWSERBASE_PROJECT_ID"))
    direct_browser_use = _any_env("BROWSER_USE_API_KEY")
    # local_available = the agent-browser CLI is present, the only local requirement for cloud providers.
    local_available = _has_agent_browser()
    local_runnable = _local_browser_runnable()
    if explicit:
        cloud_available = {
            "camofox": direct_camofox, "browserbase": local_available and direct_browserbase,
            "browser-use": local_available and direct_browser_use, "firecrawl": local_available and direct_firecrawl,
        }
        current = provider if provider in cloud_available else "local"
        available = bool(cloud_available.get(current, local_runnable))
    # Never-configured autodetect: CAMOFOX_URL activates Camofox when no selection was stored.
    elif direct_camofox:
        current, available = "camofox", True
    elif direct_browser_use:
        current, available = "browser-use", bool(local_available)
    elif direct_browserbase:
        current, available = "browserbase", bool(local_available)
    else:
        current, available = "local", bool(local_runnable)
    return _state(
        "browser", available=available, active=bool(tool_enabled and available), toolset_enabled=tool_enabled,
        current_provider=_provider_label("browser", current), explicit_configured=explicit,
    )


def _modal_feature(terminal_cfg: Dict[str, object], tool_enabled: bool) -> FeatureState:
    terminal_backend = _norm(terminal_cfg.get("backend"), "local")
    is_modal = terminal_backend == "modal"
    direct_modal = has_direct_modal_credentials()
    if not is_modal:
        available, active, direct_override = True, bool(tool_enabled), False
    else:
        available = direct_modal
        active = direct_override = bool(tool_enabled and direct_modal)
    return _state(
        "modal", available=available, active=active, direct_override=direct_override, toolset_enabled=tool_enabled,
        current_provider="Modal" if is_modal else terminal_backend or "local", explicit_configured=is_modal,
    )


def get_tool_features(config: Optional[Dict[str, object]] = None, *, force_fresh: bool = False) -> ToolFeatures:
    if config is None:
        config = load_config() or {}
    enabled = {key: _toolset_enabled(config, key) for key in ("web", "image_gen", "video_gen", "tts", "browser", "terminal")}
    selected = {
        "web": _selected_provider(_section(config, "web"), "backend"),
        "image_gen": _selected_provider(_section(config, "image_gen")),
        "video_gen": _selected_provider(_section(config, "video_gen")),
        "tts": _selected_provider(_section(config, "tts")),
        "stt": _selected_provider(_section(config, "stt")),
        "browser": _selected_provider(_section(config, "browser"), "cloud_provider"),
    }
    fal_configured = fal_key_is_configured()
    tts, stt = _audio_features(_section(config, "tts"), _section(config, "stt"), enabled["tts"], selected)
    features = {  # insertion order == _FEATURE_ORDER
        "web": _web_feature(_section(config, "web"), enabled["web"]),
        "image_gen": _fal_feature("image_gen", enabled["image_gen"], fal_configured, selected["image_gen"]),
        "video_gen": _fal_feature("video_gen", enabled["video_gen"], fal_configured, selected["video_gen"]),
        "tts": tts,
        "stt": stt,
        "browser": _browser_feature(_section(config, "browser"), enabled["browser"], selected["browser"]),
        "modal": _modal_feature(_section(config, "terminal"), enabled["terminal"]),
    }
    return ToolFeatures(features=features)
