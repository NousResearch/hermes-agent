"""Helpers for Nous subscription managed-tool capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from hermes_cli.config import get_env_value, load_config
from hermes_cli.nous_account import (
    NousPortalAccountInfo, format_nous_portal_entitlement_message, get_nous_portal_account_info,
)
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
from utils import is_truthy_value
from tools.tool_backend_helpers import (
    fal_key_is_configured, has_direct_modal_credentials, normalize_browser_cloud_provider, normalize_modal_mode,
    resolve_modal_backend_state, resolve_openai_audio_api_key
)


_DEFAULT_PLATFORM_TOOLSETS = {"cli": "hermes-cli"}


@dataclass(frozen=True)
class _FeatureSpec:
    """Per-feature parameters shared by the status, defaults and Tool Gateway offer surfaces."""

    label: str
    included_by_default: bool
    # Tool-pool coverage category (nous_account.TOOL_COVERAGE_CATEGORIES) gating per backend:
    # the free pool funds image but NOT video; STT shares TTS's "openai-audio" category.
    coverage: str
    gateway: str  # managed gateway probed for readiness (video rides image's fal-queue)
    # Config (section, selection field) written by apply_gateway_defaults; None = not offered (modal).
    section_field: Optional[tuple[str, str]] = None
    offer_label: str = ""
    direct_label: str = ""
    # Direct-credential env vars that stop apply_nous_managed_defaults switching the category to
    # managed (tts/stt also honour resolve_openai_audio_api_key()).
    default_direct_env: tuple[str, ...] = ()


_FEATURES: Dict[str, _FeatureSpec] = {
    "web": _FeatureSpec(
        "Web tools", True, "firecrawl", "firecrawl", ("web", "backend"),
        "Web search & extract (Firecrawl)", "Firecrawl/Exa/Parallel/Tavily/Perplexity/Keenable key or SearXNG",
        ("PARALLEL_API_KEY", "TAVILY_API_KEY", "PERPLEXITY_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"),
    ),
    "image_gen": _FeatureSpec(
        "Image generation", True, "fal", "fal-queue", ("image_gen", "provider"), "Image generation (FAL)", "FAL key",
    ),
    "video_gen": _FeatureSpec(
        "Video generation", False, "fal-video", "fal-queue", ("video_gen", "provider"), "Video generation (FAL)", "FAL key",
    ),
    "tts": _FeatureSpec(
        "OpenAI TTS", True, "openai-audio", "openai-audio", ("tts", "provider"),
        "Text-to-speech (OpenAI TTS)", "OpenAI/ElevenLabs key", ("ELEVENLABS_API_KEY",),
    ),
    "stt": _FeatureSpec(
        "Speech-to-text", True, "openai-audio", "openai-audio", ("stt", "provider"),
        "Speech-to-text (OpenAI Whisper)", "OpenAI/Groq/Mistral key", ("GROQ_API_KEY", "MISTRAL_API_KEY"),
    ),
    "browser": _FeatureSpec(
        "Browser automation", True, "browser-use", "browser-use", ("browser", "cloud_provider"),
        "Browser automation (Browser Use)", "Browser Use/Browserbase key or Camofox",
        ("BROWSER_USE_API_KEY", "BROWSERBASE_API_KEY"),
    ),
    "modal": _FeatureSpec("Modal execution", False, "modal", "modal"),
}

_FEATURE_ORDER = tuple(_FEATURES)
# Public / test-referenced views over the table.
MANAGED_FEATURE_COVERAGE_CATEGORY: Dict[str, str] = {k: s.coverage for k, s in _FEATURES.items()}
_GATEWAY_SECTION_FIELDS = {k: s.section_field for k, s in _FEATURES.items() if s.section_field}
_ALL_GATEWAY_KEYS = tuple(_GATEWAY_SECTION_FIELDS)
_GATEWAY_TOOL_LABELS = {k: _FEATURES[k].offer_label for k in _ALL_GATEWAY_KEYS}
# Sections apply_*_defaults always materialise before writing selections.
_DEFAULT_SECTIONS = ("web", "tts", "stt", "browser")


def _uses_gateway(section: object) -> bool:
    """True when a config section explicitly opts into the gateway (legacy ``use_gateway: true``)."""
    return isinstance(section, dict) and is_truthy_value(section.get("use_gateway"), default=False)


def _selected_provider(section: object, name_key: str = "provider") -> Optional[str]:
    """Stored provider for a section (``read_selection`` semantics): ``"nous"`` for the managed
    selection (stored ``nous`` or legacy ``use_gateway: true``), a vendor name for BYOK, else None."""
    if not isinstance(section, dict):
        return None
    if _uses_gateway(section):
        return "nous"
    value = section.get(name_key)
    return None if value is None else (str(value).strip().lower() or None)


def _selected_provider(section: object, name_key: str = "provider") -> Optional[str]:
    """Return the stored provider string for a config section dict.

    Mirrors :func:`tools.tool_backend_helpers.read_selection`'s semantics on
    an in-memory section dict: ``"nous"`` for the managed selection (stored
    ``nous`` value or legacy ``use_gateway: true``), a vendor name for BYOK
    picks, or ``None`` when no selection is stored. Keeping this in lockstep
    with the runtime resolver is what stops ``hermes status`` from lying.
    """
    if not isinstance(section, dict):
        return None
    if is_truthy_value(section.get("use_gateway"), default=False):
        return "nous"
    value = section.get(name_key)
    if value is None:
        return None
    name = str(value).strip().lower()
    return name or None


@dataclass(frozen=True)
class NousFeatureState:
    key: str
    label: str
    included_by_default: bool
    available: bool
    active: bool
    managed_by_nous: bool
    direct_override: bool
    toolset_enabled: bool
    current_provider: str = ""
    explicit_configured: bool = False


@dataclass(frozen=True)
class NousSubscriptionFeatures:
    subscribed: bool
    nous_auth_present: bool
    provider_is_nous: bool
    features: Dict[str, NousFeatureState]
    account_info: Optional[NousPortalAccountInfo] = None

    def __getattr__(self, name: str) -> NousFeatureState:  # ``features.web`` -> per-key state
        if name in _FEATURE_ORDER:
            return self.features[name]
        raise AttributeError(name)

    def items(self) -> Iterable[NousFeatureState]:
        return (self.features[key] for key in _FEATURE_ORDER)


def _section(config: Dict[str, object], key: str) -> Dict[str, object]:
    """``config[key]`` when it is a dict, else ``{}`` (read-only view)."""
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _ensure_section(config: Dict[str, object], key: str) -> Dict[str, object]:
    """Return ``config[key]`` as a dict, creating/replacing it in ``config`` when missing."""
    value = config.get(key)
    if not isinstance(value, dict):
        value = config[key] = {}
    return value


def _select_nous(config: Dict[str, object], key: str) -> None:
    """Store the managed ``nous`` selection in the ``key`` section (field per _GATEWAY_SECTION_FIELDS)."""
    section_key, field = _GATEWAY_SECTION_FIELDS[key]
    section = _ensure_section(config, section_key)
    section[field] = "nous"
    section.pop("use_gateway", None)


def _norm(value: object, default: str = "") -> str:
    return str(value or default).strip().lower()


def _provider_is_nous(config: Dict[str, object]) -> bool:
    return _norm(_section(config, "model").get("provider")) == "nous"


def _toolset_enabled(config: Dict[str, object], toolset_key: str) -> bool:
    """True when some platform's configured toolsets cover every tool of ``toolset_key``."""
    from toolsets import resolve_toolset

    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        platform_toolsets = {"cli": [_DEFAULT_PLATFORM_TOOLSETS["cli"]]}
    target_tools = set(resolve_toolset(toolset_key))
    if not target_tools:
        return False
    for platform, raw_toolsets in platform_toolsets.items():
        toolset_names = list(raw_toolsets) if isinstance(raw_toolsets, list) else []
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
    import shutil

    from hermes_constants import agent_browser_runnable

    # agent-browser resolves lazily via npx for most installs, which a bare PATH + node_modules
    # probe can't see. Mirror the local-CLI tail of tools.browser_tool_install.check_browser_requirements
    # (same cascade, same Termux carve-out) so setup/status can't diverge from runtime;
    # validate=False keeps this a cheap existence check with no subprocess spawn.
    try:
        from tools.browser_tool_install import _find_agent_browser, _requires_real_termux_browser_install
    except Exception:
        # Runtime probe unavailable: fall back to binary presence rather than crashing. Rungs: PATH;
        # Hermes-managed Node dirs ($HERMES_HOME/node, prepended to PATH at runtime but usually absent
        # from the *probe* process's PATH); local node_modules/.bin (PATHEXT-aware ``shutil.which`` so
        # Windows picks the ``.cmd`` shim). The hit must also run: a dangling symlink is reported by
        # ``which`` but fails at exec.
        # See #48521.
        from hermes_constants import with_hermes_node_path

        local_bin_dir = Path(__file__).parent.parent / "node_modules" / ".bin"
        search_paths = [None, with_hermes_node_path().get("PATH", ""), str(local_bin_dir) if local_bin_dir.is_dir() else ""]
        return any(
            (hit := shutil.which("agent-browser", **({} if path is None else {"path": path}))) and agent_browser_runnable(hit)
            for path in search_paths if path != ""
        )

    try:
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False
    # On Termux, the bare npx fallback is too fragile to advertise as ready.
    return not _requires_real_termux_browser_install(browser_cmd)


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
    if get_env_value("HERMES_LOCAL_STT_COMMAND"):
        return True
    try:
        from tools.transcription_tools import _HAS_FASTER_WHISPER

        return bool(_HAS_FASTER_WHISPER)
    except Exception:
        return False


def _resolve_browser_feature_state(
    *,
    browser_tool_enabled: bool,
    browser_provider: str,
    browser_provider_explicit: bool,
    browser_local_available: bool,
    browser_local_runnable: bool,
    direct_camofox: bool,
    direct_browserbase: bool,
    direct_browser_use: bool,
    direct_firecrawl: bool,
    managed_browser_available: bool,
) -> tuple[str, bool, bool, bool]:
    """Resolve browser availability using the same precedence as runtime.

    ``browser_local_available`` means "the agent-browser CLI is present" — the
    only local requirement for cloud providers, which host their own Chromium.
    ``browser_local_runnable`` additionally requires a usable local Chromium
    build (or the Lightpanda engine), mirroring the local-mode tail of
    :func:`tools.browser_tool.check_browser_requirements`. Local mode must gate
    on the latter, or setup/status advertise a browser that fails on first use
    when Chromium is missing.
    """
    if browser_provider_explicit:
        current_provider = browser_provider or "local"
        if current_provider == "camofox":
            # Camofox is now a stored selection (browser.cloud_provider:
            # camofox); CAMOFOX_URL is only the server address.
            available = bool(direct_camofox)
            active = bool(browser_tool_enabled and available)
            return current_provider, available, active, False
        if current_provider == "browserbase":
            available = bool(browser_local_available and direct_browserbase)
            active = bool(browser_tool_enabled and available)
            return current_provider, available, active, False
        if current_provider == "browser-use":
            provider_available = managed_browser_available or direct_browser_use
            available = bool(browser_local_available and provider_available)
            managed = bool(
                browser_tool_enabled
                and browser_local_available
                and managed_browser_available
                and not direct_browser_use
            )
            active = bool(browser_tool_enabled and available)
            return current_provider, available, active, managed
        if current_provider == "firecrawl":
            available = bool(browser_local_available and direct_firecrawl)
            active = bool(browser_tool_enabled and available)
            return current_provider, available, active, False
        if current_provider == "camofox":
            return current_provider, False, False, False

        current_provider = "local"
        available = bool(browser_local_runnable)
        active = bool(browser_tool_enabled and available)
        return current_provider, available, active, False

    # Never-configured autodetect: CAMOFOX_URL keeps activating Camofox
    # exactly as before when no cloud_provider selection was ever stored.
    if direct_camofox:
        return "camofox", True, bool(browser_tool_enabled), False

    if managed_browser_available or direct_browser_use:
        available = bool(browser_local_available)
        managed = bool(
            browser_tool_enabled
            and browser_local_available
            and managed_browser_available
            and not direct_browser_use
        )
        active = bool(browser_tool_enabled and available)
        return "browser-use", available, active, managed

    if direct_browserbase:
        available = bool(browser_local_available)
        active = bool(browser_tool_enabled and available)
        return "browserbase", available, active, False

    available = bool(browser_local_runnable)
    active = bool(browser_tool_enabled and available)
    return "local", available, active, False


def _account_info_or_none(**kwargs) -> Optional[NousPortalAccountInfo]:
    """``get_nous_portal_account_info(**kwargs)``, failing closed to ``None`` on any error."""
    try:
        return get_nous_portal_account_info(**kwargs)
    except Exception:
        return None


def _state(key: str, **fields) -> NousFeatureState:
    spec = _FEATURES[key]
    fields.setdefault("direct_override", fields["active"] and not fields["managed_by_nous"])
    return NousFeatureState(key, spec.label, spec.included_by_default, **fields)


def _web_feature(web_cfg: Dict[str, object], tool_enabled: bool, managed: bool, web_gw: bool, direct_firecrawl: bool) -> NousFeatureState:
    # Per-capability overrides decide the active search/extract backend independently of web.backend.
    backend, search_backend, extract_backend = (_norm(web_cfg.get(k)) for k in ("backend", "search_backend", "extract_backend"))
    # The "nous" selection is serviced by Firecrawl — normalize so downstream vendor checks hold.
    if backend == "nous" or web_gw:
        backend = "firecrawl"
    # Direct readiness per vendor; a stored managed selection suppresses direct credentials.
    # Keyless Tavily is opt-in: selecting it writes web.backend (or a per-capability override).
    direct = {
        "exa": _any_env("EXA_API_KEY") and not web_gw,
        "firecrawl": direct_firecrawl,
        "parallel": _any_env("PARALLEL_API_KEY") and not web_gw,
        "tavily": (_any_env("TAVILY_API_KEY") or "tavily" in {backend, search_backend, extract_backend}) and not web_gw,
        "perplexity": _any_env("PERPLEXITY_API_KEY") and not web_gw,
        "searxng": _any_env("SEARXNG_URL"),
    }
    web_managed = backend == "firecrawl" and managed and not direct_firecrawl
    active = web_managed or direct.get(backend) or direct.get(search_backend) or (extract_backend in ("tavily", "perplexity") and direct[extract_backend])
    return _state(
        "web", available=bool(managed or any(direct.values())), active=bool(tool_enabled and active),
        managed_by_nous=web_managed, toolset_enabled=tool_enabled,
        current_provider=backend or search_backend or extract_backend or "",
        explicit_configured=bool(backend or search_backend or extract_backend),
    )


def _fal_feature(key: str, tool_enabled: bool, direct: bool, managed: bool, selected: Optional[str]) -> NousFeatureState:
    # image_gen / video_gen: same FAL_KEY, independently gated managed availability.
    fal_managed = tool_enabled and managed and not direct
    if selected not in (None, "nous") or (selected is None and direct):
        label = "FAL"
    else:
        label = "Nous Subscription" if (fal_managed or selected == "nous") else ""
    return _state(
        key, available=bool(managed or direct), active=bool(tool_enabled and (fal_managed or direct)),
        managed_by_nous=fal_managed, toolset_enabled=tool_enabled, current_provider=label,
        explicit_configured=selected is not None or direct,
    )


def _audio_provider(cfg: Dict[str, object], default: str, gw: bool) -> str:
    provider = _norm(cfg.get("provider"), default)
    return "openai" if (provider == "nous" or gw) else (provider or default)


def _audio_features(
    tts_cfg: Dict[str, object], stt_cfg: Dict[str, object], tts_tool_enabled: bool,
    managed: Dict[str, bool], selected: Dict[str, Optional[str]], use_gateway: Dict[str, bool],
) -> tuple[NousFeatureState, NousFeatureState]:
    tts_gw, stt_gw = use_gateway["tts"], use_gateway["stt"]
    # STT default is "local" (faster-whisper, needs a pip install); Nous subscribers are routed to the
    # managed audio gateway by apply_nous_managed_defaults. The "nous" selection is serviced by OpenAI
    # — normalize so downstream vendor checks hold.
    tts_current = _audio_provider(tts_cfg, "edge", tts_gw)
    stt_current = _audio_provider(stt_cfg, "local", stt_gw)
    # Whisper reuses the TTS audio key (VOICE_TOOLS_OPENAI_KEY, falling back to OPENAI_API_KEY).
    audio_key = bool(resolve_openai_audio_api_key())
    direct_openai_tts, direct_openai_stt = audio_key and not tts_gw, audio_key and not stt_gw
    tts_available = bool({
        "edge": True, "neutts": True, "openai": managed["tts"] or direct_openai_tts,
        "elevenlabs": _any_env("ELEVENLABS_API_KEY") and not tts_gw, "mistral": _any_env("MISTRAL_API_KEY"),
    }.get(tts_current, False))
    tts = _state(
        "tts", available=tts_available, active=bool(tts_tool_enabled and tts_available),
        managed_by_nous=tts_tool_enabled and tts_current == "openai" and managed["tts"] and not direct_openai_tts,
        toolset_enabled=tts_tool_enabled, current_provider=_provider_label("tts", tts_current),
        # Mirrors the stored selection so status/picker markers stay in lockstep with dispatch.
        explicit_configured=selected["tts"] is not None and selected["tts"] != "edge",
    )
    # STT isn't a model-callable tool (the gateway voice middleware calls it on every inbound voice
    # message): "enabled" whenever a usable provider is configured, toolset_enabled reported True so
    # status never flags it "tool disabled".
    stt_available = bool({
        "local": _local_stt_backend_available() and not stt_gw, "openai": managed["stt"] or direct_openai_stt,
        "groq": _any_env("GROQ_API_KEY") and not stt_gw, "mistral": _any_env("MISTRAL_API_KEY") and not stt_gw,
    }.get(stt_current, False))
    stt = _state(
        "stt", available=stt_available, active=stt_available, toolset_enabled=True,
        managed_by_nous=stt_current == "openai" and managed["stt"] and not direct_openai_stt,
        current_provider=_provider_label("stt", stt_current), explicit_configured=selected["stt"] is not None,
    )
    return tts, stt


def _browser_feature(
    browser_cfg: Dict[str, object], tool_enabled: bool, managed: bool, selected: Optional[str], browser_gw: bool, direct_firecrawl: bool,
) -> NousFeatureState:
    """Resolve browser availability using the same precedence as runtime."""
    explicit = "cloud_provider" in browser_cfg
    provider = normalize_browser_cloud_provider(browser_cfg.get("cloud_provider") if explicit else None)
    if provider == "nous" or browser_gw:
        provider = "browser-use"
    # CAMOFOX_URL is the server address, not a selection: an explicit different choice wins over it.
    direct_camofox = _any_env("CAMOFOX_URL") and (selected is None or selected == "camofox")
    direct_browserbase = bool(get_env_value("BROWSERBASE_API_KEY") and get_env_value("BROWSERBASE_PROJECT_ID")) and not browser_gw
    direct_browser_use = _any_env("BROWSER_USE_API_KEY") and not browser_gw
    # local_available = the agent-browser CLI is present, the only local requirement for cloud providers.
    local_available = _has_agent_browser()
    local_runnable = _local_browser_runnable()
    browser_use_managed = bool(tool_enabled and local_available and managed and not direct_browser_use)

    if explicit:
        cloud_available = {
            "camofox": direct_camofox, "browserbase": local_available and direct_browserbase,
            "browser-use": local_available and (managed or direct_browser_use), "firecrawl": local_available and direct_firecrawl,
        }
        current = provider if provider in cloud_available else "local"
        available = bool(cloud_available.get(current, local_runnable))
        managed_now = browser_use_managed if current == "browser-use" else False
    # Never-configured autodetect: CAMOFOX_URL activates Camofox when no selection was stored.
    elif direct_camofox:
        current, available, managed_now = "camofox", True, False
    elif managed or direct_browser_use:
        current, available, managed_now = "browser-use", bool(local_available), browser_use_managed
    elif direct_browserbase:
        current, available, managed_now = "browserbase", bool(local_available), False
    else:
        current, available, managed_now = "local", bool(local_runnable), False
    return _state(
        "browser", available=available, active=bool(tool_enabled and available), managed_by_nous=managed_now,
        toolset_enabled=tool_enabled, current_provider=_provider_label("browser", current), explicit_configured=explicit,
    )


def _modal_feature(terminal_cfg: Dict[str, object], tool_enabled: bool, managed: bool, managed_tools_flag: bool) -> NousFeatureState:
    terminal_backend = _norm(terminal_cfg.get("backend"), "local")
    modal_mode = normalize_modal_mode(terminal_cfg.get("modal_mode"))
    direct_modal = has_direct_modal_credentials()
    modal_state = resolve_modal_backend_state(modal_mode, has_direct=direct_modal, managed_ready=managed, managed_enabled=managed_tools_flag)
    is_modal = terminal_backend == "modal"
    # A non-modal terminal backend, or a resolved managed/direct selection, is always "available";
    # otherwise report what the mode could use.
    selected = modal_state["selected_backend"] if is_modal else None
    if not is_modal or selected in ("managed", "direct"):
        available, active = True, bool(tool_enabled)
        managed_now = selected == "managed" and bool(tool_enabled)
        direct_override = is_modal and selected == "direct" and bool(tool_enabled)
    else:
        managed_now = direct_override = active = False
        available = bool({"managed": managed, "direct": direct_modal}.get(modal_mode, managed or direct_modal))
    return _state(
        "modal", available=available, active=active, managed_by_nous=managed_now, direct_override=direct_override,
        toolset_enabled=tool_enabled, current_provider="Modal" if is_modal else terminal_backend or "local",
        explicit_configured=is_modal,
    )


def get_nous_subscription_features(config: Optional[Dict[str, object]] = None, *, force_fresh: bool = False) -> NousSubscriptionFeatures:
    if config is None:
        config = load_config() or {}
    provider_is_nous = _provider_is_nous(config)
    account_info = _account_info_or_none(**({"force_fresh": True} if force_fresh else {}))
    # Coarse "entitled to any managed tool" gate: paid access OR a live free tool pool. Per-backend
    # availability is then narrowed by coverage (the pool funds image but not video, etc.).
    nous_auth_present = bool(account_info and account_info.logged_in)

    def _entitled_for(category: str) -> bool:
        return bool(account_info and account_info.tool_gateway_entitled_for(category))
    subscribed = provider_is_nous or nous_auth_present

    web_tool_enabled = _toolset_enabled(config, "web")
    image_tool_enabled = _toolset_enabled(config, "image_gen")
    video_tool_enabled = _toolset_enabled(config, "video_gen")
    tts_tool_enabled = _toolset_enabled(config, "tts")
    browser_tool_enabled = _toolset_enabled(config, "browser")
    modal_tool_enabled = _toolset_enabled(config, "terminal")

    web_cfg = config.get("web") if isinstance(config.get("web"), dict) else {}
    tts_cfg = config.get("tts") if isinstance(config.get("tts"), dict) else {}
    stt_cfg = config.get("stt") if isinstance(config.get("stt"), dict) else {}
    browser_cfg = config.get("browser") if isinstance(config.get("browser"), dict) else {}
    terminal_cfg = config.get("terminal") if isinstance(config.get("terminal"), dict) else {}

    web_backend = str(web_cfg.get("backend") or "").strip().lower()
    # Per-capability overrides: if set, they determine which backend is active for
    # search/extract independently of web.backend.
    web_search_backend = str(web_cfg.get("search_backend") or "").strip().lower()
    web_extract_backend = str(web_cfg.get("extract_backend") or "").strip().lower()
    tts_provider = str(tts_cfg.get("provider") or "edge").strip().lower()
    # STT default is "local" (faster-whisper) per DEFAULT_CONFIG, which
    # requires `pip install faster-whisper`. For Nous subscribers we'd
    # rather route through the managed OpenAI audio gateway — see
    # apply_nous_managed_defaults below.
    stt_provider = str(stt_cfg.get("provider") or "local").strip().lower()
    browser_provider_explicit = "cloud_provider" in browser_cfg
    browser_provider = normalize_browser_cloud_provider(
        browser_cfg.get("cloud_provider") if browser_provider_explicit else None
    )
    terminal_backend = (
        str(terminal_cfg.get("backend") or "local").strip().lower()
    )
    modal_mode = normalize_modal_mode(
        terminal_cfg.get("modal_mode")
    )

    # Stored selections (strict model): one provider string per category.
    # "nous" (stored value or legacy use_gateway: true) = managed gateway;
    # vendor name = that vendor direct; None = never configured (autodetect).
    image_gen_cfg = config.get("image_gen") if isinstance(config.get("image_gen"), dict) else {}
    video_gen_cfg = config.get("video_gen") if isinstance(config.get("video_gen"), dict) else {}
    web_selected = _selected_provider(web_cfg, "backend")
    tts_selected = _selected_provider(tts_cfg)
    stt_selected = _selected_provider(stt_cfg)
    browser_selected = _selected_provider(browser_cfg, "cloud_provider")
    image_selected = _selected_provider(image_gen_cfg)
    video_selected = _selected_provider(video_gen_cfg)

    # Lockstep with tools.tool_backend_helpers.read_selection: these are
    # merged-config sections, so the legacy DEFAULT_CONFIG-seeded
    # ``stt.provider: local`` COULD appear here without a user pick on old
    # versions. Current DEFAULT_CONFIG no longer seeds it, so a merged
    # ``local`` implies the raw file holds it — a genuine selection.

    # Managed selection flags (replace the legacy use_gateway reads —
    # use_gateway is now interpreted only inside _selected_provider).
    web_use_gateway = web_selected == "nous"
    tts_use_gateway = tts_selected == "nous"
    stt_use_gateway = stt_selected == "nous"
    browser_use_gateway = browser_selected == "nous"
    image_use_gateway = image_selected == "nous"
    video_use_gateway = video_selected == "nous"

    # The "nous" selection is serviced by a concrete vendor implementation —
    # normalize the current-provider labels so downstream vendor checks hold.
    if web_backend == "nous" or web_use_gateway:
        web_backend = "firecrawl"
    if tts_provider == "nous" or tts_use_gateway:
        tts_provider = "openai"
    if stt_provider == "nous" or stt_use_gateway:
        stt_provider = "openai"
    if browser_provider == "nous" or browser_use_gateway:
        browser_provider = "browser-use"

    direct_exa = bool(get_env_value("EXA_API_KEY"))
    direct_firecrawl = bool(get_env_value("FIRECRAWL_API_KEY") or get_env_value("FIRECRAWL_API_URL"))
    direct_parallel = bool(get_env_value("PARALLEL_API_KEY"))
    direct_tavily = bool(get_env_value("TAVILY_API_KEY"))
    # Keyless Tavily is opt-in: selecting it in `hermes tools` / setup writes
    # web.backend (or a per-capability override) without requiring a key.
    tavily_selected = "tavily" in {web_backend, web_search_backend, web_extract_backend}
    direct_searxng = bool(get_env_value("SEARXNG_URL"))
    direct_fal = fal_key_is_configured()
    direct_fal_video = direct_fal  # same FAL_KEY; separate var so use_gateway is independent
    direct_openai_tts = bool(resolve_openai_audio_api_key())
    direct_elevenlabs = bool(get_env_value("ELEVENLABS_API_KEY"))
    direct_camofox = bool(get_env_value("CAMOFOX_URL"))
    direct_browserbase = bool(get_env_value("BROWSERBASE_API_KEY") and get_env_value("BROWSERBASE_PROJECT_ID"))
    direct_browser_use = bool(get_env_value("BROWSER_USE_API_KEY"))
    direct_modal = has_direct_modal_credentials()

    # STT direct providers. OpenAI Whisper reuses the same audio key as
    # OpenAI TTS — resolve_openai_audio_api_key() reads VOICE_TOOLS_OPENAI_KEY
    # and falls back to OPENAI_API_KEY. The local provider's "direct"
    # signal is whether faster-whisper is importable; we lazy-import so
    # this module stays cheap on the happy path.
    direct_openai_stt = bool(resolve_openai_audio_api_key())
    direct_groq_stt = bool(get_env_value("GROQ_API_KEY"))
    direct_mistral_stt = bool(get_env_value("MISTRAL_API_KEY"))
    try:
        from tools.transcription_tools import _HAS_FASTER_WHISPER
        local_stt_available = bool(_HAS_FASTER_WHISPER) or bool(
            get_env_value("HERMES_LOCAL_STT_COMMAND")
        )
    except Exception:
        local_stt_available = bool(get_env_value("HERMES_LOCAL_STT_COMMAND"))

    # When use_gateway is set, suppress direct credentials for managed detection
    if web_use_gateway:
        direct_firecrawl = False
        direct_exa = False
        direct_parallel = False
        direct_tavily = False
        tavily_selected = False
    if image_use_gateway:
        direct_fal = False
    if video_use_gateway:
        direct_fal_video = False
    if tts_use_gateway:
        direct_openai_tts = False
        direct_elevenlabs = False
    if stt_use_gateway:
        direct_openai_stt = False
        direct_groq_stt = False
        direct_mistral_stt = False
        local_stt_available = False
    if browser_use_gateway:
        direct_browser_use = False
        direct_browserbase = False

    managed_web_available = (
        managed_tools_flag
        and nous_auth_present
        and is_managed_tool_gateway_ready("firecrawl")
        and _entitled_for("firecrawl")
    )
    managed_image_available = (
        managed_tools_flag
        and nous_auth_present
        and is_managed_tool_gateway_ready("fal-queue")
        and _entitled_for("fal")
    )
    # Video gen rides the same fal-queue gateway as image gen, but the free tool
    # pool funds image and NOT video — so gate it on its own coverage category
    # rather than aliasing it to image. (Paid users are entitled to both.)
    managed_video_available = (
        managed_tools_flag
        and nous_auth_present
        and is_managed_tool_gateway_ready("fal-queue")
        and _entitled_for("fal-video")
    )
    managed_tts_available = (
        managed_tools_flag
        and nous_auth_present
        and is_managed_tool_gateway_ready("openai-audio")
        and _entitled_for("openai-audio")
    )
    # STT and TTS share the same managed gateway endpoint ("openai-audio")
    # because the OpenAI audio API covers both /audio/speech (TTS) and
    # /audio/transcriptions (STT). One probe (and one entitlement), used by both.
    managed_stt_available = managed_tts_available
    managed_browser_available = (
        managed_tools_flag
        and nous_auth_present
        and is_managed_tool_gateway_ready("browser-use")
        and _entitled_for("browser-use")
    )
    managed_modal_available = (
        managed_tools_flag
        and nous_auth_present
        and is_managed_tool_gateway_ready("modal")
        and _entitled_for("modal")
    )
    modal_state = resolve_modal_backend_state(
        modal_mode,
        has_direct=direct_modal,
        managed_ready=managed_modal_available,
        managed_enabled=managed_tools_flag,
    )

    # Strict selection: a stored VENDOR selection pins the category to direct
    # credentials — managed availability must not light the feature up (the
    # runtime will error, not reroute), and camofox/local selections must not
    # be pre-empted by env credentials for other providers.
    if web_selected is not None and not web_use_gateway:
        managed_web_available = False
    if image_selected is not None and not image_use_gateway:
        managed_image_available = False
    if video_selected is not None and not video_use_gateway:
        managed_video_available = False
    if tts_selected is not None and not tts_use_gateway:
        managed_tts_available = False
    if stt_selected is not None and not stt_use_gateway:
        managed_stt_available = False
    if browser_selected is not None and not browser_use_gateway:
        managed_browser_available = False
    if browser_selected is not None and browser_selected != "camofox":
        # CAMOFOX_URL is the server address, not a selection: an explicit
        # different browser choice wins over the env var.
        direct_camofox = False


    tavily_ready = direct_tavily or tavily_selected
    web_managed = web_backend == "firecrawl" and managed_web_available and not direct_firecrawl
    web_active = bool(
        web_tool_enabled
        and (
            web_managed
            or (web_backend == "exa" and direct_exa)
            or (web_backend == "firecrawl" and direct_firecrawl)
            or (web_backend == "parallel" and direct_parallel)
            or (web_backend == "tavily" and tavily_ready)
            or (web_backend == "searxng" and direct_searxng)
            # Per-capability overrides: search_backend or extract_backend may be set
            # without web.backend (using the new split config from #20061)
            or (web_search_backend == "searxng" and direct_searxng)
            or (web_search_backend == "exa" and direct_exa)
            or (web_search_backend == "firecrawl" and direct_firecrawl)
            or (web_search_backend == "parallel" and direct_parallel)
            or (web_search_backend == "tavily" and tavily_ready)
            or (web_extract_backend == "tavily" and tavily_ready)
        )
    )
    web_available = bool(
        managed_web_available
        or direct_exa
        or direct_firecrawl
        or direct_parallel
        or tavily_ready
        or direct_searxng
    )

    image_managed = image_tool_enabled and managed_image_available and not direct_fal
    image_active = bool(image_tool_enabled and (image_managed or direct_fal))
    image_available = bool(managed_image_available or direct_fal)

    video_managed = video_tool_enabled and managed_video_available and not direct_fal_video
    video_active = bool(video_tool_enabled and (video_managed or direct_fal_video))
    video_available = bool(managed_video_available or direct_fal_video)

    tts_current_provider = tts_provider or "edge"
    tts_managed = (
        tts_tool_enabled
        and tts_current_provider == "openai"
        and managed_tts_available
        and not direct_openai_tts
    )
    tts_available = bool(
        tts_current_provider in {"edge", "neutts"}
        or (tts_current_provider == "openai" and (managed_tts_available or direct_openai_tts))
        or (tts_current_provider == "elevenlabs" and direct_elevenlabs)
        or (tts_current_provider == "mistral" and bool(get_env_value("MISTRAL_API_KEY")))
    )
    tts_active = bool(tts_tool_enabled and tts_available)

    # STT availability per provider. Unlike TTS, STT isn't a model-callable
    # tool — the gateway voice middleware calls it on every inbound voice
    # message — so toolset_enabled is N/A and we treat stt as always
    # "enabled" if a usable provider is configured.
    stt_current_provider = stt_provider or "local"
    stt_managed = (
        stt_current_provider == "openai"
        and managed_stt_available
        and not direct_openai_stt
    )
    stt_available = bool(
        (stt_current_provider == "local" and local_stt_available)
        or (stt_current_provider == "openai" and (managed_stt_available or direct_openai_stt))
        or (stt_current_provider == "groq" and direct_groq_stt)
        or (stt_current_provider == "mistral" and direct_mistral_stt)
    )
    stt_active = stt_available

    browser_local_available = _has_agent_browser()
    browser_local_runnable = _local_browser_runnable()
    (
        browser_current_provider,
        browser_available,
        browser_active,
        browser_managed,
    ) = _resolve_browser_feature_state(
        browser_tool_enabled=browser_tool_enabled,
        browser_provider=browser_provider,
        browser_provider_explicit=browser_provider_explicit,
        browser_local_available=browser_local_available,
        browser_local_runnable=browser_local_runnable,
        direct_camofox=direct_camofox,
        direct_browserbase=direct_browserbase,
        direct_browser_use=direct_browser_use,
        direct_firecrawl=direct_firecrawl,
        managed_browser_available=managed_browser_available,
    )

    if terminal_backend != "modal":
        modal_managed = False
        modal_available = True
        modal_active = bool(modal_tool_enabled)
        modal_direct_override = False
    elif modal_state["selected_backend"] == "managed":
        modal_managed = bool(modal_tool_enabled)
        modal_available = True
        modal_active = bool(modal_tool_enabled)
        modal_direct_override = False
    elif modal_state["selected_backend"] == "direct":
        modal_managed = False
        modal_available = True
        modal_active = bool(modal_tool_enabled)
        modal_direct_override = bool(modal_tool_enabled)
    elif modal_mode == "managed":
        modal_managed = False
        modal_available = bool(managed_modal_available)
        modal_active = False
        modal_direct_override = False
    elif modal_mode == "direct":
        modal_managed = False
        modal_available = bool(direct_modal)
        modal_active = False
        modal_direct_override = False
    else:
        modal_managed = False
        modal_available = bool(managed_modal_available or direct_modal)
        modal_active = False
        modal_direct_override = False

    # Explicit-configured mirrors the stored selections computed above so
    # status/picker markers stay in lockstep with runtime dispatch.
    tts_explicit_configured = tts_selected is not None and tts_selected != "edge"
    stt_explicit_configured = stt_selected is not None

    features = {
        "web": NousFeatureState(
            key="web",
            label="Web tools",
            included_by_default=True,
            available=web_available,
            active=web_active,
            managed_by_nous=web_managed,
            direct_override=web_active and not web_managed,
            toolset_enabled=web_tool_enabled,
            current_provider=web_backend or web_search_backend or web_extract_backend or "",
            explicit_configured=bool(web_backend or web_search_backend or web_extract_backend),
        ),
        "image_gen": NousFeatureState(
            key="image_gen",
            label="Image generation",
            included_by_default=True,
            available=image_available,
            active=image_active,
            managed_by_nous=image_managed,
            direct_override=image_active and not image_managed,
            toolset_enabled=image_tool_enabled,
            current_provider="FAL" if (image_selected not in (None, "nous") or (image_selected is None and direct_fal)) else ("Nous Subscription" if (image_managed or image_use_gateway) else ""),
            explicit_configured=image_selected is not None or direct_fal,
        ),
        "video_gen": NousFeatureState(
            key="video_gen",
            label="Video generation",
            included_by_default=False,
            available=video_available,
            active=video_active,
            managed_by_nous=video_managed,
            direct_override=video_active and not video_managed,
            toolset_enabled=video_tool_enabled,
            current_provider="FAL" if (video_selected not in (None, "nous") or (video_selected is None and direct_fal_video)) else ("Nous Subscription" if (video_managed or video_use_gateway) else ""),
            explicit_configured=video_selected is not None or direct_fal_video,
        ),
        "tts": NousFeatureState(
            key="tts",
            label="OpenAI TTS",
            included_by_default=True,
            available=tts_available,
            active=tts_active,
            managed_by_nous=tts_managed,
            direct_override=tts_active and not tts_managed,
            toolset_enabled=tts_tool_enabled,
            current_provider=_tts_label(tts_current_provider),
            explicit_configured=tts_explicit_configured,
        ),
        "stt": NousFeatureState(
            key="stt",
            label="Speech-to-text",
            included_by_default=True,
            available=stt_available,
            active=stt_active,
            managed_by_nous=stt_managed,
            direct_override=stt_active and not stt_managed,
            # STT isn't toolset-gated (gateway middleware calls it
            # unconditionally on inbound voice), so report True so the
            # status display doesn't flag it as "tool disabled".
            toolset_enabled=True,
            current_provider=_stt_label(stt_current_provider),
            explicit_configured=stt_explicit_configured,
        ),
        "browser": NousFeatureState(
            key="browser",
            label="Browser automation",
            included_by_default=True,
            available=browser_available,
            active=browser_active,
            managed_by_nous=browser_managed,
            direct_override=browser_active and not browser_managed,
            toolset_enabled=browser_tool_enabled,
            current_provider=_browser_label(browser_current_provider),
            explicit_configured=browser_provider_explicit,
        ),
        "modal": NousFeatureState(
            key="modal",
            label="Modal execution",
            included_by_default=False,
            available=modal_available,
            active=modal_active,
            managed_by_nous=modal_managed,
            direct_override=terminal_backend == "modal" and modal_direct_override,
            toolset_enabled=modal_tool_enabled,
            current_provider="Modal" if terminal_backend == "modal" else terminal_backend or "local",
            explicit_configured=terminal_backend == "modal",
        ),
    }
    use_gateway = {key: value == "nous" for key, value in selected.items()}
    # Managed availability per feature. A stored VENDOR selection pins the category to direct
    # credentials — managed availability must not light it up (the runtime errors, not reroutes).
    # Features without a config selection field (modal) have no pin and read as unselected.
    managed = {
        key: (
            managed_tools_flag and is_managed_tool_gateway_ready(spec.gateway)
            and account_info.tool_gateway_entitled_for(spec.coverage)
            and (selected.get(key) is None or use_gateway.get(key, False))
        )
        for key, spec in _FEATURES.items()
    }
    direct_firecrawl = _any_env("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL") and not use_gateway["web"]
    fal_configured = fal_key_is_configured()
    tts, stt = _audio_features(_section(config, "tts"), _section(config, "stt"), enabled["tts"], managed, selected, use_gateway)

    def _fal(key: str) -> NousFeatureState:
        return _fal_feature(key, enabled[key], fal_configured and not use_gateway[key], managed[key], selected[key])

    features = {  # insertion order == _FEATURE_ORDER
        "web": _web_feature(_section(config, "web"), enabled["web"], managed["web"], use_gateway["web"], direct_firecrawl),
        "image_gen": _fal("image_gen"),
        "video_gen": _fal("video_gen"),
        "tts": tts,
        "stt": stt,
        "browser": _browser_feature(
            _section(config, "browser"), enabled["browser"], managed["browser"], selected["browser"], use_gateway["browser"], direct_firecrawl,
        ),
        "modal": _modal_feature(_section(config, "terminal"), enabled["terminal"], managed["modal"], managed_tools_flag),
    }
    return NousSubscriptionFeatures(
        subscribed=provider_is_nous or nous_auth_present, nous_auth_present=nous_auth_present,
        provider_is_nous=provider_is_nous, features=features, account_info=account_info,
    )


def _has_managed_default_direct(key: str) -> bool:
    return bool(key in ("tts", "stt") and resolve_openai_audio_api_key()) or _any_env(*_FEATURES[key].default_direct_env)


def apply_nous_managed_defaults(config: Dict[str, object], *, enabled_toolsets: Optional[Iterable[str]] = None, force_fresh: bool = False) -> set[str]:
    features = get_nous_subscription_features(config, force_fresh=force_fresh)
    account_info = features.account_info
    if not (account_info and account_info.logged_in and account_info.tool_gateway_entitled and features.provider_is_nous):
        return set()

    selected_toolsets = set(enabled_toolsets or ())
    changed: set[str] = set()

    web_cfg = config.get("web")
    if not isinstance(web_cfg, dict):
        web_cfg = {}
        config["web"] = web_cfg

    tts_cfg = config.get("tts")
    if not isinstance(tts_cfg, dict):
        tts_cfg = {}
        config["tts"] = tts_cfg

    stt_cfg = config.get("stt")
    if not isinstance(stt_cfg, dict):
        stt_cfg = {}
        config["stt"] = stt_cfg

    browser_cfg = config.get("browser")
    if not isinstance(browser_cfg, dict):
        browser_cfg = {}
        config["browser"] = browser_cfg

    if "web" in selected_toolsets and not features.web.explicit_configured and not (
        get_env_value("PARALLEL_API_KEY")
        or get_env_value("TAVILY_API_KEY")
        or get_env_value("FIRECRAWL_API_KEY")
        or get_env_value("FIRECRAWL_API_URL")
    ):
        web_cfg["backend"] = "nous"
        web_cfg.pop("use_gateway", None)
        changed.add("web")

    if "tts" in selected_toolsets and not features.tts.explicit_configured and not (
        resolve_openai_audio_api_key()
        or get_env_value("ELEVENLABS_API_KEY")
    ):
        tts_cfg["provider"] = "nous"
        tts_cfg.pop("use_gateway", None)
        changed.add("tts")

    # STT: same pattern as TTS. The DEFAULT_CONFIG seed is "local"
    # (requires `pip install faster-whisper`); for Nous subscribers we
    # flip it to the managed selection so the managed audio gateway handles
    # transcription via the same auth as TTS. Skipped when the user has
    # explicitly configured STT, has direct credentials for a non-managed
    # provider, has a working local backend (faster-whisper installed or a
    # custom local command — strong intent signal that "local" was a choice,
    # not just the DEFAULT_CONFIG seed), or isn't entitled to the managed
    # "openai-audio" category (flipping would point at a gateway that
    # refuses them, silently breaking voice transcription).
    if (
        not features.stt.explicit_configured
        and not _local_stt_backend_available()
        and not (
            resolve_openai_audio_api_key()
            or get_env_value("GROQ_API_KEY")
            or get_env_value("MISTRAL_API_KEY")
        )
        and features.account_info is not None
        and features.account_info.tool_gateway_entitled_for("openai-audio")
    ):
        stt_cfg["provider"] = "nous"
        stt_cfg.pop("use_gateway", None)
        changed.add("stt")

    if "browser" in selected_toolsets and not features.browser.explicit_configured and not (
        get_env_value("BROWSER_USE_API_KEY")
        or get_env_value("BROWSERBASE_API_KEY")
    ):
        browser_cfg["cloud_provider"] = "nous"
        browser_cfg.pop("use_gateway", None)
        changed.add("browser")

    if "image_gen" in selected_toolsets and not fal_key_is_configured():
        image_cfg = config.get("image_gen")
        if not isinstance(image_cfg, dict):
            image_cfg = {}
            config["image_gen"] = image_cfg
        image_cfg["provider"] = "nous"
        image_cfg.pop("use_gateway", None)
        changed.add("image_gen")

    # Video gen is not funded by the free tool pool, so only wire managed video
    # defaults for users entitled to it (paid). Pool-only users keep video off.
    if (
        "video_gen" in selected_toolsets
        and not fal_key_is_configured()
        and features.account_info.tool_gateway_entitled_for("fal-video")
    ):
        video_cfg = config.get("video_gen")
        if not isinstance(video_cfg, dict):
            video_cfg = {}
            config["video_gen"] = video_cfg
        video_cfg["provider"] = "nous"
        video_cfg.pop("use_gateway", None)
        changed.add("video_gen")

    return changed


# Tool Gateway offer — per-tool checklist after model selection


def _get_gateway_direct_credentials() -> Dict[str, bool]:
    """tool_key -> has_direct_credentials. Env-configured keyless local backends (SearXNG, CAMOFOX_URL)
    count as configured so they are never classified "unconfigured" and pre-checked; Whisper shares
    the audio key with TTS."""
    fal_direct = fal_key_is_configured()
    audio_direct = bool(resolve_openai_audio_api_key())
    return {
        "web": bool(
            get_env_value("FIRECRAWL_API_KEY")
            or get_env_value("FIRECRAWL_API_URL")
            or get_env_value("PARALLEL_API_KEY")
            or get_env_value("TAVILY_API_KEY")
            or get_env_value("EXA_API_KEY")
            # Env-configured keyless local backend: a reachable self-hosted
            # SearXNG is a working web setup even with no stored selection
            # (the autodetect cascade in tools/web_tools.py picks it up), so
            # it must not be classified "unconfigured" and pre-checked (#92647).
            or get_env_value("SEARXNG_URL")
        ),
        "image_gen": fal_direct,
        "video_gen": fal_direct,
        "tts": bool(
            resolve_openai_audio_api_key()
            or get_env_value("ELEVENLABS_API_KEY")
        ),
        # STT direct credentials. OpenAI Whisper shares the audio key
        # with TTS via resolve_openai_audio_api_key() — counting it here
        # too is intentional: if the user has an OpenAI audio key they
        # don't need the gateway for either.
        "stt": bool(
            resolve_openai_audio_api_key()
            or get_env_value("GROQ_API_KEY")
            or get_env_value("MISTRAL_API_KEY")
        ),
        "browser": bool(
            get_env_value("BROWSER_USE_API_KEY")
            or (get_env_value("BROWSERBASE_API_KEY") and get_env_value("BROWSERBASE_PROJECT_ID"))
            # Env-configured keyless local backend: CAMOFOX_URL activates the
            # Camofox browser via never-configured autodetect (see
            # _resolve_browser_state above), so it counts as configured even
            # with no stored cloud_provider selection (#92647).
            or get_env_value("CAMOFOX_URL")
        ),
    }


_GATEWAY_DIRECT_LABELS = {
    "web": "Firecrawl/Exa/Parallel/Tavily key or SearXNG",
    "image_gen": "FAL key",
    "video_gen": "FAL key",
    "tts": "OpenAI/ElevenLabs key",
    "stt": "OpenAI/Groq/Mistral key",
    "browser": "Browser Use/Browserbase key or Camofox",
}

_ALL_GATEWAY_KEYS = ("web", "image_gen", "video_gen", "tts", "stt", "browser")

# Config section + selection field for each gateway key, matching the
# field names ``apply_gateway_defaults`` writes and
# ``get_nous_subscription_features`` reads (web uses "backend", browser
# uses "cloud_provider", everything else uses "provider").
_GATEWAY_SECTION_FIELDS = {
    "web": ("web", "backend"),
    "image_gen": ("image_gen", "provider"),
    "video_gen": ("video_gen", "provider"),
    "tts": ("tts", "provider"),
    "stt": ("stt", "provider"),
    "browser": ("browser", "cloud_provider"),
}


def get_gateway_eligible_tools(
    config: Optional[Dict[str, object]] = None,
    *,
    force_fresh: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (unconfigured, has_direct, explicit_configured, already_managed)
    tool key lists.

    - unconfigured: tools with no direct credentials and no explicit
      non-nous selection (easy switch, safe to pre-check)
    - has_direct: tools where the user has their own API keys
    - explicit_configured: tools with an explicit non-nous selection stored
      (e.g. ``web.backend: searxng``), including keyless backends that would
      otherwise look unconfigured
    - already_managed: tools already routed through the gateway

    All lists are empty when the user is not a paid Nous subscriber or
    is not using Nous as their provider.
    """
    # Fetch entitlement once: it gates the offer (paid access OR a live free tool
    # pool) AND tells us which categories are covered (the pool funds image but
    # not video, etc.). Fails closed on any error.
    try:
        account_info = get_nous_portal_account_info(force_fresh=force_fresh)
    except Exception:
        return [], [], [], []
    if not (account_info and account_info.logged_in and account_info.tool_gateway_entitled):
        return [], [], [], []

    if config is None:
        config = load_config() or {}

    # Quick provider check without the heavy get_nous_subscription_features call
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict) or str(model_cfg.get("provider") or "").strip().lower() != "nous":
        return [], [], [], []

    direct = _get_gateway_direct_credentials()

    # Check which tools the user has explicitly opted into the gateway for.
    # This is distinct from managed_by_nous which fires implicitly when
    # no direct keys exist — we only skip the prompt for tools where
    # use_gateway was explicitly set.
    opted_in = {
        "web": _uses_gateway(config.get("web")),
        "image_gen": _uses_gateway(config.get("image_gen")),
        "video_gen": _uses_gateway(config.get("video_gen")),
        "tts": _uses_gateway(config.get("tts")),
        "stt": _uses_gateway(config.get("stt")),
        "browser": _uses_gateway(config.get("browser")),
    }

    unconfigured: list[str] = []
    has_direct: list[str] = []
    explicit_configured: list[str] = []
    already_managed: list[str] = []
    for key in _ALL_GATEWAY_KEYS:
        # Only offer tools the entitlement covers (free pool: image but not video).
        if not account_info.tool_gateway_entitled_for(_FEATURES[key].coverage):
            continue
        section_key, field = _GATEWAY_SECTION_FIELDS[key]
        selected = _selected_provider(config.get(section_key), field)
        if opted_in.get(key):
            already_managed.append(key)
        elif selected is not None and selected != "nous":
            # An explicit non-nous selection (e.g. a keyless local backend
            # like SearXNG or Camofox) is configured on purpose, even
            # though it has no direct credentials to detect.
            explicit_configured.append(key)
        elif direct.get(key):
            has_direct.append(key)
        else:
            unconfigured.append(key)
    return unconfigured, has_direct, explicit_configured, already_managed


def apply_gateway_defaults(
    config: Dict[str, object],
    tool_keys: list[str],
) -> set[str]:
    """Apply Tool Gateway config for the given tool keys.

    Sets ``use_gateway: true`` in each tool's config section so the
    runtime prefers the gateway even when direct API keys are present.

    Returns the set of tools that were actually changed.
    """
    changed: set[str] = set()

    web_cfg = config.get("web")
    if not isinstance(web_cfg, dict):
        web_cfg = {}
        config["web"] = web_cfg

    tts_cfg = config.get("tts")
    if not isinstance(tts_cfg, dict):
        tts_cfg = {}
        config["tts"] = tts_cfg

    stt_cfg = config.get("stt")
    if not isinstance(stt_cfg, dict):
        stt_cfg = {}
        config["stt"] = stt_cfg

    browser_cfg = config.get("browser")
    if not isinstance(browser_cfg, dict):
        browser_cfg = {}
        config["browser"] = browser_cfg

    if "web" in tool_keys:
        web_cfg["backend"] = "nous"
        web_cfg.pop("use_gateway", None)
        changed.add("web")

    if "tts" in tool_keys:
        tts_cfg["provider"] = "nous"
        tts_cfg.pop("use_gateway", None)
        changed.add("tts")

    if "stt" in tool_keys:
        stt_cfg["provider"] = "nous"
        stt_cfg.pop("use_gateway", None)
        changed.add("stt")

    if "browser" in tool_keys:
        browser_cfg["cloud_provider"] = "nous"
        browser_cfg.pop("use_gateway", None)
        changed.add("browser")

    if "image_gen" in tool_keys:
        image_cfg = config.get("image_gen")
        if not isinstance(image_cfg, dict):
            image_cfg = {}
            config["image_gen"] = image_cfg
        image_cfg["provider"] = "nous"
        image_cfg.pop("use_gateway", None)
        changed.add("image_gen")

    if "video_gen" in tool_keys:
        video_cfg = config.get("video_gen")
        if not isinstance(video_cfg, dict):
            video_cfg = {}
            config["video_gen"] = video_cfg
        video_cfg["provider"] = "nous"
        video_cfg.pop("use_gateway", None)
        changed.add("video_gen")

    return changed


def prompt_enable_tool_gateway(
    config: Dict[str, object],
    *,
    force_fresh: bool = True,
) -> set[str]:
    """If eligible tools exist, prompt the user (per tool) to enable the Tool
    Gateway.

    "Pool enabled" is the trigger: a user with a live free tool pool (or paid
    access) is shown a per-tool checklist of the covered managed backends and
    picks which to route through the gateway. The free pool funds web/image/
    tts/browser but not video, so the checklist only lists covered tools (the
    coverage filter lives in get_gateway_eligible_tools).

    Returns the set of tools that were enabled, or empty set if the user
    declined or no tools were eligible.
    """
    # explicit_configured tools (e.g. an explicit `web.backend: searxng`) are
    # configured on purpose and are never offered here — same treatment as
    # already_managed, just for a non-nous vendor.
    unconfigured, has_direct, _explicit_configured, already_managed = get_gateway_eligible_tools(
        config,
        force_fresh=force_fresh,
    )
    if not unconfigured and not has_direct:
        return set()
    try:
        from hermes_cli.setup import prompt_checklist
    except Exception:
        return set()
    # Frame the offer by entitlement: a $0 free-tool-pool user is not on a paid plan.
    account_info = _account_info_or_none(force_fresh=False)
    pool_only = bool(
        account_info and account_info.paid_service_access is not True and account_info.tool_access is not None and account_info.tool_access.enabled
    )
    source_label = "free tool pool" if pool_only else "Nous subscription"

    # Per-tool checklist: unconfigured tools first (pre-checked for new users),
    # then tools where the user already has their own key (left unchecked so we
    # don't override their own setup unless they ask).
    #
    # Decline persistence (#92647): tools the user has previously seen offered
    # and left unchecked are recorded in ``tool_gateway_declined_tools`` and
    # are never pre-checked again — the offer downgrades to opt-in-only.
    # Acceptance used to be sticky while refusal was not, so the identical
    # pre-checked checklist re-fired on every Nous model swap.
    declined_raw = config.get("tool_gateway_declined_tools")
    declined: set[str] = (
        {str(k) for k in declined_raw} if isinstance(declined_raw, list) else set()
    )

    offer_keys: list[str] = list(unconfigured) + list(has_direct)
    labels = [_GATEWAY_TOOL_LABELS[k] for k in unconfigured] + [
        f"{_GATEWAY_TOOL_LABELS[k]} — keep using your {_FEATURES[k].direct_label}" for k in has_direct
    ]
    pre_selected = [
        i for i, k in enumerate(unconfigured) if k not in declined
    ]

    if pool_only:
        title = "Your free Nous tool pool — pick the tools to enable:"
    else:
        title = (
            "Your Nous subscription includes the Tool Gateway — "
            "pick the tools to enable:"
        )

    try:
        chosen_idx = prompt_checklist(title, labels, pre_selected)
    except (KeyboardInterrupt, EOFError, OSError, SystemExit):
        return set()
    chosen_keys = [offer_keys[i] for i in chosen_idx if 0 <= i < len(offer_keys)]

    # Persist per-tool declines: every unconfigured tool that was offered and
    # NOT chosen was actively left (or unchecked) by the user — remember that
    # so the next Nous model swap doesn't pre-check it again. Cancel paths
    # (Ctrl-C/ESC above) return before this and record nothing. Choosing a
    # previously-declined tool clears its decline.
    newly_declined = [k for k in unconfigured if k not in chosen_keys and k not in declined]
    undeclined = declined & set(chosen_keys)
    if newly_declined or undeclined:
        config["tool_gateway_declined_tools"] = sorted(
            (declined | set(newly_declined)) - set(chosen_keys)
        )

    if not chosen_keys:
        if newly_declined:
            from hermes_cli.config import save_config

            save_config(config)
        return set()

    changed = apply_gateway_defaults(config, chosen_keys)
    if changed or newly_declined:
        from hermes_cli.config import save_config

        save_config(config)
        for key in sorted(changed):
            print(f"  ✓ {_GATEWAY_TOOL_LABELS.get(key, key)}: enabled via {source_label}")
    return changed


# Inline Nous Portal login for the Tool Gateway picker (`hermes tools`)


def ensure_nous_portal_access(*, capability: str = "the Nous Tool Gateway", coverage_category: Optional[str] = None) -> bool:
    """Make sure the user is entitled to the Nous Tool Gateway, logging in if needed.

    Only performs the device-code OAuth (when not logged in) and refreshes entitlement — no model
    or provider switch. Entitlement is paid access OR a live free pool; with ``coverage_category``
    the pool must cover that category (a pool user selecting ``"fal-video"`` is denied).
    """

    def _entitled(account) -> bool:
        if account is None:
            return False
        return account.tool_gateway_entitled_for(coverage_category) if coverage_category is not None else account.tool_gateway_entitled

    info = _account_info_or_none(force_fresh=True)
    if not _entitled(info) and (info is None or not info.logged_in):
        if not _run_nous_portal_login_only(capability=capability):
            return False
        info = _account_info_or_none(force_fresh=True)
    if _entitled(info):
        return True
    # Logged in but not entitled for this capability — neutral billing guidance, do not enable.
    message = format_nous_portal_entitlement_message(info, capability=capability, coverage_category=coverage_category)
    for line in (message or "").splitlines():
        print(f"  {line}")
    return False


def _confirm(prompt: str) -> Optional[bool]:
    """Y/n prompt: True on yes/blank, False on anything else, ``None`` on EOF/Ctrl-C."""
    try:
        return input(prompt).strip().lower() in {"", "y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return None


def _run_nous_portal_login_only(*, capability: str) -> bool:
    """Run the Nous Portal device-code OAuth and persist credentials only (no model selection, no
    provider switch, no Tool Gateway bulk prompt). ``False`` if the user declined or the flow failed."""
    try:
        import hermes_cli.auth as auth
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  Could not start Nous Portal login: {exc}")
        return False
    print()
    print(f"  {capability} requires a Nous Portal login.")
    proceed = _confirm("  Log in to Nous Portal now? [Y/n]: ")
    if proceed is None:
        print()
        return False
    if not proceed:
        print("  Skipped Nous Portal login.")
        return False
    try:
        # Snapshot active_provider so a tool-config login never silently switches inference to Nous.
        with auth._auth_store_lock():
            prior_active_provider = auth._load_auth_store().get("active_provider")
        auth_state = None
        # Interrupting the import question defaults to importing.
        if auth._read_shared_nous_state() and _confirm("  Found existing Nous OAuth credentials. Import them? [Y/n]: ") is not False:
            auth_state = auth._try_import_shared_nous_state(timeout_seconds=15.0)
        if auth_state is None:
            auth_state = auth._nous_device_code_login()
        with auth._auth_store_lock():
            auth_store = auth._load_auth_store()
            auth._save_provider_state(auth_store, "nous", auth_state)
            if prior_active_provider:
                auth_store["active_provider"] = prior_active_provider
            else:
                auth_store.pop("active_provider", None)
            auth._save_auth_store(auth_store)
        auth._write_shared_nous_state(auth_state)
        auth._sync_nous_pool_from_auth_store()
        print("  Nous Portal login successful.")
        return True
    except KeyboardInterrupt:
        print("\n  Login cancelled.")
        return False
    except SystemExit:
        # _nous_device_code_login raises SystemExit on subscription_required (guidance already printed).
        return False
    except Exception as exc:
        print(f"  Nous Portal login failed: {exc}")
        return False


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'managed_nous_tools_enabled': ('tools.tool_backend_helpers', 'managed_nous_tools_enabled'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
