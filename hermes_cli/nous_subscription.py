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


def _any_env(*names: str) -> bool:
    """True when any of the named env vars (via get_env_value) is set."""
    return any(get_env_value(name) for name in names)


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
    managed_tools_flag = nous_auth_present and account_info.tool_gateway_entitled
    enabled = {key: _toolset_enabled(config, key) for key in ("web", "image_gen", "video_gen", "tts", "browser", "terminal")}
    # Stored selections (strict model): "nous" = managed gateway; vendor name = that vendor direct;
    # None = never configured (autodetect). Lockstep with tool_backend_helpers.read_selection: a
    # merged ``stt.provider: local`` here means the raw file holds it — a genuine selection.
    selected = {
        key: _selected_provider(_section(config, section_key), field)
        for key, (section_key, field) in _GATEWAY_SECTION_FIELDS.items()
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
    for key in _DEFAULT_SECTIONS:
        _ensure_section(config, key)
    for key in _DEFAULT_SECTIONS:
        if features.features[key].explicit_configured or _has_managed_default_direct(key):
            continue
        if key == "stt":
            # STT is not toolset-gated. Skip when the user has a working local backend (strong signal
            # "local" was a choice, not the DEFAULT_CONFIG seed) or isn't entitled to the managed
            # "openai-audio" category (flipping would silently break transcription).
            if _local_stt_backend_available() or not account_info.tool_gateway_entitled_for("openai-audio"):
                continue
        elif key not in selected_toolsets:
            continue
        _select_nous(config, key)
        changed.add(key)
    # Video gen is not funded by the free tool pool: only wire managed video for entitled (paid) users.
    for key, category in (("image_gen", None), ("video_gen", "fal-video")):
        if key in selected_toolsets and not fal_key_is_configured() and (category is None or account_info.tool_gateway_entitled_for(category)):
            _select_nous(config, key)
            changed.add(key)
    return changed


# Tool Gateway offer — per-tool checklist after model selection


def _get_gateway_direct_credentials() -> Dict[str, bool]:
    """tool_key -> has_direct_credentials. Env-configured keyless local backends (SearXNG, CAMOFOX_URL)
    count as configured so they are never classified "unconfigured" and pre-checked; Whisper shares
    the audio key with TTS."""
    fal_direct = fal_key_is_configured()
    audio_direct = bool(resolve_openai_audio_api_key())
    return {
        "web": _any_env("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY", "TAVILY_API_KEY", "PERPLEXITY_API_KEY", "EXA_API_KEY", "SEARXNG_URL"),
        # Env-configured keyless local backend: a reachable self-hosted SearXNG is a working web setup even
        # with no stored selection (the autodetect cascade in tools/web_tools.py picks it up), so it must
        # not be classified "unconfigured" and pre-checked (#92647).
        "image_gen": fal_direct,
        "video_gen": fal_direct,
        "tts": audio_direct or _any_env("ELEVENLABS_API_KEY"),
        "stt": audio_direct or _any_env("GROQ_API_KEY", "MISTRAL_API_KEY"),
        "browser": (
            _any_env("BROWSER_USE_API_KEY", "CAMOFOX_URL")
            or bool(get_env_value("BROWSERBASE_API_KEY") and get_env_value("BROWSERBASE_PROJECT_ID"))
        ),
    }


def get_gateway_eligible_tools(config: Optional[Dict[str, object]] = None, *, force_fresh: bool = False) -> tuple[list[str], list[str], list[str], list[str]]:
    """(unconfigured, has_direct, explicit_configured, already_managed) tool key lists: no credentials
    and no explicit non-nous selection (safe to pre-check) / own API keys / explicit non-nous selection
    stored (e.g. keyless SearXNG) even with nothing to detect / ``use_gateway`` explicitly set."""
    # Entitlement gates the offer (paid OR live free pool) and says which categories are covered.
    account_info = _account_info_or_none(force_fresh=force_fresh)
    if not (account_info and account_info.logged_in and account_info.tool_gateway_entitled):
        return [], [], [], []
    if config is None:
        config = load_config() or {}
    if not _provider_is_nous(config):
        return [], [], [], []
    direct = _get_gateway_direct_credentials()
    unconfigured, has_direct, explicit_configured, already_managed = [], [], [], []
    for key in _ALL_GATEWAY_KEYS:
        # Only offer tools the entitlement covers (free pool: image but not video).
        if not account_info.tool_gateway_entitled_for(_FEATURES[key].coverage):
            continue
        section_key, field = _GATEWAY_SECTION_FIELDS[key]
        selected = _selected_provider(config.get(section_key), field)
        if _uses_gateway(config.get(key)):
            already_managed.append(key)
        elif selected is not None and selected != "nous":
            explicit_configured.append(key)
        elif direct.get(key):
            has_direct.append(key)
        else:
            unconfigured.append(key)
    return unconfigured, has_direct, explicit_configured, already_managed


def apply_gateway_defaults(config: Dict[str, object], tool_keys: list[str]) -> set[str]:
    """Store the managed selection for ``tool_keys``; returns the set of tools actually changed."""
    for key in _DEFAULT_SECTIONS:
        _ensure_section(config, key)
    changed = [key for key in _ALL_GATEWAY_KEYS if key in tool_keys]  # table order: config key order
    for key in changed:
        _select_nous(config, key)
    return set(changed)


def prompt_enable_tool_gateway(config: Dict[str, object], *, force_fresh: bool = True) -> set[str]:
    """If eligible tools exist, show a per-tool checklist to route them through the Tool Gateway.
    Triggered by a live free pool or paid access; explicit_configured tools (e.g. ``web.backend:
    searxng``) are configured on purpose and never offered, like already_managed."""
    unconfigured, has_direct, _explicit, _managed = get_gateway_eligible_tools(config, force_fresh=force_fresh)
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

    # Unconfigured tools first (pre-checked for new users), then tools with the user's own key
    # (unchecked). Tools previously offered and left unchecked are recorded in
    # ``tool_gateway_declined_tools`` and never pre-checked again (no re-fire on every model swap).
    # Acceptance used to be sticky while refusal was not, so the identical pre-checked checklist re-fired on
    # every Nous model swap. See #92647.
    declined_raw = config.get("tool_gateway_declined_tools")
    declined: set[str] = {str(k) for k in declined_raw} if isinstance(declined_raw, list) else set()
    offer_keys: list[str] = list(unconfigured) + list(has_direct)
    labels = [_GATEWAY_TOOL_LABELS[k] for k in unconfigured] + [
        f"{_GATEWAY_TOOL_LABELS[k]} — keep using your {_FEATURES[k].direct_label}" for k in has_direct
    ]
    pre_selected = [i for i, k in enumerate(unconfigured) if k not in declined]
    title = (
        "Your free Nous tool pool — pick the tools to enable:" if pool_only
        else "Your Nous subscription includes the Tool Gateway — pick the tools to enable:"
    )
    try:
        chosen_idx = prompt_checklist(title, labels, pre_selected)
    except (KeyboardInterrupt, EOFError, OSError, SystemExit):
        return set()
    chosen_keys = [offer_keys[i] for i in chosen_idx if 0 <= i < len(offer_keys)]
    # Every offered unconfigured tool NOT chosen is a decline; choosing a previously-declined tool
    # clears it. Cancel paths above return before this and record nothing.
    newly_declined = [k for k in unconfigured if k not in chosen_keys and k not in declined]
    if newly_declined or (declined & set(chosen_keys)):
        config["tool_gateway_declined_tools"] = sorted((declined | set(newly_declined)) - set(chosen_keys))
    changed = apply_gateway_defaults(config, chosen_keys) if chosen_keys else set()
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
