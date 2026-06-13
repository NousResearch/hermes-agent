"""Helpers for Nous subscription managed-tool capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from hermes_cli.config import get_env_value, load_config
from hermes_cli.nous_account import (
    NousPortalAccountInfo,
    format_nous_portal_entitlement_message,
    get_nous_portal_account_info,
)
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
from utils import is_truthy_value
from tools.tool_backend_helpers import (
    fal_key_is_configured,
    has_direct_modal_credentials,
    managed_nous_tools_enabled,
    normalize_browser_cloud_provider,
    normalize_modal_mode,
    resolve_modal_backend_state,
    resolve_openai_audio_api_key,
)


_DEFAULT_PLATFORM_TOOLSETS = {
    "cli": "hermes-cli",
}

# Maps a tools_config provider's ``managed_nous_feature`` to the tool-pool
# coverage category (hermes_cli.nous_account.TOOL_COVERAGE_CATEGORIES). Lets the
# `hermes tools` picker scope its entitlement gate to the selected backend, so a
# free-tool-pool user is allowed image gen but denied video gen at select time —
# consistent with the per-category feature gates in get_nous_subscription_features.
MANAGED_FEATURE_COVERAGE_CATEGORY: Dict[str, str] = {
    "web": "firecrawl",
    "image_gen": "fal",
    "video_gen": "fal-video",
    "tts": "openai-audio",
    # STT shares the TTS coverage category: both ride the managed
    # "openai-audio" gateway endpoint (speech + transcriptions).
    "stt": "openai-audio",
    "browser": "browser-use",
    "modal": "modal",
}


def _uses_gateway(section: object) -> bool:
    """Return True when a config section explicitly opts into the gateway."""
    if not isinstance(section, dict):
        return False
    return is_truthy_value(section.get("use_gateway"), default=False)


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

    @property
    def web(self) -> NousFeatureState:
        return self.features["web"]

    @property
    def image_gen(self) -> NousFeatureState:
        return self.features["image_gen"]

    @property
    def tts(self) -> NousFeatureState:
        return self.features["tts"]

    @property
    def stt(self) -> NousFeatureState:
        return self.features["stt"]

    @property
    def browser(self) -> NousFeatureState:
        return self.features["browser"]

    @property
    def video_gen(self) -> NousFeatureState:
        return self.features["video_gen"]

    @property
    def modal(self) -> NousFeatureState:
        return self.features["modal"]

    def items(self) -> Iterable[NousFeatureState]:
        ordered = ("web", "image_gen", "video_gen", "tts", "stt", "browser", "modal")
        for key in ordered:
            yield self.features[key]


def _model_config_dict(config: Dict[str, object]) -> Dict[str, object]:
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return dict(model_cfg)
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def _toolset_enabled(config: Dict[str, object], toolset_key: str) -> bool:
    from toolsets import resolve_toolset

    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        platform_toolsets = {"cli": [_DEFAULT_PLATFORM_TOOLSETS["cli"]]}

    target_tools = set(resolve_toolset(toolset_key))
    if not target_tools:
        return False

    for platform, raw_toolsets in platform_toolsets.items():
        if isinstance(raw_toolsets, list):
            toolset_names = list(raw_toolsets)
        else:
            default_toolset = _DEFAULT_PLATFORM_TOOLSETS.get(platform)
            toolset_names = [default_toolset] if default_toolset else []
        if not toolset_names:
            default_toolset = _DEFAULT_PLATFORM_TOOLSETS.get(platform)
            if default_toolset:
                toolset_names = [default_toolset]

        available_tools: Set[str] = set()
        for toolset_name in toolset_names:
            if not isinstance(toolset_name, str) or not toolset_name:
                continue
            try:
                available_tools.update(resolve_toolset(toolset_name))
            except Exception:
                continue

        if target_tools and target_tools.issubset(available_tools):
            return True

    return False


def _has_agent_browser() -> bool:
    import shutil

    from hermes_constants import agent_browser_runnable

    # Validate the resolved binary actually runs — a dangling global symlink
    # (issue #48521) is reported by ``which`` but fails at exec. Fall through to
    # the local node_modules copy, which the validator also checks.
    if agent_browser_runnable(shutil.which("agent-browser")):
        return True
    local_bin = (
        Path(__file__).parent.parent / "node_modules" / ".bin" / "agent-browser"
    )
    return agent_browser_runnable(str(local_bin)) if local_bin.exists() else False


def _local_browser_runnable() -> bool:
    """Return True when the *local* browser backend would actually start.

    The ``agent-browser`` CLI being present is necessary but not sufficient for
    local mode: agent-browser also needs a Chromium build on disk (without one
    it hangs on first use until the command timeout fires), unless the
    Lightpanda engine is selected — text-only navigation needs no Chromium.

    This mirrors the local-mode tail of
    :func:`tools.browser_tool.check_browser_requirements`, so the setup/status
    surfaces advertise local browser readiness only when the runtime would
    actually run it. Cloud providers (Browserbase, Browser Use, Firecrawl) host
    their own Chromium and therefore gate on :func:`_has_agent_browser` alone.
    """
    if not _has_agent_browser():
        return False
    try:
        from tools.browser_tool import _chromium_installed, _using_lightpanda_engine
    except Exception:
        # If the runtime probe can't be imported, fall back to binary presence
        # (prior behaviour) rather than crashing the setup/status surface.
        return True
    if _using_lightpanda_engine():
        return True
    return _chromium_installed()


def _browser_label(current_provider: str) -> str:
    mapping = {
        "browserbase": "Browserbase",
        "browser-use": "Browser Use",
        "firecrawl": "Firecrawl",
        "camofox": "Camofox",
        "local": "Local browser",
    }
    return mapping.get(current_provider or "local", current_provider or "Local browser")


def _tts_label(current_provider: str) -> str:
    mapping = {
        "openai": "OpenAI TTS",
        "elevenlabs": "ElevenLabs",
        "edge": "Edge TTS",
        "xai": "xAI TTS",
        "mistral": "Mistral Voxtral TTS",
        "neutts": "NeuTTS",
    }
    return mapping.get(current_provider or "edge", current_provider or "Edge TTS")


def _stt_label(current_provider: str) -> str:
    mapping = {
        "openai": "OpenAI Whisper",
        "groq": "Groq Whisper",
        "mistral": "Mistral Voxtral Transcribe",
        "local": "Local faster-whisper",
    }
    return mapping.get(current_provider or "local", current_provider or "Local faster-whisper")


def _local_stt_backend_available() -> bool:
    """Whether a local STT backend could serve transcription right now.

    True when faster-whisper is importable or a custom local STT command
    is configured. Used both for feature detection and to stop
    ``apply_nous_managed_defaults`` from flipping a working local setup
    to the managed gateway.
    """
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
    if direct_camofox:
        return "camofox", True, bool(browser_tool_enabled), False

    if browser_provider_explicit:
        current_provider = browser_provider or "local"
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


def get_nous_subscription_features(
    config: Optional[Dict[str, object]] = None,
    *,
    force_fresh: bool = False,
) -> NousSubscriptionFeatures:
    if config is None:
        config = load_config() or {}
    config = dict(config)
    model_cfg = _model_config_dict(config)
    provider_is_nous = str(model_cfg.get("provider") or "").strip().lower() == "nous"

    try:
        if force_fresh:
            account_info = get_nous_portal_account_info(force_fresh=True)
        else:
            account_info = get_nous_portal_account_info()
    except Exception:
        account_info = None

    # Coarse "entitled to any managed tool" gate: paid access OR a live free
    # tool pool. Per-backend availability is then narrowed by coverage below
    # (the pool funds image but not video, etc.).
    managed_tools_flag = bool(
        account_info
        and account_info.logged_in
        and account_info.tool_gateway_entitled
    )
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

    # use_gateway flags — when True, the user explicitly opted into the
    # Tool Gateway via `hermes model`, so direct credentials should NOT
    # prevent gateway routing.
    web_use_gateway = _uses_gateway(web_cfg)
    tts_use_gateway = _uses_gateway(tts_cfg)
    stt_use_gateway = _uses_gateway(stt_cfg)
    browser_use_gateway = _uses_gateway(browser_cfg)
    image_gen_cfg = config.get("image_gen") if isinstance(config.get("image_gen"), dict) else {}
    image_use_gateway = _uses_gateway(image_gen_cfg)
    video_gen_cfg = config.get("video_gen") if isinstance(config.get("video_gen"), dict) else {}
    video_use_gateway = _uses_gateway(video_gen_cfg)

    direct_exa = bool(get_env_value("EXA_API_KEY"))
    direct_firecrawl = bool(get_env_value("FIRECRAWL_API_KEY") or get_env_value("FIRECRAWL_API_URL"))
    direct_parallel = bool(get_env_value("PARALLEL_API_KEY"))
    direct_tavily = bool(get_env_value("TAVILY_API_KEY"))
    direct_searxng = bool(get_env_value("SEARXNG_URL"))
    direct_crw = bool(get_env_value("CRW_API_KEY") or get_env_value("CRW_API_URL"))
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
        direct_crw = False
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

    web_managed = web_backend == "firecrawl" and managed_web_available and not direct_firecrawl
    web_active = bool(
        web_tool_enabled
        and (
            web_managed
            or (web_backend == "exa" and direct_exa)
            or (web_backend == "firecrawl" and direct_firecrawl)
            or (web_backend == "parallel" and direct_parallel)
            or (web_backend == "tavily" and direct_tavily)
            or (web_backend == "searxng" and direct_searxng)
            or (web_backend == "crw" and direct_crw)
            # Per-capability overrides: search_backend or extract_backend may be set
            # without web.backend (using the new split config from #20061)
            or (web_search_backend == "searxng" and direct_searxng)
            or (web_search_backend == "exa" and direct_exa)
            or (web_search_backend == "firecrawl" and direct_firecrawl)
            or (web_search_backend == "parallel" and direct_parallel)
            or (web_search_backend == "tavily" and direct_tavily)
            or (web_search_backend == "crw" and direct_crw)
        )
    )
    web_available = bool(
        managed_web_available or direct_exa or direct_firecrawl or direct_parallel or direct_tavily or direct_searxng or direct_crw
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

    tts_explicit_configured = False
    raw_tts_cfg = config.get("tts")
    if isinstance(raw_tts_cfg, dict) and "provider" in raw_tts_cfg:
        tts_explicit_configured = tts_provider not in {"", "edge"}

    # STT considers any non-default provider explicit. "local" is the
    # DEFAULT_CONFIG seed, so seeing it doesn't mean the user picked it.
    stt_explicit_configured = False
    raw_stt_cfg = config.get("stt")
    if isinstance(raw_stt_cfg, dict) and "provider" in raw_stt_cfg:
        stt_explicit_configured = stt_provider not in {"", "local"}

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
            current_provider=web_backend or web_search_backend or "",
            explicit_configured=bool(web_backend or web_search_backend),
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
            current_provider="FAL" if direct_fal else ("Nous Subscription" if image_managed else ""),
            explicit_configured=direct_fal,
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
            current_provider="FAL" if direct_fal_video else ("Nous Subscription" if video_managed else ""),
            explicit_configured=direct_fal_video,
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

    return NousSubscriptionFeatures(
        subscribed=subscribed,
        nous_auth_present=nous_auth_present,
        provider_is_nous=provider_is_nous,
        features=features,
        account_info=account_info,
    )





def apply_nous_managed_defaults(
    config: Dict[str, object],
    *,
    enabled_toolsets: Optional[Iterable[str]] = None,
    force_fresh: bool = False,
) -> set[str]:
    features = get_nous_subscription_features(config, force_fresh=force_fresh)
    if not (
        features.account_info
        and features.account_info.logged_in
        and features.account_info.tool_gateway_entitled
    ):
        return set()
    if not features.provider_is_nous:
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
        web_cfg["backend"] = "firecrawl"
        changed.add("web")

    if "tts" in selected_toolsets and not features.tts.explicit_configured and not (
        resolve_openai_audio_api_key()
        or get_env_value("ELEVENLABS_API_KEY")
    ):
        tts_cfg["provider"] = "openai"
        changed.add("tts")

    # STT: same pattern as TTS. The DEFAULT_CONFIG seed is "local"
    # (requires `pip install faster-whisper`); for Nous subscribers we
    # flip it to "openai" so the managed audio gateway handles transcription
    # via the same auth as TTS. Skipped when the user has explicitly
    # configured STT, has direct credentials for a non-managed provider,
    # has a working local backend (faster-whisper installed or a custom
    # local command — strong intent signal that "local" was a choice, not
    # just the DEFAULT_CONFIG seed), or isn't entitled to the managed
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
        stt_cfg["provider"] = "openai"
        changed.add("stt")

    if "browser" in selected_toolsets and not features.browser.explicit_configured and not (
        get_env_value("BROWSER_USE_API_KEY")
        or get_env_value("BROWSERBASE_API_KEY")
    ):
        browser_cfg["cloud_provider"] = "browser-use"
        changed.add("browser")

    if "image_gen" in selected_toolsets and not fal_key_is_configured():
        image_cfg = config.get("image_gen")
        if not isinstance(image_cfg, dict):
            image_cfg = {}
            config["image_gen"] = image_cfg
        image_cfg["use_gateway"] = True
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
        video_cfg["provider"] = "fal"
        video_cfg["use_gateway"] = True
        changed.add("video_gen")

    return changed


# ---------------------------------------------------------------------------
# Tool Gateway offer — single Y/n prompt after model selection
# ---------------------------------------------------------------------------

_GATEWAY_TOOL_LABELS = {
    "web": "Web search & extract (Firecrawl)",
    "image_gen": "Image generation (FAL)",
    "video_gen": "Video generation (FAL)",
    "tts": "Text-to-speech (OpenAI TTS)",
    "stt": "Speech-to-text (OpenAI Whisper)",
    "browser": "Browser automation (Browser Use)",
}


def _get_gateway_direct_credentials() -> Dict[str, bool]:
    """Return a dict of tool_key -> has_direct_credentials."""
    fal_direct = fal_key_is_configured()
    return {
        "web": bool(
            get_env_value("FIRECRAWL_API_KEY")
            or get_env_value("FIRECRAWL_API_URL")
            or get_env_value("PARALLEL_API_KEY")
            or get_env_value("TAVILY_API_KEY")
            or get_env_value("EXA_API_KEY")
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
        ),
    }


_GATEWAY_DIRECT_LABELS = {
    "web": "Firecrawl/Exa/Parallel/Tavily key",
    "image_gen": "FAL key",
    "video_gen": "FAL key",
    "tts": "OpenAI/ElevenLabs key",
    "stt": "OpenAI/Groq/Mistral key",
    "browser": "Browser Use/Browserbase key",
}

_ALL_GATEWAY_KEYS = ("web", "image_gen", "video_gen", "tts", "stt", "browser")


def get_gateway_eligible_tools(
    config: Optional[Dict[str, object]] = None,
    *,
    force_fresh: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Return (unconfigured, has_direct, already_managed) tool key lists.

    - unconfigured: tools with no direct credentials (easy switch)
    - has_direct: tools where the user has their own API keys
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
        return [], [], []
    if not (account_info and account_info.logged_in and account_info.tool_gateway_entitled):
        return [], [], []

    if config is None:
        config = load_config() or {}

    # Quick provider check without the heavy get_nous_subscription_features call
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict) or str(model_cfg.get("provider") or "").strip().lower() != "nous":
        return [], [], []

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
    already_managed: list[str] = []
    for key in _ALL_GATEWAY_KEYS:
        # Only offer tools the user's entitlement actually covers. For a free
        # tool pool that means image but not video; paid users are covered for
        # everything.
        if not account_info.tool_gateway_entitled_for(
            MANAGED_FEATURE_COVERAGE_CATEGORY[key]
        ):
            continue
        if opted_in.get(key):
            already_managed.append(key)
        elif direct.get(key):
            has_direct.append(key)
        else:
            unconfigured.append(key)
    return unconfigured, has_direct, already_managed


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
        web_cfg["backend"] = "firecrawl"
        web_cfg["use_gateway"] = True
        changed.add("web")

    if "tts" in tool_keys:
        tts_cfg["provider"] = "openai"
        tts_cfg["use_gateway"] = True
        changed.add("tts")

    if "stt" in tool_keys:
        stt_cfg["provider"] = "openai"
        stt_cfg["use_gateway"] = True
        changed.add("stt")

    if "browser" in tool_keys:
        browser_cfg["cloud_provider"] = "browser-use"
        browser_cfg["use_gateway"] = True
        changed.add("browser")

    if "image_gen" in tool_keys:
        image_cfg = config.get("image_gen")
        if not isinstance(image_cfg, dict):
            image_cfg = {}
            config["image_gen"] = image_cfg
        image_cfg["use_gateway"] = True
        changed.add("image_gen")

    if "video_gen" in tool_keys:
        video_cfg = config.get("video_gen")
        if not isinstance(video_cfg, dict):
            video_cfg = {}
            config["video_gen"] = video_cfg
        video_cfg["provider"] = "fal"
        video_cfg["use_gateway"] = True
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
    unconfigured, has_direct, already_managed = get_gateway_eligible_tools(
        config,
        force_fresh=force_fresh,
    )
    if not unconfigured and not has_direct:
        return set()

    try:
        from hermes_cli.setup import prompt_checklist
    except Exception:
        return set()

    # Frame the offer by entitlement: a $0 free-tool-pool user is not on a paid
    # plan, so don't call it "your subscription".
    try:
        account_info = get_nous_portal_account_info(force_fresh=False)
    except Exception:
        account_info = None
    pool_only = bool(
        account_info
        and account_info.paid_service_access is not True
        and account_info.tool_access is not None
        and account_info.tool_access.enabled
    )
    source_label = "free tool pool" if pool_only else "Nous subscription"

    # Per-tool checklist: unconfigured tools first (pre-checked for new users),
    # then tools where the user already has their own key (left unchecked so we
    # don't override their own setup unless they ask).
    offer_keys: list[str] = list(unconfigured) + list(has_direct)
    labels: list[str] = [_GATEWAY_TOOL_LABELS[k] for k in unconfigured]
    labels += [
        f"{_GATEWAY_TOOL_LABELS[k]} — keep using your {_GATEWAY_DIRECT_LABELS[k]}"
        for k in has_direct
    ]
    pre_selected = list(range(len(unconfigured)))

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
    if not chosen_keys:
        return set()

    changed = apply_gateway_defaults(config, chosen_keys)
    if changed:
        from hermes_cli.config import save_config

        save_config(config)
        for key in sorted(changed):
            label = _GATEWAY_TOOL_LABELS.get(key, key)
            print(f"  ✓ {label}: enabled via {source_label}")
    return changed


# ---------------------------------------------------------------------------
# Inline Nous Portal login for the Tool Gateway picker (`hermes tools`)
# ---------------------------------------------------------------------------


def ensure_nous_portal_access(
    *,
    capability: str = "the Nous Tool Gateway",
    coverage_category: Optional[str] = None,
) -> bool:
    """Make sure the user is entitled to the Nous Tool Gateway, logging in if
    needed.

    Used by ``hermes tools`` when a user selects a Nous-managed Tool Gateway
    backend (e.g. "Firecrawl (Nous Portal)").  Unlike ``hermes model``'s Nous
    login, this:

    - does NOT change the inference provider (``model.provider`` is untouched),
    - does NOT run model selection, and
    - does NOT offer the bulk "enable for all tools" Tool Gateway prompt.

    It only performs the Nous Portal device-code OAuth (when the user isn't
    already logged in) and refreshes entitlement, so the caller can enable the
    single tool the user picked.

    Entitlement is satisfied by paid service access OR a live free tool pool.
    When ``coverage_category`` is given (e.g. ``"fal"`` for image gen), the pool
    must cover that category specifically — so a pool user selecting video
    (``"fal-video"``, not pool-funded) is correctly denied.

    Returns ``True`` when the account is entitled after the flow, ``False``
    otherwise (declined login, login failed, or no entitlement).
    """

    def _entitled(account) -> bool:
        if account is None:
            return False
        if coverage_category is not None:
            return account.tool_gateway_entitled_for(coverage_category)
        return account.tool_gateway_entitled

    # Fast path: already entitled.
    try:
        info = get_nous_portal_account_info(force_fresh=True)
    except Exception:
        info = None
    if _entitled(info):
        return True

    # If not logged in at all, run the device-code login (auth only).
    if info is None or not info.logged_in:
        if not _run_nous_portal_login_only(capability=capability):
            return False
        try:
            info = get_nous_portal_account_info(force_fresh=True)
        except Exception:
            info = None

    if _entitled(info):
        return True

    # Logged in but not entitled for this capability — surface neutral billing
    # guidance, do not enable. coverage_category keeps a pool user who lacks this
    # one category from being told their credits are exhausted.
    message = format_nous_portal_entitlement_message(
        info, capability=capability, coverage_category=coverage_category
    )
    if message:
        for line in message.splitlines():
            print(f"  {line}")
    return False


def _run_nous_portal_login_only(*, capability: str) -> bool:
    """Run the Nous Portal device-code OAuth and persist credentials only.

    No model selection, no provider switch, no Tool Gateway bulk prompt.
    Returns ``True`` on a successful login, ``False`` if the user declined or
    the flow failed.
    """
    try:
        from hermes_cli.auth import (
            _auth_store_lock,
            _load_auth_store,
            _nous_device_code_login,
            _read_shared_nous_state,
            _save_auth_store,
            _save_provider_state,
            _sync_nous_pool_from_auth_store,
            _try_import_shared_nous_state,
            _write_shared_nous_state,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  Could not start Nous Portal login: {exc}")
        return False

    print()
    print(f"  {capability} requires a Nous Portal login.")
    try:
        proceed = input("  Log in to Nous Portal now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if proceed not in {"", "y", "yes"}:
        print("  Skipped Nous Portal login.")
        return False

    try:
        # Snapshot the active_provider so a tool-config login never silently
        # switches the user's inference provider to Nous.
        with _auth_store_lock():
            prior_active_provider = _load_auth_store().get("active_provider")

        auth_state = None
        shared = _read_shared_nous_state()
        if shared:
            try:
                do_import = input(
                    "  Found existing Nous OAuth credentials. Import them? [Y/n]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                do_import = "y"
            if do_import in {"", "y", "yes"}:
                auth_state = _try_import_shared_nous_state(timeout_seconds=15.0)

        if auth_state is None:
            auth_state = _nous_device_code_login()

        with _auth_store_lock():
            auth_store = _load_auth_store()
            _save_provider_state(auth_store, "nous", auth_state)
            # Preserve the user's existing inference provider — this login is
            # for tool entitlement only, not a provider switch.
            if prior_active_provider:
                auth_store["active_provider"] = prior_active_provider
            else:
                auth_store.pop("active_provider", None)
            _save_auth_store(auth_store)

        _write_shared_nous_state(auth_state)
        _sync_nous_pool_from_auth_store()
        print("  Nous Portal login successful.")
        return True
    except KeyboardInterrupt:
        print("\n  Login cancelled.")
        return False
    except SystemExit:
        # _nous_device_code_login raises SystemExit on subscription_required;
        # it already printed billing guidance.
        return False
    except Exception as exc:
        print(f"  Nous Portal login failed: {exc}")
        return False
