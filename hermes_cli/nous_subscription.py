"""Helpers for Nous subscription managed-tool capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from hermes_cli.auth import get_nous_auth_status
from hermes_cli.config import get_env_value, load_config
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
from tools.tool_backend_helpers import (
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
    def browser(self) -> NousFeatureState:
        return self.features["browser"]

    @property
    def modal(self) -> NousFeatureState:
        return self.features["modal"]

    def items(self) -> Iterable[NousFeatureState]:
        ordered = ("web", "image_gen", "tts", "browser", "modal")
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

    agent_browser_bin = shutil.which("agent-browser")
    local_bin = (
        Path(__file__).parent.parent / "node_modules" / ".bin" / "agent-browser"
    )
    return bool(agent_browser_bin or local_bin.exists())


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
        "neutts": "NeuTTS",
    }
    return mapping.get(current_provider or "edge", current_provider or "Edge TTS")


def _resolve_browser_feature_state(
    *,
    browser_tool_enabled: bool,
    browser_provider: str,
    browser_provider_explicit: bool,
    browser_local_available: bool,
    direct_camofox: bool,
    direct_browserbase: bool,
    direct_browser_use: bool,
    direct_firecrawl: bool,
    managed_browser_available: bool,
) -> tuple[str, bool, bool, bool]:
    """Resolve browser availability using the same precedence as runtime."""
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
        available = bool(browser_local_available)
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

    available = bool(browser_local_available)
    active = bool(browser_tool_enabled and available)
    return "local", available, active, False


def get_nous_subscription_features(
    config: Optional[Dict[str, object]] = None,
) -> NousSubscriptionFeatures:
    if config is None:
        config = load_config() or {}
    config = dict(config)
    model_cfg = _model_config_dict(config)
    provider_is_nous = str(model_cfg.get("provider") or "").strip().lower() == "nous"

    try:
        nous_status = get_nous_auth_status()
    except Exception:
        nous_status = {}

    managed_tools_flag = managed_nous_tools_enabled()
    nous_auth_present = bool(nous_status.get("logged_in"))
    subscribed = provider_is_nous or nous_auth_present

    web_tool_enabled = _toolset_enabled(config, "web")
    image_tool_enabled = _toolset_enabled(config, "image_gen")
    tts_tool_enabled = _toolset_enabled(config, "tts")
    browser_tool_enabled = _toolset_enabled(config, "browser")
    modal_tool_enabled = _toolset_enabled(config, "terminal")

    web_cfg = config.get("web") if isinstance(config.get("web"), dict) else {}
    tts_cfg = config.get("tts") if isinstance(config.get("tts"), dict) else {}
    browser_cfg = config.get("browser") if isinstance(config.get("browser"), dict) else {}
    terminal_cfg = config.get("terminal") if isinstance(config.get("terminal"), dict) else {}

    web_backend = str(web_cfg.get("backend") or "").strip().lower()
    tts_provider = str(tts_cfg.get("provider") or "edge").strip().lower()
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

    direct_exa = bool(get_env_value("EXA_API_KEY"))
    direct_firecrawl = bool(get_env_value("FIRECRAWL_API_KEY") or get_env_value("FIRECRAWL_API_URL"))
    direct_parallel = bool(get_env_value("PARALLEL_API_KEY"))
    direct_tavily = bool(get_env_value("TAVILY_API_KEY"))
    direct_fal = bool(get_env_value("FAL_KEY"))
    direct_openai_tts = bool(resolve_openai_audio_api_key())
    direct_elevenlabs = bool(get_env_value("ELEVENLABS_API_KEY"))
    direct_camofox = bool(get_env_value("CAMOFOX_URL"))
    direct_browserbase = bool(get_env_value("BROWSERBASE_API_KEY") and get_env_value("BROWSERBASE_PROJECT_ID"))
    direct_browser_use = bool(get_env_value("BROWSER_USE_API_KEY"))
    direct_modal = has_direct_modal_credentials()

    managed_web_available = managed_tools_flag and nous_auth_present and is_managed_tool_gateway_ready("firecrawl")
    managed_image_available = managed_tools_flag and nous_auth_present and is_managed_tool_gateway_ready("fal-queue")
    managed_tts_available = managed_tools_flag and nous_auth_present and is_managed_tool_gateway_ready("openai-audio")
    managed_browser_available = managed_tools_flag and nous_auth_present and is_managed_tool_gateway_ready("browser-use")
    managed_modal_available = managed_tools_flag and nous_auth_present and is_managed_tool_gateway_ready("modal")
    modal_state = resolve_modal_backend_state(
        modal_mode,
        has_direct=direct_modal,
        managed_ready=managed_modal_available,
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
        )
    )
    web_available = bool(
        managed_web_available or direct_exa or direct_firecrawl or direct_parallel or direct_tavily
    )

    image_managed = image_tool_enabled and managed_image_available and not direct_fal
    image_active = bool(image_tool_enabled and (image_managed or direct_fal))
    image_available = bool(managed_image_available or direct_fal)

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
    )
    tts_active = bool(tts_tool_enabled and tts_available)

    browser_local_available = _has_agent_browser()
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
            current_provider=web_backend or "",
            explicit_configured=bool(web_backend),
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
    )


def get_nous_subscription_explainer_lines() -> list[str]:
    if not managed_nous_tools_enabled():
        return []

    return [
        "Nous subscription enables managed web tools, image generation, OpenAI TTS, and browser automation by default.",
        "Those managed tools bill to your Nous subscription. Modal execution is optional and can bill to your subscription too.",
        "Change these later with: hermes setup tools, hermes setup terminal, or hermes status.",
    ]


def apply_nous_provider_defaults(config: Dict[str, object]) -> set[str]:
    """Apply provider-level Nous defaults shared by `hermes setup` and `hermes model`."""
    if not managed_nous_tools_enabled():
        return set()

    features = get_nous_subscription_features(config)
    if not features.provider_is_nous:
        return set()

    tts_cfg = config.get("tts")
    if not isinstance(tts_cfg, dict):
        tts_cfg = {}
        config["tts"] = tts_cfg

    current_tts = str(tts_cfg.get("provider") or "edge").strip().lower()
    if current_tts not in {"", "edge"}:
        return set()

    tts_cfg["provider"] = "openai"
    return {"tts"}


def apply_nous_managed_defaults(
    config: Dict[str, object],
    *,
    enabled_toolsets: Optional[Iterable[str]] = None,
) -> set[str]:
    if not managed_nous_tools_enabled():
        return set()

    features = get_nous_subscription_features(config)
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

    if "browser" in selected_toolsets and not features.browser.explicit_configured and not (
        get_env_value("BROWSER_USE_API_KEY")
        or get_env_value("BROWSERBASE_API_KEY")
    ):
        browser_cfg["cloud_provider"] = "browser-use"
        changed.add("browser")

    if "image_gen" in selected_toolsets and not get_env_value("FAL_KEY"):
        changed.add("image_gen")

    return changed
