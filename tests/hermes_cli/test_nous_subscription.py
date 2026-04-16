"""Tests for Nous subscription feature detection."""

from hermes_cli import nous_subscription as ns


def test_get_nous_subscription_features_recognizes_direct_exa_backend(monkeypatch):
    env = {"EXA_API_KEY": "exa-test"}

    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "web")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)

    features = ns.get_nous_subscription_features({"web": {"backend": "exa"}})

    assert features.web.available is True
    assert features.web.active is True
    assert features.web.managed_by_nous is False
    assert features.web.direct_override is True
    assert features.web.current_provider == "exa"


def test_get_nous_subscription_features_prefers_managed_modal_in_auto_mode(monkeypatch):
    monkeypatch.setattr("tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "terminal")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: True)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: vendor == "modal")

    features = ns.get_nous_subscription_features(
        {"terminal": {"backend": "modal", "modal_mode": "auto"}}
    )

    assert features.modal.available is True
    assert features.modal.active is True
    assert features.modal.managed_by_nous is True
    assert features.modal.direct_override is False


def test_get_nous_subscription_features_marks_browser_use_as_managed_when_gateway_ready(monkeypatch):
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "browser")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: True)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(
        ns,
        "is_managed_tool_gateway_ready",
        lambda vendor: vendor == "browser-use",
    )

    features = ns.get_nous_subscription_features(
        {"browser": {"cloud_provider": "browser-use"}}
    )

    assert features.browser.available is True
    assert features.browser.active is True
    assert features.browser.managed_by_nous is True
    assert features.browser.direct_override is False
    assert features.browser.current_provider == "Browser Use"


def test_get_nous_subscription_features_uses_direct_browserbase_when_no_managed_gateway(monkeypatch):
    """When direct Browserbase keys are set and no managed gateway is available,
    the unconfigured fallback should pick Browserbase as a direct provider."""
    env = {
        "BROWSERBASE_API_KEY": "bb-key",
        "BROWSERBASE_PROJECT_ID": "bb-project",
    }

    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "browser")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: True)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(
        ns,
        "is_managed_tool_gateway_ready",
        lambda vendor: False,  # No managed gateway available
    )

    features = ns.get_nous_subscription_features({})

    assert features.browser.available is True
    assert features.browser.active is True
    assert features.browser.managed_by_nous is False
    assert features.browser.direct_override is True
    assert features.browser.current_provider == "Browserbase"


def test_get_nous_subscription_features_prefers_camofox_over_managed_browser_use(monkeypatch):
    env = {"CAMOFOX_URL": "http://localhost:9377"}

    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "browser")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(
        ns,
        "is_managed_tool_gateway_ready",
        lambda vendor: vendor == "browser-use",
    )

    features = ns.get_nous_subscription_features(
        {"browser": {"cloud_provider": "browser-use"}}
    )

    assert features.browser.available is True
    assert features.browser.active is True
    assert features.browser.managed_by_nous is False
    assert features.browser.direct_override is True
    assert features.browser.current_provider == "Camofox"


def test_get_nous_subscription_features_requires_agent_browser_for_browserbase(monkeypatch):
    env = {
        "BROWSERBASE_API_KEY": "bb-key",
        "BROWSERBASE_PROJECT_ID": "bb-project",
    }

    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: False)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "browser")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: False)

    features = ns.get_nous_subscription_features(
        {"browser": {"cloud_provider": "browserbase"}}
    )

    assert features.browser.available is False
    assert features.browser.active is False
    assert features.browser.managed_by_nous is False
    assert features.browser.current_provider == "Browserbase"


def test_use_gateway_web_suppresses_direct_credentials(monkeypatch):
    """use_gateway=True for web must route to managed even with direct keys present.
    set_use_gateway_flags writes both backend=firecrawl and use_gateway=True,
    so both must be present for the suppress logic to work correctly.
    """
    env = {
        "FIRECRAWL_API_KEY": "fc-test",
        "EXA_API_KEY": "exa-test",
        "TAVILY_API_KEY": "tavily-test",
        "PARALLEL_API_KEY": "parallel-test",
    }
    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "web")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: True)

    features = ns.get_nous_subscription_features(
        {"web": {"backend": "firecrawl", "use_gateway": True}}
    )

    assert features.web.managed_by_nous is True
    assert features.web.direct_override is False


def test_use_gateway_tts_suppresses_direct_credentials(monkeypatch):
    """use_gateway=True for tts must route to managed even with direct keys present.
    set_use_gateway_flags writes both provider=openai and use_gateway=True.
    """
    monkeypatch.setattr(ns, "get_env_value", lambda name: "el-key" if name == "ELEVENLABS_API_KEY" else "")
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "tts")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "openai-key")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: True)

    features = ns.get_nous_subscription_features(
        {"tts": {"provider": "openai", "use_gateway": True}}
    )

    assert features.tts.managed_by_nous is True
    assert features.tts.direct_override is False


def test_use_gateway_image_suppresses_direct_fal(monkeypatch):
    """use_gateway=True for image_gen must route to managed even with FAL key present."""
    monkeypatch.setattr(ns, "get_env_value", lambda name: "fal-key" if name == "FAL_KEY" else "")
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "image_gen")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: True)

    features = ns.get_nous_subscription_features(
        {"image_gen": {"use_gateway": True}}
    )

    assert features.image_gen.managed_by_nous is True
    assert features.image_gen.direct_override is False


def test_use_gateway_browser_suppresses_direct_credentials(monkeypatch):
    """use_gateway=True for browser must route to managed even with direct keys present.
    set_use_gateway_flags writes both cloud_provider=browser-use and use_gateway=True.
    """
    env = {
        "BROWSER_USE_API_KEY": "bu-key",
        "BROWSERBASE_API_KEY": "bb-key",
        "BROWSERBASE_PROJECT_ID": "bb-proj",
    }
    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(ns, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "browser")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: True)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: True)

    features = ns.get_nous_subscription_features(
        {"browser": {"cloud_provider": "browser-use", "use_gateway": True}}
    )

    assert features.browser.managed_by_nous is True
    assert features.browser.direct_override is False
