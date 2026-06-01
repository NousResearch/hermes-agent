"""Tests for Nous subscription feature detection."""

from hermes_cli.nous_account import NousPortalAccountInfo
from hermes_cli import nous_subscription as ns


def _account(*, logged_in: bool, paid: bool | None = None) -> NousPortalAccountInfo:
    return NousPortalAccountInfo(
        logged_in=logged_in,
        source="jwt" if logged_in else "none",
        fresh=False,
        paid_service_access=paid,
    )


def test_get_nous_subscription_features_recognizes_direct_exa_backend(monkeypatch):
    env = {"EXA_API_KEY": "exa-test"}

    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=False)
    )
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


def test_get_nous_subscription_features_force_fresh_forwards_account_request(monkeypatch):
    calls = []

    def fake_account_info(*, force_fresh=False):
        calls.append(force_fresh)
        return _account(logged_in=True, paid=True)

    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(ns, "get_nous_portal_account_info", fake_account_info)
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: False)
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: False)

    features = ns.get_nous_subscription_features({}, force_fresh=True)

    assert features.account_info is not None
    assert features.account_info.paid_service_access is True
    assert calls == [True]


def test_get_nous_subscription_features_prefers_managed_modal_in_auto_mode(monkeypatch):
    monkeypatch.setattr("tools.tool_backend_helpers.managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=True, paid=True)
    )
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
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=True, paid=True)
    )
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
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=True, paid=True)
    )
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
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=True, paid=True)
    )
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
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=False)
    )
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


def test_get_nous_subscription_features_does_not_treat_quoted_false_as_gateway_opt_in(monkeypatch):
    env = {"EXA_API_KEY": "exa-test"}

    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=True, paid=True)
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "web")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: vendor == "firecrawl")

    features = ns.get_nous_subscription_features(
        {"web": {"backend": "exa", "use_gateway": "false"}}
    )

    assert features.web.available is True
    assert features.web.active is True
    assert features.web.managed_by_nous is False
    assert features.web.direct_override is True
    assert features.web.current_provider == "exa"


def test_get_gateway_eligible_tools_ignores_quoted_false_opt_in(monkeypatch):
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda: True)
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": True, "image_gen": False, "video_gen": False, "tts": False, "browser": False},
    )

    unconfigured, has_direct, already_managed = ns.get_gateway_eligible_tools(
        {
            "model": {"provider": "nous"},
            "web": {"use_gateway": "false"},
        }
    )

    assert "web" in has_direct
    assert "web" not in already_managed
    assert set(unconfigured) == {"image_gen", "video_gen", "tts", "browser"}


def test_apply_nous_managed_defaults_writes_video_gen_config(monkeypatch):
    """apply_nous_managed_defaults must write video_gen.provider and
    video_gen.use_gateway when a Nous subscriber selects video_gen
    without a direct FAL_KEY."""
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda **kw: True)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(ns, "fal_key_is_configured", lambda: False)
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=True),
    )

    config = {"model": {"provider": "nous"}}
    changed = ns.apply_nous_managed_defaults(
        config, enabled_toolsets=["video_gen"],
    )

    assert "video_gen" in changed
    assert config["video_gen"]["provider"] == "fal"
    assert config["video_gen"]["use_gateway"] is True


def test_apply_nous_managed_defaults_writes_image_gen_config(monkeypatch):
    """apply_nous_managed_defaults must write image_gen.use_gateway
    when a Nous subscriber selects image_gen without a direct FAL_KEY."""
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda **kw: True)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(ns, "fal_key_is_configured", lambda: False)
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=True),
    )

    config = {"model": {"provider": "nous"}}
    changed = ns.apply_nous_managed_defaults(
        config, enabled_toolsets=["image_gen"],
    )

    assert "image_gen" in changed
    assert config["image_gen"]["use_gateway"] is True


def test_apply_nous_managed_defaults_skips_fal_tools_when_key_present(monkeypatch):
    """When FAL_KEY is set, apply_nous_managed_defaults should not touch
    image_gen or video_gen config — the user's direct key takes precedence."""
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda **kw: True)
    monkeypatch.setenv("FAL_KEY", "fal-direct-key")
    monkeypatch.setattr(ns, "fal_key_is_configured", lambda: True)
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=True),
    )

    config = {"model": {"provider": "nous"}}
    changed = ns.apply_nous_managed_defaults(
        config, enabled_toolsets=["image_gen", "video_gen"],
    )

    assert "image_gen" not in changed
    assert "video_gen" not in changed
    assert "image_gen" not in config
    assert "video_gen" not in config


def test_apply_nous_managed_defaults_preserves_existing_video_gen_section(monkeypatch):
    """When video_gen config already exists as a dict, the function should
    update it in-place rather than replacing it."""
    monkeypatch.setattr(ns, "managed_nous_tools_enabled", lambda **kw: True)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(ns, "fal_key_is_configured", lambda: False)
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=True),
    )

    config = {
        "model": {"provider": "nous"},
        "video_gen": {"model": "pixverse-v6"},
    }
    changed = ns.apply_nous_managed_defaults(
        config, enabled_toolsets=["video_gen"],
    )

    assert "video_gen" in changed
    assert config["video_gen"]["provider"] == "fal"
    assert config["video_gen"]["use_gateway"] is True
    # Pre-existing keys should be preserved
    assert config["video_gen"]["model"] == "pixverse-v6"


# ---------------------------------------------------------------------------
# ensure_nous_portal_access — inline login gate for `hermes tools`
# ---------------------------------------------------------------------------


def test_ensure_nous_portal_access_fast_path_when_already_paid(monkeypatch):
    """Already-entitled users return True without any login prompt."""
    login_called = {"v": False}

    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=True),
    )

    def _login(**kw):
        login_called["v"] = True
        return True

    monkeypatch.setattr(ns, "_run_nous_portal_login_only", _login)

    assert ns.ensure_nous_portal_access() is True
    assert login_called["v"] is False


def test_ensure_nous_portal_access_logs_in_then_grants(monkeypatch):
    """Logged-out user logs in, then entitlement re-check shows paid access."""
    states = iter([
        _account(logged_in=False, paid=None),  # initial check
        _account(logged_in=True, paid=True),   # after login
    ])
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda **kw: next(states),
    )
    monkeypatch.setattr(ns, "_run_nous_portal_login_only", lambda **kw: True)

    assert ns.ensure_nous_portal_access() is True


def test_ensure_nous_portal_access_returns_false_when_login_declined(monkeypatch):
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=False, paid=None),
    )
    monkeypatch.setattr(ns, "_run_nous_portal_login_only", lambda **kw: False)

    assert ns.ensure_nous_portal_access() is False


def test_ensure_nous_portal_access_false_when_logged_in_but_unpaid(monkeypatch):
    """Logged in already but no paid access — no login attempt, returns False."""
    login_called = {"v": False}
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=False),
    )

    def _login(**kw):
        login_called["v"] = True
        return True

    monkeypatch.setattr(ns, "_run_nous_portal_login_only", _login)

    assert ns.ensure_nous_portal_access() is False
    # Already logged in, so no device-code login should be attempted.
    assert login_called["v"] is False
