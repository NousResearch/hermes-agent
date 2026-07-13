"""Tests for Nous subscription feature detection."""

from hermes_cli.nous_account import NousPortalAccountInfo, NousToolAccessInfo
from hermes_cli import nous_subscription as ns


_POOL_COVERAGE = {
    "firecrawl": True,
    "fal": True,
    "fal-video": False,
    "openai-audio": True,
    "browser-use": True,
    "modal": True,
}


def _account(*, logged_in: bool, paid: bool | None = None) -> NousPortalAccountInfo:
    return NousPortalAccountInfo(
        logged_in=logged_in,
        source="jwt" if logged_in else "none",
        fresh=False,
        paid_service_access=paid,
    )


def _pool_account(
    coverage: dict[str, bool] | None = None,
) -> NousPortalAccountInfo:
    """A $0 subscriber with a live free tool pool (no paid access)."""
    return NousPortalAccountInfo(
        logged_in=True,
        source="jwt",
        fresh=False,
        paid_service_access=False,
        tool_access=NousToolAccessInfo(
            enabled=True,
            coverage=_POOL_COVERAGE if coverage is None else coverage,
        ),
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
    # Paid account: entitled to every category, including video.
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=True)
    )
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {
            "web": True,
            "image_gen": False,
            "video_gen": False,
            "tts": False,
            "stt": False,
            "browser": False,
        },
    )

    unconfigured, has_direct, already_managed = ns.get_gateway_eligible_tools(
        {
            "model": {"provider": "nous"},
            "web": {"use_gateway": "false"},
        }
    )

    assert "web" in has_direct
    assert "web" not in already_managed
    assert set(unconfigured) == {
        "image_gen",
        "video_gen",
        "tts",
        "stt",
        "browser",
        "skyvern",
    }


def test_get_gateway_eligible_tools_checks_skyvern_coverage_category(monkeypatch):
    account = _pool_account()
    checked_categories = []
    monkeypatch.setattr(
        type(account),
        "tool_gateway_entitled_for",
        lambda self, category: checked_categories.append(category) or False,
    )
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: account)
    monkeypatch.setattr(ns, "_get_gateway_direct_credentials", lambda: {})

    assert ns.get_gateway_eligible_tools({"model": {"provider": "nous"}}) == (
        [],
        [],
        [],
    )
    assert "skyvern" in checked_categories


def test_get_gateway_eligible_tools_paid_account_offers_skyvern(monkeypatch):
    monkeypatch.setattr(
        ns,
        "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=True),
    )
    monkeypatch.setattr(ns, "_get_gateway_direct_credentials", lambda: {})

    unconfigured, has_direct, already_managed = ns.get_gateway_eligible_tools(
        {"model": {"provider": "nous"}}
    )

    assert "skyvern" in unconfigured
    assert "skyvern" not in has_direct
    assert "skyvern" not in already_managed


def test_get_gateway_eligible_tools_pool_without_skyvern_coverage_excludes_it(
    monkeypatch,
):
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _pool_account())
    monkeypatch.setattr(ns, "_get_gateway_direct_credentials", lambda: {})

    eligible = ns.get_gateway_eligible_tools({"model": {"provider": "nous"}})

    assert all("skyvern" not in bucket for bucket in eligible)


def test_get_gateway_eligible_tools_pool_with_skyvern_coverage_offers_it(
    monkeypatch,
):
    coverage = {**_POOL_COVERAGE, "skyvern": True}
    monkeypatch.setattr(
        ns,
        "get_nous_portal_account_info",
        lambda **kw: _pool_account(coverage),
    )
    monkeypatch.setattr(ns, "_get_gateway_direct_credentials", lambda: {})

    unconfigured, has_direct, already_managed = ns.get_gateway_eligible_tools(
        {"model": {"provider": "nous"}}
    )

    assert "skyvern" in unconfigured
    assert "skyvern" not in has_direct
    assert "skyvern" not in already_managed


def _stub_browser_probes(monkeypatch, *, has_agent_browser, chromium, lightpanda=False):
    """Common monkeypatches for local-browser readiness scenarios.

    ``chromium`` / ``lightpanda`` drive the runtime probes that
    ``_local_browser_runnable`` reuses from ``tools.browser_tool`` (lazy import,
    so patching the module attributes is enough).
    """
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=False)
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "browser")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: has_agent_browser)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: False)
    monkeypatch.setattr("tools.browser_tool._chromium_installed", lambda: chromium)
    monkeypatch.setattr(
        "tools.browser_tool._using_lightpanda_engine", lambda: lightpanda
    )


def test_local_browser_unavailable_without_chromium(monkeypatch):
    """agent-browser present but Chromium absent must NOT advertise local browser.

    The runtime (``check_browser_requirements``) refuses local mode without a
    Chromium build, so the setup/status surface must report unavailable too —
    otherwise the user sees "Browser Automation available" and the first real
    call fails. Regression for the false-positive setup bug.
    """
    _stub_browser_probes(monkeypatch, has_agent_browser=True, chromium=False)

    features = ns.get_nous_subscription_features(
        {"browser": {"cloud_provider": "local"}}
    )

    assert features.browser.available is False
    assert features.browser.active is False
    assert features.browser.managed_by_nous is False
    assert features.browser.current_provider == "Local browser"


def test_local_browser_available_with_chromium(monkeypatch):
    _stub_browser_probes(monkeypatch, has_agent_browser=True, chromium=True)

    features = ns.get_nous_subscription_features(
        {"browser": {"cloud_provider": "local"}}
    )

    assert features.browser.available is True
    assert features.browser.active is True
    assert features.browser.current_provider == "Local browser"


def test_local_browser_available_with_lightpanda_without_chromium(monkeypatch):
    """Lightpanda is text-only and needs no Chromium, so it stays available.

    Guards against the fix over-correcting into a false-negative for the
    legitimate Lightpanda-without-Chromium configuration.
    """
    _stub_browser_probes(
        monkeypatch, has_agent_browser=True, chromium=False, lightpanda=True
    )

    features = ns.get_nous_subscription_features(
        {"browser": {"cloud_provider": "local"}}
    )

    assert features.browser.available is True
    assert features.browser.active is True


def test_default_local_browser_unavailable_without_chromium(monkeypatch):
    """The implicit (no cloud_provider) local fallthrough is gated on Chromium too."""
    _stub_browser_probes(monkeypatch, has_agent_browser=True, chromium=False)

    features = ns.get_nous_subscription_features({})

    assert features.browser.available is False
    assert features.browser.current_provider == "Local browser"


def test_cloud_browserbase_available_without_local_chromium(monkeypatch):
    """Cloud providers host their own Chromium, so the new local gate must not
    regress them: agent-browser binary present + Browserbase creds is enough."""
    env = {"BROWSERBASE_API_KEY": "bb-key", "BROWSERBASE_PROJECT_ID": "bb-project"}
    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=False)
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "browser")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: True)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: False)
    # Chromium absent locally — must not matter for a cloud provider.
    monkeypatch.setattr("tools.browser_tool._chromium_installed", lambda: False)
    monkeypatch.setattr("tools.browser_tool._using_lightpanda_engine", lambda: False)

    features = ns.get_nous_subscription_features(
        {"browser": {"cloud_provider": "browserbase"}}
    )

    assert features.browser.available is True
    assert features.browser.active is True
    assert features.browser.current_provider == "Browserbase"


def test_get_gateway_eligible_tools_pool_excludes_video(monkeypatch):
    """A free-tool-pool user is offered the covered tools but NOT video gen."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _pool_account())
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "browser": False},
    )

    unconfigured, has_direct, already_managed = ns.get_gateway_eligible_tools(
        {"model": {"provider": "nous"}}
    )

    assert set(unconfigured) == {"web", "image_gen", "tts", "stt", "browser"}
    assert "video_gen" not in unconfigured
    assert "video_gen" not in has_direct
    assert "video_gen" not in already_managed


def test_apply_gateway_defaults_stores_managed_skyvern_mcp_intent():
    config = {}

    changed = ns.apply_gateway_defaults(config, ["skyvern"])

    assert changed == {"skyvern"}
    assert config["mcp_servers"]["skyvern"] == {"managed_gateway": "skyvern"}


def test_apply_gateway_defaults_preserves_direct_skyvern_mcp_config():
    direct_skyvern = {
        "url": "https://api.skyvern.com/mcp/",
        "headers": {"x-api-key": "${SKYVERN_API_KEY}"},
    }
    config = {"mcp_servers": {"skyvern": dict(direct_skyvern)}}

    changed = ns.apply_gateway_defaults(config, ["skyvern"])

    assert changed == set()
    assert config["mcp_servers"]["skyvern"] == direct_skyvern


def test_get_gateway_eligible_tools_empty_when_not_entitled(monkeypatch):
    """A logged-in free user with no pool and no paid access gets nothing."""
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=False)
    )

    unconfigured, has_direct, already_managed = ns.get_gateway_eligible_tools(
        {"model": {"provider": "nous"}}
    )

    assert (unconfigured, has_direct, already_managed) == ([], [], [])


def _capture_checklist(monkeypatch, *, selected_idx):
    """Patch prompt_checklist to capture its args and return chosen indices."""
    captured = {}

    def _fake_checklist(title, items, pre_selected=None):
        captured["title"] = title
        captured["items"] = list(items)
        captured["pre_selected"] = list(pre_selected or [])
        return list(selected_idx)

    import hermes_cli.setup as setup_mod

    monkeypatch.setattr(setup_mod, "prompt_checklist", _fake_checklist, raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.save_config", lambda cfg: None, raising=False
    )
    return captured


def test_prompt_enable_tool_gateway_pool_offers_covered_tools_only(monkeypatch):
    """Pool user's checklist lists web/image/tts/browser and never video."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _pool_account())
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "browser": False},
    )
    captured = _capture_checklist(monkeypatch, selected_idx=[])

    config = {"model": {"provider": "nous"}}
    ns.prompt_enable_tool_gateway(config)

    blob = " ".join(captured["items"]).lower()
    assert "firecrawl" in blob  # web offered
    assert "video" not in blob  # video NOT offered to a pool user
    # Pool-aware framing, not "subscription".
    assert "free" in captured["title"].lower() and "pool" in captured["title"].lower()


def test_prompt_enable_tool_gateway_writes_only_selected(monkeypatch):
    """Selecting a subset writes use_gateway only for those tools."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _pool_account())
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "browser": False},
    )
    # Offered order is _ALL_GATEWAY_KEYS filtered to covered: web, image_gen, tts, browser.
    # Select index 0 (web) and 1 (image_gen) only.
    _capture_checklist(monkeypatch, selected_idx=[0, 1])

    config = {"model": {"provider": "nous"}}
    changed = ns.prompt_enable_tool_gateway(config)

    assert changed == {"web", "image_gen"}
    assert config["web"]["use_gateway"] is True
    assert config["image_gen"]["use_gateway"] is True
    assert "tts" not in config or config.get("tts", {}).get("use_gateway") is not True
    assert "video_gen" not in config


def test_prompt_enable_tool_gateway_paid_user_offers_video(monkeypatch):
    """Paid users still get video gen in the offer (regression guard)."""
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=True)
    )
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "browser": False},
    )
    captured = _capture_checklist(monkeypatch, selected_idx=[])

    ns.prompt_enable_tool_gateway({"model": {"provider": "nous"}})

    blob = " ".join(captured["items"]).lower()
    assert "video" in blob


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


# ---------------------------------------------------------------------------
# STT — managed-by-Nous detection (Phase 4 follow-up)
# ---------------------------------------------------------------------------

def test_stt_managed_by_nous_when_provider_openai_and_no_direct_key(monkeypatch):
    """Default `stt.provider: openai` with a Nous sub + no direct OpenAI key
    should route through the managed audio gateway."""
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=True),
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: False)
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(
        ns,
        "is_managed_tool_gateway_ready",
        lambda vendor: vendor == "openai-audio",
    )

    features = ns.get_nous_subscription_features({"stt": {"provider": "openai"}})

    assert features.stt.available is True
    assert features.stt.active is True
    assert features.stt.managed_by_nous is True
    assert features.stt.direct_override is False
    assert features.stt.current_provider == "OpenAI Whisper"


def test_stt_direct_key_overrides_managed(monkeypatch):
    """When the user has VOICE_TOOLS_OPENAI_KEY set, STT should use the
    direct key, not the managed gateway — same precedence as TTS."""
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=True, paid=True),
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: False)
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "sk-direct-key")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(
        ns,
        "is_managed_tool_gateway_ready",
        lambda vendor: vendor == "openai-audio",
    )

    features = ns.get_nous_subscription_features({"stt": {"provider": "openai"}})

    assert features.stt.available is True
    assert features.stt.managed_by_nous is False
    assert features.stt.direct_override is True


def test_stt_groq_provider_requires_groq_key(monkeypatch):
    env = {"GROQ_API_KEY": "groq-key"}
    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=False),
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: False)
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda vendor: False)

    features = ns.get_nous_subscription_features({"stt": {"provider": "groq"}})

    assert features.stt.available is True
    assert features.stt.managed_by_nous is False
    assert features.stt.current_provider == "Groq Whisper"
    assert features.stt.explicit_configured is True


def test_apply_nous_managed_defaults_flips_stt_provider_to_openai_for_nous_users(monkeypatch):
    """Fresh Nous-subscribed user with the DEFAULT_CONFIG `stt.provider: local`
    seed should have it auto-flipped to "openai" so the managed audio
    gateway transcribes their voice notes without needing faster-whisper
    installed."""
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    # CI installs [all] extras, so faster-whisper is importable there —
    # force the "no local backend" case this test is about.
    monkeypatch.setattr(ns, "_local_stt_backend_available", lambda: False)
    # Avoid the heavy real probing in get_nous_subscription_features.
    monkeypatch.setattr(
        ns,
        "get_nous_subscription_features",
        lambda config, **kw: ns.NousSubscriptionFeatures(
            subscribed=True,
            nous_auth_present=True,
            provider_is_nous=True,
            account_info=_account(logged_in=True, paid=True),
            features={
                key: ns.NousFeatureState(
                    key=key, label=key, included_by_default=True,
                    available=False, active=False, managed_by_nous=False,
                    direct_override=False, toolset_enabled=False,
                    explicit_configured=False,
                )
                for key in ("web", "image_gen", "video_gen", "tts", "stt", "browser", "modal")
            },
        ),
    )

    config = {"stt": {"provider": "local"}}
    changed = ns.apply_nous_managed_defaults(config, enabled_toolsets=[])

    assert "stt" in changed
    assert config["stt"]["provider"] == "openai"


def _stt_features_stub(*, account_info):
    return ns.NousSubscriptionFeatures(
        subscribed=True,
        nous_auth_present=True,
        provider_is_nous=True,
        account_info=account_info,
        features={
            key: ns.NousFeatureState(
                key=key, label=key, included_by_default=True,
                available=False, active=False, managed_by_nous=False,
                direct_override=False, toolset_enabled=False,
                explicit_configured=False,
            )
            for key in ("web", "image_gen", "video_gen", "tts", "stt", "browser", "modal")
        },
    )


def test_apply_nous_managed_defaults_keeps_local_stt_when_backend_works(monkeypatch):
    """A working local backend (faster-whisper installed or custom command)
    is a strong intent signal — never flip it to the managed gateway."""
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "_local_stt_backend_available", lambda: True)
    monkeypatch.setattr(
        ns,
        "get_nous_subscription_features",
        lambda config, **kw: _stt_features_stub(
            account_info=_account(logged_in=True, paid=True)
        ),
    )

    config = {"stt": {"provider": "local"}}
    changed = ns.apply_nous_managed_defaults(config, enabled_toolsets=[])

    assert "stt" not in changed
    assert config["stt"]["provider"] == "local"


def test_apply_nous_managed_defaults_skips_stt_when_not_entitled(monkeypatch):
    """A subscriber whose tool pool doesn't cover openai-audio must not be
    pointed at a managed gateway that will refuse them."""
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "_local_stt_backend_available", lambda: False)
    monkeypatch.setattr(
        ns,
        "get_nous_subscription_features",
        lambda config, **kw: _stt_features_stub(
            account_info=_account(logged_in=True, paid=False)
        ),
    )

    config = {"stt": {"provider": "local"}}
    changed = ns.apply_nous_managed_defaults(config, enabled_toolsets=[])

    assert "stt" not in changed
    assert config["stt"]["provider"] == "local"


def test_apply_nous_managed_defaults_skips_stt_when_groq_key_present(monkeypatch):
    """Don't override a user who explicitly set up Groq for STT."""
    env = {"GROQ_API_KEY": "groq-key"}
    monkeypatch.setattr(ns, "get_env_value", lambda name: env.get(name, ""))
    monkeypatch.setattr(
        ns,
        "get_nous_subscription_features",
        lambda config, **kw: ns.NousSubscriptionFeatures(
            subscribed=True,
            nous_auth_present=True,
            provider_is_nous=True,
            account_info=_account(logged_in=True, paid=True),
            features={
                key: ns.NousFeatureState(
                    key=key, label=key, included_by_default=True,
                    available=False, active=False, managed_by_nous=False,
                    direct_override=False, toolset_enabled=False,
                    explicit_configured=False,
                )
                for key in ("web", "image_gen", "video_gen", "tts", "stt", "browser", "modal")
            },
        ),
    )

    config = {"stt": {"provider": "local"}}
    changed = ns.apply_nous_managed_defaults(config, enabled_toolsets=[])

    # STT was not flipped because the user has a Groq key configured.
    assert "stt" not in changed
    assert config["stt"]["provider"] == "local"


def test_apply_gateway_defaults_sets_stt_use_gateway(monkeypatch):
    config = {}
    changed = ns.apply_gateway_defaults(config, ["stt"])

    assert "stt" in changed
    assert config["stt"]["provider"] == "openai"
    assert config["stt"]["use_gateway"] is True
