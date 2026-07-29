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


def _pool_account() -> NousPortalAccountInfo:
    """A $0 subscriber with a live free tool pool (no paid access)."""
    return NousPortalAccountInfo(
        logged_in=True,
        source="jwt",
        fresh=False,
        paid_service_access=False,
        tool_access=NousToolAccessInfo(enabled=True, coverage=_POOL_COVERAGE),
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
    assert set(unconfigured) == {"image_gen", "video_gen", "tts", "stt", "browser"}


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


def test_ensure_nous_portal_access_returns_false_when_login_declined(monkeypatch):
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda **kw: _account(logged_in=False, paid=None),
    )
    monkeypatch.setattr(ns, "_run_nous_portal_login_only", lambda **kw: False)

    assert ns.ensure_nous_portal_access() is False


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


def test_has_agent_browser_resolves_via_hermes_managed_node_path(monkeypatch, tmp_path):
    """The managed-Node rung: a runnable agent-browser under the Hermes Node
    dir must count even when it's absent from the probe process's PATH (the
    Windows installer shape — install succeeded, GUI still said needs setup)."""
    import shutil as _shutil

    managed_dir = tmp_path / "node"
    managed_dir.mkdir()
    managed_bin = managed_dir / "agent-browser"
    managed_bin.write_text("#!/bin/sh\nexit 0\n")
    managed_bin.chmod(0o755)

    monkeypatch.setattr(_shutil, "which", lambda cmd, path=None: str(managed_bin) if path else None)
    monkeypatch.setattr(
        "hermes_constants.with_hermes_node_path", lambda: {"PATH": str(managed_dir)}
    )
    monkeypatch.setattr(
        "hermes_constants.agent_browser_runnable",
        lambda p: bool(p) and str(p) == str(managed_bin),
    )

    assert ns._has_agent_browser() is True


def test_has_agent_browser_false_when_nothing_runnable(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda cmd, path=None: None)
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda: {"PATH": ""})
    monkeypatch.setattr("hermes_constants.agent_browser_runnable", lambda p: False)

    assert ns._has_agent_browser() is False
