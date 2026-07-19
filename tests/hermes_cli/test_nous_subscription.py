"""Tests for Nous subscription feature detection."""

import shutil
import sys

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






# ---------------------------------------------------------------------------
# STT — managed-by-Nous detection (Phase 4 follow-up)
# ---------------------------------------------------------------------------





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






def _block_legacy_agent_browser_checks(monkeypatch):
    """Make the legacy checks (PATH lookup + local node_modules/.bin) find nothing."""
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda cmd, *args, **kwargs: (
            None if cmd == "agent-browser" else real_which(cmd, *args, **kwargs)
        ),
    )
    monkeypatch.setattr("hermes_constants.agent_browser_runnable", lambda path: False)


def test_has_agent_browser_true_for_npx_only_resolution(monkeypatch):
    """No PATH binary and no runnable node_modules copy, but the browser_tool
    cascade resolves the npx fallback: browser capability is available."""
    _block_legacy_agent_browser_checks(monkeypatch)
    import tools.browser_tool as browser_tool

    calls = []

    def fake_find_agent_browser(*, validate=True):
        calls.append({"validate": validate})
        return "npx agent-browser"

    monkeypatch.setattr(browser_tool, "_find_agent_browser", fake_find_agent_browser)
    monkeypatch.setattr(
        browser_tool, "_requires_real_termux_browser_install", lambda cmd: False
    )

    assert ns._has_agent_browser() is True
    # A readiness probe must resolve without spawning the daemon.
    assert calls and all(call["validate"] is False for call in calls)


def test_has_agent_browser_false_for_termux_local_bare_npx(monkeypatch):
    """On Termux in local mode the bare npx fallback is not a usable install."""
    _block_legacy_agent_browser_checks(monkeypatch)
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(
        browser_tool,
        "_find_agent_browser",
        lambda *, validate=True: "npx agent-browser",
    )
    monkeypatch.setattr(
        browser_tool,
        "_requires_real_termux_browser_install",
        lambda cmd: cmd.strip() == "npx agent-browser",
    )

    assert ns._has_agent_browser() is False


def test_has_agent_browser_false_when_nothing_resolvable(monkeypatch):
    _block_legacy_agent_browser_checks(monkeypatch)
    import tools.browser_tool as browser_tool

    def raise_not_found(*, validate=True):
        raise FileNotFoundError("agent-browser CLI not found")

    monkeypatch.setattr(browser_tool, "_find_agent_browser", raise_not_found)

    assert ns._has_agent_browser() is False


def test_has_agent_browser_import_failure_falls_back_to_path_check(monkeypatch):
    """If tools.browser_tool cannot be imported, the old PATH + node_modules
    check must still answer (prior behaviour), not crash."""
    monkeypatch.setitem(sys.modules, "tools.browser_tool", None)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda cmd, *args, **kwargs: (
            "/fake/bin/agent-browser"
            if cmd == "agent-browser"
            else real_which(cmd, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        "hermes_constants.agent_browser_runnable",
        lambda path: path == "/fake/bin/agent-browser",
    )

    assert ns._has_agent_browser() is True


def test_has_agent_browser_import_failure_and_no_binary_is_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "tools.browser_tool", None)
    _block_legacy_agent_browser_checks(monkeypatch)

    assert ns._has_agent_browser() is False
