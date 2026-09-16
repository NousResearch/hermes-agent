"""Tests for Nous subscription feature detection."""

import shutil
import sys

from hermes_cli.nous_account import NousPortalAccountInfo, NousToolAccessInfo
from hermes_cli import nous_subscription as ns
from tools import tool_backend_helpers
from tools import browser_tool_install as bt_install


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


def test_get_nous_subscription_features_recognizes_keyless_tavily_backend(monkeypatch):
    """Selecting Tavily in setup/tools counts as available with no API key.

    Mirrors tools.web_tools._is_backend_available('tavily'): keyless is
    opt-in via web.backend / search_backend / extract_backend, not a
    silent empty-install default. The setup summary previously required
    TAVILY_API_KEY and printed a false 'missing' after a skipped key prompt.
    """
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=False)
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "web")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)

    features = ns.get_nous_subscription_features({"web": {"backend": "tavily"}})

    assert features.web.available is True
    assert features.web.active is True
    assert features.web.managed_by_nous is False
    assert features.web.direct_override is True
    assert features.web.current_provider == "tavily"
    assert features.web.explicit_configured is True


def test_keyless_tavily_search_backend_without_shared_backend(monkeypatch):
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=False)
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "web")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)

    features = ns.get_nous_subscription_features(
        {"web": {"search_backend": "tavily"}}
    )

    assert features.web.available is True
    assert features.web.active is True
    assert features.web.current_provider == "tavily"


def test_unconfigured_web_without_keys_is_unavailable(monkeypatch):
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info", lambda: _account(logged_in=False)
    )
    monkeypatch.setattr(ns, "_toolset_enabled", lambda config, key: key == "web")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)

    features = ns.get_nous_subscription_features({})

    assert features.web.available is False
    assert features.web.active is False
    assert features.web.explicit_configured is False
def _stub_browser_probes(monkeypatch, *, has_agent_browser, chromium, lightpanda=False):
    """Common monkeypatches for local-browser readiness scenarios.

    ``chromium`` / ``lightpanda`` drive the runtime probes that
    ``_local_browser_runnable`` reuses from the ``tools.browser_tool_*`` siblings (lazy import,
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
    monkeypatch.setattr("tools.browser_tool_install._chromium_installed", lambda: chromium)
    monkeypatch.setattr(
        "tools.browser_tool_lightpanda_fallback._using_lightpanda_engine", lambda: lightpanda
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


def test_logged_in_entitled_account_yields_a_state_for_every_feature(monkeypatch):
    """The logged-in + entitled branch must produce a state for EVERY feature, including
    those with no config selection field (modal). Regression: the managed-availability
    table indexed the selection map by every feature key and raised KeyError('modal'),
    crashing `hermes status`, `hermes tools`, and the dashboard toolsets API."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _pool_account())
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda gateway: True)
    monkeypatch.setattr(ns, "get_env_value", lambda name: "")
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(ns, "has_direct_modal_credentials", lambda: False)

    result = ns.get_nous_subscription_features({"model": {"provider": "nous"}})

    assert set(result.features) == set(ns._FEATURES)
    assert result.modal.available is True  # entitled + gateway ready → managed modal is offered


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


def test_get_gateway_eligible_tools_treats_explicit_backend_as_configured(monkeypatch):
    """A keyless local backend (e.g. searxng) has no credentials to detect,
    but an explicit non-nous selection must still keep it out of
    'unconfigured' — regression for #92647, where it was pre-checked and a
    single Enter during `hermes model` overwrote it to `web.backend: nous`.
    """
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=True))
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "browser": False},
    )

    config = {"model": {"provider": "nous"}, "web": {"backend": "searxng"}}
    unconfigured, has_direct, explicit_configured, already_managed = ns.get_gateway_eligible_tools(config)

    assert "web" not in unconfigured
    assert "web" not in has_direct
    assert "web" in explicit_configured
    assert "web" not in already_managed


def test_get_gateway_eligible_tools_treats_browser_use_selection_as_explicit(monkeypatch):
    """An explicit BYOK `browser.cloud_provider: browser-use` selection must
    land in explicit_configured, not unconfigured/has_direct — the same
    protection as searxng above. This is distinct from the gateway's own
    managed selection, which is always stored as `cloud_provider: nous`
    (see apply_gateway_defaults); "browser-use" only appears here when the
    user picked it directly, so it must never be treated as up for grabs.
    """
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=True))
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "browser": True},
    )

    config = {"model": {"provider": "nous"}, "browser": {"cloud_provider": "browser-use"}}
    unconfigured, has_direct, explicit_configured, already_managed = ns.get_gateway_eligible_tools(config)

    assert "browser" not in unconfigured
    assert "browser" not in has_direct
    assert "browser" in explicit_configured
    assert "browser" not in already_managed


def test_get_gateway_eligible_tools_not_entitled_returns_four_empty_lists(monkeypatch):
    """A logged-in Nous account with no paid access and no free tool pool
    must fail closed with a 4-tuple, not a 3-tuple — regression for a crash
    where the early 'not entitled' return still had the pre-refactor arity
    while the happy path and every caller had moved to 4 values."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=False))

    config = {"model": {"provider": "nous"}}
    result = ns.get_gateway_eligible_tools(config)

    assert result == ([], [], [], [])


def test_prompt_enable_tool_gateway_not_entitled_does_not_crash(monkeypatch):
    """The unconditional call site in model_setup_flows (no try/except) must
    not raise when a Nous account is logged in but not entitled to the Tool
    Gateway (i.e. an ordinary non-paid, non-pool account)."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=False))

    config = {"model": {"provider": "nous"}}
    assert ns.prompt_enable_tool_gateway(config) == set()


def test_prompt_enable_tool_gateway_never_offers_explicit_backend(monkeypatch):
    """The checklist itself must not list (let alone pre-check) a tool with
    an explicit non-nous selection, so it can never be silently overwritten
    by an accidental Enter."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=True))
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "browser": False},
    )
    captured = _capture_checklist(monkeypatch, selected_idx=[])

    config = {"model": {"provider": "nous"}, "web": {"backend": "searxng"}}
    ns.prompt_enable_tool_gateway(config)

    blob = " ".join(captured["items"]).lower()
    assert "firecrawl" not in blob  # web (searxng-configured) NOT offered
    assert "image" in blob  # other unconfigured tools still offered


def test_gateway_direct_credentials_honor_env_configured_local_backends(monkeypatch):
    """SEARXNG_URL / CAMOFOX_URL are env-configured keyless local backends
    with no stored selection — they must still count as direct credentials
    so the tool is offered unchecked, never pre-checked (#92647)."""
    monkeypatch.setattr(
        ns,
        "get_env_value",
        lambda name: "http://localhost:9377" if name in ("SEARXNG_URL", "CAMOFOX_URL") else "",
    )
    monkeypatch.setattr(ns, "fal_key_is_configured", lambda: False)
    monkeypatch.setattr(ns, "resolve_openai_audio_api_key", lambda: None)

    direct = ns._get_gateway_direct_credentials()

    assert direct["web"] is True
    assert direct["browser"] is True
    assert direct["image_gen"] is False


def test_prompt_enable_tool_gateway_persists_decline(monkeypatch):
    """Submitting the checklist with a tool left unchecked records it in
    tool_gateway_declined_tools and never pre-checks it again (#92647:
    acceptance was sticky, refusal was not)."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=True))
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "stt": False, "browser": False},
    )
    saved = []
    captured = _capture_checklist(monkeypatch, selected_idx=[])
    monkeypatch.setattr(
        "hermes_cli.config.save_config", lambda cfg: saved.append(dict(cfg)), raising=False
    )

    config = {"model": {"provider": "nous"}}
    assert ns.prompt_enable_tool_gateway(config) == set()

    # First offer: everything pre-checked, decline recorded and saved.
    assert captured["pre_selected"] == list(range(len(captured["items"])))
    declined = config.get("tool_gateway_declined_tools")
    assert isinstance(declined, list) and "web" in declined and "browser" in declined
    assert saved, "decline must be persisted via save_config"

    # Second offer with the recorded declines: nothing is pre-checked.
    captured2 = _capture_checklist(monkeypatch, selected_idx=[])
    monkeypatch.setattr(
        "hermes_cli.config.save_config", lambda cfg: saved.append(dict(cfg)), raising=False
    )
    ns.prompt_enable_tool_gateway(config)
    assert captured2["pre_selected"] == []


def test_prompt_enable_tool_gateway_choosing_declined_tool_clears_decline(monkeypatch):
    """Opting in to a previously-declined tool removes it from the decline
    list, so state tracks the user's latest explicit choice."""
    monkeypatch.setattr(ns, "get_nous_portal_account_info", lambda **kw: _account(logged_in=True, paid=True))
    monkeypatch.setattr(
        ns,
        "_get_gateway_direct_credentials",
        lambda: {"web": False, "image_gen": False, "video_gen": False, "tts": False, "stt": False, "browser": False},
    )
    captured = _capture_checklist(monkeypatch, selected_idx=[0])

    config = {
        "model": {"provider": "nous"},
        "tool_gateway_declined_tools": ["browser", "web"],
    }
    ns.prompt_enable_tool_gateway(config)

    # The first offered key was chosen; it must leave the decline list.
    chosen_key = None
    for key, label in ns._GATEWAY_TOOL_LABELS.items():
        if captured["items"][0].startswith(label):
            chosen_key = key
            break
    assert chosen_key is not None
    assert chosen_key not in config["tool_gateway_declined_tools"]


def test_apply_nous_managed_defaults_writes_video_gen_config(monkeypatch):
    """apply_nous_managed_defaults must store the managed 'nous' selection
    when a Nous subscriber selects video_gen without a direct FAL_KEY."""
    monkeypatch.setattr(tool_backend_helpers, "managed_nous_tools_enabled", lambda **kw: True)
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
    assert config["video_gen"]["provider"] == "nous"
    assert "use_gateway" not in config["video_gen"]


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

    calls = []

    def fake_find_agent_browser(*, validate=True):
        calls.append({"validate": validate})
        return "npx agent-browser"

    monkeypatch.setattr(bt_install, "_find_agent_browser", fake_find_agent_browser)
    monkeypatch.setattr(
        "tools.browser_tool_install._requires_real_termux_browser_install", lambda cmd: False
    )

    assert ns._has_agent_browser() is True
    # A readiness probe must resolve without spawning the daemon.
    assert calls and all(call["validate"] is False for call in calls)


def test_has_agent_browser_false_for_termux_local_bare_npx(monkeypatch):
    """On Termux in local mode the bare npx fallback is not a usable install."""
    _block_legacy_agent_browser_checks(monkeypatch)

    monkeypatch.setattr(
        bt_install,
        "_find_agent_browser",
        lambda *, validate=True: "npx agent-browser",
    )
    monkeypatch.setattr(
        "tools.browser_tool_install._requires_real_termux_browser_install",
        lambda cmd: cmd.strip() == "npx agent-browser",
    )

    assert ns._has_agent_browser() is False


def test_has_agent_browser_false_when_nothing_resolvable(monkeypatch):
    _block_legacy_agent_browser_checks(monkeypatch)

    def raise_not_found(*, validate=True):
        raise FileNotFoundError("agent-browser CLI not found")

    monkeypatch.setattr(bt_install, "_find_agent_browser", raise_not_found)

    assert ns._has_agent_browser() is False


def test_has_agent_browser_import_failure_falls_back_to_path_check(monkeypatch):
    """If tools.browser_tool_install cannot be imported, the old PATH + node_modules
    check must still answer (prior behaviour), not crash."""
    monkeypatch.setitem(sys.modules, "tools.browser_tool_install", None)
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


def test_has_agent_browser_import_failure_falls_back_to_hermes_managed_node_path(
    monkeypatch, tmp_path
):
    """If tools.browser_tool_install cannot be imported, the managed-Node rung must
    still find a runnable agent-browser under the Hermes Node dir even when
    it's absent from the probe process's PATH — the Windows installer shape
    where install succeeded but the GUI still said needs setup."""
    monkeypatch.setitem(sys.modules, "tools.browser_tool_install", None)
    managed_dir = tmp_path / "node"
    managed_dir.mkdir()
    managed_bin = managed_dir / "agent-browser"
    managed_bin.write_text("#!/bin/sh\nexit 0\n")
    managed_bin.chmod(0o755)

    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda cmd, *args, **kwargs: (
            None
            if cmd == "agent-browser" and not kwargs.get("path")
            else real_which(cmd, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        "hermes_constants.with_hermes_node_path", lambda: {"PATH": str(managed_dir)}
    )
    monkeypatch.setattr(
        "hermes_constants.agent_browser_runnable",
        lambda p: bool(p) and str(p) == str(managed_bin),
    )

    assert ns._has_agent_browser() is True


def test_has_agent_browser_import_failure_and_no_binary_is_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "tools.browser_tool_install", None)
    _block_legacy_agent_browser_checks(monkeypatch)

    assert ns._has_agent_browser() is False
