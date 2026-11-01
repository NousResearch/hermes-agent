"""Turn-override seam: optional, bounded, handle-free, armed through the canonical one-turn owner."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway import turn_overrides as T
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, get_plugin_manager

OVERRIDE = {"provider": "moa", "model": "default", "base_url": "moa://local",
            "api_key": "moa-virtual-provider", "api_mode": "chat_completions"}


def _owner(prior=None):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    conversation = runner._session_state("k").conversation
    conversation.model_override = dict(prior) if prior is not None else None
    return runner, conversation


def _source(**kw):
    base = dict(platform=SimpleNamespace(value="telegram"), chat_type="dm", chat_id="1", user_id="1", is_bot=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _call(runner, conversation, event=None, evict=None, full_config=None):
    event = event or SimpleNamespace(text="hello")
    return event, T.apply_turn_override(
        event=event, source=_source(), session_key="k", conversation=conversation,
        snapshot_override=runner._snapshot_session_model_override, evict_cached_agent=evict or MagicMock(),
        full_config=full_config or {"x": {"y": [1]}})


@pytest.fixture
def slot(monkeypatch):
    manager = get_plugin_manager()
    seen = []

    def install(provider, *, enabled=True):
        info = SimpleNamespace(enabled=enabled, error=None, manifest=PluginManifest(name="example-override"))
        monkeypatch.setitem(manager._plugins, "example-override", info)
        monkeypatch.setitem(manager._turn_override_providers, "example", ("example-override", provider))
        import hermes_cli.config as config
        real = config.load_config
        def configured(*a, **kw):
            cfg = real(*a, **kw)
            cfg.setdefault("plugins", {})["enabled"] = ["example-override"]
            return cfg
        monkeypatch.setattr(config, "load_config", configured)
    for key in list(manager._turn_override_providers):
        monkeypatch.delitem(manager._turn_override_providers, key)
    return install, seen


def test_no_provider_means_no_override():
    runner, conversation = _owner()
    event, applied = _call(runner, conversation)
    assert applied is False and conversation.model_override is None and conversation.one_turn_restore is None


@pytest.mark.parametrize("mode", ["raises", "slow", "dict", "bad_keys", "non_str", "empty_reason", "disabled"])
def test_broken_provider_fails_open_without_mutation(slot, monkeypatch, mode):
    install, _ = slot
    monkeypatch.setattr(T, "DECIDE_TIMEOUT_SECONDS", 0.05)
    import threading
    release = threading.Event()
    class Provider:
        def decide(self, request):
            if mode == "raises":
                raise RuntimeError("example")
            if mode == "slow":
                release.wait(5)
                return T.TurnOverride(OVERRIDE, "late")
            if mode == "dict":
                return {"override": OVERRIDE, "reason": "x"}
            if mode == "bad_keys":
                return T.TurnOverride({**OVERRIDE, "extra": "x"}, "x")
            if mode == "non_str":
                return T.TurnOverride({**OVERRIDE, "api_key": 1}, "x")
            if mode == "empty_reason":
                return T.TurnOverride(OVERRIDE, "")
            return T.TurnOverride(OVERRIDE, "x")
    install(Provider(), enabled=(mode != "disabled"))
    runner, conversation = _owner(prior={"provider": "p", "model": "m"})
    started = time.monotonic()
    event, applied = _call(runner, conversation)
    assert time.monotonic() - started < 2.0  # never waits for a hung provider (it blocks up to 5 s)
    release.set()
    assert applied is False
    assert conversation.model_override == {"provider": "p", "model": "m"} and conversation.one_turn_restore is None
    assert not getattr(event, "_moa_disable_after_turn", False)
    if mode == "slow":  # the late result can never arm the already-running turn
        time.sleep(0.1)
        assert conversation.model_override == {"provider": "p", "model": "m"}


def test_valid_provider_arms_canonical_owner_and_gets_no_handles(slot):
    install, seen = slot
    class Provider:
        def decide(self, request):
            seen.append(request)
            request.config["x"]["y"].append(2)  # detached copy: must not reach host config
            return T.TurnOverride(OVERRIDE, "probe")
    install(Provider())
    runner, conversation = _owner(prior={"provider": "p", "model": "m"})
    host_config = {"x": {"y": [1]}}
    evict = MagicMock()
    event, applied = _call(runner, conversation, evict=evict, full_config=host_config)
    assert applied is True and conversation.model_override == OVERRIDE
    assert conversation.one_turn_restore == {"had_override": True, "override": {"provider": "p", "model": "m"}}
    evict.assert_called_once_with("k")
    assert event._moa_disable_after_turn is True and event._turn_override_reason == "probe"
    assert host_config == {"x": {"y": [1]}}
    request = seen[0]
    assert set(vars(request)) == {"text", "platform", "chat_type", "chat_id", "user_id", "is_bot", "is_internal",
                                  "is_command", "session_key", "config"}
    with pytest.raises(Exception):
        request.text = "forged"


def test_earliest_pending_restore_is_preserved(slot):
    install, _ = slot
    class Provider:
        def decide(self, request):
            return T.TurnOverride(OVERRIDE, "probe")
    install(Provider())
    runner, conversation = _owner()
    conversation.one_turn_restore = {"had_override": True, "override": {"provider": "earliest", "model": "m"}}
    _call(runner, conversation)
    assert conversation.one_turn_restore == {"had_override": True, "override": {"provider": "earliest", "model": "m"}}


def test_explicit_one_turn_override_wins_and_provider_not_consulted(slot):
    install, seen = slot
    class Provider:
        def decide(self, request):
            seen.append(request)
            return T.TurnOverride(OVERRIDE, "probe")
    install(Provider())
    runner, conversation = _owner()
    event = SimpleNamespace(text="x", _moa_disable_after_turn=True)
    _, applied = _call(runner, conversation, event=event)
    assert applied is False and seen == [] and conversation.model_override is None


def test_unload_revokes_registration():
    manager = PluginManager()
    class Provider:
        def decide(self, request):
            return None
    registration = PluginContext(PluginManifest(name="example"), manager).register_turn_override_provider("h", Provider())
    assert "h" in manager._turn_override_providers
    registration.dispose()
    assert "h" not in manager._turn_override_providers


def test_registration_rejects_invalid_and_duplicate():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="example"), manager)
    with pytest.raises(ValueError):
        ctx.register_turn_override_provider("bad", object())
    class Provider:
        def decide(self, request):
            return None
    ctx.register_turn_override_provider("h", Provider())
    with pytest.raises(ValueError):
        ctx.register_turn_override_provider("h", Provider())
