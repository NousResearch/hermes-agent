"""Feishu model-picker contracts: native card controls and safe selection routing."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from hermes_constants import VALID_REASONING_EFFORTS
from plugins.platforms.feishu.adapter import FeishuAdapter


def _adapter():
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = SimpleNamespace()
    adapter._owner_profile = "picker-profile"
    adapter._feishu_send_with_retry = AsyncMock(return_value=SimpleNamespace(
        success=lambda: True, data=SimpleNamespace(message_id="picker-message"),
    ))
    adapter._card_response = lambda card_data=None: card_data
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._picker_test_queue = []
    adapter._picker_test_patches = []
    adapter._submit_on_loop = lambda loop, coro: adapter._picker_test_queue.append(coro) or True
    def patch_card(request):
        adapter._picker_test_patches.append(json.loads(request.request_body.content))
        return SimpleNamespace(success=lambda: True)
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(patch=patch_card))))
    async def run_blocking(fn, *args):
        return fn(*args)
    adapter._run_blocking = run_blocking
    return adapter


def _actions(card):
    return [action for element in card["elements"] if element["tag"] == "action"
            for action in element["actions"]]


async def _click(adapter, card, action, option=None):
    control = next(a for a in _actions(card) if a["value"]["hermes_model_picker"] == action)
    response = adapter._handle_model_picker_action(event=_event(control, option=option),
                        action_value=control["value"], loop=asyncio.get_running_loop())
    assert response is None  # SDK acknowledgement must never overwrite an async card PATCH.
    while adapter._picker_test_queue:
        await adapter._picker_test_queue.pop(0)
    return adapter._picker_test_patches[-1]


def _event(action, *, option=None, chat_id="picker-chat", message_id="picker-message", open_id="picker-user"):
    return SimpleNamespace(
        context=SimpleNamespace(open_chat_id=chat_id, open_message_id=message_id),
        operator=SimpleNamespace(open_id=open_id, user_id=None, union_id=None),
        action=SimpleNamespace(value=action["value"], tag=action["tag"], option=option),
    )


async def _send_picker(adapter, providers, callback=None):
    result = await adapter.send_model_picker(
        chat_id="picker-chat", providers=providers,
        current_model="model-0", current_provider="provider-0",
        session_key="agent:picker-profile:feishu:dm:picker-chat",
        on_model_selected=callback or AsyncMock(return_value="Model switched."),
        picker_context={"user_id": "picker-user", "chat_type": "dm", "reasoning_effort": "high"},
    )
    assert result.success
    return json.loads(adapter._feishu_send_with_retry.call_args.kwargs["payload"])


@pytest.mark.asyncio
async def test_duplicate_provider_names_are_numbered_in_menu():
    """One relay, two keys: both groups stay listed and are told apart by route suffix."""
    adapter = _adapter()
    relay_one = {"slug": "custom:relay", "name": "Relay", "models": ["shared-model"], "total_models": 1}
    relay_two = {"slug": "custom:relay-2", "name": "Relay", "models": ["shared-model"], "total_models": 1}

    card = await _send_picker(adapter, [relay_one, relay_two])
    selector = next(action for action in _actions(card) if action["tag"] == "select_static")
    assert [option["text"]["content"] for option in selector["options"]] == ["Relay", "Relay（2）"]

    # Which route is listed first must not change the labels.
    card = await _send_picker(adapter, [relay_two, relay_one])
    selector = next(action for action in _actions(card) if action["tag"] == "select_static")
    assert [option["text"]["content"] for option in selector["options"]] == ["Relay（2）", "Relay"]

    # Unique names stay untouched.
    card = await _send_picker(adapter, [relay_one, {"slug": "custom:other", "name": "Other",
                                                    "models": ["other-model"], "total_models": 1}])
    selector = next(action for action in _actions(card) if action["tag"] == "select_static")
    assert [option["text"]["content"] for option in selector["options"]] == ["Relay", "Other"]


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,title,provider_label,model_label,effort_label", [
    ("zh", "模型与推理强度", "选择 Provider", "选择模型", "推理强度"),
    ("en", "Model & Reasoning", "Select Provider", "Select model", "Reasoning effort"),
])
async def test_card_text_follows_resolved_language(monkeypatch, lang, title, provider_label, model_label, effort_label):
    """The card speaks the profile's display language (both catalogs ship these strings)."""
    monkeypatch.setattr("agent.i18n.get_language", lambda: lang)
    adapter = _adapter()
    card = await _send_picker(adapter, [{"slug": "relay", "name": "Relay",
                                        "models": ["model-a"], "total_models": 1}])
    assert card["header"]["title"]["content"] == title
    selector = next(a for a in _actions(card) if a["tag"] == "select_static")
    assert selector["placeholder"]["content"] == provider_label
    menu_card = await _click(adapter, card, "provider", "0")
    placeholders = [a["placeholder"]["content"] for a in _actions(menu_card) if a["tag"] == "select_static"]
    assert model_label in placeholders and effort_label in placeholders


@pytest.mark.asyncio
async def test_card_language_pinned_when_sent(monkeypatch):
    """An open card keeps the language captured at send time; a later change must not repaint it."""
    monkeypatch.setattr("agent.i18n.get_language", lambda: "zh")
    adapter = _adapter()
    card = await _send_picker(adapter, [{"slug": "relay", "name": "Relay",
                                        "models": ["model-a"], "total_models": 1}])
    assert "选择 Provider" in json.dumps(card, ensure_ascii=False)
    monkeypatch.setattr("agent.i18n.get_language", lambda: "en")
    refreshed = await _click(adapter, card, "provider", "0")
    blob = json.dumps(refreshed, ensure_ascii=False)
    assert "选择模型和档位后" in blob
    assert "Pick a model and effort" not in blob


@pytest.mark.asyncio
async def test_picker_preserves_providers_and_bounds_each_model_menu():
    """Later providers remain reachable without an oversized row of model buttons."""
    adapter = _adapter()
    providers = [
        {"slug": f"provider-{index}", "name": f"Provider {index}",
         "models": [f"model-{number}" for number in range(25)], "total_models": 25}
        for index in range(3)
    ]
    card = await _send_picker(adapter, providers)
    selector = next(action for action in _actions(card) if action["tag"] == "select_static")
    assert len(selector["options"]) == len(providers)
    assert "\n" in card["elements"][0]["content"]
    assert "\\n" not in card["elements"][0]["content"]
    for option in selector["options"]:
        models_card = await _click(adapter, card, "provider", option["value"])
        model_selector = next(action for action in _actions(models_card) if action["tag"] == "select_static")
        assert len(model_selector["options"]) == 20
        assert any(action["value"]["hermes_model_picker"] == "back" for action in _actions(models_card))
        assert all(len(element["actions"]) <= 5 for element in models_card["elements"]
                   if element["tag"] == "action")
        card = await _click(adapter, models_card, "back")


@pytest.mark.asyncio
@pytest.mark.parametrize("effort,scenario", [(e, "session") for e in (*VALID_REASONING_EFFORTS, "none", "keep")] +
                         [("ultra", s) for s in ("global", "global-failure", "rotated", "revoked", "failure")])
async def test_gateway_binds_requester_and_commits_effort_with_model(tmp_path, monkeypatch, effort, scenario):
    """Every visible effort (including ultra) reaches the real gateway commit unchanged."""
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore
    from hermes_cli.model_switch import ModelSwitchResult
    import gateway.run as gateway_run

    home = tmp_path / "picker-home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: old-model\n  provider: custom\nagent:\n  reasoning_effort: high\n")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = _adapter()
    adapter._loop = asyncio.get_running_loop()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    patches = adapter._picker_test_patches
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.FEISHU: adapter}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._running_agents = {}
    runner._session_db = None
    runner._is_user_authorized = lambda *_a, **_kw: True
    runner.session_store = SessionStore(sessions_dir=home / "sessions", config=runner.config)
    source = SessionSource(platform=Platform.FEISHU, chat_id="picker-chat", chat_type="dm", user_id="picker-user")
    key = runner._session_key_for_source(source)
    runner.session_store.get_or_create_session(source)
    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers", lambda **kwargs: [
        {"slug": "provider-0", "name": "Provider", "models": ["model-0", "model-1"]}])
    # The card copies come from the i18n catalogs (zh profile display language here).
    monkeypatch.setattr("agent.i18n.get_language", lambda: "zh")
    # Resolution is the external seam; commit, session storage and reasoning parsing are real.
    async def perform(ctx, model_id, provider, source):
        if scenario == "failure":
            return None, "Resolution rejected."
        return ModelSwitchResult(success=True, new_model=model_id, target_provider=provider, is_global=ctx.persist_global), None
    runner._perform_model_switch = perform
    runner._model_switch_confirmation = AsyncMock(return_value="Model applied.")
    runner._delivery_adapter_for = lambda source: adapter
    if scenario == "global-failure":
        def fail_write(*args, **kwargs):
            raise OSError("synthetic config write failure")
        # The paired model+effort persistence is ONE targeted read-modify-write; fail it there.
        monkeypatch.setattr("utils.atomic_roundtrip_yaml_update_multi", fail_write)

    for effort in (effort,):
        event = MessageEvent(text="/model --global" if scenario.startswith("global") else "/model", source=source)
        assert await runner._handle_model_command(event) is None
        card = json.loads(adapter._feishu_send_with_retry.call_args.kwargs["payload"])
        if scenario.startswith("global"):
            assert "此机器人的默认设置" in card["elements"][0]["content"]
        for action_name, option in (("provider", "0"), ("model", "1"), ("effort", effort)):
            control = next(a for a in _actions(card) if a["value"]["hermes_model_picker"] == action_name)
            if action_name == "effort":
                assert {o["value"] for o in control["options"]} == {"keep", "none", *VALID_REASONING_EFFORTS}
                assert effort in {o["value"] for o in control["options"]}
            assert adapter._on_card_action_trigger(SimpleNamespace(event=_event(control, option=option))) is None
            while adapter._picker_test_queue:
                await adapter._picker_test_queue.pop(0)
            card = patches[-1]
        apply = next(a for a in _actions(card) if a["value"]["hermes_model_picker"] == "apply")
        submitted = []
        def submit(loop, coro):
            submitted.append(coro)
            return True
        adapter._submit_on_loop = submit
        if scenario == "rotated":
            runner.session_store.get_or_create_session(source, force_new=True)
        if scenario == "revoked":
            runner._is_user_authorized = lambda *_a, **_kw: False
        adapter._on_card_action_trigger(SimpleNamespace(event=_event(apply)))
        adapter._on_card_action_trigger(SimpleNamespace(event=_event(apply)))
        assert len(submitted) == 1
        await submitted[0]
        if scenario in {"rotated", "revoked", "failure"}:
            assert not runner._session_model_overrides
            assert not runner._session_reasoning_overrides
            assert "old-model" in (home / "config.yaml").read_text()
            return
        if scenario == "global":
            import hermes_yaml as yaml
            saved = yaml.safe_load((home / "config.yaml").read_text())
            assert saved["model"]["default"] == "model-1"
            assert saved["agent"]["reasoning_effort"] == effort
            assert not runner._session_model_overrides
            assert not runner._session_reasoning_overrides
            return
        assert runner._session_model_overrides[key]["model"] == "model-1"
        if effort == "keep":
            assert key not in runner._session_reasoning_overrides
        else:
            expected = {"enabled": False} if effort == "none" else {"enabled": True, "effort": effort}
            assert runner._session_reasoning_overrides[key] == expected
        restored = SessionStore(sessions_dir=home / "sessions", config=runner.config)
        assert restored.get_model_override(key)["model"] == "model-1"
        if effort != "keep":
            assert effort in patches[-1]["elements"][0]["content"]
        assert not adapter._model_picker_state
    assert "old-model" in (home / "config.yaml").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["user", "chat", "message", "profile", "auth", "expired", "superseded"])
async def test_picker_rejects_unbound_actions(invalid, monkeypatch):
    adapter = _adapter()
    callback = AsyncMock()
    providers = [{"slug": "relay", "models": ["model-1"]}]
    card = await _send_picker(adapter, providers, callback)
    control = _actions(card)[0]
    event = _event(control, option="0")
    before = next(iter(adapter._model_picker_state.values()))
    if invalid == "user":
        event.operator.open_id = "other-user"
    elif invalid == "chat":
        event.context.open_chat_id = "other-chat"
    elif invalid == "message":
        event.context.open_message_id = "other-message"
    elif invalid == "profile":
        adapter._owner_profile = "other-profile"
    elif invalid == "auth":
        adapter.set_authorization_check(lambda *_a, **_kw: False)
    elif invalid == "expired":
        before["created"] -= 1000
    elif invalid == "superseded":
        await _send_picker(adapter, providers, callback)
    adapter._on_card_action_trigger(SimpleNamespace(event=event))
    assert before["provider_index"] is None
    callback.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_picker_adapter_keeps_original_signature(monkeypatch):
    from gateway.slash_commands_model import GatewayModelCommandsMixin
    class LegacyAdapter:
        async def send_model_picker(self, chat_id, providers, current_model, current_provider,
                                    session_key, on_model_selected, metadata=None):
            return SimpleNamespace(success=True)
    runner = GatewayModelCommandsMixin()
    runner._thread_metadata_for_source = lambda *_a: None
    runner._reply_anchor_for_event = lambda *_a: None
    runner._resolve_session_reasoning_config = lambda **_kw: {}
    source = SimpleNamespace(chat_id="picker-chat")
    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers", lambda **_k: [{"slug": "relay", "models": ["model"]}])
    assert await runner._send_model_picker(None, source, LegacyAdapter(), "session-key",
                                          {"current_provider": "relay", "current_model": "model"}, AsyncMock())


@pytest.mark.asyncio
async def test_picker_uses_bound_primary_identity_and_thread():
    adapter = _adapter()
    seen = []
    adapter.set_authorization_check(lambda user_id, chat_type, chat_id, **kw:
                                    seen.append((user_id, kw.get("thread_id"))) or user_id == "tenant-user")
    await adapter.send_model_picker(
        "picker-chat", [{"slug": "relay", "models": ["model"]}], "model", "relay", "session", AsyncMock(),
        picker_context={"user_id": "tenant-user", "user_id_alt": "union-user", "chat_type": "group", "thread_id": "topic-a"})
    card = json.loads(adapter._feishu_send_with_retry.call_args.kwargs["payload"])
    control = _actions(card)[0]
    event = _event(control, option="0", open_id="app-user")
    event.operator.user_id = "tenant-user"
    event.operator.union_id = "union-user"
    adapter._handle_model_picker_action(event=event, action_value=control["value"], loop=asyncio.get_running_loop())
    while adapter._picker_test_queue:
        await adapter._picker_test_queue.pop(0)
    assert seen == [("tenant-user", "topic-a")]
    assert next(iter(adapter._model_picker_state.values()))["provider_index"] == 0


@pytest.mark.asyncio
async def test_delayed_model_callback_cannot_select_a_different_provider():
    adapter = _adapter()
    card = await _send_picker(adapter, [{"slug": "a", "models": ["model-a"]},
                                       {"slug": "b", "models": ["model-b"]}])
    first = await _click(adapter, card, "provider", "0")
    delayed = next(a for a in _actions(first) if a["value"]["hermes_model_picker"] == "model")
    back = await _click(adapter, first, "back")
    await _click(adapter, back, "provider", "1")
    adapter._handle_model_picker_action(event=_event(delayed, option="0"),
                action_value=delayed["value"], loop=asyncio.get_running_loop())
    state = next(iter(adapter._model_picker_state.values()))
    assert state["provider_index"] == 1
    assert state["model_index"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("write_fails", [False, True])
async def test_model_and_effort_are_one_config_write(tmp_path, monkeypatch, write_fails):
    import hermes_yaml as yaml
    import utils
    from gateway.slash_commands_model import _persist_model_switch_to_config
    from hermes_cli.model_switch import ModelSwitchResult
    path = tmp_path / "config.yaml"
    before = {"model": {"default": "old", "provider": "custom:old"},
              "agent": {"reasoning_effort": "low"}, "marker": "keep"}
    path.write_text(yaml.safe_dump(before))
    writes = []
    original = utils.atomic_roundtrip_yaml_update_multi
    def write(target, updates, **kwargs):
        writes.append(dict(updates))
        assert updates["model.default"] == "new"
        assert updates["agent.reasoning_effort"] == "ultra"
        if write_fails:
            raise OSError("synthetic disk failure")
        return original(target, updates, **kwargs)
    monkeypatch.setattr(utils, "atomic_roundtrip_yaml_update_multi", write)
    result = ModelSwitchResult(success=True, new_model="new", target_provider="custom:new")
    if write_fails:
        with pytest.raises(OSError):
            await _persist_model_switch_to_config(result, path, reasoning_effort="ultra")
        assert yaml.safe_load(path.read_text()) == before
    else:
        await _persist_model_switch_to_config(result, path, reasoning_effort="ultra")
        saved = yaml.safe_load(path.read_text())
        assert saved["model"]["default"] == "new"
        assert saved["agent"]["reasoning_effort"] == "ultra"
        assert saved["marker"] == "keep"
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_global_persist_does_not_revert_concurrent_writer(tmp_path, monkeypatch):
    """A /fast-style --global save landing between the selection's read and its write must survive
    that write: the selection only applies its own dotted keys, never a stale whole-doc snapshot."""
    import hermes_yaml as yaml
    import hermes_cli.config as config
    from hermes_cli.model_switch import ModelSwitchResult, persist_model_selection
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"default": "old", "provider": "custom:old"}, "marker": "keep"}))
    real_read = config.read_user_config_raw
    def read_then_other_writer_saves(*args, **kwargs):
        data = real_read(*args, **kwargs)
        # Another --global writer (/fast) persists right after this read, before ours lands.
        interleaved = yaml.safe_load(path.read_text())
        interleaved.setdefault("agent", {})["service_tier"] = "fast"
        path.write_text(yaml.safe_dump(interleaved))
        return data
    monkeypatch.setattr(config, "read_user_config_raw", read_then_other_writer_saves)
    result = ModelSwitchResult(success=True, new_model="new", target_provider="custom:new")
    persist_model_selection(result, path)
    saved = yaml.safe_load(path.read_text())
    assert saved["agent"]["service_tier"] == "fast"
    assert saved["model"]["default"] == "new"
    assert saved["marker"] == "keep"


@pytest.mark.asyncio
async def test_picker_apply_rejected_while_agent_running(tmp_path, monkeypatch):
    """A delayed card click must not hot-swap the agent an in-flight turn is streaming from —
    the typed /model path is rejected mid-run with the same text (run_busy._BUSY_REJECT_TEXT)."""
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore
    from hermes_cli.model_switch import ModelSwitchResult
    import gateway.run as gateway_run

    home = tmp_path / "picker-home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: old-model\n  provider: custom\n")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = _adapter()
    adapter._loop = asyncio.get_running_loop()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    patches = adapter._picker_test_patches
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.FEISHU: adapter}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._running_agents = {}
    runner._session_db = None
    runner._is_user_authorized = lambda *_a, **_kw: True
    runner.session_store = SessionStore(sessions_dir=home / "sessions", config=runner.config)
    source = SessionSource(platform=Platform.FEISHU, chat_id="picker-chat", chat_type="dm", user_id="picker-user")
    key = runner._session_key_for_source(source)
    runner.session_store.get_or_create_session(source)
    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers", lambda **kwargs: [
        {"slug": "provider-0", "name": "Provider", "models": ["model-0", "model-1"]}])
    performed = []
    async def perform(ctx, model_id, provider, source):
        performed.append(model_id)
        return ModelSwitchResult(success=True, new_model=model_id, target_provider=provider,
                                 is_global=ctx.persist_global), None
    runner._perform_model_switch = perform
    runner._model_switch_confirmation = AsyncMock(return_value="Model applied.")
    runner._delivery_adapter_for = lambda source: adapter

    assert await runner._handle_model_command(MessageEvent(text="/model", source=source)) is None
    card = json.loads(adapter._feishu_send_with_retry.call_args.kwargs["payload"])
    for action_name, option in (("provider", "0"), ("model", "1")):
        control = next(a for a in _actions(card) if a["value"]["hermes_model_picker"] == action_name)
        adapter._on_card_action_trigger(SimpleNamespace(event=_event(control, option=option)))
        while adapter._picker_test_queue:
            await adapter._picker_test_queue.pop(0)
        card = patches[-1]
    runner._running_agents[key] = object()  # a turn is streaming for this session
    apply = next(a for a in _actions(card) if a["value"]["hermes_model_picker"] == "apply")
    submitted = []
    adapter._submit_on_loop = lambda loop, coro: submitted.append(coro) or True
    adapter._on_card_action_trigger(SimpleNamespace(event=_event(apply)))
    assert len(submitted) == 1
    await submitted[0]
    assert performed == []  # resolution never ran
    assert "Agent is running" in patches[-1]["elements"][0]["content"]
    assert not runner._session_model_overrides
    assert not runner._session_reasoning_overrides
    assert "old-model" in (home / "config.yaml").read_text()


@pytest.mark.asyncio
async def test_new_session_during_commit_does_not_take_the_old_effort(tmp_path, monkeypatch):
    """`/new` can land while the card's commit awaits metadata; the fresh session must not
    receive the old card's model override or reasoning pin."""
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore
    from hermes_cli.model_switch import ModelSwitchResult
    import gateway.run as gateway_run

    home = tmp_path / "picker-home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: old-model\n  provider: custom\n")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = _adapter()
    adapter._loop = asyncio.get_running_loop()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    patches = adapter._picker_test_patches
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.FEISHU: adapter}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._running_agents = {}
    runner._session_db = None
    runner._is_user_authorized = lambda *_a, **_kw: True
    runner.session_store = SessionStore(sessions_dir=home / "sessions", config=runner.config)
    source = SessionSource(platform=Platform.FEISHU, chat_id="picker-chat", chat_type="dm", user_id="picker-user")
    key = runner._session_key_for_source(source)
    runner.session_store.get_or_create_session(source)
    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers", lambda **kwargs: [
        {"slug": "provider-0", "name": "Provider", "models": ["model-0", "model-1"]}])
    # The stale-session notice also comes from the i18n catalogs.
    monkeypatch.setattr("agent.i18n.get_language", lambda: "zh")
    async def perform(ctx, model_id, provider, source):
        return ModelSwitchResult(success=True, new_model=model_id, target_provider=provider,
                                 is_global=ctx.persist_global), None
    runner._perform_model_switch = perform
    async def confirm(*_a, **_kw):
        # `/new` lands mid-commit: the session rotates (its conversation scope clears) while
        # the commit awaits its metadata lookup.
        runner.session_store.get_or_create_session(source, force_new=True)
        runner._session_model_overrides.clear()
        runner._session_reasoning_overrides.clear()
        return "Model applied."
    runner._model_switch_confirmation = confirm
    runner._delivery_adapter_for = lambda source: adapter

    assert await runner._handle_model_command(MessageEvent(text="/model", source=source)) is None
    card = json.loads(adapter._feishu_send_with_retry.call_args.kwargs["payload"])
    for action_name, option in (("provider", "0"), ("model", "1"), ("effort", "ultra")):
        control = next(a for a in _actions(card) if a["value"]["hermes_model_picker"] == action_name)
        adapter._on_card_action_trigger(SimpleNamespace(event=_event(control, option=option)))
        while adapter._picker_test_queue:
            await adapter._picker_test_queue.pop(0)
        card = patches[-1]
    apply = next(a for a in _actions(card) if a["value"]["hermes_model_picker"] == "apply")
    submitted = []
    adapter._submit_on_loop = lambda loop, coro: submitted.append(coro) or True
    adapter._on_card_action_trigger(SimpleNamespace(event=_event(apply)))
    assert len(submitted) == 1
    await submitted[0]
    assert key not in runner._session_reasoning_overrides
    assert key not in runner._session_model_overrides
    assert "已变化" in patches[-1]["elements"][0]["content"]


@pytest.mark.asyncio
async def test_guarded_model_requires_card_confirmation(monkeypatch):
    """A cost/data-policy guarded pick must confirm in the card before switching — parity with
    the typed guard and the Telegram/Discord pickers' "Switch anyway" step."""
    adapter = _adapter()
    callback = AsyncMock(return_value="Model switched.")
    card = await _send_picker(adapter, [{"slug": "relay", "models": ["guarded-model"]}], callback)
    card = await _click(adapter, card, "provider", "0")
    card = await _click(adapter, card, "model", "0")
    import hermes_cli.model_selection_guards as guards
    def fake_warning(model_name, **kwargs):
        if model_name == "guarded-model":
            return SimpleNamespace(title="Data-Training Tier Warning",
                                   message="This model may train on your data.")
        return None
    monkeypatch.setattr(guards, "combined_selection_warning", fake_warning)

    confirm_card = await _click(adapter, card, "apply")
    callback.assert_not_called()
    assert "train on your data" in confirm_card["elements"][0]["content"]
    assert any(a["value"]["hermes_model_picker"] == "apply_confirm" for a in _actions(confirm_card))

    await _click(adapter, confirm_card, "apply_confirm")
    callback.assert_called_once()

