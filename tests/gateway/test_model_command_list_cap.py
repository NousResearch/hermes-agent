"""Gateway ``/model`` list cap: ``--list-by-provider`` (per-user) and
``model.max_models_per_provider`` (operator) must size EVERY listing surface.

The cap has two renderers — the interactive picker (Telegram/Discord/Slack/Matrix) and the
text fallback — and one persisted home (the per-session override store). A cap wired into only
one of them reads as accepted and does nothing; a cap dropped by the next model switch, or by a
restart, breaks the promise the flag makes on screen.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.slash_commands_model import _ModelSwitchContext

# The built-in default, pinned so a silent change to it is a test failure, not a surprise.
DEFAULT_CAP = 50


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _make_runner(store=None):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._running_agents = {}
    runner._session_model_overrides = {}
    runner.session_store = None
    runner._async_session_store = SimpleNamespace(
        _store=None, set_model_override=AsyncMock(return_value=None))
    runner._thread_metadata_for_source = lambda *a, **k: None
    runner._reply_anchor_for_event = lambda *a, **k: None
    runner._evict_cached_agent = lambda *a, **k: None
    runner._channel_override_for = lambda *a, **k: None
    return runner, (runner._async_session_store.set_model_override)


def _make_event(args: str = ""):
    return MessageEvent(
        text=f"/model{(' ' + args) if args else ''}",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Isolated home so config loading is cheap and deterministic (no real creds, no network)."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: gpt-x\n  provider: openrouter\nproviders: {}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


def _write_config(home, extra: str):
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-x\n  provider: openrouter\n" + extra + "providers: {}\n",
        encoding="utf-8")


class _FakePickerAdapter:
    """Adapter whose *type* exposes ``send_model_picker`` (the gate the handler checks)."""

    async def send_model_picker(self, **kwargs):
        return SimpleNamespace(success=True)


def _capture_picker(monkeypatch):
    seen = []
    monkeypatch.setattr("hermes_cli.model_switch_providers.list_picker_providers",
                        lambda **kw: seen.append(kw) or [{"slug": "openrouter", "name": "OpenRouter",
                                                          "is_current": True, "models": ["gpt-x"],
                                                          "total_models": 1}])
    return seen


def _capture_text(monkeypatch):
    seen = []
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers",
                        lambda **kw: seen.append(kw) or [{"slug": "openrouter", "name": "OpenRouter",
                                                          "is_current": True, "models": ["gpt-x"],
                                                          "total_models": 1}])
    return seen


# --------------------------------------------------------------------------- #
# The flag sizes both renderers
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_flag_sizes_the_interactive_picker(isolated_config, monkeypatch):
    """/model --list-by-provider 5 on a picker surface: the picker is the ONLY renderer there, so a
    cap wired just into the text fallback is a no-op the user cannot see (#124162 review P2)."""
    seen = _capture_picker(monkeypatch)
    runner, _ = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}

    assert await runner._handle_model_command(_make_event("--list-by-provider 5")) is None
    assert seen, "picker listing never ran"
    assert seen[0]["max_models"] == 5


@pytest.mark.asyncio
async def test_flag_sizes_the_text_fallback(isolated_config, monkeypatch):
    seen = _capture_text(monkeypatch)
    runner, _ = _make_runner()

    reply = await runner._handle_model_command(_make_event("--list-by-provider 5"))
    assert isinstance(reply, str) and seen
    assert seen[0]["max_models"] == 5


@pytest.mark.asyncio
async def test_default_cap_is_unchanged_when_nothing_is_configured(isolated_config, monkeypatch):
    picker_seen, text_seen = _capture_picker(monkeypatch), _capture_text(monkeypatch)

    runner, _ = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}
    assert await runner._handle_model_command(_make_event()) is None
    assert picker_seen[0]["max_models"] == DEFAULT_CAP

    runner, _ = _make_runner()
    await runner._handle_model_command(_make_event())
    assert text_seen[0]["max_models"] == DEFAULT_CAP


# --------------------------------------------------------------------------- #
# Precedence: flag > session > config > default
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_config_key_sizes_both_renderers(isolated_config, monkeypatch):
    _write_config(isolated_config, "  max_models_per_provider: 7\n")
    picker_seen, text_seen = _capture_picker(monkeypatch), _capture_text(monkeypatch)

    runner, _ = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}
    assert await runner._handle_model_command(_make_event()) is None
    assert picker_seen[0]["max_models"] == 7

    runner, _ = _make_runner()
    await runner._handle_model_command(_make_event())
    assert text_seen[0]["max_models"] == 7


@pytest.mark.asyncio
async def test_flag_beats_the_config_key_for_this_call(isolated_config, monkeypatch):
    _write_config(isolated_config, "  max_models_per_provider: 7\n")
    seen = _capture_picker(monkeypatch)
    runner, _ = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}

    assert await runner._handle_model_command(_make_event("--list-by-provider 3")) is None
    assert seen[0]["max_models"] == 3


@pytest.mark.asyncio
async def test_a_broken_config_value_falls_back_to_the_default(isolated_config, monkeypatch):
    # A non-int (or a bool, which is an int in Python) must not size the list to 1.
    _write_config(isolated_config, "  max_models_per_provider: true\n")
    seen = _capture_text(monkeypatch)
    runner, _ = _make_runner()

    await runner._handle_model_command(_make_event())
    assert seen[0]["max_models"] == DEFAULT_CAP


@pytest.mark.asyncio
async def test_the_stored_session_value_is_used_on_the_next_call(isolated_config, monkeypatch):
    seen = _capture_picker(monkeypatch)
    runner, _ = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}

    assert await runner._handle_model_command(_make_event("--list-by-provider 4")) is None
    assert seen[0]["max_models"] == 4
    # Second bare /model: no flag, so the stored per-session value is the cap.
    assert await runner._handle_model_command(_make_event()) is None
    assert seen[1]["max_models"] == 4


@pytest.mark.asyncio
async def test_the_cap_is_per_user_and_never_written_to_config(isolated_config, monkeypatch):
    """One user's cap rides the per-session override store; config.yaml is the operator's file, so
    a chat command must not edit it (that is what --global is for, and it is refused)."""
    _capture_picker(monkeypatch)
    runner, set_override = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _FakePickerAdapter()}

    await runner._handle_model_command(_make_event("--list-by-provider 6"))

    persisted = [call.args[1] for call in set_override.await_args_list if len(call.args) > 1]
    assert {"list_by_provider": "6"} in persisted, persisted
    assert "max_models_per_provider" not in (isolated_config / "config.yaml").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The cap survives what the reviewer found it did not
# --------------------------------------------------------------------------- #
def _switch_result(model="gpt-y"):
    return SimpleNamespace(
        new_model=model, target_provider="openrouter", api_key="", base_url="", api_mode="chat",
        request_overrides={}, runtime_capabilities={}, provider_label="OpenRouter")


def _ctx(session_key="telegram:12345", persist_global=False):
    return _ModelSwitchContext(session_key=session_key, source=None, config_path=None,
                               persist_global=persist_global)


@pytest.mark.asyncio
async def test_a_model_switch_keeps_the_stored_cap(monkeypatch):
    """/model --list-by-provider 5 then /model gpt-y. _record_model_switch REPLACES the override
    dict, so a key missing from its literal is erased from memory and, via the write-through,
    from sessions.json (#124162 review P2)."""
    monkeypatch.setattr("gateway.slash_commands_model._persist_model_switch_to_config",
                        AsyncMock(return_value=None))
    runner, set_override = _make_runner()
    runner._session_model_overrides["telegram:12345"] = {"list_by_provider": "5"}

    reply = await runner._record_model_switch(_switch_result(), _ctx(), source=None,
                                              one_turn=False, picker=False)

    assert reply is None
    override = runner._session_model_overrides["telegram:12345"]
    assert override["model"] == "gpt-y"
    assert override["list_by_provider"] == "5"
    # The write-through must persist the cap, not the reduced dict.
    persisted = [c.args[1] for c in set_override.await_args_list if len(c.args) > 1]
    assert any(p.get("list_by_provider") == "5" for p in persisted), persisted


@pytest.mark.asyncio
async def test_a_global_model_switch_keeps_the_stored_cap(monkeypatch):
    """--global drops the MODEL route (config.yaml is its one durable authority). The cap has no
    durable home in config.yaml, so dropping the whole override discards a setting the switch is
    not about."""
    persist = AsyncMock(return_value=None)
    monkeypatch.setattr("gateway.slash_commands_model._persist_model_switch_to_config", persist)
    runner, set_override = _make_runner()
    runner._session_model_overrides["telegram:12345"] = {"model": "gpt-x", "list_by_provider": "9"}

    reply = await runner._record_model_switch(_switch_result(), _ctx(persist_global=True),
                                              source=None, one_turn=False, picker=False)

    assert reply is None
    assert runner._session_model_overrides["telegram:12345"] == {"list_by_provider": "9"}
    persisted = [c.args[1] for c in set_override.await_args_list if len(c.args) > 1]
    assert persisted[-1] == {"list_by_provider": "9"}, persisted


@pytest.mark.asyncio
async def test_once_snapshot_keeps_the_cap_the_same_command_set(isolated_config, monkeypatch):
    """``/model gpt-y --once --list-by-provider 5``: the one-turn revert restores the pre-command
    override, which was snapshotted before the cap was stored. Reverting the model is not a revert
    of the list size, so the snapshot must carry the cap."""
    _capture_picker(monkeypatch)
    monkeypatch.setattr("gateway.slash_commands_model._persist_model_switch_to_config",
                        AsyncMock(return_value=None))
    runner, _ = _make_runner()
    claims = []
    runner._claim_one_turn_restore = lambda key, snapshot=None: claims.append((key, snapshot))
    runner._switch_cached_agent_model = lambda *_a, **_k: None
    runner._model_switch_confirmation = AsyncMock(return_value="switched")
    runner._perform_model_switch = AsyncMock(return_value=(_switch_result("gpt-y"), None))
    runner._model_selection_guard_reply = AsyncMock(return_value=(False, None))

    reply = await runner._handle_model_command(_make_event("gpt-y --once --list-by-provider 5"))

    assert reply == "switched"
    assert claims, "the one-turn restore was never armed"
    snapshot = claims[0][1]
    assert snapshot["had_override"] is True
    assert snapshot["override"]["list_by_provider"] == "5", snapshot


@pytest.mark.asyncio
async def test_a_switch_without_a_stored_cap_stores_none(monkeypatch):
    """No cap set: the override dict must not grow a key the store would then persist as noise."""
    monkeypatch.setattr("gateway.slash_commands_model._persist_model_switch_to_config",
                        AsyncMock(return_value=None))
    runner, _ = _make_runner()

    await runner._record_model_switch(_switch_result(), _ctx(), source=None, one_turn=False, picker=False)

    assert "list_by_provider" not in runner._session_model_overrides["telegram:12345"]


def test_a_restart_rehydrates_the_stored_cap():
    """The flag promises to survive a gateway restart. Rehydration rebuilt the override from the
    route keys only, so the persisted cap never came back (#124162 review, third instance)."""
    runner, _ = _make_runner()
    state = SimpleNamespace(conversation=SimpleNamespace(model_override=None))
    runner._session_state = lambda session_key: state
    runner._peek_session_state = lambda session_key: None
    runner.session_store = SimpleNamespace(
        get_model_override=lambda key: {"model": "gpt-x", "provider": "", "list_by_provider": "8"})

    runner._rehydrate_session_model_override("telegram:12345")

    assert state.conversation.model_override["list_by_provider"] == "8"
    assert state.conversation.model_override["model"] == "gpt-x"


def test_a_cap_only_override_does_not_blank_the_configured_model():
    """A cap with no switch (and what a --global switch leaves behind) persists as a cap-only
    override. Rehydrating it must not write model=None: the key would then exist, and the next
    turn's ``.get("model", default)`` would read None instead of the configured model."""
    runner, _ = _make_runner()
    state = SimpleNamespace(conversation=SimpleNamespace(model_override=None))
    runner._session_state = lambda session_key: state
    runner._peek_session_state = lambda session_key: None
    runner.session_store = SimpleNamespace(get_model_override=lambda key: {"list_by_provider": "20"})

    runner._rehydrate_session_model_override("telegram:12345")

    override = state.conversation.model_override
    assert override == {"list_by_provider": "20"}, override
    assert "model" not in override

    # The two readers that resolve the model must fall back, not return the absent key.
    runner._session_model_override = lambda key: override
    model, _kw = runner._apply_session_model_override("telegram:12345", "configured-model", {})
    assert model == "configured-model", model


def test_the_text_preview_slice_honours_the_cap():
    """The text renderer slices the list itself. A cap applied to the listing call only is thrown
    away by the code that prints the list."""
    from gateway.slash_commands_model import _model_provider_listing_lines

    providers = [{"slug": "llamacpp", "name": "llama.cpp", "is_current": True,
                  "models": [f"m{i}" for i in range(10)], "total_models": 10}]

    lines = "\n".join(_model_provider_listing_lines(providers, 7))
    assert "`m5`" in lines and "`m6`" in lines          # 7 shown: m0..m6
    assert "`m7`" not in lines and "`m9`" not in lines  # the tail the cap hides
    assert "(+3 more)" in lines, lines

    # The default is the built-in cap, not a second hardcoded 5.
    from gateway.slash_commands_model import _DEFAULT_LIST_CAP
    assert len("\n".join(_model_provider_listing_lines(providers)).split("`m")) - 1 == 10
    assert _DEFAULT_LIST_CAP == DEFAULT_CAP


def test_a_cap_only_override_is_not_a_model_selection_for_the_api_server():
    """The API server treats a standing /model selection as outranking the request's own
    model/route. A cap pins no route, so it must not swallow them."""
    from gateway.platforms.api_server import _override_selects_model

    assert _override_selects_model({"model": "gpt-x", "list_by_provider": "5"}) is True
    assert _override_selects_model({"provider": "nous"}) is True
    assert _override_selects_model({"list_by_provider": "5"}) is False
    assert _override_selects_model(None) is False
    assert _override_selects_model({}) is False


def test_apply_override_reads_the_cap_from_a_string_and_ignores_junk():
    from gateway.slash_commands_model import _ModelSwitchContext

    ctx = _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False)
    ctx.apply_override({"list_by_provider": "12"})
    assert ctx.list_cap == 12

    ctx = _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False)
    ctx.apply_override({"list_by_provider": "many"})
    assert ctx.list_cap == DEFAULT_CAP

    # A cap-only override (the shape a --global switch leaves behind) must not blank the route.
    ctx = _ModelSwitchContext(session_key="k", source=None, config_path=None, persist_global=False)
    ctx.current_model = "gpt-x"
    ctx.apply_override({"list_by_provider": "3"})
    assert ctx.current_model == "gpt-x" and ctx.list_cap == 3
