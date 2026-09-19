"""Surface persistence for autonomous model selections."""

from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner
from tui_gateway import server


SELECTION = {
    "runtime": {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": "https://example.invalid",
        "api_mode": "responses",
    },
    "reasoning_config": {"enabled": True, "effort": "high"},
    "reasoning_effort": "high",
    "message": "Model selected: Sol — high-consequence review. Reasoning: high — route default.",
}


def test_tui_callback_pins_and_persists_selected_runtime(monkeypatch):
    session = {"agent": object()}
    persisted = []
    emitted = []
    markers = []
    monkeypatch.setitem(server._sessions, "model-route", session)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda value: persisted.append(value))
    monkeypatch.setattr(server, "_emit_session_info", lambda sid, value: emitted.append((sid, value)))
    monkeypatch.setattr(server, "_append_model_switch_marker", lambda *args, **kwargs: markers.append((args, kwargs)))
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload: emitted.append((event, sid, payload)))

    callback = server._agent_cbs("model-route")["model_selection_callback"]
    callback(SELECTION)

    assert session["model_override"] == SELECTION["runtime"]
    assert session["create_reasoning_override"] == SELECTION["reasoning_config"]
    assert persisted == [session]
    assert markers[0][1]["detail"] == SELECTION["message"]
    assert ("model-route", session) in emitted
    assert ("message.interim", "model-route", {"text": SELECTION["message"], "already_streamed": False}) in emitted


def test_gateway_callback_pins_and_persists_selected_runtime():
    persisted = []
    delivered = []

    class Store:
        async def set_model_override(self, session_key, runtime):
            persisted.append((session_key, runtime))

    async def deliver(source, message):
        delivered.append((source, message))

    runner = SimpleNamespace(
        _session_model_overrides={},
        _session_reasoning_overrides={},
        async_session_store=Store(),
        _deliver_platform_notice=deliver,
    )
    runner._set_session_reasoning_override = (
        lambda key, value: runner._session_reasoning_overrides.__setitem__(key, value)
    )
    source = SimpleNamespace(platform="telegram")
    context = SimpleNamespace(session_key="telegram:42", source=source)
    turn_runner = TurnRunner(runner, context)

    def schedule(coro, _message):
        import asyncio

        asyncio.run(coro)

    turn_runner._schedule = schedule
    turn_runner._status_live = lambda: True
    turn_runner._model_selection_callback_sync(SELECTION)

    expected_runtime = {**SELECTION["runtime"], "reasoning_effort": "high"}
    assert runner._session_model_overrides["telegram:42"] == expected_runtime
    assert runner._session_reasoning_overrides["telegram:42"] == SELECTION["reasoning_config"]
    assert persisted == [("telegram:42", expected_runtime)]
    assert delivered == [(source, SELECTION["message"])]
