"""``dashboard.replay_coalesce_deltas`` in config.yaml reaches the replay ring through the listener builder."""

import pytest

from tui_gateway import event_replay


@pytest.fixture(autouse=True)
def _clean_ring():
    event_replay.reset_replay_state()
    yield
    event_replay.configure(coalesce_deltas=False)
    event_replay.reset_replay_state()


def _replayed_after_streaming(tmp_path, monkeypatch, config_text):
    from hermes_cli.web_server_lifecycle import build_listener_server

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    build_listener_server(lambda scope, receive, send: None, "127.0.0.1", 0, auth_required=False)
    for offset, chunk in enumerate("abc"):
        frame = {"jsonrpc": "2.0", "method": "event", "params": {
            "type": "message.delta", "session_id": "s",
            "payload": {"text": chunk, "turn_id": "t", "segment": 0, "offset": offset}}}
        event_replay._stamp_event(frame)
    return [event["payload"]["text"] for event in event_replay.events_since("s", 0)]


def test_the_key_on_coalesces_a_block_into_one_replayed_delta(tmp_path, monkeypatch):
    assert _replayed_after_streaming(tmp_path, monkeypatch, "dashboard:\n  replay_coalesce_deltas: true\n") == ["abc"]


def test_the_default_config_keeps_one_replayed_delta_per_chunk(tmp_path, monkeypatch):
    assert _replayed_after_streaming(tmp_path, monkeypatch, "model:\n  default: x\n") == ["a", "b", "c"]
