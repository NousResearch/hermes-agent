"""Turns rebuilt from the session DB re-attach the newest retained user images (agent/image_replay.py)."""

import json
from pathlib import Path

import pytest

from agent.image_routing import build_native_content_parts
from agent.session_persistence import _durable_content
from agent.turn_context import build_api_messages
from gateway.session_ingress_media import capture_native_media
from hermes_state import SessionDB

_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8cfc0f00f00050201ff2ecf3ca10000000049454e44ae426082")


class _Agent:
    api_mode = "chat_completions"
    provider, model = "custom", "fake-vision"
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = 10_000.0

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _config(**vision):
    from hermes_constants import get_hermes_home  # sandboxed per test by conftest
    cfg = {"agent": {"image_input_mode": vision.pop("mode", "native")}, "vision": vision}
    (get_hermes_home() / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")  # JSON is valid YAML


def _live_turn(tmp_path, name, text, data=_PNG):
    """A native user turn exactly as the CLI/gateway/API server build it: bytes retained under
    native-inputs, then ``build_native_content_parts`` on the retained path."""
    staged = tmp_path / name
    staged.write_bytes(data)
    path = capture_native_media([staged])[0]["path"]
    parts, skipped = build_native_content_parts(text, [path])
    assert not skipped
    return {"role": "user", "content": parts}, Path(path)


def _rebuilt_request(tmp_path, live_turns):
    """Persist ``live_turns`` through the DB projection, reload as the API server does, send a follow-up."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s", source="api_server")
    for i, turn in enumerate(live_turns):
        db.append_message("s", role="user", content=_durable_content(turn["content"]), timestamp=1000.0 + 2 * i)
        db.append_message("s", role="assistant", content=f"reply {i}", timestamp=1001.0 + 2 * i)
    history = db.get_messages_as_conversation("s") + [{"role": "user", "content": "and the expiry date?"}]
    request, _ = build_api_messages(
        _Agent(), history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="")
    return [m for m in request if m["role"] == "user"]


def test_replay_window_is_resolved_once_per_turn(tmp_path):
    """Every request of one turn (each tool-loop iteration) sends the same bytes, even if the config
    changes mid-turn; the next turn picks the change up."""
    _config()
    live, _ = _live_turn(tmp_path, "label.png", "What is on this label?")
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s", source="api_server")
    db.append_message("s", role="user", content=_durable_content(live["content"]), timestamp=1000.0)
    db.append_message("s", role="assistant", content="reply", timestamp=1001.0)
    history = db.get_messages_as_conversation("s") + [{"role": "user", "content": "and the expiry date?"}]
    agent = _Agent()

    def first_user_row():
        request, _ = build_api_messages(
            agent, history, current_turn_user_idx=len(history) - 1,
            ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="")
        return next(m for m in request if m["role"] == "user")["content"]

    assert first_user_row() == live["content"]
    _config(mode="text")
    assert first_user_row() == live["content"]  # same turn: unchanged wire bytes
    agent._current_turn_timestamp += 60  # next turn
    assert isinstance(first_user_row(), str)


@pytest.mark.parametrize("mode,replayed", [("native", True), ("text", False)])
def test_rebuilt_turn_resends_the_image_byte_identical_only_for_native_input(tmp_path, mode, replayed):
    _config(mode=mode)
    live, _ = _live_turn(tmp_path, "label.png", "What is on this label?")

    users = _rebuilt_request(tmp_path, [live])

    assert users[-1]["content"] == "and the expiry date?"
    if replayed:
        # Same bytes as the request that first sent it: the prompt-cache prefix holds.
        assert users[0]["content"] == live["content"]
    else:
        assert isinstance(users[0]["content"], str) and users[0]["content"].endswith("\n[screenshot]")


def test_replay_window_skips_untrusted_bytes_and_typed_markers(tmp_path):
    _config(replay_recent_images=2)
    oldest, _ = _live_turn(tmp_path, "oldest.png", "first photo")
    older, _ = _live_turn(tmp_path, "older.png", "second photo")
    typed = {"role": "user", "content": "I literally typed\n[screenshot]"}  # no attachment hint: never paired
    changed, changed_path = _live_turn(tmp_path, "changed.png", "third photo", data=_PNG + b"\x00")
    newest, _ = _live_turn(tmp_path, "new.png", "fourth photo")
    changed_path.write_bytes(b"tampered")  # retained bytes no longer match their digest

    users = _rebuilt_request(tmp_path, [oldest, older, typed, changed, newest])

    assert users[4]["content"] == newest["content"]
    assert users[3]["content"].endswith("\n[screenshot]")  # tampered bytes are never replayed
    assert users[2]["content"] == "I literally typed\n[screenshot]"
    assert users[1]["content"] == older["content"]
    assert users[0]["content"].endswith("\n[screenshot]")  # outside the 2-image window


def test_replay_keeps_part_order_and_appended_turn_context(tmp_path):
    """Turn-time memory/hook context lands after the image (``_append_multimodal_context``); the
    rebuilt row still carries the image in place (the reload scrubs the memory fence itself)."""
    from agent.turn_context import _append_multimodal_context

    _config()
    live, _ = _live_turn(tmp_path, "label.png", "What is on this label?")
    _append_multimodal_context(_Agent(), live, "recalled memory", "plugin hook context", preflight_compressed=False)
    assert live["content"][-1]["type"] == "text"

    rebuilt = _rebuilt_request(tmp_path, [live])[0]["content"]
    assert rebuilt[:2] == live["content"][:2]
    assert rebuilt[2]["type"] == "text" and rebuilt[2]["text"].endswith("plugin hook context")


@pytest.mark.parametrize("parts", [
    [{"type": "image_url", "image_url": {"url": "IMG"}}, {"type": "text", "text": "what is this?"}],
    [{"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": "IMG", "detail": "low"}}],
], ids=["image-first", "detail-low"])
def test_api_server_image_replays_with_its_position_and_detail(tmp_path, parts):
    """API callers send the image first or pick ``detail``; the API ingress (commit + restore) row
    rebuilt from the DB replays the same parts."""
    import base64

    from gateway.session_api_media import commit_api_images, restore_api_images

    _config()
    url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    parts = [{**p, "image_url": {**p["image_url"], "url": url}} if p["type"] == "image_url" else p for p in parts]
    live = {"role": "user", "content": restore_api_images(parts, commit_api_images(parts))}

    assert _rebuilt_request(tmp_path, [live])[0]["content"] == live["content"]
