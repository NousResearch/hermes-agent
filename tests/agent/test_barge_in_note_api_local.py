"""The barge-in note is API-call-local: it must reach the model without ever being persisted
into a user row. Otherwise a compaction copy of the live turn dict re-materialises the note into
a second user row (same ``message_uid`` and timestamp, different ``content``), the display
collapse keys the two logical turns apart, and auto-TTS speaks the reply twice (#131990).

RED while the live turn dict keeps the note in ``content``. GREEN once the note rides the
``api_content`` sidecar and ``content`` stays durable.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import agent.conversation_loop  # pre-import: avoids a lazily-activated bootstrap mid-test
from agent.context_compressor import _INFLIGHT_TASK_REPLAY_HEADER, ContextCompressor
from agent.session_persistence import durable_user_row_content
from agent.turn_context import (
    _stage_turn_user_message,
    _stamp_api_content_sidecar,
    build_api_messages,
    reanchor_current_turn_user_idx,
)
from hermes_state import SessionDB
from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE

NOTE = SPEECH_INTERRUPTED_NOTE
CLEAN = "What's the weather like in Lisbon this weekend?"
TIMESTAMP = 1791543218.2296236


def _view():
    """A SessionDB instance with no durable backing: only the pure display projection is used."""
    return object.__new__(SessionDB)


def _row(rid, content, timestamp, active, message_uid=None):
    # message_uid/_compressed_summary are read unconditionally by the display dedupe's
    # summary-carrier fold (hermes_state_messages._fold_summary_carrier, 5a282c6c3b).
    return {"id": rid, "role": "user", "content": content, "timestamp": timestamp,
            "tool_call_id": None, "tool_calls": None, "tool_name": None, "active": active,
            "display_kind": None, "display_metadata": None, "message_uid": message_uid,
            "_compressed_summary": None}


def _agent(override):
    agent = types.SimpleNamespace(_pending_cli_user_message=None)
    agent._persist_user_message_override = override
    return agent


def _stage_barge_in_turn():
    """Run the real staging path for a barge-in turn; return the live dict plus the
    (content, api_content) the durable flush would write."""
    api_text = f"{NOTE}\n\n{CLEAN}"
    agent = _agent(CLEAN)
    live, _ = _stage_turn_user_message(agent, api_text, CLEAN, TIMESTAMP, None, None, None)
    durable_content, api_content = durable_user_row_content(
        agent, live, live["content"], live.get("api_content"))
    return live, durable_content, api_content


def test_barge_in_note_never_persists_into_a_second_user_turn():
    live, durable_content, api_content = _stage_barge_in_turn()

    # The note must still reach the model on the wire copy.
    assert NOTE in (api_content or "")
    # The durable transcript copy must not carry the note.
    assert NOTE not in (durable_content or "")

    # Fold the durable row against the compaction copy re-materialised from the live turn dict,
    # exactly as the display projection does: one user message -> ONE logical turn.
    merged = _view()._dedupe_display_generations([
        _row(1, durable_content, TIMESTAMP, active=0, message_uid="u-bar-in"),
        _row(2, live["content"], TIMESTAMP, active=1, message_uid="u-bar-in"),
    ])
    assert len(merged) == 1, "one user message must collapse to ONE display turn"


class _SendAgent:
    api_mode = "chat_completions"
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = TIMESTAMP + 60

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def test_noted_turn_replays_the_same_wire_bytes_on_later_turns():
    """Prompt-cache invariant: moving the note into the sidecar must not change what the model
    was sent. On a LATER turn the historical noted row still goes out with the note."""
    live, _, _ = _stage_barge_in_turn()
    history = [live, {"role": "assistant", "content": "Sunny, around 24C."},
               {"role": "user", "content": "And on Monday?"}]

    request, _ = build_api_messages(
        _SendAgent(), history, current_turn_user_idx=2, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )

    assert request[0]["role"] == "user"
    assert request[0]["content"] == f"{NOTE}\n\n{CLEAN}"
    assert "api_content" not in request[0]


def test_injected_context_composes_onto_the_noted_sidecar():
    """Memory/plugin context is appended to the staged API variant, not to the clean content:
    the model gets note + prompt + context, the transcript keeps the prompt alone."""
    live, _, _ = _stage_barge_in_turn()
    agent = _agent(CLEAN)
    agent.session_id = "s1"
    agent._session_db = None

    _stamp_api_content_sidecar(agent, [live], 0, "MEMORY: the user lives in Lisbon", "",
                               preflight_compressed=False)

    assert live["content"] == CLEAN
    assert live["api_content"].startswith(f"{NOTE}\n\n{CLEAN}\n\n")
    assert "MEMORY: the user lives in Lisbon" in live["api_content"]


def test_moa_turn_keeps_injected_context_on_a_staged_sidecar():
    """MoA turns are never stamped by the prologue, so a staged sidecar is note-only: the
    wire copy must compose the injections onto it, or memory/plugin context is silently
    dropped on exactly the turns that stage a note."""
    live, _, _ = _stage_barge_in_turn()

    request, _ = build_api_messages(
        _SendAgent(), [live], current_turn_user_idx=0,
        ext_prefetch_cache="MEMORY: the user lives in Lisbon",
        plugin_user_context="PLUGIN: current task is voice cloning",
        moa_config={"models": ["a", "b"]}, active_system_prompt="",
    )

    wire = request[0]["content"]
    assert NOTE in wire
    assert "MEMORY: the user lives in Lisbon" in wire
    assert "PLUGIN: current task is voice cloning" in wire
    assert "api_content" not in request[0]
    # The durable transcript keeps the clean prompt, and the staged bytes are untouched.
    assert live["content"] == CLEAN
    assert live["api_content"] == f"{NOTE}\n\n{CLEAN}"

    # Byte-stability: a second pass with nothing injected replays the staged bytes as-is.
    quiet, _ = build_api_messages(
        _SendAgent(), [live], current_turn_user_idx=0, ext_prefetch_cache="",
        plugin_user_context="", moa_config={"models": ["a", "b"]}, active_system_prompt="",
    )
    assert quiet[0]["content"] == f"{NOTE}\n\n{CLEAN}"


def test_codex_turn_keeps_injected_context_on_a_staged_sidecar():
    """codex_app_server turns are never stamped by the prologue either (same skip
    predicate as MoA): with a staged sidecar the wire copy must compose the injections,
    or memory/plugin context is dropped on codex turns that stage a note."""
    live, _, _ = _stage_barge_in_turn()

    class _CodexSendAgent(_SendAgent):
        api_mode = "codex_app_server"

    request, _ = build_api_messages(
        _CodexSendAgent(), [live], current_turn_user_idx=0,
        ext_prefetch_cache="MEMORY: the user lives in Lisbon",
        plugin_user_context="PLUGIN: current task is voice cloning",
        moa_config=None, active_system_prompt="",
    )

    wire = request[0]["content"]
    assert NOTE in wire
    assert "MEMORY: the user lives in Lisbon" in wire
    assert "PLUGIN: current task is voice cloning" in wire
    assert live["content"] == CLEAN
    assert live["api_content"] == f"{NOTE}\n\n{CLEAN}"

    # Byte-stability: nothing injected -> the staged bytes go out untouched.
    quiet, _ = build_api_messages(
        _CodexSendAgent(), [live], current_turn_user_idx=0, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    assert quiet[0]["content"] == f"{NOTE}\n\n{CLEAN}"


def test_noted_turn_reanchors_on_the_staged_dict_despite_a_later_user_row():
    """Post-staging ``content`` is the CLEAN text while this turn's text is the noted one:
    the sidecar must exact-match, or the reanchor falls through to the last user-originated
    row and anchors the injection stamp on the wrong dict after compression."""
    live, _, _ = _stage_barge_in_turn()
    noted = f"{NOTE}\n\n{CLEAN}"
    assert live["content"] == CLEAN and live["api_content"] == noted
    messages = [
        live,
        {"role": "assistant", "content": "Sunny, around 24C."},
        {"role": "user", "content": "an older restored prompt"},
    ]

    assert reanchor_current_turn_user_idx(messages, noted) == 0
    # Control: the clean persist text still anchors the same dict.
    assert reanchor_current_turn_user_idx(messages, CLEAN) == 0


def test_compaction_replay_restates_the_task_without_the_note():
    """The in-flight task replay after a compaction restates ``content``; the one-shot note was
    for the turn it interrupted and must not be re-delivered after the handoff."""
    live, _, _ = _stage_barge_in_turn()
    tool_turns = []
    for i in range(40):
        tool_turns.append({"role": "assistant", "content": f"step {i}",
                           "tool_calls": [{"id": f"c{i}", "function": {"name": "terminal", "arguments": "{}"}}]})
        tool_turns.append({"role": "tool", "tool_call_id": f"c{i}", "content": ("tool output " * 200) + f" {i}"})
    transcript = [{"role": "system", "content": "You are Hermes."}, live, *tool_turns]

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Summary\nChecked the forecast sources."
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)
    compressor.tail_token_budget = 500
    with patch("agent.context_compressor.call_llm", return_value=response):
        compressed = compressor.compress(transcript, current_tokens=200_000, force=True)

    replay_idx = next(i for i, m in enumerate(compressed)
                      if m.get("role") == "user" and _INFLIGHT_TASK_REPLAY_HEADER in str(m.get("content")))
    replay = compressed[replay_idx]
    assert CLEAN in replay["content"]
    assert NOTE not in replay["content"]
    assert NOTE not in str(replay.get("api_content") or "")

    # On the wire: the restated task goes out without the note, while the protected original row
    # still replays the bytes the model was sent (prompt-cache prefix).
    request, _ = build_api_messages(
        _SendAgent(), compressed, current_turn_user_idx=replay_idx, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire_users = [m["content"] for m in request if m.get("role") == "user"]
    assert NOTE not in wire_users[-1]
    assert wire_users[0] == f"{NOTE}\n\n{CLEAN}"


def test_genuine_repeat_of_identical_text_still_speaks_twice():
    """Control: the same text sent as two GENUINE turns (distinct timestamps) must remain two
    user turns -- a fix that swallows real repeats is a regression."""
    merged = _view()._dedupe_display_generations([
        _row(1, "ok", 100.0, active=0, message_uid="u-genuine-1"),
        _row(2, "ok", 200.0, active=1, message_uid="u-genuine-2"),
    ])
    assert len(merged) == 2
