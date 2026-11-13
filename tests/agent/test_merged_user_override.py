"""A replayed merged row keeps the unanswered request and exact wire sidecar."""

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.message_metadata import TYPED_ASK
from agent.session_persistence import SessionPersistenceMixin, _db_flush_row
from agent.transcript_repair import sync_flushed_message_markers


def _agent_persisting(override):
    agent = object.__new__(SessionPersistenceMixin)
    agent._persist_user_message_idx = 1
    agent._persist_user_message_override = override
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    return agent


def test_replay_flush_row_keeps_prefix_and_exact_api_sidecar():
    from agent.model_metadata import estimate_messages_tokens_rough
    from gateway.message_timestamps import strip_leading_message_timestamps

    wire = "[2026-04-13T17:02:06+0200] can you also run the smoke tests"
    clean, _epoch = strip_leading_message_timestamps(wire)
    messages = [
        {"role": "assistant", "content": "previous response"},
        {"role": "user", "content": "please deploy build 42 to staging"},
        {"role": "user", "content": wire},
    ]
    assert repair_message_sequence(None, messages) == 1
    merged_wire = messages[1]["content"]
    # The recorded prefix is bookkeeping: the estimate prices only the row's content.
    assert estimate_messages_tokens_rough([messages[1]]) == estimate_messages_tokens_rough(
        [{"role": "user", "content": merged_wire}]
    )
    agent = _agent_persisting(clean)
    expected = "please deploy build 42 to staging\n\ncan you also run the smoke tests"
    row = _db_flush_row(agent, messages[1], True)
    assert row["content"] == expected
    assert row["api_content"] == merged_wire
    agent._apply_persist_user_message_override(messages)
    assert messages[1]["content"] == expected


def test_empty_gateway_timestamp_turn_preserves_all_unanswered_rows_on_live_and_replay():
    from gateway.message_timestamps import strip_leading_message_timestamps

    clean, _epoch = strip_leading_message_timestamps("[2026-04-13T17:02:06+0200]")
    assert clean == ""
    for prior in (("request R",), ("request R", "request U")):
        messages = [{"role": "assistant", "content": "previous response"}]
        messages.extend({"role": "user", "content": text} for text in prior)
        messages.append({"role": "user", "content": clean})
        assert repair_message_sequence(None, messages) == len(prior)
        expected = "\n\n".join(prior)
        assert messages[1]["content"] == expected
        agent = _agent_persisting(clean)
        agent._apply_persist_user_message_override(messages)
        assert messages[1]["content"] == expected
        row = _db_flush_row(agent, messages[1], True)
        assert row["content"] == expected
        agent._apply_persist_user_message_override(messages)
        assert messages[1]["content"] == expected
        assert _db_flush_row(agent, messages[1], True)["content"] == expected
        # A later non-empty override on the same row joins like the merge does.
        assert _db_flush_row(_agent_persisting("later"), messages[1], True)["content"] == expected + "\n\nlater"


_STORED_IMAGE_TURN = "first ask\n[screenshot]\n\nPLUGIN NOTE"  # a reloaded stamped multimodal row


@pytest.mark.parametrize("rows", [
    [{"content": _STORED_IMAGE_TURN, TYPED_ASK: "first ask"}, {"content": "second ask"}],
    [{"content": _STORED_IMAGE_TURN, TYPED_ASK: "first ask"}, {"content": ""}],
    [{"content": ""}, {"content": _STORED_IMAGE_TURN, TYPED_ASK: "first ask"}],
], ids=["stamped-then-text", "stamped-then-empty-keeps-stamp", "empty-then-stamped-no-transfer"])
def test_a_merged_row_forgets_the_typed_ask_and_keeps_its_other_metadata(rows):
    """An unanswered stored image turn folded with the next prompt on repair: the row now holds more than
    that one ask, so the stamp goes (the row reads as before stamps existed); other metadata stays. An
    empty turn folded in changes nothing, so the stamp stays."""
    from agent.agent_runtime_helpers import _merge_consecutive_users
    from agent.turn_context import _previous_user_message_text

    def history(stamped):
        return [{"role": "user", "content": row["content"], "display_metadata": {"keep": 1, **(
            {TYPED_ASK: row[TYPED_ASK]} if stamped and TYPED_ASK in row else {})}} for row in rows] + [
            {"role": "assistant", "content": "x"}]
    merged, _ = _merge_consecutive_users(history(True))
    plain, _ = _merge_consecutive_users(history(False))

    if rows[1]["content"] == "":
        assert merged[0]["display_metadata"] == {"keep": 1, TYPED_ASK: "first ask"}
        assert _previous_user_message_text(merged) == "first ask"
    else:
        assert merged[0]["display_metadata"] == {"keep": 1}
        assert _previous_user_message_text(merged) == _previous_user_message_text(plain)


@pytest.mark.parametrize("folded", ["text", "text-unloaded", "empty", None],
                         ids=["fold-drops-stamp", "fold-without-loaded-metadata", "empty-fold-keeps-stamp", "rewrite-keeps-stamp"])
def test_a_folded_rows_flush_removes_its_stored_stamp_and_keeps_a_reaction(tmp_path, monkeypatch, folded):
    """A row-addressed flush rewrites only owned columns, so a fold's dropped stamp would come back on reload
    (previous_message would name only the first ask); the rewrite removes it, keeping a concurrent reaction.
    A rewrite of the same content leaves it, even from a live dict that lacks the row's metadata."""
    from types import SimpleNamespace
    from agent.agent_runtime_helpers import _merge_consecutive_users
    from agent.turn_context import _previous_user_message_text
    from hermes_agent.hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "test")
    row = {"role": "user", "content": _STORED_IMAGE_TURN, "display_metadata": {TYPED_ASK: "first ask"}}
    db.append_messages_batch("s1", [row])
    db.set_message_reaction("s1", row["_row_id"], "👍")
    rows = [row, {"role": "user", "content": "" if folded == "empty" else "second ask"}] if folded else [row]
    merged, _ = _merge_consecutive_users([*rows, {"role": "assistant", "content": "x"}])
    if folded == "text-unloaded":  # the flush hands stored metadata back to a dict that never loaded it
        merged[0]["display_metadata"] = None
    if not folded:  # an owned column changes, content stays; this live dict never loaded the row's metadata
        merged[0]["finish_reason"] = "stop"
        merged[0].pop("display_metadata")
    agent = SimpleNamespace(_persist_user_message_override=None, _persist_user_message_idx=None)
    flushed = [{**_db_flush_row(agent, m, False), **{k: v for k, v in m.items() if k.startswith("_")}} for m in merged]
    db.append_messages_batch("s1", flushed)
    sync_flushed_message_markers(merged, flushed)  # what the agent flush hands back to its live dicts

    reloaded = db.get_messages_as_conversation("s1")
    assert db.get_message_reactions("s1", row["_row_id"])
    if folded:  # restart parity: the reloaded row and the live one after the flush read the same
        assert _previous_user_message_text(reloaded) == _previous_user_message_text(merged)
    if folded in ("text", "text-unloaded"):
        assert TYPED_ASK not in reloaded[0]["display_metadata"]
    else:
        assert reloaded[0]["display_metadata"][TYPED_ASK] == "first ask"
