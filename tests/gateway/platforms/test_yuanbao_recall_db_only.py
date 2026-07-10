"""Yuanbao recall: branch A1 (exact id) and A2 (content-match) against DB-only transcripts.

state.db persists the platform-side ``message_id`` via the
``platform_message_id`` column (added in the salvage of PR #29211) and
``load_transcript`` surfaces it back on each message dict as ``message_id``
— so the recall guard's exact-id match path stays canonical even with the
JSONL file gone.  When a row has no platform id (e.g. agent-processed
@bot messages whose adapter didn't carry a msg_id, or pre-column legacy
rows), recall falls through to content-match.
"""
from types import SimpleNamespace

from gateway.platforms.yuanbao import RecallGuardMiddleware
from gateway.session import SessionSource, SessionStore
from gateway.config import GatewayConfig, Platform


def _pin_db(monkeypatch, tmp_path):
    """Force SessionDB() to write into tmp_path instead of the real ~/.hermes."""
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")


def test_recall_branch_a1_exact_id_match_round_trips_through_db(tmp_path, monkeypatch):
    """A user message persisted with ``message_id`` must round-trip through
    state.db so recall can find and redact it by exact id (branch A1)."""
    _pin_db(monkeypatch, tmp_path)

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-yuanbao-recall-a1"
    store._db.create_session(session_id=sid, source="yuanbao:group:G")
    store.append_to_transcript(sid, {
        "role": "user",
        "content": "sensitive content",
        "timestamp": 1.0,
        "message_id": "platform-msg-abc",
    })
    store.append_to_transcript(sid, {
        "role": "assistant",
        "content": "ack",
        "timestamp": 2.0,
    })

    history = store.load_transcript(sid)
    # The user row must carry its platform id back so the recall guard can
    # match by exact id; the assistant row had no platform id so it should
    # not gain one spuriously.
    user_msg = next(m for m in history if m["role"] == "user")
    assistant_msg = next(m for m in history if m["role"] == "assistant")
    assert user_msg.get("message_id") == "platform-msg-abc"
    assert "message_id" not in assistant_msg

    # Branch A1: locate the row by exact platform id — no content heuristics.
    target = next(
        (m for m in history if m.get("message_id") == "platform-msg-abc"),
        None,
    )
    assert target is not None
    assert target["content"] == "sensitive content"


def test_recall_branch_a2_content_match_when_no_platform_id(tmp_path, monkeypatch):
    """Rows that lack a platform_message_id (e.g. agent-processed @bot
    messages) still match by content as a fallback."""
    _pin_db(monkeypatch, tmp_path)

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    sid = "test-yuanbao-recall-a2"
    store._db.create_session(session_id=sid, source="yuanbao:group:G")
    # No message_id on the dict — simulates an agent-processed message
    # that did not carry the platform msg_id through.
    store.append_to_transcript(sid, {
        "role": "user",
        "content": "sensitive content",
        "timestamp": 1.0,
    })

    history = store.load_transcript(sid)
    assert all("message_id" not in m for m in history)

    # Branch A2: content match recovers the target.
    target = next(
        (m for m in history
         if m.get("role") == "user" and m.get("content") == "sensitive content"),
        None,
    )
    assert target is not None


def _make_adapter(store, group_code: str, from_account: str):
    """Minimal adapter stub: only what RecallGuardMiddleware touches."""
    source = SessionSource(
        platform=Platform.YUANBAO,
        chat_id=(f"group:{group_code}" if group_code else f"direct:{from_account}"),
        chat_type="group" if group_code else "dm",
        user_id=from_account or None,
        thread_id="main" if group_code else None,
    )
    return SimpleNamespace(
        name="yuanbao",
        _session_store=store,
        build_source=lambda **kw: source,
    )


class TestRecallPreservesArchivedTranscript:
    """A recall/redaction must not destroy in-place-compacted (archived) history.

    ``_patch_transcript`` loads the ACTIVE-only view via ``load_transcript()``
    and rewrites it via ``rewrite_transcript()``, which defaults to a full
    DELETE of every row for the session (see gateway/session.py). A session
    that has already been in-place-compacted has soft-archived (active=0)
    rows sitting alongside the live ones; without checking
    ``has_archived_messages()`` first, a recall event on such a session
    silently wipes that archived pre-compaction history — the same
    destructive-rewrite class fixed for the hygiene/``/compress`` paths.
    """

    def test_patch_transcript_preserves_archived_rows(self, tmp_path, monkeypatch):
        _pin_db(monkeypatch, tmp_path)
        config = GatewayConfig()
        store = SessionStore(sessions_dir=tmp_path, config=config)

        sid = "test-yuanbao-recall-archive"
        store._db.create_session(session_id=sid, source="yuanbao:direct:acct1")
        # Simulate a prior in-place compaction: archive the original turns,
        # then insert a compacted summary as the new active set.
        store.append_to_transcript(sid, {"role": "user", "content": "pre-compaction turn"})
        store.append_to_transcript(sid, {"role": "assistant", "content": "pre-compaction reply"})
        store._db.archive_and_compact(sid, [
            {"role": "system", "content": "[compacted summary]"},
        ])
        assert store.has_archived_messages(sid) is True

        # Now a recall event arrives for a message in the (post-compaction)
        # active transcript.
        store.append_to_transcript(sid, {
            "role": "user",
            "content": "sensitive content",
            "message_id": "platform-msg-recall",
        })

        # _patch_transcript resolves the session via get_or_create_session(),
        # which keys off the routing table, not the literal session_id we
        # created directly through store._db. Point it at our session so the
        # test targets the transcript-rewrite behavior, not routing.
        monkeypatch.setattr(store, "get_or_create_session", lambda source: SimpleNamespace(session_id=sid))

        adapter = _make_adapter(store, group_code="", from_account="acct1")
        RecallGuardMiddleware._patch_transcript(
            adapter, recalled_id="platform-msg-recall",
            group_code="", from_account="acct1",
        )

        # The archived pre-compaction turns must survive the rewrite.
        assert store.has_archived_messages(sid) is True
        full_history = store._db.get_messages_as_conversation(sid, include_inactive=True)
        contents = [m["content"] for m in full_history]
        assert "pre-compaction turn" in contents
        assert "pre-compaction reply" in contents

        # And the redaction itself must still have taken effect on the
        # active view.
        active_history = store.load_transcript(sid)
        redacted = next(m for m in active_history if m.get("message_id") == "platform-msg-recall")
        assert redacted["content"] != "sensitive content"

    def test_patch_transcript_no_archived_rows_still_wipes_correctly(self, tmp_path, monkeypatch):
        """Sessions with NO archived history keep the simple full-replace
        behavior (active_only=False is a no-op when nothing is archived)."""
        _pin_db(monkeypatch, tmp_path)
        config = GatewayConfig()
        store = SessionStore(sessions_dir=tmp_path, config=config)

        sid = "test-yuanbao-recall-no-archive"
        store._db.create_session(session_id=sid, source="yuanbao:direct:acct2")
        store.append_to_transcript(sid, {
            "role": "user",
            "content": "sensitive content",
            "message_id": "platform-msg-recall-2",
        })
        assert store.has_archived_messages(sid) is False

        monkeypatch.setattr(store, "get_or_create_session", lambda source: SimpleNamespace(session_id=sid))

        adapter = _make_adapter(store, group_code="", from_account="acct2")
        RecallGuardMiddleware._patch_transcript(
            adapter, recalled_id="platform-msg-recall-2",
            group_code="", from_account="acct2",
        )

        active_history = store.load_transcript(sid)
        assert len(active_history) == 1
        assert active_history[0]["content"] != "sensitive content"
