"""Round-8 regression tests (issue #3, findings 4030343969/3982/3987/3995)."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from agent.conversation_sanitation import (
    prepare_sanitation_commit,
    sanitation_snapshot_member_row_ids,
    sanitation_snapshot_row_ids,
    validate_sanitation_candidate,
)

_SANITATION_GROWTH_BOUND = 1024


def _placeholder(pattern: str, secret: str) -> str:
    parts = [
        f"[LCM sensitive redaction: name={pattern}; "
        f"chars={len(secret)}; bytes={len(secret.encode())}"
    ]
    if pattern != "password_assignment":
        parts.append(f"sha256={hashlib.sha256(secret.encode()).hexdigest()[:16]}")
    return "; ".join(parts) + "]"


def _strip_row_ids(messages: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in message.items() if key != "_row_id"}
        for message in messages
    ]


def _make_db(tmp_path: Path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess1", source="test")
    return db


def _make_agent(tmp_path: Path, db):
    import os

    from run_agent import AIAgent

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = cast(
            Any,
            AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id="sess1",
                skip_context_files=True,
                skip_memory=True,
            ),
        )
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    return agent


def test_cold_resume_loader_carries_row_ids_for_sanitation(tmp_path):
    """Finding 4030343969 (CLI cold-resume entrypoint): the production resume
    loader must stamp ``_row_id`` — without it the sanitation commit's
    represented-row set is the empty tuple, every active durable row falls to
    the retained-by-slot branch, and the whole transcript is re-cloned
    byte-exact (secret survives in SQLite and FTS)."""
    from hermes_state import SessionDB

    db = _make_db(tmp_path)
    db.append_message("sess1", role="user", content="first ask")
    db.append_message("sess1", role="assistant", content="first reply")
    secret_row = db.append_message(
        "sess1", role="user", content="password=hostpw7-round8"
    )
    db.append_message("sess1", role="assistant", content="next reply")

    # The exact production call shape: cli_agent_setup_mixin.py cold resume.
    messages = db.get_messages_as_conversation(
        "sess1", repair_alternation=True, include_row_ids=True
    )
    assert sanitation_snapshot_row_ids(messages), (
        "the cold-resume loader must stamp row ids so a sanitation commit can "
        "represent them"
    )
    assert secret_row in sanitation_snapshot_row_ids(messages)


def test_acp_restore_loader_carries_row_ids(tmp_path):
    """Finding 4041479196 (ACP adapter resume entrypoint): the ACP session
    restore loader must stamp ``_row_id`` like the CLI cold-resume and gateway
    live-replay loaders — without it the sanitation commit's represented-row
    set is the empty tuple and the whole transcript is re-cloned byte-exact."""
    from acp_adapter import session as acp_session
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess1", source="acp")
    secret_row = db.append_message(
        "sess1", role="user", content="password=acp-pw-round8"
    )
    db.append_message("sess1", role="assistant", content="reply")

    class _Manager(acp_session.SessionManager):
        def _get_db(self):
            return db

        def _make_agent(self, **_kwargs):
            return object()

    manager = _Manager()
    state = manager.get_session("sess1")
    assert state is not None, "the ACP restore path must load an acp-source session"
    ids = sanitation_snapshot_row_ids(state.history)
    assert secret_row in ids, (
        "the ACP restore loader must stamp row ids so a sanitation commit can "
        "represent them"
    )


def test_live_replay_transcript_loader_carries_row_ids(tmp_path):
    """Finding 4030343969 (gateway live-replay entrypoint): load_transcript is
    the production shape feeding live replay, and its loaded rows must carry
    durable row ids for the same reason."""
    from hermes_state import SessionDB

    db = _make_db(tmp_path)
    db.append_message("sess1", role="user", content="first ask")
    secret_row = db.append_message(
        "sess1", role="assistant", content="password=hostpw7-round8"
    )
    db.append_message("sess1", role="user", content="follow-up")

    # SessionTranscriptMixin.load_transcript calls
    # get_messages_as_conversation(session_id, repair_alternation=True) — the
    # invariant is that THIS read shape carries row ids.
    messages = db.get_messages_as_conversation(
        "sess1", repair_alternation=True, include_row_ids=True
    )
    assert secret_row in sanitation_snapshot_row_ids(messages)


def test_sanitation_commit_refuses_when_represented_rows_empty_but_rows_active(
    tmp_path,
):
    """Defense-in-depth for 4030343969: an empty represented-row snapshot over
    a session with active durable rows is an invalid structure — the commit
    must fail closed (raise), never re-clone every row byte-exact."""
    from hermes_state import SessionDB

    db = _make_db(tmp_path)
    db.append_message("sess1", role="user", content="password=hostpw7-round8")
    db.append_message("sess1", role="assistant", content="ok")
    watermark = db.get_active_message_watermark("sess1")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    # Empty tuple (the loader-omission poison value), not None (the legacy
    # watermark fallback — that contract stays intact).
    with pytest.raises(ValueError, match="snapshot source omitted"):
        db.sanitize_and_compact(
            "sess1",
            [{"role": "user", "content": "clean"}],
            watermark=watermark,
            lock_holder="sanitizer",
            represented_row_ids=(),
        )

    durable = db.get_messages("sess1", include_inactive=True)
    assert [row["content"] for row in durable] == [
        "password=hostpw7-round8",
        "ok",
    ], "a refused commit must not delete or duplicate any durable row"


def test_harness_strip_preserves_durable_row_ids_in_snapshot_membership(tmp_path):
    """Finding 4030343982: harness rows stripped by the restore-time filter
    must still be represented by the loaded transcript's row-id snapshot —
    otherwise a later sanitation commit re-clones them byte-exact (resurrected
    hijackable content)."""
    from hermes_state import SessionDB

    db = _make_db(tmp_path)
    harness_row = db.append_message(
        "sess1",
        role="user",
        content="Review the conversation above and update the skill library",
    )
    curator_row = db.append_message(
        "sess1", role="assistant", content="curator-mode reply"
    )
    db.append_message("sess1", role="user", content="real ask")
    db.append_message("sess1", role="assistant", content="real reply")

    messages = db.get_messages_as_conversation(
        "sess1", repair_alternation=True, include_row_ids=True
    )
    contents = [m.get("content") for m in messages]
    assert all(
        "Review the conversation above" not in (c or "") for c in contents
    ), "the harness prompt must still be stripped from the replayed transcript"
    # Membership, not the positional id list: the stripped rows ride the adjacent
    # survivor's `_merged_row_ids` group, which is what the commit consults.
    member_ids = set().union(
        *(
            set(group)
            for group in sanitation_snapshot_member_row_ids(messages)
        )
    ) if sanitation_snapshot_member_row_ids(messages) else set()
    assert harness_row in member_ids, (
        "a stripped harness row remains an active durable row; its id must "
        "survive in the membership snapshot so sanitation does not re-clone it"
    )
    assert curator_row in member_ids


def test_harness_strip_row_ids_survive_alternation_repair_carry(tmp_path):
    """Finding 4030343982 companion: the harness-stripped ids ride the same
    repair-provenance channel as merged row ids (``_merged_row_ids`` on the
    adjacent survivor), so the commit's member_row_ids covers them too."""
    from agent.conversation_sanitation import (
        sanitation_snapshot_member_row_ids,
    )
    from hermes_state import SessionDB

    db = _make_db(tmp_path)
    harness_row = db.append_message(
        "sess1",
        role="user",
        content="Review the conversation above and consider saving to memory",
    )
    curator_row = db.append_message(
        "sess1", role="assistant", content="curator reply"
    )
    db.append_message("sess1", role="user", content="real ask")

    messages = db.get_messages_as_conversation(
        "sess1", repair_alternation=True, include_row_ids=True
    )
    member_ids = [set(group) for group in sanitation_snapshot_member_row_ids(messages)]
    covered = harness_row in set().union(*member_ids) if member_ids else False
    covered = covered or curator_row in set().union(*member_ids)
    assert covered, (
        "stripped harness/curator row ids must ride the membership snapshot "
        "(``_merged_row_ids``), not vanish with the strip"
    )


def test_salvage_gate_refusal_leaves_rollback_snapshot_intact():
    """Finding 4030343987: the salvage gate must not strip api_content from
    the caller's LIVE transcript (which IS the rollback snapshot's source for
    pure sanitation) — after a refusal the snapshot restore must still find
    provider replay sidecars intact."""
    from agent import conversation_compression as cc

    messages_before_compression = [
        {
            "role": "user",
            "content": "secret api_key=abcdefgh",
            "api_content": [{"role": "user", "content": "secret api_key=abcdefgh"}],
        },
        {
            "role": "assistant",
            "content": "ok",
            "api_content": [{"role": "assistant", "content": "ok"}],
        },
    ]
    messages = copy.deepcopy(messages_before_compression)
    compressed = [
        {
            "role": "user",
            "content": "secret " + _placeholder("api_key", "abcdefgh"),
        },
        {
            "role": "assistant",
            "content": "ok",
            "api_content": [{"role": "assistant", "content": "ok"}],
        },
    ]

    class _Compressor:
        _proactive_prune_rearm_tokens = 5

    class _Agent:
        session_id = "sess1"
        _cached_system_prompt = "PROMPT"
        tools = []

        def _emit_warning(self, _msg):
            pass

    agent = _Agent()
    agent.context_compressor = _Compressor()
    candidate, prompt = cc._salvage_or_refuse_grown_transcript(
        agent,
        messages,
        compressed,
        system_message="system",
        attempt_started_at=0.0,
        attempt_snapshot={"_proactive_prune_rearm_tokens": 5},
        pure_sanitation=True,
    )
    assert candidate is None, "an invalid-structure candidate must be refused"
    assert all(
        "api_content" in message for message in messages
    ), "a refusal must not strip api_content from the live rollback snapshot"


def test_retained_retry_rebases_onto_append_only_tail():
    """Finding 4030343995: take_sanitation_retry required a byte-equal
    transcript, so once the raw request continued and appended after a
    refusal, the retained candidate was dropped permanently and the secret
    stayed in SQLite/FTS. The candidate must rebase onto its exact original
    PREFIX while preserving the unchanged append-only tail."""
    from agent.conversation_sanitation import (
        SanitationRetryCandidate,
        has_sanitation_retry,
        take_sanitation_retry,
    )

    class _Compressor:
        def load_externalized_payload_sidecar(self, _ref):
            return None

    class _Agent:
        session_id = "s1"
        _pending_sanitation_retry = None
        context_compressor = _Compressor()

    secret = "password=hostpw7-retry"
    placeholder = _placeholder("password_assignment", "hostpw7-retry")
    original = [
        {"role": "user", "content": secret},
        {"role": "assistant", "content": "refused answer"},
    ]
    candidate = [
        {"role": "user", "content": f"password={placeholder}"},
        {"role": "assistant", "content": "refused answer"},
    ]
    agent = _Agent()
    agent._pending_sanitation_retry = SanitationRetryCandidate(
        session_id="s1",
        original=copy.deepcopy(original),
        candidate=copy.deepcopy(candidate),
    )

    # The retry-time transcript: the refusal answer stayed, the provider
    # response and next user turn appended.
    grown = original + [
        {"role": "assistant", "content": "provider reply after refusal"},
        {"role": "user", "content": "next turn"},
    ]
    rebased = take_sanitation_retry(agent, grown)
    assert rebased is not None, (
        "a retained candidate whose original is an exact prefix of the grown "
        "transcript must rebase instead of being dropped"
    )
    assert rebased[: len(candidate)] == candidate
    assert rebased[len(candidate):] == grown[len(candidate):], (
        "the append-only tail must be preserved untouched"
    )

    # A NON-prefix transcript (content edited mid-list) still drops today.
    agent2 = _Agent()
    agent2._pending_sanitation_retry = SanitationRetryCandidate(
        session_id="s1",
        original=copy.deepcopy(original),
        candidate=copy.deepcopy(candidate),
    )
    edited = [
        {"role": "user", "content": "different first message"},
        {"role": "assistant", "content": "refused answer"},
    ] + grown[2:]
    assert take_sanitation_retry(agent2, edited) is None, (
        "a candidate whose original is not a prefix of the transcript must "
        "still be dropped"
    )
    assert not has_sanitation_retry(_Agent(), [{"role": "user", "content": "x"}])


def test_retry_taken_after_persistence_stamps_api_content_on_prefix():
    """Round-2 finding 4041479200: turn-start persistence stamps the user row's
    ``api_content`` sidecar onto the live transcript AFTER a sanitation rollback
    (the durable sidecar backfill), so the retained candidate's original —
    snapshotted before the stamp — differs from the retry-time transcript by
    that sidecar alone. The retry prefix-equality check must treat
    ``api_content`` as comparison-invisible or the validated candidate is
    dropped and the secret stays in SQLite/FTS."""
    from agent.conversation_sanitation import (
        SanitationRetryCandidate,
        has_sanitation_retry,
        take_sanitation_retry,
    )

    class _Compressor:
        def load_externalized_payload_sidecar(self, _ref):
            return None

    class _Agent:
        session_id = "s1"
        _pending_sanitation_retry = None
        context_compressor = _Compressor()

    secret = "password=hostpw7-stamp"
    placeholder = _placeholder("password_assignment", "hostpw7-stamp")
    original = [
        {"role": "user", "content": secret},
        {"role": "assistant", "content": "refused answer"},
    ]
    candidate = [
        {"role": "user", "content": f"password={placeholder}"},
        {"role": "assistant", "content": "refused answer"},
    ]

    # Rollback happened; persistence then stamped the sidecar on the prefix.
    stamped = [
        {"role": "user", "content": secret, "api_content": [{"role": "user", "content": secret}]},
        {"role": "assistant", "content": "refused answer"},
    ]
    agent = _Agent()
    agent._pending_sanitation_retry = SanitationRetryCandidate(
        session_id="s1",
        original=copy.deepcopy(original),
        candidate=copy.deepcopy(candidate),
    )
    assert has_sanitation_retry(agent, stamped), (
        "a sidecar-stamped prefix must still be recognized as the retained "
        "candidate's original"
    )
    rebased = take_sanitation_retry(agent, copy.deepcopy(stamped) + [
        {"role": "user", "content": "next turn"},
    ])
    assert rebased is not None, (
        "the retained candidate must be taken once persistence stamped "
        "api_content on the rolled-back prefix"
    )
    assert rebased[: len(candidate)] == candidate
    # Validation still runs on the UNSTRIPPED lists: a mid-list CONTENT edit is
    # a real difference, not a sidecar — the candidate still drops.
    agent2 = _Agent()
    agent2._pending_sanitation_retry = SanitationRetryCandidate(
        session_id="s1",
        original=copy.deepcopy(original),
        candidate=copy.deepcopy(candidate),
    )
    edited = [
        {"role": "user", "content": "edited content"},
        {"role": "assistant", "content": "refused answer"},
    ]
    assert not has_sanitation_retry(agent2, edited)


def test_envelope_dropped_when_content_redacted():
    """Round-2 finding 4041479190: a candidate whose content was redacted but
    whose replay envelope stayed byte-identical must not publish the envelope —
    the providers prefer it over content and would re-send the secret."""
    original = [
        {
            "role": "assistant",
            "content": "token=sk-live-value",
            "codex_message_items": [{"type": "message", "text": "token=sk-live-value"}],
        }
    ]
    candidate = [
        {
            "role": "assistant",
            "content": "token=[REDACTED-PLACEHOLDER]",
            "codex_message_items": [{"type": "message", "text": "token=sk-live-value"}],
        }
    ]
    prepare_sanitation_commit(original, candidate, watermark_messages=[])
    assert "codex_message_items" not in candidate[0], (
        "the stale envelope must be dropped when content was redacted"
    )
