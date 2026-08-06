"""Contract for the durable ``SessionStore`` mirror behind the Claude runtime.

Three layers are pinned down here:

1. **The storage contracts the SDK depends on** — entries are opaque and come
   back deep-equal in append order, a re-delivered batch is stored once,
   deleting a session cascades to its subagent transcripts and drops the
   summary. These run with or without the optional extra, because the
   guarantees are Hermes' SQLite behaviour, not the SDK's.
2. **The SDK's own conformance suite** run against the adapter, which is the
   strongest available evidence the adapter is correct.
3. **Resume through the sanitized transport** — the one thing the SDK does
   *not* do for us, because supplying a custom transport makes it skip
   ``materialize_resume_session()``.

``claude-agent-sdk`` is an optional extra; layers 2 and 3 skip without it.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from agent.claude_session_store import (
    RUNTIME,
    HermesClaudeSessionStore,
    build_claude_session_store,
    is_visible_user_entry,
)
from hermes_state import SessionDB



def _sdk_or_skip():
    return pytest.importorskip(
        "claude_agent_sdk", reason="claude-code extra not installed"
    )


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


@pytest.fixture()
def store(db):
    """Adapter with a stand-in fold, so the storage contracts run SDK-free."""
    return HermesClaudeSessionStore(db, summary_fold=_counting_fold)


def _counting_fold(previous, key, entries):
    """A pure fold with the same shape as the SDK's, for SDK-free tests."""
    data = dict((previous or {}).get("data") or {})
    data["batches"] = int(data.get("batches", 0)) + 1
    data["entries"] = int(data.get("entries", 0)) + len(entries)
    return {"session_id": key["session_id"], "mtime": 0, "data": data}


def _entry(**fields):
    return {"type": "user", **fields}


KEY = {"project_key": "proj", "session_id": "sess"}
SUB = {"project_key": "proj", "session_id": "sess", "subpath": "subagents/agent-1"}


# ---------------------------------------------------------------------------
# Storage contracts (no SDK required)
# ---------------------------------------------------------------------------


def test_entries_come_back_in_append_order(store):
    asyncio.run(store.append(KEY, [_entry(uuid="b", n=1), _entry(uuid="a", n=2)]))
    asyncio.run(store.append(KEY, [_entry(uuid="c", n=3)]))

    loaded = asyncio.run(store.load(KEY))

    # Append order, not uuid order — the SDK replays these as JSONL lines.
    assert [e["uuid"] for e in loaded] == ["b", "a", "c"]


def test_entries_round_trip_deep_equal_including_nested_payloads(store):
    original = _entry(
        uuid="u1",
        message={"role": "user", "content": [{"type": "text", "text": "héllo"}]},
        nested={"a": [1, 2, {"b": None}], "z": True},
    )
    asyncio.run(store.append(KEY, [original]))

    loaded = asyncio.run(store.load(KEY))

    assert loaded == [original]


def test_a_redelivered_batch_is_stored_once(store):
    """A failed append is retried, and a retry can re-deliver what landed."""
    first = [_entry(uuid="u1"), _entry(uuid="u2")]
    asyncio.run(store.append(KEY, first))
    # The retry overlaps the previous batch and adds one new entry.
    asyncio.run(store.append(KEY, first + [_entry(uuid="u3")]))

    loaded = asyncio.run(store.load(KEY))

    assert [e["uuid"] for e in loaded] == ["u1", "u2", "u3"]


def test_entries_without_a_uuid_are_never_deduplicated(store):
    """Titles, tags and mode markers carry no uuid and legitimately repeat."""
    asyncio.run(store.append(KEY, [{"type": "tag", "tag": "x"}]))
    asyncio.run(store.append(KEY, [{"type": "tag", "tag": "x"}]))

    assert len(asyncio.run(store.load(KEY))) == 2


def test_dedup_is_scoped_per_key_because_a_fork_reuses_uuids(store):
    other = {"project_key": "proj", "session_id": "other"}
    asyncio.run(store.append(KEY, [_entry(uuid="shared", side="a")]))
    asyncio.run(store.append(other, [_entry(uuid="shared", side="b")]))

    assert asyncio.run(store.load(KEY))[0]["side"] == "a"
    assert asyncio.run(store.load(other))[0]["side"] == "b"


def test_load_distinguishes_never_written_from_emptied(store):
    assert asyncio.run(store.load(KEY)) is None
    asyncio.run(store.append(KEY, []))
    assert asyncio.run(store.load(KEY)) == []


def test_subagent_transcripts_are_stored_and_listed_separately(store):
    asyncio.run(store.append(KEY, [_entry(uuid="main")]))
    asyncio.run(store.append(SUB, [_entry(uuid="sub")]))

    assert [e["uuid"] for e in asyncio.run(store.load(KEY))] == ["main"]
    assert asyncio.run(store.list_subkeys(KEY)) == ["subagents/agent-1"]
    # A subagent transcript is not a session in its own right.
    assert [s["session_id"] for s in asyncio.run(store.list_sessions("proj"))] == [
        "sess"
    ]


def test_deleting_a_session_cascades_to_subkeys_and_drops_the_summary(store):
    sub2 = {**KEY, "subpath": "subagents/agent-2"}
    survivor = {"project_key": "proj", "session_id": "survivor"}
    asyncio.run(store.append(KEY, [_entry(uuid="m")]))
    asyncio.run(store.append(SUB, [_entry(uuid="s1")]))
    asyncio.run(store.append(sub2, [_entry(uuid="s2")]))
    asyncio.run(store.append(survivor, [_entry(uuid="k")]))

    asyncio.run(store.delete(KEY))

    assert asyncio.run(store.load(KEY)) is None
    assert asyncio.run(store.load(SUB)) is None
    assert asyncio.run(store.load(sub2)) is None
    assert asyncio.run(store.list_subkeys(KEY)) == []
    assert asyncio.run(store.list_session_summaries("proj")) == [] or [
        s["session_id"] for s in asyncio.run(store.list_session_summaries("proj"))
    ] == ["survivor"]
    # Untouched neighbours survive.
    assert asyncio.run(store.load(survivor)) is not None


def test_deleting_one_subpath_leaves_the_session_intact(store):
    sub2 = {**KEY, "subpath": "subagents/agent-2"}
    asyncio.run(store.append(KEY, [_entry(uuid="m")]))
    asyncio.run(store.append(SUB, [_entry(uuid="s1")]))
    asyncio.run(store.append(sub2, [_entry(uuid="s2")]))

    asyncio.run(store.delete(SUB))

    assert asyncio.run(store.load(SUB)) is None
    assert asyncio.run(store.load(sub2)) is not None
    assert asyncio.run(store.load(KEY)) is not None
    assert asyncio.run(store.list_subkeys(KEY)) == ["subagents/agent-2"]


def test_the_summary_fold_runs_once_per_main_batch_and_never_for_subagents(store):
    asyncio.run(store.append(KEY, [_entry(uuid="a"), _entry(uuid="b")]))
    asyncio.run(store.append(KEY, [_entry(uuid="c")]))
    before = asyncio.run(store.list_session_summaries("proj"))[0]["data"]

    asyncio.run(store.append(SUB, [_entry(uuid="s")]))
    after = asyncio.run(store.list_session_summaries("proj"))[0]["data"]

    assert before == {"batches": 2, "entries": 3}
    # A subagent transcript must not contribute to the main session's summary.
    assert after == before


def test_summary_mtime_shares_a_clock_with_the_session_listing(store):
    """The staleness fast-path compares these two directly."""
    asyncio.run(store.append(KEY, [_entry(uuid="a", timestamp="2024-01-01T00:00:00Z")]))

    listed = {s["session_id"]: s["mtime"] for s in asyncio.run(store.list_sessions("proj"))}
    summary = asyncio.run(store.list_session_summaries("proj"))[0]

    assert summary["mtime"] >= listed["sess"]
    # Epoch milliseconds, not seconds — anything smaller would read as 1970.
    assert summary["mtime"] > 1e12


def test_project_keys_are_isolated(store):
    asyncio.run(store.append({"project_key": "A", "session_id": "s"}, [_entry(k="A")]))
    asyncio.run(store.append({"project_key": "B", "session_id": "s"}, [_entry(k="B")]))

    # Same session_id under two project keys must not collide.
    assert asyncio.run(store.load({"project_key": "A", "session_id": "s"}))[0]["k"] == "A"
    assert asyncio.run(store.load({"project_key": "B", "session_id": "s"}))[0]["k"] == "B"
    assert len(asyncio.run(store.list_sessions("A"))) == 1
    assert len(asyncio.run(store.list_sessions("B"))) == 1
    assert asyncio.run(store.list_sessions("never-used")) == []


def test_a_concurrent_append_burst_folds_the_summary_exactly_once_per_batch(store):
    """Read-fold-write on the sidecar must not interleave."""

    async def _burst():
        await asyncio.gather(
            *(store.append(KEY, [_entry(uuid=f"u{i}")]) for i in range(12))
        )

    asyncio.run(_burst())

    summary = asyncio.run(store.list_session_summaries("proj"))[0]
    assert summary["data"] == {"batches": 12, "entries": 12}
    assert len(asyncio.run(store.load(KEY))) == 12


def test_the_store_is_unavailable_without_a_session_db():
    """No DB (persistence off, a background-review fork) means no mirror."""
    assert build_claude_session_store(None) is None


def test_a_downgrade_can_leave_mirror_rows_in_place(db, store):
    """The rollback contract: never require deleting SDK transcript data."""
    asyncio.run(store.append(KEY, [_entry(uuid="a")]))
    db.bind_provider_runtime_session("hermes-1", RUNTIME, "sess", project_key="proj")

    # Dropping the binding is all a rollback needs; the transcript survives.
    db.clear_provider_runtime_session("hermes-1", RUNTIME)

    assert asyncio.run(store.load(KEY)) is not None


# ---------------------------------------------------------------------------
# Visible-user-turn boundary detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry,expected",
    [
        ({"type": "user", "message": {"role": "user", "content": "hi"}}, True),
        (
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
            },
            True,
        ),
        # A tool result is a type="user" entry too, but nothing the user sees.
        (
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "ok"}],
                },
            },
            False,
        ),
        ({"type": "user", "isMeta": True, "message": {"role": "user"}}, False),
        (
            {"type": "user", "isCompactSummary": True, "message": {"role": "user"}},
            False,
        ),
        ({"type": "user", "isSidechain": True, "message": {"role": "user"}}, False),
        ({"type": "assistant", "message": {"role": "assistant"}}, False),
        ("not a dict", False),
    ],
)
def test_only_a_real_user_message_is_a_rewind_boundary(entry, expected):
    assert is_visible_user_entry(entry) is expected


# ---------------------------------------------------------------------------
# The SDK's own conformance suite
# ---------------------------------------------------------------------------


def test_adapter_passes_the_sdk_session_store_conformance_suite(tmp_path):
    _sdk_or_skip()
    from claude_agent_sdk.testing import run_session_store_conformance

    opened: list[SessionDB] = []

    def _make_store():
        # Each contract gets EMPTY backing storage, as the suite requires.
        session_db = SessionDB(db_path=tmp_path / f"conformance-{len(opened)}.db")
        opened.append(session_db)
        return build_claude_session_store(session_db)

    try:
        asyncio.run(run_session_store_conformance(_make_store))
    finally:
        for session_db in opened:
            session_db.close()

    assert len(opened) > 1, "the suite must build a fresh store per contract"


# ---------------------------------------------------------------------------
# Resume through the sanitized transport
# ---------------------------------------------------------------------------


def _seed_resumable_session(store, cwd):
    """Write a main transcript plus a subagent transcript for *cwd*."""
    from claude_agent_sdk import project_key_for_directory

    project_key = project_key_for_directory(cwd)
    session_id = str(uuid.uuid4())
    key = {"project_key": project_key, "session_id": session_id}
    asyncio.run(
        store.append(
            key,
            [
                {
                    "type": "user",
                    "uuid": str(uuid.uuid4()),
                    "sessionId": session_id,
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "message": {"role": "user", "content": "hello"},
                }
            ],
        )
    )
    asyncio.run(
        store.append(
            {**key, "subpath": "subagents/agent-1"},
            [
                {
                    "type": "user",
                    "uuid": str(uuid.uuid4()),
                    "sessionId": session_id,
                    "message": {"role": "user", "content": "sub"},
                }
            ],
        )
    )
    return project_key, session_id


def test_resume_materializes_the_stored_transcript_for_the_sanitized_transport(
    db, tmp_path, monkeypatch
):
    """The blocker PR7 flagged: a custom transport skips materialization.

    The session facade runs it explicitly instead, one step earlier, and hands
    the repointed options to BOTH the transport and the client.
    """
    _sdk_or_skip()
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk._internal.sessions import _get_projects_dir

    from agent.transports.claude_agent_session import ClaudeAgentSession
    from agent.transports.claude_sanitized_transport import (
        build_child_env,
        sanitized_transport_class,
    )

    # A credential that outranks the subscription. It must not survive into the
    # child even on the resume path.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-stripped")

    cwd = str(tmp_path)
    store = build_claude_session_store(db)
    _project_key, session_id = _seed_resumable_session(store, cwd)

    options = ClaudeAgentOptions(
        cwd=cwd,
        env={},
        session_store=store,
        session_store_flush="eager",
        resume=session_id,
        stderr=lambda line: None,
    )
    session = ClaudeAgentSession(options_factory=lambda: options)
    resolved, materialized = asyncio.run(session._materialize(options))
    assert materialized is not None
    # What _connect() does; the facade owns the teardown because the SDK
    # cleans up only what it materialized itself.
    session._materialized = materialized
    try:
        config_dir = materialized.config_dir
        # The CLI can only resume from a local file, so the stored transcript
        # is written back out — including every subagent transcript, which is
        # what list_subkeys() exists for.
        written = {str(p.relative_to(config_dir)) for p in config_dir.rglob("*.jsonl")}
        assert f"projects/{_project_key}/{session_id}.jsonl" in written
        assert (
            f"projects/{_project_key}/{session_id}/subagents/agent-1.jsonl" in written
        )

        transport = sanitized_transport_class()(
            prompt=_empty_prompt(), options=resolved
        )
        transport._cli_path = "/usr/bin/claude"
        command = transport._build_command()
        assert f"--resume={session_id}" in command
        assert "--session-mirror" in command

        child_env = build_child_env(resolved, cwd=cwd)
        assert child_env["CLAUDE_CONFIG_DIR"] == str(config_dir)
        # PR7's guarantee is intact: sanitization still wins.
        assert "ANTHROPIC_API_KEY" not in child_env
        assert "HOME" not in resolved.env

        # The mirror batcher resolves its projects_dir from options.env, so the
        # repointed options must reach the client too or mirrored entries land
        # under a key resume will never look in.
        assert _get_projects_dir(resolved.env) == config_dir / "projects"
    finally:
        asyncio.run(session._cleanup_materialized())
    assert not materialized.config_dir.exists()


def test_a_fresh_session_does_not_redirect_the_config_dir(db, tmp_path):
    """No resume, no temp dir — CLAUDE_CONFIG_DIR passes through as PR7 set it."""
    _sdk_or_skip()
    from claude_agent_sdk import ClaudeAgentOptions

    from agent.transports.claude_agent_session import ClaudeAgentSession

    options = ClaudeAgentOptions(
        cwd=str(tmp_path), env={}, session_store=build_claude_session_store(db)
    )
    session = ClaudeAgentSession(options_factory=lambda: options)

    resolved, materialized = asyncio.run(session._materialize(options))

    assert materialized is None
    assert resolved is options
    assert "CLAUDE_CONFIG_DIR" not in resolved.env


def test_a_fork_rewrites_uuids_rather_than_copying_rows(db, tmp_path):
    """``fork_session_via_store`` must transform, and must leave the source alone."""
    _sdk_or_skip()
    from claude_agent_sdk import fork_session_via_store

    cwd = str(tmp_path)
    store = build_claude_session_store(db)
    project_key, session_id = _seed_resumable_session(store, cwd)
    source_key = {"project_key": project_key, "session_id": session_id}
    before = asyncio.run(store.load(source_key))

    forked = asyncio.run(fork_session_via_store(store, session_id, directory=cwd))

    assert forked.session_id != session_id
    # The parent transcript is immutable across a fork.
    assert asyncio.run(store.load(source_key)) == before
    child = asyncio.run(
        store.load({"project_key": project_key, "session_id": forked.session_id})
    )
    assert child
    child_uuids = {e.get("uuid") for e in child if e.get("uuid")}
    assert not (child_uuids & {e.get("uuid") for e in before})
    assert all(
        e.get("sessionId") in (None, forked.session_id) for e in child
    ), "every entry's sessionId must be rewritten to the fork"


async def _empty_prompt_impl():
    return
    yield {}


def _empty_prompt():
    return _empty_prompt_impl()


def test_stored_entries_are_valid_json_documents(store, db):
    """The store is a mirror, so what goes in must survive a JSONL rewrite."""
    asyncio.run(store.append(KEY, [_entry(uuid="a", text="line\nbreak — dash")]))

    loaded = asyncio.run(store.load(KEY))

    assert json.loads(json.dumps(loaded)) == loaded
