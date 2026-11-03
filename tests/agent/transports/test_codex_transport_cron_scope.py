"""Unified-runtime cron fires share one Codex prompt-cache scope per job (#139434)."""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.codex
    return get_transport("codex_responses")


@pytest.fixture
def execution_context():
    """Binds through ``session_cron.bind_execution`` -- the call ``execute()`` makes -- so the
    admission layout the transport reads is the production one."""
    from gateway.session_cron import _execution, bind_execution

    token = _execution.set(None)
    try:
        yield lambda sid, job_id, execution_id="fire": bind_execution(object(), sid, job_id, execution_id)
    finally:
        _execution.reset(token)


def test_cache_key_stable_for_unified_cron_session_ids(transport, execution_context):
    """A scheduler admission, not the opaque local physical id, scopes cron cache reuse."""
    messages = [{"role": "user", "content": "Hi"}]

    def build(sid):
        return transport.build_kwargs(model="gpt-5.4", messages=messages, tools=[], session_id=sid)

    execution_context("local-first-fire", "job42", "fire-1")
    first = build("local-first-fire")
    execution_context("local-second-fire", "job42", "fire-2")
    second = build("local-second-fire")
    # A session that is not the bound admission (e.g. a delegated child) keeps its own scope.
    child = build("local-child-of-fire")
    execution_context("local-other-job", "job99", "fire-3")
    other = build("local-other-job")

    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["prompt_cache_key"] != other["prompt_cache_key"]
    assert child["prompt_cache_key"] != first["prompt_cache_key"]


@pytest.mark.parametrize("override", [None, "caller-cache-key"])
def test_cron_backend_headers_and_legacy_scope(transport, execution_context, override):
    """Logical routing is stable while physical headers and request content stay intact."""
    from copy import deepcopy

    from agent.transports.codex import _cache_scope_from_session_id

    messages = [
        {"role": "system", "content": "Keep this static instruction."},
        {"role": "user", "content": "Run the scheduled task."},
    ]
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Read a value.",
        "parameters": {"type": "object", "properties": {}},
    }}]
    original = deepcopy((messages, tools))

    def build(sid):
        return transport.build_kwargs(
            model="gpt-5.4", messages=messages, tools=tools, session_id=sid,
            is_codex_backend=True,
            request_overrides={"prompt_cache_key": override} if override else {},
        )

    legacy_sid = "cron_job42_20261015_090000"
    legacy = build(legacy_sid)
    assert _cache_scope_from_session_id(legacy_sid) == "cron_job42"
    for sid in ("local-first-fire", "local-second-fire"):
        execution_context(sid, "job42")
        current = build(sid)
        assert _cache_scope_from_session_id(sid) == "cron_job42"
        assert current["prompt_cache_key"] == legacy["prompt_cache_key"]
        assert current["extra_headers"]["session_id"] == sid
        assert current["extra_headers"]["x-client-request-id"] == current["prompt_cache_key"]
        assert current["instructions"] == messages[0]["content"]
        assert current["input"] == legacy["input"]
        assert current["tools"] == legacy["tools"]
        assert current["tools"][0]["name"] == "lookup"
        if override:
            assert current["prompt_cache_key"] == override
    assert (messages, tools) == original


def test_cron_compression_rotation_resolves_admitted_scope(transport, execution_context, tmp_path):
    """A real lineage lookup reconnects the rotated physical ID to the cron admission."""
    from types import SimpleNamespace

    from agent.prompt_cache_scope import resolve_prompt_cache_scope
    from hermes_state import SessionDB

    root, rotated = "local-admitted-fire", "local-rotated-fire"
    execution_context(root, "job42", "fire-1")
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session(root, source="cli")
        agent = SimpleNamespace(session_id=root, _session_db=db)

        def build():
            scope = resolve_prompt_cache_scope(agent)
            assert scope == root
            return transport.build_kwargs(
                model="gpt-5.4", messages=[{"role": "user", "content": "Hi"}],
                tools=[], session_id=agent.session_id, cache_scope_id=scope,
                is_codex_backend=True,
            )

        before = build()
        db.end_session(root, "compression")
        db.create_session(rotated, source="cli", parent_session_id=root)
        agent.session_id = rotated
        after = build()

    assert before["prompt_cache_key"] == after["prompt_cache_key"]
    assert before["extra_headers"]["session_id"] == root
    assert after["extra_headers"]["session_id"] == rotated
    assert after["extra_headers"]["x-client-request-id"] == after["prompt_cache_key"]


@pytest.mark.parametrize("sid,admission", [
    ("local-without-admission", None),
    ("ordinary-session", None),
    ("local-child", ("local-parent", "job42")),
    ("local-empty-job", ("local-empty-job", "")),
    ("local-blank-job", ("local-blank-job", "  ")),
    ("local-missing-job", ("local-missing-job", None)),
])
def test_unadmitted_or_jobless_sessions_keep_scope(execution_context, sid, admission):
    from agent.transports.codex import _cache_scope_from_session_id

    if admission is not None:
        execution_context(*admission, "fire-1")
    assert _cache_scope_from_session_id(sid) == sid
