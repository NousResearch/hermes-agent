"""A viewer's /compress on a messaging chat (#144422).

``session.mutate`` ``compress`` on a Telegram chat a viewer resumed was refused with
``runtime_coordination_required``: the prepare pass reads the LOCAL frozen policy, and a messaging chat
has none, so nothing was summarized and the transcript stayed unchanged. The mutation now runs the
chat's own ``/compress`` on the gateway that owns its lane. Real GatewayRunner, SessionAuthority and
SQLite store; the summarizer is stubbed — the compression core has its own tests.
"""

import pytest

from gateway.run import GatewayRunner
from gateway.slash_commands_session import ManualCompressionOutcome


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


async def _compressed_chat(tmp_path, *, messages=6):
    """A Telegram chat compressed once; the viewer resumes it by its segment id, as history pickers do."""
    import json
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    runner = GatewayRunner(GatewayConfig())
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="owner"
    )
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="4242", user_id="u-4242", chat_type="dm"
    )
    root = authority.register(source).session_id
    route = runner.session_store._generate_session_key(source)
    db = authority.db
    db.append_message(root, "user", "first")
    db.publish_compression_child(
        parent_session_id=root,
        child_session_id="seg-1",
        source="telegram",
        messages=[{"role": "user", "content": "summary"}],
        require_compression_lease=False,
    )
    assert (
        runner.session_store.advance_compression_session(route, root, "seg-1")
        is not None
    )
    db.record_gateway_session_peer(
        "seg-1",
        source="telegram",
        user_id=source.user_id,
        session_key=route,
        chat_id=source.chat_id,
        chat_type="dm",
        origin_json=json.dumps(source.to_dict()),
    )
    for index in range(messages):
        db.append_message(
            "seg-1", "user" if index % 2 == 0 else "assistant", f"turn {index}"
        )
    # The segment's own fences moved on (its turns ran there); the viewer holds the mount's.
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET runtime_revision=7, runtime_generation=3 WHERE id='seg-1'"
        )
    )
    viewer = AuthorityConnection(
        authority,
        object(),
        {
            "provider": "local",
            "user_id": "viewer",
            "profile_id": "default",
            "instance_id": "owner",
            "capabilities": ["session:read", "session:control"],
        },
    )
    resumed = await viewer.dispatch(
        {"id": 1, "method": "session.resume", "params": {"session_id": "seg-1"}}
    )
    assert resumed.get("result", {}).get("session_id") == root, resumed
    return runner, authority, source, route, viewer, root, resumed["result"]


def _mutate(sid, fences, payload, request_id):
    return {
        "id": request_id,
        "method": "session.mutate",
        "params": {
            "session_id": sid,
            "request_id": request_id,
            "operation": "compress",
            "payload": payload,
            "expected_revision": fences["revision"],
            "expected_generation": fences["execution_generation"],
        },
    }


def _stub_compress(runner, calls):
    """The chat's own manual compression, without a provider: report a committed rewrite."""

    async def _run(source, session_entry, history, request):
        calls.append(
            (source, session_entry.session_id, len(history), request.focus_topic)
        )
        return ManualCompressionOutcome("Compressed: 45 → 12 messages", "compressed")

    runner._run_manual_compression = _run


@pytest.mark.asyncio
async def test_a_viewer_compresses_a_messaging_chat_instead_of_being_refused(tmp_path):
    runner, _authority, _source, _route, viewer, root, fences = await _compressed_chat(
        tmp_path
    )
    calls = []
    _stub_compress(runner, calls)
    try:
        result = await viewer.dispatch(_mutate(root, fences, {}, "compress-root"))

        assert "error" not in result, result
        assert result["result"]["status"] == "compressed", result
        # The lane's current conversation (the compression tip) is what got compressed.
        assert result["result"]["target_session_id"] == "seg-1", result
        assert result["result"]["previous_target_session_id"] == "seg-1", result
        assert result["result"]["lines"] == ["Compressed: 45 → 12 messages"], result
        assert len(calls) == 1 and calls[0][0].platform.value == "telegram", calls
        assert calls[0][2] >= 4, calls  # the chat's own transcript, not the named row's
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_segment_id_resolves_to_the_mount_the_chat_is_compressed_on(tmp_path):
    """The app names whatever row it shows; a compression segment routes to its lineage root's lane."""
    runner, _authority, _source, _route, viewer, _root, fences = await _compressed_chat(
        tmp_path
    )
    calls = []
    _stub_compress(runner, calls)
    try:
        result = await viewer.dispatch(_mutate("seg-1", fences, {}, "compress-segment"))

        assert "error" not in result, result
        assert result["result"]["status"] == "compressed", result
        assert len(calls) == 1, calls
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_preview_reads_the_chat_and_summarizes_nothing(tmp_path):
    runner, _authority, _source, _route, viewer, root, fences = await _compressed_chat(
        tmp_path
    )
    calls = []
    _stub_compress(runner, calls)
    try:
        result = await viewer.dispatch(
            _mutate(root, fences, {"preview": True}, "compress-preview")
        )

        assert "error" not in result, result
        assert result["result"]["status"] == "preview", result
        assert result["result"]["lines"], result
        assert calls == [], calls
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_local_compress_is_not_routed_to_the_chat_path(tmp_path):
    """The routing is for messaging chats only: a LOCAL row keeps its own (prepare-pass) refusal."""
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    runner = GatewayRunner(GatewayConfig())
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="owner"
    )
    source = SessionSource(
        platform=Platform.LOCAL, chat_id="local-1", user_id="uid:1000"
    )
    sid = authority.register(source).session_id
    authority.db.append_message(sid, "user", "hi")
    calls = []
    _stub_compress(runner, calls)
    viewer = AuthorityConnection(
        authority,
        object(),
        {
            "provider": "local",
            "user_id": "viewer",
            "profile_id": "default",
            "instance_id": "owner",
            "capabilities": ["session:read", "session:control"],
        },
    )
    try:
        result = await viewer.dispatch(
            {
                "id": 2,
                "method": "session.mutate",
                "params": {
                    "session_id": sid,
                    "request_id": "compress-local",
                    "operation": "compress",
                    "payload": {},
                    "expected_revision": 0,
                    "expected_generation": 0,
                },
            }
        )

        assert calls == [], calls
        assert result.get("error", {}).get("data", {}).get("reason") in {
            "runtime_coordination_required",
            "permission_denied",
        }, result
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
