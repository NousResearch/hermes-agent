"""Classic CLI ``/undo`` on the shared gateway: the transcript rewind commits first, so a failing
file half (``rollback.undo_files`` refused) must never hide it, and with checkpoints off the file
half is a no-op that cannot fail. Real authority, SQLite and dispatch; no inference."""
from types import SimpleNamespace

import pytest


async def _run_undo(tmp_path, monkeypatch, *, checkpoints: bool):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    (tmp_path / "home").mkdir()
    (tmp_path / "user").mkdir()
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli.gateway_chat_commands import run_command
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_mutations import PreparedMutations
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"platform_toolsets": {"cli": []}})
    store = SessionStore(tmp_path / "home" / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id="fixture", instance_id="owner")
    try:
        conn = AuthorityConnection(authority, object(), {"user_id": "human", "profile_id": "fixture"})
        ref = create_local_session(authority, conn.actor, {
            "request_id": "cli", "source": "cli", "cwd": str(tmp_path), "model": "frozen", "toolsets": [],
            "checkpoints": checkpoints})
        for role, text in [("user", "first question"), ("assistant", "first answer"),
                           ("user", "second question"), ("assistant", "second answer")]:
            authority.db.append_message(ref.session_id, role, text)

        class Client:
            async def rpc(self, method, **params):
                if method == "rollback.undo_files":
                    # Another viewer starts a turn between the rewind and the file half.
                    from hermes_state_runtime import admit_session_input, claim_session_input
                    admit_session_input(authority.db, epoch=authority.epoch, principal_id="other-viewer",
                                        session_id=ref.session_id, request_id="other-viewer",
                                        payload={"text": "new turn"})
                    assert claim_session_input(authority.db, epoch=authority.epoch,
                                               session_id=ref.session_id)["status"] == "started"
                reply = await conn.dispatch({"id": 1, "method": method, "params": params})
                if "error" in reply:
                    raise GatewayClientError(reply["error"]["message"])
                return reply["result"]

        view = SimpleNamespace(client=Client(), session_id=ref.session_id, mutations=PreparedMutations(),
                               generation=0, prefill="")
        await run_command(view, "/undo", "")
        return view, [r["content"] for r in authority.db.get_messages(ref.session_id)]
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoints", [True, False])
async def test_undo_reports_the_committed_rewind_even_when_the_file_half_is_refused(
        tmp_path, monkeypatch, capsys, checkpoints):
    view, remaining = await _run_undo(tmp_path, monkeypatch, checkpoints=checkpoints)
    printed = capsys.readouterr().out

    assert remaining == ["first question", "first answer"]
    assert "Undid 1 turn" in printed and view.prefill == "second question"
    assert not view.mutations.pending  # the refresh landed; nothing left to replay
    # Checkpoints on: the refusal is reported. Off: the file half had nothing to do and said nothing.
    assert ("session_busy" in printed) is checkpoints
