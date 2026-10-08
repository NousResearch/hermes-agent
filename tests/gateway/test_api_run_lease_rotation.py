"""An API-run registry lease follows a Hermes compression tip and ignores Codex compaction."""

from types import SimpleNamespace

from gateway.platforms import api_server_runs as runs


class _Lease:
    def __init__(self, session_id):
        self.session_id = session_id
        self.released = False
        self.lease_id = "lease-1"


def _run(lease):
    return SimpleNamespace(
        run_id="run_1", session_id="old", session_lease=lease, owner=SimpleNamespace(
            _run_statuses={"run_1": {"status": "running", "session_id": "old"}},
            _set_run_status=lambda run_id, status, **fields: owner_update(run_id, status, fields)))


def owner_update(run_id, status, fields):
    owner_update.calls.append((run_id, status, fields))


def test_rotation_transfers_the_lease_and_codex_compaction_does_not(monkeypatch):
    moved = []

    def _transfer(lease, *, session_id, metadata):
        moved.append((lease, session_id, metadata))
        lease.session_id = session_id
        return True

    import hermes_cli.active_sessions as active_sessions
    monkeypatch.setattr(active_sessions, "transfer_active_session", _transfer)

    lease = _Lease("old")
    owner_update.calls = []
    run = _run(lease)
    agent = SimpleNamespace(event_callback=None)
    runs._bind_run_lease_to_compression(agent, run)

    agent.event_callback("session:compress", {
        "session_id": "old", "old_session_id": "", "in_place": False, "runtime": "codex_app_server"})
    assert moved == []
    assert run.session_id == "old"

    agent.event_callback("session:compress", {
        "session_id": "tip", "old_session_id": "old", "in_place": False})
    assert moved == [(lease, "tip", {"live_session_id": "run_1"})]
    assert run.session_id == "tip"
    assert lease.session_id == "tip"
    assert owner_update.calls == [("run_1", "running", {"session_id": "tip"})]

    agent.event_callback("session:compress", {
        "session_id": "tip", "old_session_id": "old", "in_place": True})
    assert len(moved) == 1


def test_failed_rotation_interrupts_instead_of_continuing_without_ownership(monkeypatch):
    import hermes_cli.active_sessions as active_sessions
    monkeypatch.setattr(active_sessions, "transfer_active_session", lambda *args, **kwargs: False)
    lease = _Lease("old")
    run = _run(lease)
    interrupted = []
    agent = SimpleNamespace(event_callback=None, interrupt=interrupted.append)
    runs._bind_run_lease_to_compression(agent, run)
    agent.event_callback("session:compress", {"session_id": "tip", "old_session_id": "old"})
    assert interrupted == [run.coordination_error]
    assert run.session_id == lease.session_id == "old"
    assert not lease.released
