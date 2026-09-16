"""Mid-chat /resume honours per-session exclusivity: a session another surface holds is shown read-only,
an unowned one moves this CLI's lease with it (inspired by openai/codex#43253)."""
from cli import HermesCLI
from hermes_cli.active_sessions import active_session_registry_snapshot, try_acquire_active_session


class _FakeDB:
    def __init__(self):
        self.rows = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]

    def get_session(self, sid):
        return {"session_id": sid, "title": None} if sid in ("S", "T") else None

    def get_resume_conversations(self, sid):
        return list(self.rows), list(self.rows)

    def reopen_session(self, sid):
        pass

    def end_session(self, *a, **k):
        pass


def _cli(tmp_path, monkeypatch, printed):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cli = object.__new__(HermesCLI)
    cli.session_id, cli.config, cli._active_session_lease = "mine", {}, None
    cli._console_print = lambda text: printed.append(str(text))
    cli._session_db, cli.agent, cli.conversation_history = _FakeDB(), None, []
    cli._pending_resume_sessions = cli._pending_title = None
    cli._resumed, cli.resume_display = False, "full"
    for name in ("_restore_session_cwd", "_restore_session_yolo", "_restore_session_model"):
        setattr(cli, name, lambda *a, **k: None)
    assert cli._claim_active_session("cli") is True
    return cli


def test_resume_of_session_held_elsewhere_is_read_only(tmp_path, monkeypatch):
    printed: list[str] = []
    cli = _cli(tmp_path, monkeypatch, printed)
    held, message = try_acquire_active_session(
        session_id="S", surface="tui", config={}, metadata={"live_session_id": "other"})
    assert message is None
    try:
        cli._handle_resume_command("/resume S")
        assert cli.session_id == "mine"  # nothing switched
        assert cli.conversation_history == []
        joined = "\n".join(printed)
        assert "already has a live owner (tui" in joined
        assert "Read-only view" in joined and "retry /resume S" in joined
        assert sorted((e["session_id"], e["surface"]) for e in active_session_registry_snapshot()) == [
            ("S", "tui"), ("mine", "cli")]
    finally:
        held.release()
        cli._release_active_session()


def test_resume_of_unowned_session_moves_the_lease(tmp_path, monkeypatch):
    printed: list[str] = []
    cli = _cli(tmp_path, monkeypatch, printed)
    try:
        cli._handle_resume_command("/resume T")
        assert cli.session_id == "T"
        assert [e["session_id"] for e in active_session_registry_snapshot()] == ["T"]
        # A second surface now sees T as owned.
        other, refusal = try_acquire_active_session(
            session_id="T", surface="tui", config={}, metadata={"live_session_id": "other"})
        assert other is None and getattr(refusal, "reason", "") == "SESSION_NOT_OWNED"
    finally:
        cli._release_active_session()
