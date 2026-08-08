import pytest


def test_oneshot_resume_fails_before_reopen_when_recorded_cwd_is_missing(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    reopened = []

    class FakeSessionDB:
        def get_session(self, session_id):
            return {"id": session_id, "cwd": "/recorded/workspace"}

        def resolve_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, session_id):
            return session_id

        def get_resume_conversations(self, _session_id):
            return ([{"role": "user", "content": "prior context"}], [])

        def reopen_session(self, session_id):
            reopened.append(session_id)

    monkeypatch.setattr(oneshot_mod.os.path, "isdir", lambda _path: False)

    with pytest.raises(
        FileNotFoundError,
        match="Recorded session working directory is unavailable",
    ):
        oneshot_mod._load_oneshot_resume(
            FakeSessionDB(),
            resume_session_id="session-1",
            continue_last=False,
            restore_resume_cwd=True,
        )

    assert reopened == []


def test_oneshot_resume_allows_explicit_cwd_restore_opt_out(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    reopened = []

    class FakeSessionDB:
        def get_session(self, session_id):
            return {"id": session_id, "cwd": "/recorded/workspace"}

        def resolve_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, session_id):
            return session_id

        def get_resume_conversations(self, _session_id):
            return ([{"role": "user", "content": "prior context"}], [])

        def reopen_session(self, session_id):
            reopened.append(session_id)

    monkeypatch.setattr(oneshot_mod.os.path, "isdir", lambda _path: False)

    session_id, history = oneshot_mod._load_oneshot_resume(
        FakeSessionDB(),
        resume_session_id="session-1",
        continue_last=False,
        restore_resume_cwd=False,
    )

    assert session_id == "session-1"
    assert history == [{"role": "user", "content": "prior context"}]
    assert reopened == ["session-1"]
