"""Pasted session IDs resolve through the real CLI lookup and SQLite store."""
from argparse import Namespace

import pytest


@pytest.fixture
def resume_store(tmp_path, monkeypatch):
    import hermes_state

    real_db = hermes_state.SessionDB
    path = tmp_path / "resume.db"
    class LocalSessionDB(real_db):
        def __init__(self, **kwargs):
            super().__init__(db_path=path, **kwargs)

    db = LocalSessionDB()
    monkeypatch.setattr(hermes_state, "SessionDB", LocalSessionDB)
    yield db
    db.close()


@pytest.mark.parametrize("padding", [("", ""), (" ", ""), ("", " "), ("\t", "\n")])
def test_resume_padded_session_id(resume_store, padding):
    from hermes_cli.main import _resolve_session_by_name_or_id

    session_id = "20261009_130000_abc123"
    resume_store.create_session(session_id, source="cli")
    before, after = padding
    assert _resolve_session_by_name_or_id(before + session_id + after) == session_id


@pytest.mark.parametrize("value", ["", " \t\n", "missing", "20261009_130000", "20261009_ 130000_abc123"])
def test_missing_blank_prefix_and_internal_space_do_not_select_session(resume_store, value):
    from hermes_cli.main import _resolve_session_by_name_or_id

    resume_store.create_session("20261009_130000_abc123", source="cli")
    assert _resolve_session_by_name_or_id(value) is None


def test_title_fallback_and_exact_id_precedence(resume_store):
    from hermes_cli.main import _resolve_session_by_name_or_id

    resume_store.create_session("exact-id", source="cli")
    resume_store.create_session("titled-id", source="cli")
    resume_store.set_session_title("titled-id", "My Project")
    assert _resolve_session_by_name_or_id("  My Project  ") == "titled-id"
    resume_store.set_session_title("titled-id", "exact-id")
    assert _resolve_session_by_name_or_id(" exact-id ") == "exact-id"


def test_padded_root_still_resolves_compression_tip(resume_store):
    from hermes_cli.main import _resolve_session_by_name_or_id

    resume_store.create_session("root", source="cli")
    resume_store.end_session("root", "compression")
    resume_store.create_session("tip", source="cli", parent_session_id="root")
    assert _resolve_session_by_name_or_id(" root ") == "tip"


def test_lookup_exception_remains_best_effort(monkeypatch):
    import hermes_state
    from hermes_cli.main import _resolve_session_by_name_or_id

    class BrokenDB:
        closed = False

        def get_session(self, _value):
            raise RuntimeError("unavailable")

        def close(self):
            self.closed = True

    db = BrokenDB()
    monkeypatch.setattr(hermes_state, "SessionDB", lambda **kw: db)
    assert _resolve_session_by_name_or_id(" id ") is None
    assert db.closed


@pytest.mark.parametrize("flag", ["resume", "continue_last"])
def test_startup_passes_resolved_id_to_tui(resume_store, monkeypatch, flag):
    from hermes_cli import main

    resume_store.create_session("startup-id", source="cli")
    captured = {}

    def launch(resume_session_id=None, **kwargs):
        captured["id"] = resume_session_id
        raise SystemExit(0)

    monkeypatch.setattr(main, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main, "_launch_tui", launch)
    args = Namespace(continue_last=None, model=None, provider=None, resume=None,
                     toolsets=None, tui=True, tui_dev=False)
    setattr(args, flag, " startup-id ")
    with pytest.raises(SystemExit) as exc:
        main.cmd_chat(args)
    assert exc.value.code == 0
    assert captured["id"] == "startup-id"
