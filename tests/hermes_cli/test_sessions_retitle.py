"""CLI ``hermes sessions retitle`` — manual retitle entry point.

Third trigger surface for the retitle worker, alongside the exactly-N
auto-fire and the ``/retitle`` slash command. All three write through
``retitle_session`` in ``agent.title_generator``, so these tests focus on
the CLI adapter: arg parsing, provenance guard, dry-run mode, and error
paths. The worker itself is covered by tests/agent/test_title_generator.py.
"""

import sys
from typing import Any, Dict, List, Optional
from unittest.mock import patch


class _FakeDB:
    """Minimal SessionDB double for the retitle CLI action."""

    def __init__(
        self,
        *,
        known=("20260827_120000_abc123",),
        title="Old Title",
        source="llm",
        history: Optional[List[Dict[str, Any]]] = None,
    ):
        self.known = set(known)
        self._titles = {sid: title for sid in known}
        self._sources = {sid: source for sid in known}
        self._history = history if history is not None else [
            {"role": "user", "content": "hello", "id": 1},
            {"role": "assistant", "content": "hi there", "id": 2},
        ]
        self.get_messages_calls: List[Dict[str, Any]] = []
        self.closed = False

    def resolve_session_id(self, session_id):
        for k in self.known:
            if k.startswith(session_id):
                return k
        return None

    def get_session_title(self, session_id):
        return self._titles.get(session_id)

    def get_session_title_source(self, session_id):
        return self._sources.get(session_id)

    def get_messages(self, session_id, *, latest=False, limit=None, **_kwargs):
        self.get_messages_calls.append({"session_id": session_id, "latest": latest, "limit": limit})
        return list(self._history)

    def close(self):
        self.closed = True


def _run(monkeypatch, capsys, argv_tail, db, *, retitle_return: Optional[str] = "New Better Title"):
    """Invoke the CLI with retitle_session mocked out."""
    import hermes_cli.main as main_mod
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: db)
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *argv_tail])

    # The action patches retitle_session AND _retitle_config at their
    # source module — the CLI imports lazily inside the action.
    with patch("agent.title_generator.retitle_session", return_value=retitle_return) as m_ret, \
         patch("agent.title_generator._retitle_config", return_value={"turns_window": 10}):
        try:
            main_mod.main()
            code = 0
        except SystemExit as e:
            code = e.code or 0

    return code, capsys.readouterr().out, m_ret


def test_retitle_writes_new_title(monkeypatch, capsys):
    db = _FakeDB()
    code, out, m_ret = _run(monkeypatch, capsys, ["retitle", "20260827_120000_abc123"], db)
    assert code == 0
    assert m_ret.called
    assert "retitled" in out.lower()
    assert "New Better Title" in out
    assert "Old Title" in out


def test_retitle_accepts_unique_prefix(monkeypatch, capsys):
    db = _FakeDB()
    code, out, _m = _run(monkeypatch, capsys, ["retitle", "20260827_120000"], db)
    assert code == 0
    assert "20260827_120000_abc123" in out


def test_retitle_unknown_session_id(monkeypatch, capsys):
    db = _FakeDB()
    code, out, m_ret = _run(monkeypatch, capsys, ["retitle", "nope"], db)
    assert code == 1
    assert "not found" in out.lower()
    assert not m_ret.called


def test_retitle_skips_user_set_title_without_force(monkeypatch, capsys):
    db = _FakeDB(title="Deb's Custom Title", source="user")
    code, out, m_ret = _run(monkeypatch, capsys, ["retitle", "20260827_120000_abc123"], db)
    assert code == 1
    assert "manual title" in out.lower()
    assert "--force" in out
    assert not m_ret.called


def test_retitle_force_overwrites_user_title(monkeypatch, capsys):
    db = _FakeDB(title="Deb's Custom Title", source="user")
    code, out, m_ret = _run(
        monkeypatch, capsys, ["retitle", "20260827_120000_abc123", "--force"], db
    )
    assert code == 0
    assert m_ret.called
    # Force flag propagates to the worker
    _args, kwargs = m_ret.call_args
    assert kwargs.get("force") is True


def test_retitle_dry_run_skips_llm_call(monkeypatch, capsys):
    db = _FakeDB()
    code, out, m_ret = _run(
        monkeypatch, capsys, ["retitle", "20260827_120000_abc123", "--dry-run"], db
    )
    assert code == 0
    assert not m_ret.called
    assert "would retitle" in out.lower()
    assert "old title" in out.lower()
    assert "not touched" in out.lower()


def test_retitle_no_history_fails_cleanly(monkeypatch, capsys):
    db = _FakeDB(history=[])
    code, out, m_ret = _run(monkeypatch, capsys, ["retitle", "20260827_120000_abc123"], db)
    assert code == 1
    assert "no history" in out.lower()
    assert not m_ret.called


def test_retitle_custom_turns_flag_propagates(monkeypatch, capsys):
    db = _FakeDB()
    code, _out, m_ret = _run(
        monkeypatch,
        capsys,
        ["retitle", "20260827_120000_abc123", "--turns", "20"],
        db,
    )
    assert code == 0
    _args, kwargs = m_ret.call_args
    assert kwargs.get("turns_window") == 20
    # get_messages fetched 2 * turns
    assert db.get_messages_calls[0]["limit"] == 40


def test_retitle_worker_returning_none_reports_no_change(monkeypatch, capsys):
    db = _FakeDB()
    code, out, m_ret = _run(
        monkeypatch,
        capsys,
        ["retitle", "20260827_120000_abc123"],
        db,
        retitle_return=None,
    )
    assert code == 1
    assert m_ret.called
    assert "no title change" in out.lower()


def test_retitle_never_touches_platform_names(monkeypatch, capsys):
    """DB-only-by-default is the whole point of the manual CLI too."""
    db = _FakeDB()
    _code, _out, m_ret = _run(
        monkeypatch, capsys, ["retitle", "20260827_120000_abc123"], db
    )
    _args, kwargs = m_ret.call_args
    assert kwargs.get("touch_platform_names") is False
