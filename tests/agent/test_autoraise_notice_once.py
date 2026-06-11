"""The Codex gpt-5.5 autoraise notice must fire once per profile, not once
per agent init — gateways rebuild the agent on restart/session rotation and
the banner was spamming chat platforms on every rebuild."""

import importlib


def _mod():
    return importlib.import_module("agent.agent_init")


def test_pending_then_marked(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    m = _mod()
    assert m._autoraise_notice_pending()
    m._mark_autoraise_notice_shown()
    assert not m._autoraise_notice_pending()
    marker = m._autoraise_notice_marker_path()
    assert marker.exists()
    assert str(marker).startswith(str(tmp_path))


def test_mark_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    m = _mod()
    m._mark_autoraise_notice_shown()
    m._mark_autoraise_notice_shown()
    assert not m._autoraise_notice_pending()


def test_notice_text_keeps_optout_command():
    m = _mod()
    text = m._build_codex_gpt55_autoraise_notice({"from": 0.5, "to": 0.85})
    assert "85%" in text and "50%" in text
    assert "compression.codex_gpt55_autoraise false" in text
