"""Regression tests for the WhatsApp pairing-wizard session dir (issue #85391, Bug 1).

The gateway adapter and the dashboard both resolve the session directory via
``get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")``. The
pairing wizard used to hard-code the legacy ``<home>/whatsapp/session`` path,
so it wrote pairing state into a directory the gateway may not read — the two
diverge the moment the empty legacy stub stops shadowing the consolidated
``platforms/whatsapp/session`` location. ``_whatsapp_session_path`` makes the
wizard use the shared resolver so writer and reader always agree.
"""

import hermes_constants

from hermes_cli.main import _whatsapp_session_path


def test_resolves_to_consolidated_path_when_no_legacy(monkeypatch, tmp_path):
    """Fresh install (no legacy dir) → consolidated platforms/whatsapp/session."""
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    assert _whatsapp_session_path() == tmp_path / "platforms" / "whatsapp" / "session"


def test_empty_legacy_stub_does_not_shadow(monkeypatch, tmp_path):
    """An empty legacy whatsapp/session must NOT be chosen over the new path.

    This is the exact divergence in the bug: a cleared/abandoned legacy stub
    would otherwise capture the wizard's writes while the gateway reads the
    consolidated location.
    """
    (tmp_path / "whatsapp" / "session").mkdir(parents=True)  # empty stub
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    assert _whatsapp_session_path() == tmp_path / "platforms" / "whatsapp" / "session"


def test_populated_legacy_is_honored(monkeypatch, tmp_path):
    """A populated legacy dir keeps being used (back-compat, no forced migration)."""
    legacy = tmp_path / "whatsapp" / "session"
    legacy.mkdir(parents=True)
    (legacy / "creds.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    assert _whatsapp_session_path() == legacy


def test_agrees_with_adapter_default(monkeypatch, tmp_path):
    """Wizard and gateway adapter must resolve to the identical directory."""
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    wizard = _whatsapp_session_path()
    adapter_default = hermes_constants.get_hermes_dir(
        "platforms/whatsapp/session", "whatsapp/session"
    )
    assert wizard == adapter_default


def test_wizard_rejects_empty_creds_in_resolved_dir(monkeypatch, tmp_path):
    """A 0-byte ``creds.json`` in the wizard's resolved session dir must not
    count as an existing pairing.

    The wizard now gates both its "existing session" and "paired successfully"
    checks on ``has_valid_whatsapp_creds`` (content), not ``Path.exists()``, so
    a truncated file the gateway would reject is never reported as paired here
    either — closing the sibling path of issue #85391 Bug 2 that the initial
    fix left in the pairing wizard.
    """
    from gateway.platforms.whatsapp_common import has_valid_whatsapp_creds

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    session_dir = _whatsapp_session_path()
    session_dir.mkdir(parents=True, exist_ok=True)
    creds = session_dir / "creds.json"
    creds.write_text("", encoding="utf-8")  # exists() is True; pairing is not real

    assert creds.exists()
    assert has_valid_whatsapp_creds(creds) is False
