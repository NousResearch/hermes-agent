"""Invariants for the read-only real-profile browser login source (``browser:``).

Contracts that must never regress (all exercised through the package facade / tool, not the bare
class):
1. The source appears in ``enabled_backends()`` and routes through ``backend_for_handle`` IFF
   ``browser.use_real_profile`` is on.
2. ``list_items`` carries origin + username only, read from the plaintext columns of the snapshot's
   ``Login Data`` — never the encrypted password.
3. A fill resolves the password server-side and password-blind: the value never appears in the tool
   result and IS registered with the redaction boundary. Mutant-proof: with the registration removed
   the no-leak assertion fails.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.vault_backends import backend_for_handle, enabled_backends

_PASSWORD = "correct horse battery staple 42"


def _seed_login_data(copy_dir: Path, origin: str = "https://example.com/login",
                     username: str = "jane@example.com") -> None:
    """A minimal Chrome ``Login Data`` with one row. password_value holds opaque bytes: the backend
    must never read it (it resolves over CDP), so the test proves the bytes never surface."""
    default = copy_dir / "Default"
    default.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(default / "Login Data")
    con.execute("CREATE TABLE logins (origin_url TEXT, username_value TEXT, password_value BLOB, "
                "blacklisted_by_user INTEGER DEFAULT 0)")
    con.execute("INSERT INTO logins (origin_url, username_value, password_value, blacklisted_by_user) "
                "VALUES (?, ?, ?, 0)", (origin, username, b"v10ENCRYPTED-NEVER-READ"))
    con.commit()
    con.close()


@pytest.fixture
def real_profile(tmp_path, monkeypatch):
    """Scratch HERMES_HOME with a seeded snapshot for the 'chrome' default browser and consent on."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    copy_dir = tmp_path / ".hermes" / "browser-profile" / "chrome"
    _seed_login_data(copy_dir)
    with patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
         patch("hermes_cli.browser_connect.real_profile_copy_dir", return_value=str(copy_dir)):
        yield copy_dir


def _set_consent(on: bool):
    return patch("agent.vault_backends.base._browser_profile_enabled", return_value=on)


def test_source_present_and_routes_only_with_consent(real_profile):
    with _set_consent(True):
        names = [b.name for b in enabled_backends()]
        assert "browser" in names
        b = next(b for b in enabled_backends() if b.name == "browser")
        assert b.display_name == "Browser profile — Chrome"
        handle = b.list_items()[0].id
        assert handle.startswith("browser:")
        assert backend_for_handle(handle) is not None
        assert backend_for_handle(handle).name == "browser"

    with _set_consent(False):
        assert "browser" not in [b.name for b in enabled_backends()]
        # same handle no longer routes once the source is gone
        assert backend_for_handle(handle) is None


def test_list_is_metadata_only_never_the_password(real_profile):
    with _set_consent(True):
        b = next(b for b in enabled_backends() if b.name == "browser")
        items = b.list_items()
        assert len(items) == 1
        meta = items[0]
        assert meta.origin == "https://example.com"
        assert meta.identifier == "jane@example.com"
        # The encrypted blob and any decrypted form must never appear in list output.
        blob = json.dumps([m.to_dict() for m in items])
        assert "ENCRYPTED" not in blob and "v10" not in blob
        assert meta.allowed_origins == ("https://example.com",)


def test_fill_is_password_blind_and_redaction_registered(real_profile):
    """A preview fill resolves the password over CDP (mocked) and must: (a) never put it in the tool
    result, (b) register it with the redaction boundary. Removing the registration flips (b)."""
    from tools import browser_vault_tool

    with _set_consent(True):
        b = next(b for b in enabled_backends() if b.name == "browser")
        handle = b.list_items()[0].id

    registered: list[str] = []
    # The real browser autofills + we read it over CDP; mock that one hop, assert blindness downstream.
    with _set_consent(True), \
         patch("tools.browser_real_profile_logins.resolve_login_password", return_value=_PASSWORD), \
         patch("tools.browser_session_real_profile.real_profile_cdp", return_value=("ws://x", None)), \
         patch("agent.redact.register_vault_redaction_value", side_effect=registered.append), \
         patch("tools.browser_vault_tool._bound_page_origin", return_value="https://example.com"), \
         patch("tools.browser_vault_tool._eval_js", return_value={"success": True, "result": json.dumps(
             [{"index": 0, "tag": "input", "type": "password", "name": "password", "autocomplete": "current-password"}])}), \
         patch("tools.browser_vault_tool._eval_js_secret", return_value={"success": True, "result": json.dumps({"filled": 1})}):
        out = browser_vault_tool.browser_vault_fill(handle, task_id="t")

    result = json.loads(out)
    assert _PASSWORD not in out, "password leaked into the fill tool result"
    assert result.get("success") is True or result.get("filled_fields") == 1 or "error" in result
    # The resolved password was registered with the redaction boundary before touching the page.
    assert _PASSWORD in registered, "resolved password was not registered for redaction"


def test_resolve_refuses_wrong_account_on_multi_account_origin():
    """A saved login carrying a username only yields its password when the browser autofilled THAT
    username — otherwise a site with several saved accounts could hand back a different account's
    password. The CDP read + autofill hop is mocked at the DOM-read boundary."""
    from tools import browser_real_profile_logins as rpl

    calls = {}

    def fake_cdp_call(ws, mid, method, params=None):
        if method == "Runtime.evaluate":
            expr = (params or {}).get("expression", "")
            if "getBoundingClientRect" in expr:
                return {"result": {"value": '{"x": 1, "y": 1}'}}  # field-point read
            return {"result": {"value": calls["read_value"]}}      # password/username value read
        return {"result": {}}

    with patch.object(rpl, "_page_cdp_url", return_value="ws://page"), \
         patch("websockets.sync.client.connect"), \
         patch.object(rpl, "_cdp_call", side_effect=fake_cdp_call), \
         patch("time.sleep"):
        calls["read_value"] = '{"password": "other-accts-pw", "username": "bob@example.com"}'
        assert rpl.resolve_login_password("ws://b", "https://example.com", "alice@example.com") is None
        calls["read_value"] = '{"password": "alice-pw", "username": "alice@example.com"}'
        assert rpl.resolve_login_password("ws://b", "https://example.com", "alice@example.com") == "alice-pw"
