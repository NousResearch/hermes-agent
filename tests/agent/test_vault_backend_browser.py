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

from hermes_agent.agent.vault_backends import backend_for_handle, enabled_backends

_PASSWORD = "correct horse battery staple 42"


def _fake_page_target(page_url):
    """A stand-in for the real ``_page_target`` context manager that yields a page CDP URL without
    opening a browser tab (the resolver's CDP calls are mocked separately in these unit tests)."""
    import contextlib

    @contextlib.contextmanager
    def _cm(cdp_url, nav_url):
        yield page_url
    return _cm


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
    return patch("hermes_agent.agent.vault_backends.base._browser_profile_enabled", return_value=on)


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
    from hermes_agent.tools import browser_vault_tool

    with _set_consent(True):
        b = next(b for b in enabled_backends() if b.name == "browser")
        handle = b.list_items()[0].id

    registered: list[str] = []
    # The real browser autofills + we read it over CDP; mock that one hop, assert blindness downstream.
    with _set_consent(True), \
         patch("hermes_agent.tools.browser_real_profile_logins.resolve_login_password", return_value=_PASSWORD), \
         patch("hermes_agent.tools.browser_session_real_profile.real_profile_cdp", return_value=("ws://x", None)), \
         patch("hermes_agent.agent.redact.register_vault_redaction_value", side_effect=registered.append), \
         patch("hermes_agent.tools.browser_vault_tool._bound_page_origin", return_value="https://example.com"), \
         patch("hermes_agent.tools.browser_vault_tool._eval_js", return_value={"success": True, "result": json.dumps(
             [{"index": 0, "tag": "input", "type": "password", "name": "password", "autocomplete": "current-password"}])}), \
         patch("hermes_agent.tools.browser_vault_tool._eval_js_secret", return_value={"success": True, "result": json.dumps({"filled": 1})}):
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
    from hermes_agent.tools import browser_real_profile_logins as rpl

    calls = {}

    def fake_cdp_call(ws, mid, method, params=None):
        if method == "Runtime.evaluate":
            expr = (params or {}).get("expression", "")
            # The value-read JS is the only one that reads ``location.origin``; the field-point JS is
            # everything else (both now share the visibility helper's getBoundingClientRect call).
            if "location.origin" not in expr:
                return {"result": {"value": '{"x": 1, "y": 1}'}}  # field-point read
            return {"result": {"value": calls["read_value"]}}      # password/username value read
        return {"result": {}}

    with patch.object(rpl, "_page_target", _fake_page_target("ws://page")), \
         patch("websockets.sync.client.connect"), \
         patch.object(rpl, "_cdp_call", side_effect=fake_cdp_call), \
         patch("time.sleep"):
        calls["read_value"] = ('{"password": "other-accts-pw", "username": "bob@example.com",'
                               ' "origin": "https://example.com"}')
        assert rpl.resolve_login_password("ws://b", "https://example.com/login",
                                          "https://example.com", "alice@example.com") is None
        calls["read_value"] = ('{"password": "alice-pw", "username": "alice@example.com",'
                               ' "origin": "https://example.com"}')
        assert rpl.resolve_login_password("ws://b", "https://example.com/login",
                                          "https://example.com", "alice@example.com") == "alice-pw"


# ── finding 1 (security): a redirect off the saved origin must never return another site's password ──

def test_resolve_refuses_when_page_redirected_off_saved_origin():
    """The saved root redirects to another site that has its OWN saved login; Chrome autofills THAT
    site's password. The resolver must refuse because the final page's origin is not the handle's
    saved origin, so a redirect can never leak a different site's password for this handle."""
    from hermes_agent.tools import browser_real_profile_logins as rpl

    def fake_cdp_call(ws, mid, method, params=None):
        if method == "Runtime.evaluate":
            if "location.origin" not in (params or {}).get("expression", ""):
                return {"result": {"value": '{"x": 1, "y": 1}'}}
            # Autofilled a password, but the page is now on evil.example, NOT the saved origin.
            return {"result": {"value": '{"password": "victim-site-pw", "username": "",'
                                        ' "origin": "https://evil.example"}'}}
        return {"result": {}}

    with patch.object(rpl, "_page_target", _fake_page_target("ws://page")), \
         patch("websockets.sync.client.connect"), \
         patch.object(rpl, "_cdp_call", side_effect=fake_cdp_call), \
         patch("time.sleep"):
        # No username on the handle, so only the origin recheck can stop the leak.
        assert rpl.resolve_login_password(
            "ws://b", "https://bank.example/login", "https://bank.example", "") is None
        # Control: same page, origin back on the saved origin -> the password is returned.

    def fake_same_origin(ws, mid, method, params=None):
        if method == "Runtime.evaluate":
            if "location.origin" not in (params or {}).get("expression", ""):
                return {"result": {"value": '{"x": 1, "y": 1}'}}
            return {"result": {"value": '{"password": "victim-site-pw", "username": "",'
                                        ' "origin": "https://bank.example"}'}}
        return {"result": {}}

    with patch.object(rpl, "_page_target", _fake_page_target("ws://page")), \
         patch("websockets.sync.client.connect"), \
         patch.object(rpl, "_cdp_call", side_effect=fake_same_origin), \
         patch("time.sleep"):
        assert rpl.resolve_login_password(
            "ws://b", "https://bank.example/login", "https://bank.example", "") == "victim-site-pw"


# ── finding 3: the saved login PATH is preserved end to end (handle -> nav_url) ──────────────────

def test_handle_preserves_saved_login_path_and_resolve_navigates_there(real_profile, monkeypatch):
    """A login saved at ``https://example.com/login`` must list with an origin of
    ``https://example.com`` (for the exact-origin fill guard) yet resolve by navigating to the full
    saved URL (path included), not the bare root which may have no form."""
    from hermes_agent.tools import browser_real_profile_logins as rpl

    with _set_consent(True):
        b = next(b for b in enabled_backends() if b.name == "browser")
        item = b.list_items()[0]
        assert item.origin == "https://example.com"
        assert item.allowed_origins == ("https://example.com",)

        navigated = {}

        def fake_resolve(cdp_url, nav_url, origin, username):
            navigated["nav_url"] = nav_url
            navigated["origin"] = origin
            return "pw"

        monkeypatch.setattr(rpl, "resolve_login_password", fake_resolve)
        monkeypatch.setattr("hermes_agent.tools.browser_session_real_profile.real_profile_cdp",
                            lambda: ("ws://x", None))
        assert b.resolve_password(item.id) == "pw"
    # The full saved URL (with its path) drives navigation; the normalized origin guards the match.
    assert navigated["nav_url"] == "https://example.com/login"
    assert navigated["origin"] == "https://example.com"


# ── finding 4: a freshly-enabled source with no snapshot prepares one before listing ────────────

def test_list_prepares_snapshot_when_none_exists(tmp_path, monkeypatch):
    """On a fresh profile with no prior browser session the snapshot does not exist yet; listing must
    drive the real-profile lifecycle (``real_profile_cdp``) to create it, then read it — not silently
    return empty so the user can never select a login to initialise the source."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    copy_dir = tmp_path / ".hermes" / "browser-profile" / "chrome"  # does NOT exist yet

    def prepare_snapshot():
        _seed_login_data(copy_dir)  # the lifecycle call is what materialises the snapshot
        return ("ws://x", None)

    prepared = {"n": 0}

    def counted():
        prepared["n"] += 1
        return prepare_snapshot()

    with patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
         patch("hermes_cli.browser_connect.real_profile_copy_dir", return_value=str(copy_dir)), \
         patch("hermes_agent.tools.browser_session_real_profile.real_profile_cdp", side_effect=counted), \
         _set_consent(True):
        b = next(b for b in enabled_backends() if b.name == "browser")
        items = b.list_items()
    assert prepared["n"] == 1, "listing did not prepare a missing snapshot"
    assert len(items) == 1 and items[0].origin == "https://example.com"


# ── finding 5: the temporary autofill tab is closed (Target.closeTarget), not just the socket ────

def test_page_target_closes_the_tab_on_exit():
    """``_page_target`` opens a tab for the autofill and must close it with ``Target.closeTarget`` on
    exit — on both success and failure — so repeated fills never leak live pages."""
    from hermes_agent.tools import browser_real_profile_logins as rpl

    closed = []
    created = {"id": "TARGET-123"}

    def fake_cdp_call(ws, mid, method, params=None):
        if method == "Target.createTarget":
            return {"targetId": created["id"]}
        if method == "Target.closeTarget":
            closed.append((params or {}).get("targetId"))
        return {}

    import contextlib

    @contextlib.contextmanager
    def fake_connect(url, **kwargs):
        yield object()

    with patch("websockets.sync.client.connect", fake_connect), \
         patch.object(rpl, "_cdp_call", side_effect=fake_cdp_call):
        with rpl._page_target("ws://browser/devtools/browser/abc", "https://example.com/login") as page_url:
            assert created["id"] in page_url
        assert closed == ["TARGET-123"], "tab was not closed on success"
        # And it is still closed when the body raises.
        closed.clear()
        with pytest.raises(RuntimeError):
            with rpl._page_target("ws://browser/devtools/browser/abc", "https://example.com/login"):
                raise RuntimeError("boom")
        assert closed == ["TARGET-123"], "tab was not closed on failure"


# ── finding 6: drive the REAL resolver against a faithful CDP-level fake (no whole-function mock) ──

class _FakeAutofillWS:
    """A faithful fake of the real-profile browser's DevTools WebSocket for ONE page. It implements
    the exact JSON-RPC methods ``resolve_login_password`` drives (``Page.enable``, ``Runtime.enable``,
    the field-point ``Runtime.evaluate``, the trusted ``Input.dispatchMouseEvent`` click, and the
    value ``Runtime.evaluate``) so the real resolver — field discovery, the trusted-click gesture, the
    settle wait, the read-back, and the origin/username guards — runs unmodified. Chrome fills the
    password ONLY after the trusted click, exactly like PasswordAutofillAgent; with the click removed
    the password reads back empty and the resolver returns None (the mutant-proof check)."""

    def __init__(self, *, origin, username, password):
        self._origin, self._username, self._password = origin, username, password
        self._clicked = False
        self._pending = None

    def send(self, payload):
        self._pending = json.loads(payload)

    def recv(self, timeout=None):
        req = self._pending
        mid, method, params = req["id"], req["method"], req.get("params", {})
        result = {}
        if method == "Input.dispatchMouseEvent":
            self._clicked = True
        elif method == "Runtime.evaluate":
            expr = params.get("expression", "")
            if "location.origin" not in expr:
                result = {"result": {"value": '{"x": 10, "y": 20}'}}
            else:
                # The browser only fills after a trusted gesture landed on the form.
                pw = self._password if self._clicked else ""
                user = self._username if self._clicked else ""
                result = {"result": {"value": json.dumps(
                    {"password": pw, "username": user, "origin": self._origin})}}
        return json.dumps({"id": mid, "result": result})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_resolve_drives_real_autofill_mechanism_over_cdp_level_fake():
    from hermes_agent.tools import browser_real_profile_logins as rpl

    ws = _FakeAutofillWS(origin="https://example.com", username="jane@example.com",
                         password="s3cret-from-browser")

    def fake_connect(url, **kwargs):
        return ws  # same fake page socket for the resolve connection

    import contextlib

    @contextlib.contextmanager
    def fake_page_target(cdp_url, nav_url):
        yield "ws://page"

    with patch.object(rpl, "_page_target", fake_page_target), \
         patch("websockets.sync.client.connect", fake_connect), \
         patch("time.sleep"):
        got = rpl.resolve_login_password(
            "ws://browser", "https://example.com/login", "https://example.com", "jane@example.com")
    assert got == "s3cret-from-browser"


def test_resolve_returns_none_without_the_trusted_click_mutant():
    """Mutant-proof for finding 6: with the trusted click suppressed (PasswordAutofillAgent never
    fires), the real resolver reads an empty password and returns None — proving the test exercises
    the click mechanism, not a stubbed value."""
    from hermes_agent.tools import browser_real_profile_logins as rpl

    ws = _FakeAutofillWS(origin="https://example.com", username="jane@example.com",
                         password="s3cret-from-browser")
    ws.send = lambda payload, _w=ws: setattr(_w, "_pending", _swallow_click(payload))

    def fake_connect(url, **kwargs):
        return ws

    import contextlib

    @contextlib.contextmanager
    def fake_page_target(cdp_url, nav_url):
        yield "ws://page"

    with patch.object(rpl, "_page_target", fake_page_target), \
         patch("websockets.sync.client.connect", fake_connect), \
         patch("time.sleep"):
        got = rpl.resolve_login_password(
            "ws://browser", "https://example.com/login", "https://example.com", "jane@example.com")
    assert got is None


def _swallow_click(payload):
    """Turn a dispatched mouse-event request into a no-op the fake acknowledges but never treats as a
    trusted click, so the autofill never fires (used by the mutant test)."""
    req = json.loads(payload)
    if req.get("method") == "Input.dispatchMouseEvent":
        req["method"] = "Input.ignored"
    return req


# ── finding 8: committed-but-WAL changes are visible (no immutable=1 on a live DB) ──────────────

def test_read_login_metadata_sees_committed_wal_changes(tmp_path):
    """The real browser mutates its own ``Login Data`` while running: a saved/changed login is
    committed into the WAL file first. ``immutable=1`` tells SQLite to ignore the WAL, so such a login
    would be missing or stale in the picker. Reading with ``mode=ro`` (no immutable) must see it."""
    from hermes_agent.tools.browser_real_profile_logins import read_login_metadata

    default = tmp_path / "Default"
    default.mkdir()
    db = default / "Login Data"
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE logins (origin_url TEXT, username_value TEXT, password_value BLOB, "
                "blacklisted_by_user INTEGER DEFAULT 0)")
    con.execute("INSERT INTO logins VALUES ('https://old.example/login', 'a', X'00', 0)")
    con.commit()
    # A SECOND connection commits another row and stays OPEN, so the change lives in the WAL and is
    # not checkpointed into the main DB file — exactly the running-browser situation.
    live = sqlite3.connect(db)
    live.execute("INSERT INTO logins VALUES ('https://new.example/login', 'b', X'00', 0)")
    live.commit()
    assert (db.parent / "Login Data-wal").exists(), "fixture did not leave a WAL file"
    try:
        rows = read_login_metadata(str(tmp_path))
    finally:
        con.close()
        live.close()
    urls = {r["origin_url"] for r in rows}
    assert "https://new.example/login" in urls, "committed WAL change was not seen (immutable=1 regression)"
    assert "https://old.example/login" in urls


# ── finding 5: a '#' (or '?'/space) in the DB path must not truncate the sqlite URI ──────────────

def test_read_login_metadata_with_hash_in_path(tmp_path):
    """The DB path is interpolated into a ``file:`` SQLite URI. A directory component containing '#'
    (common in cloned Chrome profiles, synced folders) makes a bare ``f"file:{db}?mode=ro"`` truncate
    at the '#' fragment, so SQLite opens the wrong/empty path and EVERY login silently drops. Building
    the URI with ``Path.as_uri`` percent-escapes the '#', so the real logins are read."""
    from hermes_agent.tools.browser_real_profile_logins import read_login_metadata

    base = tmp_path / "Profile #1 (work)" / "copy"
    default = base / "Default"
    default.mkdir(parents=True)
    con = sqlite3.connect(default / "Login Data")
    con.execute("CREATE TABLE logins (origin_url TEXT, username_value TEXT, password_value BLOB, "
                "blacklisted_by_user INTEGER DEFAULT 0)")
    con.execute("INSERT INTO logins VALUES ('https://example.com/login', 'jane', X'00', 0)")
    con.commit()
    con.close()
    rows = read_login_metadata(str(base))
    assert [r["origin_url"] for r in rows] == ["https://example.com/login"], (
        "logins under a '#'-containing path were dropped (unescaped sqlite URI)")


# ── finding 3 (round 3): a RELATIVE copy dir (HERMES_HOME=./root) must still list, not ValueError ──

def test_read_login_metadata_with_relative_copy_dir(tmp_path, monkeypatch):
    """``HERMES_HOME=./root`` (a supported setting) makes ``real_profile_copy_dir()`` relative, so
    the DB path reaches ``read_login_metadata`` relative. ``Path.as_uri()`` raises ValueError on a
    relative path, which escapes the ``sqlite3.Error`` handler and makes listing fail for EVERY
    source once a snapshot exists. Resolving the path to absolute first must let the real logins
    read back — the picker WORKS, not merely swallows the error."""
    from hermes_agent.tools.browser_real_profile_logins import read_login_metadata

    base = tmp_path / "copy"
    default = base / "Default"
    default.mkdir(parents=True)
    con = sqlite3.connect(default / "Login Data")
    con.execute("CREATE TABLE logins (origin_url TEXT, username_value TEXT, password_value BLOB, "
                "blacklisted_by_user INTEGER DEFAULT 0)")
    con.execute("INSERT INTO logins VALUES ('https://example.com/login', 'jane', X'00', 0)")
    con.commit()
    con.close()

    # Make the copy dir RELATIVE to cwd, exactly as a relative HERMES_HOME would.
    monkeypatch.chdir(tmp_path)
    rel = Path("copy")
    assert not rel.is_absolute()
    rows = read_login_metadata(str(rel))
    assert [r["origin_url"] for r in rows] == ["https://example.com/login"], (
        "a relative copy dir failed to list logins (as_uri ValueError on a relative path)")


# ── finding 4: listing refreshes when the snapshot's source profile no longer matches the pin ────

def _write_snapshot(copy_dir: Path, *, source: str, origin: str, username: str) -> None:
    """A completed snapshot dir: Default/Login Data with one row + the done marker naming its source
    profile (what ``snapshot_real_profile`` writes)."""
    from hermes_cli.browser_connect import _SNAPSHOT_DONE_MARKER
    default = copy_dir / "Default"
    default.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(default / "Login Data")
    con.execute("CREATE TABLE logins (origin_url TEXT, username_value TEXT, password_value BLOB, "
                "blacklisted_by_user INTEGER DEFAULT 0)")
    con.execute("INSERT INTO logins VALUES (?, ?, X'00', 0)", (origin, username))
    con.commit()
    con.close()
    (copy_dir / _SNAPSHOT_DONE_MARKER).write_text(source, encoding="utf-8")


def test_list_refreshes_when_pin_changed_to_another_profile(tmp_path, monkeypatch):
    """A snapshot exists for source profile 'Default' (alice). The user changes ``browser.real_profile_pin``
    to 'Profile 2' (bob). The base trusted the snapshot by file existence and skipped the source-profile
    lifecycle, so a new backend still listed ALICE. The fix validates the snapshot's source against the
    currently-selected pin and refreshes through the lifecycle when they differ — listing BOB.

    Real sqlite + real config reads; only the live-profile root resolution and the lifecycle's snapshot
    *rebuild* step are stubbed (no real Chrome needed for this bug, which is in the trust check)."""
    from hermes_agent.agent.vault_backends import enabled_backends

    live_root = tmp_path / "chrome-data"           # the real browser's user-data dir
    (live_root / "Default").mkdir(parents=True)
    (live_root / "Profile 2").mkdir(parents=True)
    copy_dir = tmp_path / ".hermes" / "browser-profile" / "chrome"
    # Existing snapshot was taken from 'Default' and lists alice.
    _write_snapshot(copy_dir, source="Default", origin="https://alice.example/login", username="alice")

    # Config pins the source to 'Profile 2' now.
    monkeypatch.setattr("hermes_cli.browser_connect._real_profile_pin", lambda: "Profile 2")

    def refresh():
        # The lifecycle rebuild: re-snapshot from the pinned 'Profile 2' (bob) and re-stamp the marker.
        import shutil
        shutil.rmtree(copy_dir, ignore_errors=True)
        _write_snapshot(copy_dir, source="Profile 2", origin="https://bob.example/login", username="bob")
        return ("ws://x", None)

    refreshed = {"n": 0}

    def counted():
        refreshed["n"] += 1
        return refresh()

    with patch("hermes_agent.agent.vault_backends.base._browser_profile_enabled", return_value=True), \
         patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
         patch("hermes_cli.browser_connect.real_profile_copy_dir", return_value=str(copy_dir)), \
         patch("hermes_cli.browser_connect.real_profile_data_dir", return_value=str(live_root)), \
         patch("hermes_agent.tools.browser_session_real_profile.real_profile_cdp", side_effect=counted):
        backend = next(b for b in enabled_backends() if b.name == "browser")
        items = backend.list_items()

    assert refreshed["n"] == 1, "listing did not refresh a snapshot whose source profile no longer matches the pin"
    origins = {m.origin for m in items}
    assert origins == {"https://bob.example"}, f"listed the stale profile's logins: {origins}"


def test_list_does_not_refresh_when_snapshot_source_still_matches(tmp_path, monkeypatch):
    """Control for finding 4: when the snapshot's source profile still matches the current pin, the
    backend serves it WITHOUT re-running the lifecycle (no needless refresh on every list)."""
    from hermes_agent.agent.vault_backends import enabled_backends

    live_root = tmp_path / "chrome-data"
    (live_root / "Default").mkdir(parents=True)
    copy_dir = tmp_path / ".hermes" / "browser-profile" / "chrome"
    _write_snapshot(copy_dir, source="Default", origin="https://alice.example/login", username="alice")

    monkeypatch.setattr("hermes_cli.browser_connect._real_profile_pin", lambda: "Default")
    refreshed = {"n": 0}

    with patch("hermes_agent.agent.vault_backends.base._browser_profile_enabled", return_value=True), \
         patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
         patch("hermes_cli.browser_connect.real_profile_copy_dir", return_value=str(copy_dir)), \
         patch("hermes_cli.browser_connect.real_profile_data_dir", return_value=str(live_root)), \
         patch("hermes_agent.tools.browser_session_real_profile.real_profile_cdp",
               side_effect=lambda: refreshed.__setitem__("n", refreshed["n"] + 1) or ("ws://x", None)):
        backend = next(b for b in enabled_backends() if b.name == "browser")
        items = backend.list_items()

    assert refreshed["n"] == 0, "listing needlessly refreshed a snapshot whose source still matches"
    assert {m.origin for m in items} == {"https://alice.example"}


# ── round-3 finding (2): a profile switch while the old browser is RUNNING must not serve stale ───

class _FakeRunningBrowser:
    def __init__(self, cdp_url):
        self.cdp_url = cdp_url


def test_list_refreshes_when_pin_changed_and_browser_is_running(tmp_path, monkeypatch):
    """A snapshot exists for 'Default' (alice) AND the real-profile browser is still RUNNING. The
    user switches ``browser.real_profile_pin`` to 'Profile 2' (bob).

    This drives the REAL ``real_profile_cdp()``: while a browser is running it ATTACHES and returns
    the existing connection WITHOUT re-syncing, so the round-2 marker check alone kept listing
    ALICE. The fix stops the running browser first on a source mismatch, so the relaunch re-snapshots
    BOB. Only ``get_browser`` / ``_launch`` / ``stop_browser`` are stubbed — the attach-vs-relaunch
    decision is the production code's.
    """
    import hermes_agent.tools.browser_session_real_profile as rpf
    from hermes_agent.agent.vault_backends import enabled_backends

    live_root = tmp_path / "chrome-data"
    (live_root / "Default").mkdir(parents=True)
    (live_root / "Profile 2").mkdir(parents=True)
    copy_dir = tmp_path / ".hermes" / "browser-profile" / "chrome"
    _write_snapshot(copy_dir, source="Default", origin="https://alice.example/login", username="alice")

    monkeypatch.setattr("hermes_cli.browser_connect._real_profile_pin", lambda: "Profile 2")

    # Browser lifecycle state: a browser is running, attached to the old (alice) snapshot.
    state = {"running": True}

    def fake_get_browser(key):
        return _FakeRunningBrowser("ws://running") if state["running"] else None

    def fake_stop_browser(key):
        state["running"] = False  # terminate_real_profile_browser's effect

    def fake_launch(browser_name):
        # A real relaunch re-snapshots from the now-pinned 'Profile 2' (bob) and re-stamps the marker.
        import shutil
        shutil.rmtree(copy_dir, ignore_errors=True)
        _write_snapshot(copy_dir, source="Profile 2", origin="https://bob.example/login", username="bob")
        state["running"] = True
        return ("ws://relaunched", None)

    monkeypatch.setattr("hermes_agent.tools.browser_session_cloud.use_real_profile", lambda: True)
    monkeypatch.setattr("hermes_agent.tools.browser_lightpanda.using_lightpanda_engine", lambda: False)
    monkeypatch.setattr("hermes_agent.tools.browser_launcher.get_browser", fake_get_browser)
    monkeypatch.setattr("hermes_agent.tools.browser_launcher.stop_browser", fake_stop_browser)
    monkeypatch.setattr(rpf, "_launch", fake_launch)

    with patch("hermes_agent.agent.vault_backends.base._browser_profile_enabled", return_value=True), \
         patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
         patch("hermes_cli.browser_connect.real_profile_copy_dir", return_value=str(copy_dir)), \
         patch("hermes_cli.browser_connect.real_profile_data_dir", return_value=str(live_root)):
        backend = next(b for b in enabled_backends() if b.name == "browser")
        items = backend.list_items()

    origins = {m.origin for m in items}
    assert origins == {"https://bob.example"}, (
        f"profile switch with a running browser listed the stale profile's logins: {origins}")
