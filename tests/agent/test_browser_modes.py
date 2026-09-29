"""Browser modes: real config in a temp HERMES_HOME, real sqlite cookie DBs, browser launch faked."""

import sqlite3

import pytest

from agent import browser_modes as bm


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _cookie_db(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT)")
    con.executemany("INSERT INTO cookies VALUES (?,?,?)", rows)
    con.commit()
    con.close()


def test_login_check_sees_auth_cookie_names_but_never_returns_values(tmp_path):
    prof = tmp_path / "Default"
    _cookie_db(prof / "Network" / "Cookies", [(".google.com", "SID", "TOPSECRET"), (".example.com", "a", "b")])

    assert bm.google_signed_in(str(prof)) is True
    assert "TOPSECRET" not in repr(bm.google_signed_in(str(prof)))


def test_login_check_is_false_for_tracking_only_cookies_and_none_without_db(tmp_path):
    prof = tmp_path / "Default"
    _cookie_db(prof / "Cookies", [(".google.com", "NID", "x"), (".evil.com", "SID", "x")])

    assert bm.google_signed_in(str(prof)) is False
    assert bm.google_signed_in(str(tmp_path / "missing")) is None


def test_modes_round_trip_through_config_and_are_mutually_exclusive(home):
    assert bm.current_mode() == "managed"

    assert bm.set_mode("copy")["mode"] == "copy"
    assert bm._browser_cfg().get("use_real_profile") is True

    assert bm.set_mode("own")["mode"] == "own"
    assert not bm._browser_cfg().get("use_real_profile") and bm._browser_cfg()["cdp_url"]

    assert bm.set_mode("managed")["mode"] == "managed"
    with pytest.raises(ValueError):
        bm.set_mode("live")


def test_clear_copy_removes_logins_and_leaves_copy_mode(home):
    from hermes_cli import browser_connect as hb
    copy = home / "browser-profile" / "chrome"
    copy.mkdir(parents=True)
    (copy / hb._SNAPSHOT_DONE_MARKER).write_text("Default")
    bm.set_mode("copy")

    got = bm.clear_copy()

    assert not (home / "browser-profile").exists() and got["mode"] == "managed"


def test_sign_in_switches_to_own_mode_and_reports_a_missing_browser(home, monkeypatch):
    from hermes_cli import browser_connect as hb
    monkeypatch.setattr(hb, "is_browser_debug_ready", lambda *a, **k: False)
    monkeypatch.setattr(hb, "launch_chrome_debug", lambda port: hb.ChromeDebugLaunch(launched=False))
    with pytest.raises(RuntimeError):
        bm.open_sign_in()

    monkeypatch.setattr(hb, "launch_chrome_debug", lambda port: hb.ChromeDebugLaunch(launched=True))
    monkeypatch.setattr(bm, "_open_tab", lambda *a: None)

    assert bm.open_sign_in()["mode"] == "own"
