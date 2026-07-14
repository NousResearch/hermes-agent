import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest


def _http_error(status, message):
    import requests
    response = Mock(status_code=status, text=json.dumps({"error": message}))
    response.json.return_value = {"error": message}
    return requests.HTTPError(response=response)


@pytest.mark.parametrize("status,message", [
    (404, "Tab not found"),
    (410, "Tab not found"),
    (503, "Target page, context or browser has been closed"),
])
def test_navigate_recreates_stale_tab_once(monkeypatch, status, message):
    from tools import browser_camofox as mod
    instance = object()
    session = {
        "tab_id": "old", "user_id": "user", "session_key": "scope",
        "instance": instance,
    }
    pool = Mock()
    monkeypatch.setattr(mod, "_get_session", lambda task_id: session)
    monkeypatch.setattr(mod, "_rewrite_loopback_url_for_camofox", lambda url: (url, None))
    monkeypatch.setattr(mod, "get_vnc_url", lambda: None)
    monkeypatch.setattr(mod, "_get_camofox_config", lambda: {"mode": "per_thread_instances"})
    monkeypatch.setattr(mod, "_get_instance_pool", lambda cfg: pool)
    monkeypatch.setattr(mod, "_post", Mock(side_effect=_http_error(status, message)))
    monkeypatch.setattr(mod, "_ensure_tab", Mock(side_effect=lambda task_id, url: {**session, "tab_id": "fresh"}))
    monkeypatch.setattr(mod, "_get", Mock(return_value={"snapshot": "ok", "refsCount": 1}))
    result = json.loads(mod.camofox_navigate("https://example.com", "declared-scope"))
    assert result["success"] is True
    mod._ensure_tab.assert_called_once_with("declared-scope", "https://example.com")
    pool.ensure_viewer.assert_called_once_with(instance, force=True)


def test_firefox_import_is_domain_scoped_and_merges_exact_profile(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    db = sqlite3.connect(profile / "cookies.sqlite")
    db.execute("CREATE TABLE moz_cookies (name, value, host, path, expiry, isSecure, isHttpOnly, sameSite)")
    db.executemany("INSERT INTO moz_cookies VALUES (?,?,?,?,?,?,?,?)", [
        ("c_user", "secret-a", ".facebook.com", "/", 123, 1, 1, 1),
        ("xs", "secret-b", ".facebook.com", "/", 123, 1, 1, 2),
        ("other", "do-not-copy", ".example.com", "/", 123, 0, 0, 0),
    ])
    db.commit(); db.close()
    from tools import browser_camofox, camofox_auth
    profile_root = tmp_path / "camofox-profiles"
    monkeypatch.setattr(browser_camofox, "camofox_identity_for_scope", lambda scope: {"user_id": "derived"})
    stop = Mock()
    monkeypatch.setattr(browser_camofox, "stop_camofox_scope", stop)
    monkeypatch.setattr(browser_camofox, "_get_camofox_config", lambda: {"profile_dir": str(profile_root)})
    assert camofox_auth.import_firefox_cookies(
        profile, task_id="declared-scope", domains=["facebook.com"],
        required_names=["c_user", "xs"],
    ) == 2
    stop.assert_called_once_with("declared-scope")
    path = camofox_auth._profile_state_path(profile_root, "derived")
    state = json.loads(path.read_text())
    assert {c["name"] for c in state["cookies"]} == {"c_user", "xs"}
    assert path.stat().st_mode & 0o777 == 0o600


def test_navigate_cannot_restart_same_scope_until_cookie_import_finishes(tmp_path, monkeypatch):
    from tools import browser_camofox, camofox_auth
    from tools.camofox_instance_pool import CamofoxInstancePool

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    (server_dir / "server.js").write_text("// test")
    source = tmp_path / "firefox"
    source.mkdir()
    db = sqlite3.connect(source / "cookies.sqlite")
    db.execute("CREATE TABLE moz_cookies (name, value, host, path, expiry, isSecure, isHttpOnly, sameSite)")
    db.execute("INSERT INTO moz_cookies VALUES (?,?,?,?,?,?,?,?)", (
        "session", "secret", ".example.com", "/", 123, 1, 1, 1,
    ))
    db.commit()
    db.close()

    pool = CamofoxInstancePool(server_dir, log_root=tmp_path / "logs")
    processes = [Mock(pid=101), Mock(pid=102)]
    for process in processes:
        process.poll.return_value = None
    cfg = {
        "mode": "per_thread_instances",
        "server_dir": str(server_dir),
        "profile_dir": str(tmp_path / "profiles"),
    }
    monkeypatch.setattr(browser_camofox, "_get_camofox_config", lambda: cfg)
    monkeypatch.setattr(browser_camofox, "_get_instance_pool", lambda ignored: pool)
    monkeypatch.setattr(browser_camofox, "camofox_identity_for_scope", lambda scope: {"user_id": "derived"})
    monkeypatch.setattr(browser_camofox, "_sessions", {})
    monkeypatch.setattr(browser_camofox, "get_vnc_url", lambda: None)
    monkeypatch.setattr(browser_camofox, "_get", lambda *args, **kwargs: {"snapshot": "", "refsCount": 0})

    def create_tab(task_id, url):
        session = browser_camofox._sessions[task_id]
        session["tab_id"] = "new-tab"
        return session

    monkeypatch.setattr(browser_camofox, "_ensure_tab", create_tab)

    merge_entered = threading.Event()
    allow_merge = threading.Event()
    real_merge = camofox_auth._atomic_merge_storage_state

    def blocking_merge(path, cookies):
        merge_entered.set()
        assert allow_merge.wait(timeout=5)
        real_merge(path, cookies)

    popen = Mock(side_effect=processes)
    with monkeypatch.context() as context:
        context.setattr(camofox_auth, "_atomic_merge_storage_state", blocking_merge)
        context.setattr("tools.camofox_instance_pool.subprocess.Popen", popen)
        context.setattr(pool, "_wait_until_ready", Mock())
        context.setattr("tools.camofox_instance_pool.os.getpgid", lambda pid: pid)
        context.setattr("tools.camofox_instance_pool.os.killpg", Mock())
        pool.get_or_start("scope")

        with ThreadPoolExecutor(max_workers=2) as executor:
            importing = executor.submit(
                camofox_auth.import_firefox_cookies, source, task_id="scope",
                domains=["example.com"], required_names=["session"],
            )
            assert merge_entered.wait(timeout=5)
            navigating = executor.submit(
                browser_camofox.camofox_navigate, "https://example.com", "scope"
            )
            time.sleep(0.05)
            assert not navigating.done()
            assert popen.call_count == 1
            allow_merge.set()
            assert importing.result(timeout=5) == 1
            assert json.loads(navigating.result(timeout=5))["success"] is True
            assert browser_camofox._sessions["scope"]["instance"].process is processes[1]


def test_cookie_import_requires_explicit_source_profile():
    from tools import camofox_auth
    with pytest.raises(ValueError, match="explicit Firefox source_profile"):
        camofox_auth._handle_import(
            {"domains": ["facebook.com"], "required_names": ["c_user"]},
            task_id="scope",
        )


def test_storage_state_merge_preserves_origins_and_unrelated_cookies(tmp_path):
    from tools.camofox_auth import _atomic_merge_storage_state
    path = tmp_path / "profile" / "storage-state.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "cookies": [
            {"name": "keep", "value": "old", "domain": ".example.com", "path": "/"},
            {"name": "xs", "value": "stale", "domain": ".facebook.com", "path": "/"},
        ],
        "origins": [{"origin": "https://example.com", "localStorage": []}],
    }))
    _atomic_merge_storage_state(path, [
        {"name": "xs", "value": "fresh", "domain": ".facebook.com", "path": "/"},
    ])
    state = json.loads(path.read_text())
    assert [(c["name"], c["value"]) for c in state["cookies"]] == [("keep", "old"), ("xs", "fresh")]
    assert state["origins"][0]["origin"] == "https://example.com"


def test_stop_scope_invalidates_tab_and_stops_only_scope(monkeypatch):
    from tools import browser_camofox as mod
    session = {"tab_id": "old"}
    pool = Mock()
    monkeypatch.setattr(mod, "_sessions", {"scope-a": session, "scope-b": {"tab_id": "other"}})
    monkeypatch.setattr(mod, "_get_camofox_config", lambda: {"mode": "per_thread_instances"})
    monkeypatch.setattr(mod, "_get_instance_pool", lambda cfg: pool)
    mod.stop_camofox_scope("scope-a")
    pool.stop.assert_called_once_with("scope-a")
    assert session["tab_id"] is None
    assert "scope-a" not in mod._sessions
    assert mod._sessions["scope-b"]["tab_id"] == "other"


def test_cached_session_restarts_dead_server_and_invalidates_tab(monkeypatch):
    from tools import browser_camofox as mod
    old_instance, new_instance = object(), Mock(api_url="http://new", viewer_url="http://viewer")
    session = {"tab_id": "old", "instance": old_instance, "api_url": "http://old", "adopt_existing_tab": False}
    pool = Mock()
    pool.get_or_start.return_value = new_instance
    pool.scope_lifecycle.side_effect = lambda scope: nullcontext()
    monkeypatch.setattr(mod, "_sessions", {"scope": session})
    monkeypatch.setattr(mod, "_get_camofox_config", lambda: {"mode": "per_thread_instances"})
    monkeypatch.setattr(mod, "_get_instance_pool", lambda cfg: pool)
    recovered = mod._get_session("scope")
    assert recovered["tab_id"] is None
    assert recovered["api_url"] == "http://new"
    pool.ensure_viewer.assert_called_once_with(new_instance, force=True)


def test_cron_job_persists_browser_scope(tmp_path, monkeypatch):
    from cron import jobs
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "jobs.json")
    monkeypatch.setattr(jobs, "compute_next_run", lambda schedule: "2099-01-01T00:00:00")
    job = jobs.create_job("continue browser work", "every 1h", browser_scope="facebook-leo")
    assert job["browser_scope"] == "facebook-leo"


def test_cron_current_browser_scope_resolves_or_refuses():
    from tools.cronjob_tools import _resolve_cron_browser_scope
    assert _resolve_cron_browser_scope("current", "session-scope") == "session-scope"
    with pytest.raises(ValueError, match="active browser scope"):
        _resolve_cron_browser_scope("current", None)
