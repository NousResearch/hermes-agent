import pytest

from agent import secret_scope
from agent.secret_scope import UnscopedSecretError
from hermes_constants import (
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from plugins.browser.browser_use.provider import BrowserUseBrowserProvider
from plugins.browser.browserbase.provider import BrowserbaseBrowserProvider
from plugins.browser.firecrawl.provider import FirecrawlBrowserProvider


@pytest.fixture(autouse=True)
def _reset_secret_scope():
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)


def test_browserbase_uses_profile_secret_scope_not_process_env(monkeypatch):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "foreign-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "foreign-project")
    monkeypatch.setenv("BROWSERBASE_BASE_URL", "https://foreign.example")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "BROWSERBASE_API_KEY": "scoped-key",
        "BROWSERBASE_PROJECT_ID": "scoped-project",
        "BROWSERBASE_BASE_URL": "https://scoped.example/",
    })
    try:
        config = BrowserbaseBrowserProvider()._get_config_or_none()
    finally:
        secret_scope.reset_secret_scope(token)

    assert config == {
        "api_key": "scoped-key",
        "project_id": "scoped-project",
        "base_url": "https://scoped.example",
    }


def test_browser_use_uses_profile_secret_scope_not_process_env(monkeypatch):
    monkeypatch.setenv("BROWSER_USE_API_KEY", "foreign-key")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"BROWSER_USE_API_KEY": "scoped-key"})
    try:
        config = BrowserUseBrowserProvider()._get_config_or_none(refresh_token=False)
    finally:
        secret_scope.reset_secret_scope(token)

    assert config == {
        "api_key": "scoped-key",
        "base_url": "https://api.browser-use.com/api/v3",
        "managed_mode": False,
    }


def test_firecrawl_uses_profile_secret_scope_not_process_env(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "foreign-key")
    monkeypatch.setenv("FIRECRAWL_API_URL", "https://foreign.example")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "FIRECRAWL_API_KEY": "scoped-key",
        "FIRECRAWL_API_URL": "https://scoped.example",
    })
    try:
        provider = FirecrawlBrowserProvider()
        assert provider.is_available() is True
        assert provider._api_url() == "https://scoped.example"
        assert provider._headers()["Authorization"] == "Bearer scoped-key"
    finally:
        secret_scope.reset_secret_scope(token)


def test_cloud_browser_provider_unscoped_multiplex_read_fails_closed(monkeypatch):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "foreign-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "foreign-project")

    secret_scope.set_multiplex_active(True)

    with pytest.raises(UnscopedSecretError):
        BrowserbaseBrowserProvider()._get_config_or_none()


@pytest.mark.parametrize("cleanup_path", ["inactivity", "atexit"])
def test_cloud_cleanup_restores_each_session_owner_context(
    monkeypatch,
    tmp_path,
    cleanup_path,
):
    from tools import browser_tool

    closed = []

    class Provider:
        name = "browserbase"

        def create_session(self, task_id):
            return {
                "session_name": f"session-{task_id}",
                "bb_session_id": f"remote-{task_id}",
                "cdp_url": None,
            }

        def close_session(self, session_id):
            closed.append((
                session_id,
                secret_scope.current_secret_scope(),
                get_hermes_home_override(),
            ))
            return True

    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: Provider())
    monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda _task_id: None)
    monkeypatch.setattr(browser_tool, "_run_browser_command", lambda *args, **kwargs: {})
    monkeypatch.setattr(browser_tool.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(browser_tool, "_reap_orphaned_browser_sessions", lambda: None)
    monkeypatch.setattr(browser_tool, "_cleanup_done", False)
    monkeypatch.setattr(browser_tool, "_cleanup_running", False)

    browser_tool._active_sessions.clear()
    browser_tool._session_last_activity.clear()
    browser_tool._cloud_cleanup_states.clear()
    browser_tool._cloud_cleanup_pending.clear()
    browser_tool._cloud_cleanup_in_progress.clear()
    try:
        for task_id in ("a", "b"):
            scope_token = secret_scope.set_secret_scope({
                "BROWSERBASE_API_KEY": f"key-{task_id}",
                "BROWSERBASE_PROJECT_ID": f"project-{task_id}",
                "UNRELATED_SECRET": "excluded",
            })
            home_token = set_hermes_home_override(str(tmp_path / task_id))
            try:
                browser_tool._get_session_info(task_id)
            finally:
                reset_hermes_home_override(home_token)
                secret_scope.reset_secret_scope(scope_token)

        if cleanup_path == "inactivity":
            browser_tool._session_last_activity.update({"a": 0.0, "b": 0.0})
            monkeypatch.setattr(browser_tool.time, "time", lambda: 100.0)
            monkeypatch.setattr(browser_tool, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 1)
            browser_tool._cleanup_inactive_browser_sessions()
        else:
            browser_tool._emergency_cleanup_all_sessions()
    finally:
        browser_tool._active_sessions.clear()
        browser_tool._session_last_activity.clear()
        browser_tool._cloud_cleanup_states.clear()
        browser_tool._cloud_cleanup_pending.clear()
        browser_tool._cloud_cleanup_in_progress.clear()

    assert closed == [
        (
            "remote-a",
            {"BROWSERBASE_API_KEY": "key-a", "BROWSERBASE_PROJECT_ID": "project-a"},
            str(tmp_path / "a"),
        ),
        (
            "remote-b",
            {"BROWSERBASE_API_KEY": "key-b", "BROWSERBASE_PROJECT_ID": "project-b"},
            str(tmp_path / "b"),
        ),
    ]
