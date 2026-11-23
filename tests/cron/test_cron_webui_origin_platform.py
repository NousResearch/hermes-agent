"""Regression: WebUI cron origin must not be stamped with a non-deliverable platform.

WebUI (hermes-webui/api/streaming.py) binds ``HERMES_SESSION_PLATFORM='webui'`` and a chat id
in the agent subprocess. ``webui`` is NOT a Platform enum member / registered adapter, so
fire-time delivery resolves it with ``Platform('webui')`` -> ValueError -> ``unknown platform
'webui'`` and every ``deliver=origin`` fire silently drops. WebUI is also NOT stateless
(``async_delivery_supported()`` is True there), so the old api_server guard missed it.

``_origin_from_env`` must validate the origin platform via the SAME ``Platform(...)``
resolution the fire-time delivery path uses, and drop unresolvable ones so the
home-channel fallback + creation-time notice apply. Static members (telegram/qqbot) and
dynamically-registered plugin adapters must keep being stamped.
"""

import json
import logging

import pytest

from gateway.session_context import reset_session_vars

_SESSION_ENV_KEYS = (
    "HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_THREAD_ID",
    "HERMES_SESSION_MESSAGE_ID", "HERMES_SESSION_USER_ID", "HERMES_SESSION_SCOPE_ID",
    "HERMES_SESSION_CHAT_NAME", "HERMES_SESSION_SOURCE", "HERMES_CRON_SESSION",
)


@pytest.fixture(autouse=True)
def _clean_session_env(monkeypatch):
    """WebUI injects real env vars; clear ContextVars so get_session_env reads os.environ,
    and guarantee async_delivery_supported() is True (WebUI is NOT stateless)."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    for key in _SESSION_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    reset_session_vars()
    yield
    reset_session_vars()


def _bind_session(monkeypatch, platform, chat_id="chat-abc"):
    """Bind a WebUI-style session identity via real env vars (as streaming.py:3605 does)."""
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", chat_id)
    reset_session_vars()  # ContextVars unset -> get_session_env falls back to os.environ


# --- unit: the guard ---------------------------------------------------------------

def test_webui_session_stamps_no_origin(monkeypatch, caplog):
    """HERMES_SESSION_PLATFORM=webui -> _origin_from_env returns None, with a debug reason."""
    from gateway.session_context import async_delivery_supported
    from cron.job_args import _origin_from_env

    _bind_session(monkeypatch, "webui", "e9e6831c44b3")
    # This is exactly why the old guard missed it: WebUI supports async delivery.
    assert async_delivery_supported() is True
    with caplog.at_level(logging.DEBUG, logger="tools.cronjob_tools"):
        origin = _origin_from_env("every 10m")
    assert origin is None
    assert any(
        r.levelno == logging.DEBUG and "webui" in r.getMessage()
        for r in caplog.records
    ), "a debug log must explain why the origin was dropped"


def test_arbitrary_unknown_platform_is_dropped(monkeypatch):
    """The guard is registry-based, not a hardcoded 'webui' denylist."""
    from cron.job_args import _origin_from_env

    _bind_session(monkeypatch, "totally_fake_platform_xyz")
    assert _origin_from_env("every 10m") is None


@pytest.mark.parametrize("platform", ["telegram", "qqbot"])
def test_deliverable_enum_platform_keeps_origin(monkeypatch, platform):
    """Reverse control: static Platform members are still stamped (no regression)."""
    from cron.job_args import _origin_from_env

    _bind_session(monkeypatch, platform, "chat-123")
    origin = _origin_from_env("every 10m")
    assert origin is not None
    assert origin["platform"] == platform
    assert origin["chat_id"] == "chat-123"


def test_registered_plugin_platform_keeps_origin(monkeypatch):
    """A bundled plugin resolved dynamically by Platform._missing_ is still stamped."""
    from gateway.config import Platform
    from cron.job_args import _origin_from_env

    assert Platform("irc").value == "irc"  # dynamic member, not a static enum member
    _bind_session(monkeypatch, "irc", "irc-channel")
    origin = _origin_from_env("every 10m")
    assert origin is not None
    assert origin["platform"] == "irc"


# --- E2E: real jobs.json through the cronjob() tool --------------------------------

class _NoopScheduler:
    def register_job(self, job):
        return None


def _create_in_session(monkeypatch, tmp_home, platform, chat_id, name):
    from unittest.mock import patch

    from cron.jobs import use_cron_store
    from tools.cronjob_tools import cronjob

    _bind_session(monkeypatch, platform, chat_id)
    with patch("cron.scheduler_provider.resolve_cron_scheduler", return_value=_NoopScheduler()):
        with use_cron_store(tmp_home):
            return cronjob(
                action="create", schedule="every 10m",
                prompt=f"WebUI origin probe {name}", name=name, deliver="origin",
            )


def test_e2e_webui_create_persists_null_origin_and_no_webui_target(monkeypatch, tmp_path):
    """Create deliver=origin in a WebUI session; jobs.json origin must be null and
    _resolve_delivery_targets must not produce a webui target."""
    from cron import scheduler_delivery as sd
    from cron.jobs import load_jobs, use_cron_store

    tmp_home = tmp_path / "home"
    result = json.loads(_create_in_session(
        monkeypatch, tmp_home, "webui", "e9e6831c44b3", "probe-webui-origin"))
    assert result["success"] is True, result

    raw = (tmp_home / "cron" / "jobs.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    webui_job = next(j for j in data["jobs"] if j["name"] == "probe-webui-origin")
    assert webui_job["origin"] is None
    assert '"origin": null' in raw
    with use_cron_store(tmp_home):
        stored = load_jobs()[0]
    targets = sd._resolve_delivery_targets(stored)
    assert all(str(t.get("platform", "")).lower() != "webui" for t in targets), targets


@pytest.mark.parametrize("platform", ["telegram", "qqbot"])
def test_e2e_messaging_create_still_stamps_origin(monkeypatch, tmp_path, platform):
    """Reverse control E2E: same create in a Telegram/qqbot session stamps origin."""
    from cron.jobs import load_jobs, use_cron_store

    tmp_home = tmp_path / "home"
    name = f"probe-{platform}-origin"
    result = json.loads(_create_in_session(
        monkeypatch, tmp_home, platform, "chat-555", name))
    assert result["success"] is True, result

    raw = (tmp_home / "cron" / "jobs.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    job = next(j for j in data["jobs"] if j["name"] == name)
    assert job["origin"]["platform"] == platform
    assert job["origin"]["chat_id"] == "chat-555"
    with use_cron_store(tmp_home):
        assert load_jobs()[0]["origin"]["platform"] == platform
