"""Google Chat bot-id cache must resolve under get_hermes_home()."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import PlatformConfig

# Install google-* shims and import the adapter the same way the main suite does.
import tests.gateway.test_google_chat as _gc_suite  # noqa: F401
from plugins.platforms.google_chat.adapter import (  # noqa: E402
    GoogleChatAdapter,
    _ThreadCountStore,
)


def _base_config():
    cfg = PlatformConfig(enabled=True)
    cfg.extra.update(
        {
            "project_id": "test-project",
            "subscription_name": "projects/test-project/subscriptions/test-sub",
            "service_account_json": "/tmp/fake-sa.json",
        }
    )
    return cfg


@pytest.fixture()
def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default-home"))
    a = GoogleChatAdapter(_base_config())
    a._thread_count_store = _ThreadCountStore(tmp_path / "thread_counts.json")
    return a


def test_bot_id_cache_path_follows_hermes_home_env(adapter, tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "coder"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    path = adapter._bot_id_cache_path()

    assert path == home / "google_chat_bot_id.json"
    # Must not hardcode the non-profile ~/.hermes fallback when HERMES_HOME is set.
    assert Path.home() / ".hermes" / "google_chat_bot_id.json" != path


def test_bot_id_cache_path_honors_context_override(adapter, tmp_path):
    """Context-local override is invisible to raw os.getenv(HERMES_HOME)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    override = tmp_path / "override-home"
    override.mkdir()
    token = set_hermes_home_override(str(override))
    try:
        assert adapter._bot_id_cache_path() == override / "google_chat_bot_id.json"
    finally:
        reset_hermes_home_override(token)
