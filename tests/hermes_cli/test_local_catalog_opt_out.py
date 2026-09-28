"""Local catalog refresh respects the profile's opt-out (#126888)."""

import io
import json
import threading
from unittest.mock import Mock

import pytest

from hermes_cli.local_runtime import catalog
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.parametrize("force", [False, True])
def test_disabled_refresh_preserves_catalog_and_retry_budget(monkeypatch, tmp_path, force):
    home = tmp_path / "disabled"
    home.mkdir()
    (home / "config.yaml").write_text("model_catalog:\n  enabled: false\n", encoding="utf-8")
    token = set_hermes_home_override(home)
    original = catalog.CATALOG
    monkeypatch.setattr(catalog, "CATALOG", original)
    monkeypatch.setattr(catalog, "_last_refresh_attempt", 0.0)
    fetch = Mock(return_value=io.BytesIO(b'{"schema_version": 1, "models": []}'))
    monkeypatch.setattr(catalog.urllib.request, "urlopen", fetch)
    try:
        assert catalog.refresh_catalog(force=force) is False
        fetch.assert_not_called()
        assert catalog.CATALOG is original
        assert catalog._last_refresh_attempt == 0.0
    finally:
        reset_hermes_home_override(token)


def test_background_refresh_keeps_calling_profile_and_offline_fallback(monkeypatch, tmp_path):
    disabled = tmp_path / "disabled"
    enabled = tmp_path / "enabled"
    for home, value in ((disabled, "false"), (enabled, "true")):
        home.mkdir()
        (home / "config.yaml").write_text(f"model_catalog:\n  enabled: {value}\n", encoding="utf-8")
    # The process default opts out; a scoped caller opts in. The worker must not
    # silently inherit the process default instead of its initiating profile.
    monkeypatch.setenv("HERMES_HOME", str(disabled))
    original = catalog.CATALOG
    monkeypatch.setattr(catalog, "CATALOG", original)
    monkeypatch.setattr(catalog, "_last_refresh_attempt", 0.0)
    monkeypatch.setattr(catalog, "_REFRESH_TTL_S", 0)
    started = []
    real_thread = threading.Thread

    def track_thread(*args, **kwargs):
        worker = real_thread(*args, **kwargs)
        started.append(worker)
        return worker

    monkeypatch.setattr(catalog.threading, "Thread", track_thread)
    fetch = Mock(return_value=io.BytesIO(json.dumps({"schema_version": 1, "models": []}).encode()))
    monkeypatch.setattr(catalog.urllib.request, "urlopen", fetch)
    try:
        for home, expected_workers in ((disabled, 0), (enabled, 1), (disabled, 1)):
            token = set_hermes_home_override(home)
            try:
                catalog.refresh_catalog_soon()
            finally:
                reset_hermes_home_override(token)
            for worker in started:
                worker.join(timeout=5)
                assert not worker.is_alive()
            assert len(started) == expected_workers
        fetch.assert_called_once()
        assert catalog.CATALOG == ()  # An enabled empty response is still a valid catalog.
        token = set_hermes_home_override(enabled)
        try:
            fetch.side_effect = OSError("offline")
            assert catalog.refresh_catalog(force=True) is False
            assert catalog.CATALOG == ()
        finally:
            reset_hermes_home_override(token)
    finally:
        for worker in started:
            worker.join(timeout=5)
