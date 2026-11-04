"""Gateway watcher uses the owning profile scope for automatic catalog refresh."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.run_watchers import _refresh_profile_provider_models


def test_refresh_reads_profile_config_and_secrets_in_own_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "first", tmp_path / "second"]
    for home in homes:
        home.mkdir()

    @contextmanager
    def scoped(home):
        token = set_hermes_home_override(str(home))
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    monkeypatch.setattr("gateway.run._profile_runtime_scope", scoped)

    def fake_load_config():
        from hermes_constants import get_hermes_home
        if "first" in str(get_hermes_home()):
            return {
                "provider_models": {
                    "background_refresh": {
                        "interval_seconds": 60,
                        "providers": ["custom:local-cpa"],
                    },
                },
            }
        return {}

    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)

    from hermes_cli.provider_model_refresh import RefreshSummary

    with patch(
        "hermes_cli.provider_model_refresh.refresh_configured_provider_models",
        return_value=RefreshSummary(1, 0, ()),
    ):
        assert _refresh_profile_provider_models(str(homes[0])) == (60, 1, 0)
        assert _refresh_profile_provider_models(str(homes[1])) == (0, 0, 0)
