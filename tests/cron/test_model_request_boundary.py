"""Regression coverage for the maintainer-only model-request overlay."""

from types import SimpleNamespace

import cron.scheduler as scheduler


def test_unbound_agent_skips_maintainer_timeout_authority(monkeypatch, tmp_path):
    """Generic cron agents must not load Civic Assure profile-only state."""
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    agent = SimpleNamespace(_civic_assure_model_request_binding=None)

    assert scheduler._civic_assure_check_model_request_timeout(agent) is None
