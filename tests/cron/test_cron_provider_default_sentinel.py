"""``cron.model`` may name the provider's current default instead of a model id.

A model id pinned in config.yaml goes stale the day the provider retires that SKU, and
the job then fails with HTTP 404 instead of following the provider. The sentinel is an
explicit opt-in (the operator wrote it), so a scheduled job never silently starts
running on a model nobody chose - and the fail-fast still applies when the default
cannot be resolved.
"""
from __future__ import annotations

import pytest

from cron import scheduler as sched


def _write_config(tmp_path, body: str):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")
    return home


def _prepare(monkeypatch, tmp_path, body: str):
    home = _write_config(tmp_path, body)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.setattr(sched, "_get_hermes_home", lambda: home, raising=False)
    return home


def test_the_sentinel_resolves_to_the_providers_current_default(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path, "model:\n  provider: openrouter\ncron:\n  model: provider-default\n")
    monkeypatch.setattr(
        "hermes_cli.models.get_default_model_for_provider",
        lambda provider: "vendor/current:free",
    )

    config = sched._load_cron_job_config({"id": "j", "name": "j", "prompt": "x"}, "j", "j")

    assert config.model == "vendor/current:free"


def test_the_sentinel_uses_the_cron_model_provider(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path,
             "model:\n  provider: openrouter\ncron:\n  model: provider-default\n  model_provider: nous\n")
    seen = []

    def _default(provider):
        seen.append(provider)
        return "vendor/nous-default"

    monkeypatch.setattr("hermes_cli.models.get_default_model_for_provider", _default)

    config = sched._load_cron_job_config({"id": "j", "name": "j", "prompt": "x"}, "j", "j")

    assert config.model == "vendor/nous-default"
    assert seen == ["nous"]


def test_a_per_job_pin_still_wins_over_the_sentinel(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path, "model:\n  provider: openrouter\ncron:\n  model: provider-default\n")
    monkeypatch.setattr(
        "hermes_cli.models.get_default_model_for_provider",
        lambda provider: "vendor/current:free",
    )

    config = sched._load_cron_job_config(
        {"id": "j", "name": "j", "prompt": "x", "model": "vendor/pinned:free"}, "j", "j")

    assert config.model == "vendor/pinned:free"


def test_the_sentinel_still_fails_fast_when_no_default_can_be_resolved(monkeypatch, tmp_path):
    """An empty model reaches the provider as an opaque 400 (#23979)."""
    _prepare(monkeypatch, tmp_path, "model:\n  provider: openrouter\ncron:\n  model: provider-default\n")
    monkeypatch.setattr("hermes_cli.models.get_default_model_for_provider", lambda provider: "")

    with pytest.raises(RuntimeError, match="has no model configured"):
        sched._load_cron_job_config({"id": "j", "name": "j", "prompt": "x"}, "j", "j")


def test_no_model_anywhere_still_fails_fast(monkeypatch, tmp_path):
    """The fail-fast is unchanged: a job with no model at all is refused."""
    _prepare(monkeypatch, tmp_path, "model:\n  provider: openrouter\n")
    monkeypatch.setattr(
        "hermes_cli.models.get_default_model_for_provider",
        lambda provider: "vendor/current:free",
    )

    with pytest.raises(RuntimeError, match="has no model configured"):
        sched._load_cron_job_config({"id": "j", "name": "j", "prompt": "x"}, "j", "j")


def test_the_error_points_at_the_sentinel(monkeypatch, tmp_path):
    _prepare(monkeypatch, tmp_path, "model:\n  provider: openrouter\n")

    with pytest.raises(RuntimeError, match="provider-default"):
        sched._load_cron_job_config({"id": "j", "name": "j", "prompt": "x"}, "j", "j")
