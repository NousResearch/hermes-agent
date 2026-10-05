"""The cron failure notice delivered to a chat follows the profile's active language, resolved per
call: a real temp HERMES_HOME with a user locale overlay and ``display.language`` in config.yaml."""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

from agent import i18n, i18n_layers
from cron.scheduler_failure_copy import provider_failure_notice


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()
    yield home
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


def _notice() -> str:
    return provider_failure_notice("Nightly", "abc123", "model_not_found", backup_provider_phrase="")


def test_provider_failure_notice_uses_overlay_of_active_language(home):
    overlay = {"gateway": {"cron": {"failure": {
        "provider": "DE[{job_name}|{job_id}|{action}]",
        "action_model_not_found": "Modell wechseln: {job_id}",
    }}}}
    (home / "locales" / "de.yaml").write_text(yaml.safe_dump(overlay, allow_unicode=True), encoding="utf-8")
    rendered = "DE[Nightly|abc123|Modell wechseln: abc123]"

    # The module was imported before the language was chosen: no import-time freeze.
    i18n.reset_language_cache()
    assert _notice() != rendered

    (home / "config.yaml").write_text(yaml.safe_dump({"display": {"language": "de"}}), encoding="utf-8")
    i18n.reset_language_cache()
    assert _notice() == rendered
