"""A ``pinned`` cron job must pin the durable provider identity, not the
resolved billing class.

``_main_model_pin`` stored the resolved runtime's provider tag. Every named
custom entry resolves to the literal ``"custom"``, so ``create_job(pinned=True)``
/ ``update_job({"pinned": True})`` pinned ``"custom"``; at fire time the
scheduler resolves that bare string through the ladder — to the OpenRouter
fallback when a key exists, or the credentialless-bare-custom AuthError when it
does not — instead of the configured entry (#109765). The pin must heal bare
``custom`` to the ``custom:<name>`` identity, the same lookup session
persistence uses.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cron import jobs
from hermes_cli import runtime_provider as rp

PROVIDER_KEY = "omniroute"
BASE_URL = "https://omniroute.invalid/v1"
CANONICAL = f"custom:{PROVIDER_KEY}"


def _named_custom_runtime(base_url: str, requested: str) -> dict:
    """What the resolver's named-custom rung reports for a configured entry:
    the literal billing tag ``custom`` plus the endpoint and the requested name."""
    return {"provider": "custom", "base_url": base_url, "requested_provider": requested,
            "api_key": "sk-test"}


@pytest.fixture
def named_custom_config(monkeypatch):
    config = {
        "model": {"default": "hermes-smart-stack"},
        "custom_providers": [
            {
                "name": PROVIDER_KEY,
                "base_url": BASE_URL,
                "api_key": "sk-test",
                "model": "hermes-smart-stack",
            },
        ],
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    return config


def _pin(monkeypatch, tmp_path, runtime):
    """Drive ``_main_model_pin`` with the config layer mocked: the model axis reads
    ``load_user_config_effective`` under a temp home, the heal reads ``rp.load_config``."""
    (tmp_path / "config.yaml").write_text("model:\n  default: main-model\n")
    monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)
    monkeypatch.setattr("hermes_cli.config_effective.load_user_config_effective",
                        lambda *a, **k: {"model": {"default": "main-model"}})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                        MagicMock(return_value=dict(runtime)))
    return jobs._main_model_pin()


def test_pinned_custom_provider_heals_to_named_identity(named_custom_config, monkeypatch, tmp_path):
    """The regression (#109765): the resolved tag `custom` must not be pinned as-is."""
    provider, _model = _pin(monkeypatch, tmp_path, _named_custom_runtime(BASE_URL, CANONICAL))
    assert provider == CANONICAL


def test_unresolvable_bare_custom_keeps_current_behaviour(monkeypatch, tmp_path):
    """No configured entry to heal to: keep the bare tag instead of inventing an identity."""
    config = {"model": {}}
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    provider, _model = _pin(
        monkeypatch, tmp_path, _named_custom_runtime("https://unconfigured.invalid/v1", "custom"))
    assert provider == "custom"


def test_non_custom_tag_is_stored_unchanged(monkeypatch, tmp_path):
    """Providers that are their own identity (openrouter, anthropic, ...) need no healing."""
    provider, _model = _pin(
        monkeypatch, tmp_path,
        {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
         "requested_provider": "auto"},
    )
    assert provider == "openrouter"


PROVIDERS_KEY = "tokenrhythm"
PROVIDERS_BASE_URL = "https://api.tokenrhythm.invalid/v1"
PROVIDERS_CANONICAL = f"custom:{PROVIDERS_KEY}"


def test_providers_only_named_entry_heals(monkeypatch, tmp_path):
    """A keyed ``providers:`` entry with no ``custom_providers`` at all (#109765 review):
    ``_find_custom_identity`` scans ``providers:`` first, so the pin still heals."""
    config = {
        "model": {"default": "hermes-smart-stack"},
        "providers": {
            PROVIDERS_KEY: {
                "name": "TokenRhythm",
                "api": PROVIDERS_BASE_URL,
                "key_env": "TOKENRHYTHM_API_KEY",
                "model": "hermes-smart-stack",
            },
        },
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    provider, _model = _pin(monkeypatch, tmp_path, _named_custom_runtime(PROVIDERS_BASE_URL, "custom"))
    assert provider == PROVIDERS_CANONICAL


BARE_NAME_KEY = "llamacpp-qwen38"
BARE_NAME_BASE_URL = "http://llamacpp.local:8080/v1"


def test_bare_name_provider_spelling_heals(monkeypatch, tmp_path):
    """A bare-name ``model.provider`` spelling (no ``custom:`` prefix), as independently
    verified on #109765: the resolve layer still reports tag ``custom`` with the bare
    name as ``requested_provider`` — the heal must recover the identity regardless."""
    config = {
        "model": {"default": "qwen3.8-27b", "provider": BARE_NAME_KEY},
        "providers": {
            BARE_NAME_KEY: {
                "base_url": BARE_NAME_BASE_URL,
                "model": "qwen3.8-27b",
                "key_env": "LLAMACPP_QWEN38_API_KEY",
            },
        },
    }
    monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(rp, "_get_model_config", lambda: config["model"])
    provider, _model = _pin(monkeypatch, tmp_path, _named_custom_runtime(BARE_NAME_BASE_URL, BARE_NAME_KEY))
    assert provider == f"custom:{BARE_NAME_KEY}"


class TestPinnedJobStoresHealedIdentity:
    """End to end through the job store: ``pinned=True`` writes the healed identity onto
    the record, both at create and on a later update."""

    @staticmethod
    def _store(monkeypatch, tmp_path, runtime):
        (tmp_path / "config.yaml").write_text("model:\n  default: main-model\n")
        monkeypatch.setattr(jobs, "get_hermes_home", lambda: tmp_path, raising=True)
        state = {"jobs": []}
        monkeypatch.setattr(jobs, "load_jobs", lambda: list(state["jobs"]), raising=True)
        monkeypatch.setattr(jobs, "save_jobs", lambda j: state.__setitem__("jobs", list(j)), raising=True)
        monkeypatch.setattr(jobs, "resolve_job_ref", lambda ref: next(
            (j for j in state["jobs"] if j["id"] == ref), None), raising=True)
        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider",
                            MagicMock(return_value=dict(runtime)))
        return jobs

    def test_create_pinned_stores_healed_identity(self, named_custom_config, monkeypatch, tmp_path):
        jobs = self._store(monkeypatch, tmp_path,
                           _named_custom_runtime(BASE_URL, CANONICAL))
        job = jobs.create_job(prompt="do a thing", schedule="every 1 hour", pinned=True)
        assert job["provider"] == CANONICAL

    def test_update_pinned_stores_healed_identity(self, named_custom_config, monkeypatch, tmp_path):
        jobs = self._store(monkeypatch, tmp_path,
                           _named_custom_runtime(BASE_URL, CANONICAL))
        job = jobs.create_job(prompt="do a thing", schedule="every 1 hour")
        assert job["provider"] is None
        locked = jobs.update_job(job["id"], {"pinned": True})
        assert locked["provider"] == CANONICAL
