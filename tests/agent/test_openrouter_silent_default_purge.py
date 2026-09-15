"""Regression tests for issue #81952 — aux free default + env-ingestion WARNING.

KILL 2: the auxiliary lane's BUILT-IN OpenRouter fallback model (used only
when the user never set ``auxiliary.openrouter_model``) must be a ``:free``
SKU — the paid ``google/gemini-3.6-flash`` default meant silent real spend on
a lane the user never opted into.

KILL 3: auto-ingesting OPENROUTER_API_KEY from the environment into the
credential pool arms silent spend; it must log a WARNING naming the provider
and env var when a credential is newly ingested.
"""

import logging
import uuid

import pytest


class TestAuxiliaryOpenrouterDefaultIsFree:
    def test_builtin_openrouter_default_is_free_sku(self):
        from agent import auxiliary_client as ac

        assert ac._is_free_model(ac._OPENROUTER_MODEL), (
            "the built-in auxiliary OpenRouter fallback model must be a :free "
            "SKU — a paid built-in default is silent real spend (#81952)"
        )

    def test_builtin_default_matches_free_only_warning_recommendation(self):
        """The default is the same SKU the free_only warning tells users to set."""
        from agent import auxiliary_client as ac

        assert ac._OPENROUTER_MODEL == "nvidia/nemotron-3-ultra-550b-a55b:free"

    def test_user_configured_model_still_honored(self, monkeypatch):
        """auxiliary.openrouter_model from config wins over the built-in default."""
        from agent import auxiliary_client as ac

        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"auxiliary": {"openrouter_model": "google/gemini-3.6-flash"}},
        )
        free_only, model = ac._aux_openrouter_settings()
        assert model == "google/gemini-3.6-flash"


class TestEnvIngestionWarning:
    def _fresh_home(self, tmp_path, monkeypatch, token="sk-or-FAKEINGEST123"):
        home = tmp_path / "hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("OPENROUTER_API_KEY", token)
        return home

    def test_warns_on_new_openrouter_env_ingestion(self, tmp_path, monkeypatch, caplog):
        self._fresh_home(tmp_path, monkeypatch)
        from agent import credential_pool as cp

        monkeypatch.setattr(cp, "_ENV_INGESTION_WARNED", set())
        entries = []
        with caplog.at_level(logging.WARNING, logger=cp.logger.name):
            changed, sources = cp._seed_from_env("openrouter", entries)

        assert changed is True
        assert "env:OPENROUTER_API_KEY" in sources
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "Ingested OPENROUTER_API_KEY" in r.getMessage()
        ]
        assert warnings, "expected a WARNING for env->pool openrouter ingestion"
        assert "OpenRouter spend" in warnings[0]
        assert "hermes auth remove openrouter" in warnings[0]

    def test_warning_once_per_process(self, tmp_path, monkeypatch, caplog):
        self._fresh_home(tmp_path, monkeypatch)
        from agent import credential_pool as cp

        monkeypatch.setattr(cp, "_ENV_INGESTION_WARNED", set())
        with caplog.at_level(logging.WARNING, logger=cp.logger.name):
            cp._warn_env_ingestion_once("openrouter", "OPENROUTER_API_KEY")
            cp._warn_env_ingestion_once("openrouter", "OPENROUTER_API_KEY")
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "Ingested OPENROUTER_API_KEY" in r.getMessage()
        ]
        assert len(warnings) == 1

    def test_no_warning_when_entry_unchanged(self, tmp_path, monkeypatch, caplog):
        """Re-seeding an already-present, identical credential stays silent."""
        self._fresh_home(tmp_path, monkeypatch)
        from agent import credential_pool as cp

        entries = []
        changed, _ = cp._seed_from_env("openrouter", entries)
        assert changed is True

        monkeypatch.setattr(cp, "_ENV_INGESTION_WARNED", set())
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=cp.logger.name):
            changed_again, _ = cp._seed_from_env("openrouter", entries)
        assert changed_again is False
        assert not [
            r for r in caplog.records if "Ingested OPENROUTER_API_KEY" in r.getMessage()
        ]
