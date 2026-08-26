"""Third-party memory imports must not rewrite the Hermes process environment.

``hindsight_api.config`` executes, at module import time::

    load_dotenv(find_dotenv(usecwd=True), override=True)

and every ``hindsight`` package pulls it in transitively. ``find_dotenv`` walks
up from the *process cwd*, which for a Hermes gateway is the hermes-agent
checkout — so it resolves to the default profile's ``~/.hermes/.env`` no matter
which profile is running, and ``override=True`` then replaces variables the
process already loaded from its own ``$HERMES_HOME/.env``.

That is not confined to Hindsight's own settings: a profile ``.env`` also holds
``DISCORD_BOT_TOKEN``, ``DISCORD_ALLOWED_CHANNELS``,
``DISCORD_FREE_RESPONSE_CHANNELS`` and ``MATRIX_ACCESS_TOKEN``. Memory
initializes lazily, on the first turn that uses it, so a non-default profile
answers exactly one message and then goes permanently silent: every later
message is dropped by a channel allow-list that now belongs to another bot.

The damage lands on the *first* import anywhere in the process, so every import
site has to be guarded — guarding only the obvious one leaves whichever path
runs first free to do the damage, and the guarded call then finds the module
already in ``sys.modules`` and nothing left to restore. That is exactly how the
first attempt at this fix failed in production: ``_check_local_runtime()`` runs
before ``initialize()`` reaches its import.
"""

import importlib
import os
import types

import pytest

hindsight = importlib.import_module("plugins.memory.hindsight")


@pytest.fixture
def clobbering_import(monkeypatch):
    """Replace importlib.import_module with one that rewrites the env.

    Stands in for the real packages: the env damage happens while the module
    body executes, i.e. inside ``import_module`` itself.
    """
    calls = []

    def fake_import_module(name):
        calls.append(name)
        os.environ["DISCORD_BOT_TOKEN"] = "default-profile-token"
        os.environ["DISCORD_ALLOWED_CHANNELS"] = "111,222"
        os.environ["HINDSIGHT_LEAKED_KEY"] = "from-foreign-dotenv"
        module = types.ModuleType(name)
        module.HindsightEmbedded = type("HindsightEmbedded", (), {})
        module.Hindsight = type("Hindsight", (), {})
        return module

    monkeypatch.setattr(hindsight.importlib, "import_module", fake_import_module)
    return calls


@pytest.fixture
def profile_env(monkeypatch):
    """A non-default profile's own environment."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "oracle-token")
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "999")
    monkeypatch.delenv("HINDSIGHT_LEAKED_KEY", raising=False)


def _assert_profile_env_intact():
    assert os.environ["DISCORD_BOT_TOKEN"] == "oracle-token"
    assert os.environ["DISCORD_ALLOWED_CHANNELS"] == "999"
    assert "HINDSIGHT_LEAKED_KEY" not in os.environ


class TestPreservedProcessEnv:
    def test_restores_an_overwritten_value(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "mine")
        with hindsight._preserved_process_env("fake"):
            os.environ["DISCORD_ALLOWED_CHANNELS"] = "someone-elses"
        assert os.environ["DISCORD_ALLOWED_CHANNELS"] == "mine"

    def test_removes_a_key_the_body_added(self, monkeypatch):
        monkeypatch.delenv("HERMES_TEST_ONLY_KEY", raising=False)
        with hindsight._preserved_process_env("fake"):
            os.environ["HERMES_TEST_ONLY_KEY"] = "leaked"
        assert "HERMES_TEST_ONLY_KEY" not in os.environ

    def test_restores_a_key_the_body_deleted(self, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "oracle-token")
        with hindsight._preserved_process_env("fake"):
            del os.environ["DISCORD_BOT_TOKEN"]
        assert os.environ["DISCORD_BOT_TOKEN"] == "oracle-token"

    def test_restores_even_when_the_body_raises(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "mine")
        with pytest.raises(RuntimeError):
            with hindsight._preserved_process_env("fake"):
                os.environ["DISCORD_ALLOWED_CHANNELS"] = "someone-elses"
                raise RuntimeError("import blew up")
        assert os.environ["DISCORD_ALLOWED_CHANNELS"] == "mine"

    def test_quiet_when_nothing_changed(self, monkeypatch, caplog):
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "mine")
        with caplog.at_level("INFO", logger=hindsight.logger.name):
            with hindsight._preserved_process_env("fake"):
                pass
        assert not caplog.records

    def test_logs_what_it_restored(self, monkeypatch, caplog):
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "mine")
        with caplog.at_level("INFO", logger=hindsight.logger.name):
            with hindsight._preserved_process_env("fake-runtime"):
                os.environ["DISCORD_ALLOWED_CHANNELS"] = "someone-elses"
        assert any(
            "fake-runtime" in r.getMessage() and "DISCORD_ALLOWED_CHANNELS" in r.getMessage()
            for r in caplog.records
        )


class TestImportGuarded:
    def test_returns_the_module(self, clobbering_import, profile_env):
        module = hindsight._import_guarded("hindsight")
        assert module.__name__ == "hindsight"

    def test_does_not_leak_the_foreign_env(self, clobbering_import, profile_env):
        hindsight._import_guarded("hindsight")
        _assert_profile_env_intact()


class TestEveryImportSiteIsGuarded:
    """Each production entry point that pulls in a hindsight package."""

    def test_check_local_runtime(self, clobbering_import, profile_env):
        # This is the one that regressed: it runs *before* initialize() reaches
        # its own import, so leaving it unguarded made every later guard a no-op.
        assert hindsight._check_local_runtime() == (True, None)
        assert clobbering_import == [
            "hindsight",
            "hindsight_embed.daemon_embed_manager",
        ]
        _assert_profile_env_intact()

    def test_check_local_runtime_still_reports_failure(self, monkeypatch, profile_env):
        def boom(name):
            raise RuntimeError("numpy exploded")

        monkeypatch.setattr(hindsight.importlib, "import_module", boom)
        available, reason = hindsight._check_local_runtime()
        assert available is False
        assert "numpy exploded" in reason
        _assert_profile_env_intact()

    def test_import_hindsight_embedded(self, clobbering_import, profile_env):
        assert hindsight._import_hindsight_embedded().__name__ == "HindsightEmbedded"
        _assert_profile_env_intact()
