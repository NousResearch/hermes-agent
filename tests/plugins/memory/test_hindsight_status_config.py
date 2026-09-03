"""`hermes memory status` must show Hindsight's REAL resolved config.

Hindsight keeps its native config in $HERMES_HOME/hindsight/config.json rather
than under config.yaml's ``memory.hindsight`` key, so the ``provider_config``
cmd_status passes in is normally empty or stale. ``get_status_config`` loads
the real file instead; these tests pin that, and that the mode-specific fields
are scoped to the active mode (#68073) — a cloud user should not be told to set
``HINDSIGHT_LLM_PROVIDER``.
"""

import json

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider


@pytest.fixture
def hindsight_home(tmp_path, monkeypatch):
    """Point the provider at an isolated $HERMES_HOME/hindsight/config.json.

    ``get_hermes_home()`` resolves and caches the home directory, so setting
    ``HERMES_HOME`` after import is not enough — patch the symbol the provider
    module actually calls, or the tests silently read the developer's own
    Hindsight config and pass no matter what the code does.
    """

    def _write(cfg: dict):
        home = tmp_path / "hermes_home"
        (home / "hindsight").mkdir(parents=True, exist_ok=True)
        (home / "hindsight" / "config.json").write_text(
            json.dumps(cfg), encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: home
        )
        return home

    return _write


def test_status_reads_the_real_config_not_the_passed_in_one(hindsight_home):
    """A stale/empty provider_config must not win over the real config file."""
    hindsight_home({"mode": "cloud", "api_url": "https://real.example", "bank_id": "b"})

    display = HindsightMemoryProvider().get_status_config(
        {"mode": "local_embedded", "api_url": "https://stale.example"}
    )

    assert display["mode"] == "cloud"
    assert display["api_url"] == "https://real.example"


def test_cloud_mode_hides_local_embedded_llm_fields(hindsight_home):
    """The env-var hint list is filtered by mode: cloud has no LLM provider.

    This is the point of #68073 — ``hermes memory status`` listed every schema
    field carrying an env_var regardless of the active mode, so a cloud user
    was told to set ``HINDSIGHT_LLM_PROVIDER``, which does nothing for them.
    """
    hindsight_home({"mode": "cloud", "api_url": "https://api.example"})

    display = HindsightMemoryProvider().get_status_config({})

    assert display["mode"] == "cloud"
    assert "llm_provider" not in display
    assert "llm_model" not in display


def test_local_embedded_mode_shows_llm_fields_and_hides_api_url(hindsight_home):
    """Conversely, the embedded daemon has no api_url to report."""
    hindsight_home(
        {
            "mode": "local_embedded",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        }
    )

    display = HindsightMemoryProvider().get_status_config({})

    assert display["mode"] == "local_embedded"
    assert display["llm_provider"] == "openai"
    assert display["llm_model"] == "gpt-4o-mini"
    assert "api_url" not in display


def test_openai_compatible_reports_its_base_url(hindsight_home):
    """base_url only matters for the openai_compatible provider."""
    hindsight_home(
        {
            "mode": "local_embedded",
            "llm_provider": "openai_compatible",
            "llm_model": "local",
            "llm_base_url": "http://localhost:1234/v1",
        }
    )

    display = HindsightMemoryProvider().get_status_config({})

    assert display["llm_base_url"] == "http://localhost:1234/v1"
