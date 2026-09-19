"""`hermes memory status` must show Hindsight's REAL resolved config.

Hindsight keeps its native config in $HERMES_HOME/hindsight/config.json rather
than under config.yaml's ``memory.hindsight`` key, so the ``provider_config``
cmd_status passes in is normally empty or stale. ``get_status_config`` loads
the real file instead; these tests pin that, and that the mode-specific fields
are scoped to the active mode (#68073) — a cloud user should not be told to set
``HINDSIGHT_LLM_PROVIDER``. The final test drives the shared env-var
filter in ``cmd_status`` itself — the path teknium1 asked for regression
coverage on.
"""

import json

import pytest

import hermes_cli.memory_setup as memory_setup
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


def _run_status_with_real_provider(hindsight_home, monkeypatch, capsys, native_cfg):
    """Run ``cmd_status`` against the REAL Hindsight provider and return its stdout.

    The provider is real (real ``get_config_schema``/``get_status_config``), and
    only its availability is stubbed so ``cmd_status`` takes the branch that
    prints the env-var hints. Everything under test — the "when"-clause filter
    in ``cmd_status`` — is the production code path.
    """
    hindsight_home(native_cfg)
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    monkeypatch.delenv("HINDSIGHT_LLM_API_KEY", raising=False)

    provider = HindsightMemoryProvider()
    monkeypatch.setattr(HindsightMemoryProvider, "is_available", lambda self: False)
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("hindsight", "API key / local", provider)],
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "hindsight", "hindsight": {}}},
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools", lambda *a, **k: {"memory"}
    )

    memory_setup.cmd_status(object())
    out = capsys.readouterr().out
    assert "  Missing:" in out, out
    return out.split("  Missing:")[1].split("Note:")[0]


def test_local_embedded_cmd_status_filters_env_hints_by_mode(
    hindsight_home, monkeypatch, capsys
):
    """The shared filter in ``cmd_status`` must drop other modes' env vars.

    Regression test for #68073's local/embedded half. ``get_status_config``
    resolves the native ``local_embedded`` config, and ``cmd_status`` then
    filters the schema's env-var hints against it: both ``HINDSIGHT_API_KEY``
    entries (cloud and local_external) are meaningless to the embedded daemon
    and must be omitted, while the unset ``HINDSIGHT_LLM_API_KEY`` — the one
    the user actually has to set here — must be retained.
    """
    missing = _run_status_with_real_provider(
        hindsight_home,
        monkeypatch,
        capsys,
        {"mode": "local_embedded", "llm_provider": "openai"},
    )

    assert "HINDSIGHT_LLM_API_KEY" in missing
    assert "HINDSIGHT_API_KEY" not in missing.replace("HINDSIGHT_LLM_API_KEY", "")


def test_cloud_cmd_status_filters_env_hints_by_mode(hindsight_home, monkeypatch, capsys):
    """The same filter keeps cloud's ``HINDSIGHT_API_KEY`` and drops the LLM one.

    The mirror image of the test above, pinning that the filter is genuinely
    mode-driven in both directions rather than special-casing local_embedded.
    """
    missing = _run_status_with_real_provider(
        hindsight_home,
        monkeypatch,
        capsys,
        {"mode": "cloud", "api_url": "https://api.example"},
    )

    assert "HINDSIGHT_API_KEY" in missing
    assert "HINDSIGHT_LLM_API_KEY" not in missing
