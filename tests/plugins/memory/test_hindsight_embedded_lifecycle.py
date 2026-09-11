"""Embedded-daemon lifecycle contracts for the Hindsight provider.

Covers two update-time regressions that silently disable local_embedded memory:

* the profile env comparison must ignore keys hindsight-embed writes itself, or a healthy
  daemon is SIGTERMed on every session start (NousResearch/hermes-agent#82943);
* local_embedded must install a fastmcp branch that can coexist with Hermes' own mcp pin
  (NousResearch/hermes-agent#95855).
"""

import pytest

from plugins.memory.hindsight.embedded import (
    _build_embedded_profile_env,
    _embedded_profile_env_is_current,
    _embedded_profile_env_path,
    _materialize_embedded_profile_env,
)
from plugins.memory.hindsight.settings import _LOCAL_EMBEDDED_DEPS


@pytest.fixture
def embedded_config(tmp_path, monkeypatch):
    """A local_embedded config whose profile env lands under a temp HOME."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return {
        "mode": "local_embedded",
        "profile": "hermes",
        "llm_provider": "openrouter",
        "llm_model": "google/gemini-2.5-flash-lite",
        "llm_api_key": "test-key",
        "idle_timeout": 0,
    }


def test_freshly_materialized_env_is_current(embedded_config):
    """What the plugin just wrote must read back as current — else it restarts in a loop."""
    _materialize_embedded_profile_env(embedded_config)

    assert _embedded_profile_env_is_current(embedded_config)


def test_foreign_runtime_key_does_not_invalidate_env(embedded_config):
    """hindsight-embed appends HINDSIGHT_API_PORT; the plugin owns none of it (#82943).

    A whole-mapping equality check reports "config changed" forever and stops the daemon on
    every session start, so recall/retain hit a restarting API for the rest of the window.
    """
    _materialize_embedded_profile_env(embedded_config)
    profile_env = _embedded_profile_env_path(embedded_config)
    profile_env.write_text(profile_env.read_text() + "HINDSIGHT_API_PORT=9177\n", encoding="utf-8")

    assert _embedded_profile_env_is_current(embedded_config)


def test_changed_plugin_owned_value_invalidates_env(embedded_config):
    """The restart path must still fire for a key the plugin actually owns."""
    _materialize_embedded_profile_env(embedded_config)
    assert _embedded_profile_env_is_current(embedded_config)

    embedded_config["llm_model"] = "google/gemini-2.5-pro"

    assert not _embedded_profile_env_is_current(embedded_config)


def test_missing_plugin_owned_key_invalidates_env(embedded_config):
    """A truncated/hand-edited env file is not current even though nothing conflicts."""
    _materialize_embedded_profile_env(embedded_config)
    profile_env = _embedded_profile_env_path(embedded_config)
    kept = [
        line for line in profile_env.read_text(encoding="utf-8").splitlines()
        if not line.startswith("HINDSIGHT_API_LLM_MODEL=")
    ]
    profile_env.write_text("\n".join(kept) + "\n", encoding="utf-8")

    assert not _embedded_profile_env_is_current(embedded_config)


def test_local_embedded_deps_constrain_fastmcp_above_the_mcp_conflict():
    """local_embedded must pin fastmcp to the branch that requires mcp>=2.0 (#95855).

    hindsight-all -> hindsight-api-slim -> fastmcp>=3.2.0 (no ceiling). fastmcp<4 caps
    mcp<2.0, which cannot coexist with Hermes' mcp==2.0.0, so an unconstrained resolve
    yields fastmcp 3.x and `import hindsight` dies on a missing mcp 1.x symbol.
    """
    assert "hindsight-all" in _LOCAL_EMBEDDED_DEPS

    fastmcp_specs = [spec for spec in _LOCAL_EMBEDDED_DEPS if spec.startswith("fastmcp")]
    assert fastmcp_specs, "local_embedded must constrain fastmcp explicitly"

    spec = fastmcp_specs[0]
    assert ">=4" in spec, f"fastmcp floor must clear the mcp<2.0 branch, got {spec!r}"
    assert "<5" in spec, f"dependency policy requires an upper bound, got {spec!r}"


def test_env_builder_owns_only_hindsight_api_keys(embedded_config):
    """The plugin's env surface stays namespaced, so foreign keys are recognisably foreign."""
    built = _build_embedded_profile_env(embedded_config)

    assert built
    assert all(key.startswith("HINDSIGHT_") for key in built)
