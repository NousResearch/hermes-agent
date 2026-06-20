"""Regression tests for max_tokens propagation from config.yaml to AIAgent.

Covers #20741: `model.max_tokens` was silently dropped before reaching the
gateway-spawned agent, so providers without a hardcoded default (OpenRouter
free models, Ollama Cloud, custom OpenAI-compatible endpoints) truncated long
generations with `finish_reason="length"`.

Precedence verified here:
    HERMES_MAX_TOKENS env  >  model.max_tokens  >  per-provider
    max_output_tokens  >  None
"""

import importlib
import sys
import textwrap

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a writable config.yaml and a clean module cache.

    These tests deliberately re-import ``hermes_cli`` / ``gateway`` so each
    config write is read fresh. To avoid leaking that purge into sibling test
    files in the same worker (which breaks their import-time mocks), we snapshot
    the affected modules and restore them on teardown.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)

    _saved = {
        k: v
        for k, v in sys.modules.items()
        if k.startswith(("hermes_cli", "gateway"))
    }

    def write_cfg(body: str) -> None:
        (hermes_home / "config.yaml").write_text(textwrap.dedent(body))

    def fresh_gateway():
        for mod in list(sys.modules.keys()):
            if mod.startswith(("hermes_cli", "gateway")):
                del sys.modules[mod]
        return importlib.import_module("gateway.run")

    try:
        yield write_cfg, fresh_gateway
    finally:
        # Drop anything we (re)imported, then restore the pre-test snapshot so
        # the next test file sees the module objects it was loaded with.
        for k in list(sys.modules.keys()):
            if k.startswith(("hermes_cli", "gateway")):
                del sys.modules[k]
        sys.modules.update(_saved)


def test_top_level_max_tokens_propagates(isolated_home):
    """model.max_tokens is read into the gateway runtime kwargs (#20741)."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: openrouter
          max_tokens: 16384
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs()
    assert kw["max_tokens"] == 16384


def test_per_provider_max_output_tokens_fallback(isolated_home):
    """A custom provider's max_output_tokens fills in when no global is set."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: mylocal
        providers:
          mylocal:
            api: http://localhost:11434/v1
            api_key: sk-test
            default_model: glm-5.1
            max_output_tokens: 12000
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs()
    assert kw["max_tokens"] == 12000


def test_global_max_tokens_beats_per_provider(isolated_home):
    """The documented global model.max_tokens wins over a provider cap."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: mylocal
          max_tokens: 16384
        providers:
          mylocal:
            api: http://localhost:11434/v1
            api_key: sk-test
            default_model: glm-5.1
            max_output_tokens: 12000
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs()
    assert kw["max_tokens"] == 16384


def test_env_override_beats_everything(isolated_home, monkeypatch):
    """HERMES_MAX_TOKENS is the internal override mechanism (highest priority)."""
    write_cfg, fresh_gateway = isolated_home
    monkeypatch.setenv("HERMES_MAX_TOKENS", "2048")
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: mylocal
          max_tokens: 16384
        providers:
          mylocal:
            api: http://localhost:11434/v1
            api_key: sk-test
            default_model: glm-5.1
            max_output_tokens: 12000
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs()
    assert kw["max_tokens"] == 2048


def test_no_config_leaves_max_tokens_none(isolated_home):
    """No cap configured anywhere -> max_tokens is None (no spurious limit)."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: openrouter
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs()
    assert kw["max_tokens"] is None


def test_lift_helper_accepts_alias_and_rejects_garbage(isolated_home):
    """_lift_max_output_tokens accepts both keys, ignores non-positive/non-int."""
    write_cfg, _ = isolated_home
    write_cfg("model:\n  provider: openrouter\n")
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli"):
            del sys.modules[mod]
    rp = importlib.import_module("hermes_cli.runtime_provider")

    out: dict = {}
    rp._lift_max_output_tokens({"max_output_tokens": 8192}, out)
    assert out["max_output_tokens"] == 8192

    out = {}
    rp._lift_max_output_tokens({"max_tokens": 4096}, out)
    assert out["max_output_tokens"] == 4096

    for bad in ({"max_output_tokens": 0}, {"max_output_tokens": "x"}, {}):
        out = {}
        rp._lift_max_output_tokens(bad, out)
        assert "max_output_tokens" not in out
