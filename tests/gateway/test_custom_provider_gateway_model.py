"""E2E: custom_providers[].model reaches gateway session runtime (#9702)."""

import importlib
import sys
import textwrap

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a writable config.yaml and a clean module cache.

    Copied from tests/gateway/test_max_tokens_propagation.py so this file
    re-imports hermes_cli/gateway against a real config without leaking
    that purge into sibling test files.
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
        for k in list(sys.modules.keys()):
            if k.startswith(("hermes_cli", "gateway")):
                del sys.modules[k]
        sys.modules.update(_saved)


def test_empty_default_adopts_custom_providers_model(isolated_home):
    """A named custom_providers entry with model, and no model.default,
    must surface that model on the gateway session runtime path."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          provider: local-ollama
        custom_providers:
          - name: local-ollama
            base_url: http://127.0.0.1:11434/v1
            api_key: sk-test
            model: qwen3:32b
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs()
    assert kw.get("model") == "qwen3:32b"

    runner = object.__new__(grun.GatewayRunner)
    runner._session_model_overrides = {}
    model, session_kw = runner._resolve_session_agent_runtime()
    assert model == "qwen3:32b"
    assert "model" not in session_kw
