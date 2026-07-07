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
import os
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


def _pinned_provider_returns(runtime_kwargs: dict):
    """Convert a configure_runtime-* style dict into what
    _resolve_runtime_agent_kwargs_for_provider should return.

    Lets the regression tests below assert #59763 without spinning up the
    full ``resolve_runtime_provider`` plumbing — the helper itself is
    covered by the test_top_level_* tests above for the default route.
    """
    return {
        "api_key": runtime_kwargs.get("api_key"),
        "base_url": runtime_kwargs.get("base_url"),
        "provider": runtime_kwargs.get("provider"),
        "api_mode": runtime_kwargs.get("api_mode"),
        "command": runtime_kwargs.get("command"),
        "args": list(runtime_kwargs.get("args") or []),
        "credential_pool": runtime_kwargs.get("credential_pool"),
        "max_tokens": runtime_kwargs.get("max_tokens"),
    }


def test_pinned_provider_resolves_max_tokens_via_helper(isolated_home, monkeypatch):
    """Regression for #59763: ``_resolve_runtime_agent_kwargs_for_provider``
    must resolve ``max_tokens`` the same way the default route does, so
    channel overrides and persisted /model session overrides honour the
    same global cap instead of silently sending uncapped requests.
    """
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: openrouter
          max_tokens: 8192
        """
    )
    grun = fresh_gateway()
    runtime_kwargs = grun._resolve_max_tokens_cap(
        {"api_key": "sk-test", "provider": "openrouter", "max_output_tokens": 9999}
    )

    # Smoke: helper surfaces 8192 from model.max_tokens even though a
    # higher per-provider cap (9999) is also configured — global wins.
    assert runtime_kwargs == 8192

    # Now verify both return-dict shapes carry the cap.
    default = grun._resolve_runtime_agent_kwargs()
    assert default["max_tokens"] == 8192

    # The pinned-provider path uses the same helper, so its return must
    # also carry the cap. This is what previous code silently dropped.
    pinned = grun._resolve_runtime_agent_kwargs_for_provider("openrouter")
    assert pinned["max_tokens"] == 8192

    # The two returns must agree on every other field too (modulo
    # credential-pool identity which may differ).
    for key in ("api_mode", "provider", "base_url", "command", "args"):
        assert pinned.get(key) == default.get(key), key


def test_pinned_provider_env_var_wins(isolated_home, monkeypatch):
    """``HERMES_MAX_TOKENS`` overrides model.max_tokens on the pinned-provider
    route too — operators expecting a single env var to cap every route
    shouldn't need provider-specific config files."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: openrouter
          max_tokens: 8192
        """
    )
    monkeypatch.setenv("HERMES_MAX_TOKENS", "1024")
    grun = fresh_gateway()

    # Default route reads the env var (existing behaviour).
    assert grun._resolve_runtime_agent_kwargs()["max_tokens"] == 1024

    # Pinned-provider route reads it too.
    pinned = grun._resolve_runtime_agent_kwargs_for_provider("openrouter")
    assert pinned["max_tokens"] == 1024


def test_pinned_provider_no_cap_means_none(isolated_home):
    """When no global cap is set, the pinned-provider route returns
    ``max_tokens=None`` to mean "let the provider default apply" — same
    contract as the default route."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg("model:\n  default: glm-5.1\n  provider: openrouter\n")
    grun = fresh_gateway()

    assert grun._resolve_runtime_agent_kwargs_for_provider("openrouter")["max_tokens"] is None
