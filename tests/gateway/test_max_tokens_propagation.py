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


# ---------------------------------------------------------------------------
# Paths that previously DROPPED the cap and fell to the custom-profile 65536
# floor: the per-provider kwargs resolver (channel overrides + /model
# rehydration), the /model override application, and CLI credential
# resolution. Regression coverage for the off-by-one ContextWindowExceeded
# incident (cap dropped -> 65536 reserved -> input trimmed to window-65536 by
# the client tokenizer -> vLLM chat template pushed it 1 token over the
# backend window).
# ---------------------------------------------------------------------------

_MYLOCAL_CFG = """
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


def test_provider_specific_path_carries_max_tokens(isolated_home):
    """_resolve_runtime_agent_kwargs_for_provider must not drop the cap."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(_MYLOCAL_CFG)
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs_for_provider("mylocal")
    assert kw["max_tokens"] == 12000


def test_provider_specific_path_global_wins(isolated_home):
    """model.max_tokens beats the provider cap on the per-provider path too."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg("""
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
        """)
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs_for_provider("mylocal")
    assert kw["max_tokens"] == 16384


def test_apply_session_model_override_forwards_max_tokens(isolated_home):
    """/model overrides that carry max_tokens propagate it to runtime kwargs."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(_MYLOCAL_CFG)
    grun = fresh_gateway()

    class _Stub:
        _session_model_overrides = {
            "sess-1": {"model": "other", "provider": "mylocal", "max_tokens": 9999}
        }

    model, kwargs = grun.GatewayRunner._apply_session_model_override(
        _Stub(), "sess-1", "glm-5.1", {"max_tokens": None}
    )
    assert model == "other"
    assert kwargs["max_tokens"] == 9999


def test_resolve_configured_max_tokens_helper(isolated_home, monkeypatch):
    """Direct helper contract: env > model.max_tokens > runtime arg > None."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg("model:\n  default: glm-5.1\n  provider: openrouter\n")
    grun = fresh_gateway()

    assert grun._resolve_configured_max_tokens(12000) == 12000
    assert grun._resolve_configured_max_tokens(None) is None
    assert grun._resolve_configured_max_tokens(0) is None
    assert grun._resolve_configured_max_tokens("x") is None
    monkeypatch.setenv("HERMES_MAX_TOKENS", "2048")
    assert grun._resolve_configured_max_tokens(12000) == 2048


def test_cli_credential_resolution_lifts_provider_cap(isolated_home, monkeypatch):
    """CLI _ensure_runtime_credentials adopts the provider cap when the global
    keys are unset — previously it ignored max_output_tokens entirely and the
    transport fell to the custom-profile 65536 floor."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(_MYLOCAL_CFG)
    fresh_gateway()  # re-import hermes_cli against the isolated config
    import importlib as _il

    mixin_mod = _il.import_module("hermes_cli.cli_agent_setup_mixin")

    class _StubCLI(mixin_mod.CLIAgentSetupMixin):
        def __init__(self):
            self.requested_provider = "mylocal"
            self._explicit_api_key = None
            self._explicit_base_url = None
            self._fallback_model = []
            self.api_key = None
            self.base_url = None
            self.provider = "mylocal"
            self.api_mode = "chat_completions"
            self.acp_command = None
            self.acp_args = []
            self.model = "glm-5.1"
            self.agent = None
            self.max_tokens = None

        def _normalize_model_for_provider(self, provider):
            return False

    cli_obj = _StubCLI()
    assert cli_obj._ensure_runtime_credentials() is True
    assert cli_obj.max_tokens == 12000
    assert cli_obj._max_tokens_from_provider is True

    # Global key set in __init__ (simulated) must not be overwritten.
    cli_obj2 = _StubCLI()
    cli_obj2.max_tokens = 32768  # as if model.max_tokens resolved it
    assert cli_obj2._ensure_runtime_credentials() is True
    assert cli_obj2.max_tokens == 32768

    # Provider-sourced cap re-resolves (clears) when the provider stops
    # advertising one.
    cli_obj.requested_provider = "openrouter"
    import hermes_cli.cli_agent_setup_mixin as _m

    def _fake_resolve(**kwargs):
        return {
            "api_key": "sk-x",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
        }

    import hermes_cli.runtime_provider as _rp

    monkeypatch.setattr(_rp, "resolve_runtime_provider", _fake_resolve)
    assert cli_obj._ensure_runtime_credentials() is True
    assert cli_obj.max_tokens is None
    assert cli_obj._max_tokens_from_provider is False


def test_turn_agent_config_carries_max_tokens(isolated_home):
    """_resolve_turn_agent_config runtime dict includes the session cap so
    background agents don't fall back to the profile default."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(_MYLOCAL_CFG)
    fresh_gateway()
    import importlib as _il

    mixin_mod = _il.import_module("hermes_cli.cli_agent_setup_mixin")

    class _StubCLI(mixin_mod.CLIAgentSetupMixin):
        def __init__(self):
            self.api_key = "sk-test"
            self.base_url = "http://localhost:11434/v1"
            self.provider = "mylocal"
            self.api_mode = "chat_completions"
            self.acp_command = None
            self.acp_args = []
            self.model = "glm-5.1"
            self.max_tokens = 12000
            self._fast_mode = False

    route = _StubCLI()._resolve_turn_agent_config("hi")
    assert route["runtime"]["max_tokens"] == 12000
