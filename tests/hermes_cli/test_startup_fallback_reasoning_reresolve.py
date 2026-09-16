"""Startup provider fallback must re-resolve the CLI-level reasoning_config.

``self.reasoning_config`` is resolved once at init time for the launch model
(``cli._init_prompt_and_reasoning``) and is passed verbatim to the lazily
built agent. ``_resolve_fallback_runtime`` swaps ``requested_provider`` /
``model`` but never touched ``reasoning_config``, so when the fallback model
has a stricter effort contract than the primary, the first request went out
with the primary's effort and the provider rejected it (e.g. always-thinking
models 400 on a mismatched effort). The mid-session fallback path already
re-resolves (``_reresolve_fallback_reasoning_config``); the startup path must
go through the same chokepoint so per-model ``reasoning_overrides`` apply to
the fallback model too.
"""

from __future__ import annotations

import pytest


class _StubCLI:
    """Minimal surface ``_resolve_fallback_runtime`` touches."""

    def __init__(self, *, model, fallback_chain, runtime, config):
        self.requested_provider = "primary-provider"
        self.model = model
        self._fallback_model = fallback_chain
        # The launch-model resolution the init path performs (primary accepts "medium").
        self.reasoning_config = {"enabled": True, "effort": "medium"}
        self._console_logs: list[str] = []
        self._runtime = runtime
        self._config = config

    def _console_print(self, msg):
        self._console_logs.append(msg)


def _make_harness(inner):
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    class _Harness(CLIAgentSetupMixin):
        def __init__(self, inner):
            self.__dict__.update(inner.__dict__)

    return _Harness(inner)


@pytest.fixture()
def stub_cli(monkeypatch, tmp_path):
    """Wire the fallback entry + config the chokepoint reads, without touching disk state."""
    import sys

    repo_root = tmp_path  # unused; keep signature simple
    config = {
        "agent": {
            "reasoning_effort": "medium",
            "reasoning_overrides": {"glm-5.3-flash": "high"},
        },
    }
    cli = _StubCLI(
        model="codex-gpt55",
        fallback_chain=[{"provider": "zai", "model": "glm-5.3-flash"}],
        runtime={
            "provider": "zai",
            "api_key": "sk-test",
            "base_url": "https://api.z.ai",
        },
        config=config,
    )

    import hermes_cli.cli_agent_setup_mixin as mixin_mod

    # The chain: AuthError on the primary -> resolve_entry_api_key ok ->
    # resolve_runtime_provider returns the stub runtime -> load_config returns our config.
    def fake_resolve_runtime_provider(**kwargs):
        return cli._runtime

    def fake_load_config():
        return cli._config

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: "sk-fallback"
    )
    # ``_resolve_fallback_runtime`` imports ``_cprint``/``logger`` from the real ``cli``
    # module at call time; provide a stub with the same surface so the method runs
    # without booting the full CLI.
    cli_mod = type(sys)("cli")
    import logging

    def _cprint(msg):
        pass

    cli_mod._cprint = _cprint
    cli_mod.logger = logging.getLogger("cli-test")
    monkeypatch.setitem(sys.modules, "cli", cli_mod)
    return cli


def test_startup_fallback_reresolves_reasoning_config(stub_cli):
    """After the model swap the CLI-level config must reflect the fallback model's override."""
    from hermes_cli.auth_constants import AuthError

    harness = _make_harness(stub_cli)
    runtime = harness._resolve_fallback_runtime(
        AuthError("primary auth failed", provider="primary-provider")
    )
    assert runtime is not None
    assert harness.requested_provider == "zai"
    assert harness.model == "glm-5.3-flash"
    # The regression: this stayed {"enabled": True, "effort": "medium"} before the fix.
    assert harness.reasoning_config == {"enabled": True, "effort": "high"}


def test_startup_fallback_reasoning_resolution_failure_keeps_current(stub_cli):
    """A config load failure must not kill the swap — the old config rides on."""
    from hermes_cli.auth_constants import AuthError

    harness = _make_harness(stub_cli)

    def boom():
        raise RuntimeError("config load failed")

    # load_config is imported inside the method from hermes_cli.config; patch there.
    import hermes_cli.config as config_mod

    orig = config_mod.load_config
    config_mod.load_config = boom
    try:
        runtime = harness._resolve_fallback_runtime(
            AuthError("primary auth failed", provider="primary-provider")
        )
        assert runtime is not None
        assert harness.model == "glm-5.3-flash"
        assert harness.reasoning_config == {"enabled": True, "effort": "medium"}
    finally:
        config_mod.load_config = orig
