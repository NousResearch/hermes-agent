"""Regression test for #105979: agent rebuilt on every turn after /model switch.

After a /model switch to a model on a different api_mode route than the config
default, _ensure_runtime_credentials() must NOT overwrite self.api_mode with the
default's route.  Without this, the route signature mismatches the one stored at
agent init and the agent is torn down and rebuilt on every turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


# A minimal CLI stub that mimics a post-/model-switch state where the session
# model is on a different api_mode route than the config default.
class _ModelSwitchedCLI(CLIAgentSetupMixin):
    def __init__(self, *, model: str, provider: str, api_mode: str):
        self.model = model
        self.requested_provider = provider
        self.provider = provider
        self.api_key = "test-key"
        self.base_url = "https://opencode.ai/zen/go"
        self.api_mode = api_mode  # set by /model switch (e.g. anthropic_messages)
        self.acp_command = None
        self.acp_args = []
        self.agent = SimpleNamespace()  # simulate existing agent
        self._fallback_model = []
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._credential_pool = None
        self.service_tier = None
        self._model_is_default = False  # post /model switch

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


def _write_profile_config(hermes_home) -> None:
    """Config where the default model uses chat_completions (glm on opencode-go)."""
    (hermes_home / "config.yaml").write_text(
        """
model:
  default: glm-5.3-flash
  provider: opencode-go
providers:
  opencode-go:
    base_url: https://opencode.ai/zen/go
    api_key: test-key
""",
        encoding="utf-8",
    )


def test_ensure_runtime_credentials_passes_target_model(monkeypatch):
    """_ensure_runtime_credentials must pass target_model=self.model to
    resolve_runtime_provider so the re-resolved api_mode matches the session
    model, not the config default."""
    from hermes_constants import get_hermes_home

    _write_profile_config(get_hermes_home())

    # Simulate: session model is qwen3.7-max on opencode-go (anthropic_messages),
    # but config default is glm-5.3-flash (chat_completions).
    cli = _ModelSwitchedCLI(
        model="qwen3.7-max", provider="opencode-go", api_mode="anthropic_messages"
    )

    # Track what resolve_runtime_provider was called with
    _call_kwargs = {}

    def _tracking_resolve(**kwargs):
        _call_kwargs.update(kwargs)
        # Return a runtime with the CORRECT api_mode for the session model
        return {
            "provider": "opencode-go",
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go",
            "api_mode": "anthropic_messages",  # correct for qwen3.7-max
            "requested_provider": "opencode-go",
        }

    # Mock at the source module since the import is lazy inside the method
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", _tracking_resolve):
        result = cli._ensure_runtime_credentials()

    assert result is True
    # The critical assertion: target_model must be passed
    assert "target_model" in _call_kwargs, (
        "resolve_runtime_provider must receive target_model to avoid api_mode "
        "mismatch after /model switch (see #105979)"
    )
    assert _call_kwargs["target_model"] == "qwen3.7-max"


def test_api_mode_preserved_after_ensure_credentials(monkeypatch):
    """After a /model switch sets api_mode=anthropic_messages,
    _ensure_runtime_credentials must not overwrite it with chat_completions."""
    from hermes_constants import get_hermes_home

    _write_profile_config(get_hermes_home())

    cli = _ModelSwitchedCLI(
        model="qwen3.7-max", provider="opencode-go", api_mode="anthropic_messages"
    )

    def _returning_correct_mode(**kwargs):
        return {
            "provider": "opencode-go",
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go",
            "api_mode": "anthropic_messages",  # correct for qwen3.7-max
            "requested_provider": "opencode-go",
        }

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", _returning_correct_mode):
        cli._ensure_runtime_credentials()

    # api_mode must still be the correct value from /model switch
    assert cli.api_mode == "anthropic_messages", (
        f"api_mode was overwritten to {cli.api_mode!r} after _ensure_runtime_credentials; "
        f"expected 'anthropic_messages' (see #105979)"
    )


def test_route_signature_matches_after_ensure_credentials(monkeypatch):
    """The route signature computed after _ensure_runtime_credentials must match
    the one stored at agent init when the session model hasn't changed."""
    from hermes_constants import get_hermes_home
    from hermes_cli.cli_agent_setup_mixin import _route_signature

    _write_profile_config(get_hermes_home())

    cli = _ModelSwitchedCLI(
        model="qwen3.7-max", provider="opencode-go", api_mode="anthropic_messages"
    )

    def _returning_correct_mode(**kwargs):
        return {
            "provider": "opencode-go",
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go",
            "api_mode": "anthropic_messages",  # correct for qwen3.7-max
            "requested_provider": "opencode-go",
        }

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", _returning_correct_mode):
        cli._ensure_runtime_credentials()

    # Compute the signature the same way _resolve_turn_agent_config does
    runtime = {
        "api_key": cli.api_key,
        "base_url": cli.base_url,
        "provider": cli.provider,
        "requested_provider": getattr(cli, "requested_provider", cli.provider),
        "api_mode": cli.api_mode,
        "command": cli.acp_command,
        "args": list(cli.acp_args or []),
        "credential_pool": getattr(cli, "_credential_pool", None),
    }
    current_signature = _route_signature(cli.model, runtime)

    # The signature stored at agent init (simulating what _init_agent sets)
    stored_signature = _route_signature(
        "qwen3.7-max",
        {"provider": "opencode-go", "requested_provider": "opencode-go",
         "base_url": "https://opencode.ai/zen/go", "api_mode": "anthropic_messages",
         "command": None, "args": []},
    )

    assert current_signature == stored_signature, (
        f"Route signature mismatch after _ensure_runtime_credentials: "
        f"current={current_signature} stored={stored_signature}. "
        f"This would cause agent rebuild every turn (#105979)."
    )
