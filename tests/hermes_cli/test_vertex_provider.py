"""Tests for Vertex AI runtime-provider resolution and profile registration.

Covers: provider-profile registration + aliases, alias canonicalization,
resolve_runtime_provider(vertex) minting an OAuth token, and the friendly
AuthError when credentials can't be resolved. No network calls.
"""

from __future__ import annotations

import pytest


def test_resolve_runtime_provider_raises_autherror_when_unresolved(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(va, "get_vertex_config", lambda: (None, None))
    with pytest.raises(AuthError):
        rp.resolve_runtime_provider(requested="vertex")


def test_vertex_registered_in_provider_registry():
    """PROVIDER_REGISTRY (hermes_cli.auth) is what agent/auxiliary_client.py's
    resolve_provider_client() looks up before dispatching on auth_type. Without
    an entry here, the ``elif pconfig.auth_type == "vertex":`` branch there is
    unreachable dead code — every auxiliary Vertex call (vision, title
    generation, MoA reference/aggregator slots, ...) fails at the
    ``pconfig is None`` guard before ever reaching it."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get("vertex")
    assert cfg is not None
    assert cfg.auth_type == "vertex"


def test_vertex_registered_in_hermes_overlays():
    """hermes_cli.providers.get_provider("vertex") backs
    _preserve_provider_with_base_url() in agent/auxiliary_named_route.py, which
    decides whether a MoA slot's resolved Vertex (base_url, api_key) pair
    keeps its "vertex" provider identity or silently collapses to "custom" —
    losing the identity _refresh_provider_credentials() needs to re-mint an
    expired OAuth2 token on a 401."""
    from hermes_cli.providers import get_provider

    resolved = get_provider("vertex")
    assert resolved is not None
    assert resolved.auth_type == "vertex"


class _CountingCreds:
    """google-auth Credentials stand-in: every refresh() is one OAuth JWT exchange (a mint)."""

    def __init__(self, mints: list):
        self.token, self.expired, self.expiry, self._mints = None, False, None, mints

    def refresh(self, _request):
        self._mints.append(1)
        self.token = f"ya29.minted-{len(self._mints)}"


def _vertex_cli_shell(monkeypatch, tmp_path, *, loadable: bool):
    import types

    import agent.vertex_adapter as va
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    mints: list = []
    sa = tmp_path / "sa.json"
    sa.write_text('{"project_id": "sa-proj"}')
    va._creds_cache.clear()
    monkeypatch.setattr(va, "_ensure_google_auth", lambda: True)
    request_mod = types.SimpleNamespace(Request=lambda: None)
    monkeypatch.setattr(va, "google", types.SimpleNamespace(auth=types.SimpleNamespace(
        transport=types.SimpleNamespace(requests=request_mod))), raising=False)
    monkeypatch.setattr(va, "_resolve_credentials_path", lambda explicit=None: str(sa))
    monkeypatch.setattr(va, "_vertex_config", lambda: {"region": "us-central1"})
    monkeypatch.setattr(va, "_load_credentials",
                        lambda path, raw: (_CountingCreds(mints), "sa-proj") if loadable else None)

    shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
    shell.model, shell.requested_provider = "google/gemini-3-flash-preview", "vertex"
    shell._explicit_api_key = shell._explicit_base_url = None
    shell._fallback_model = []
    shell.tool_progress_mode = "off"
    shell.agent = shell.api_key = shell.base_url = shell.provider = shell.acp_command = None
    shell.api_mode, shell.acp_args = "chat_completions", []
    shell._normalize_model_for_provider = lambda _provider: False
    shell._maybe_print_free_tier_available_notice = lambda: None
    return shell, mints


def test_gateway_backed_cli_resolves_vertex_route_without_minting(monkeypatch, tmp_path):
    """The classic CLI is a thin client: the gateway session mints the Vertex token for the turn it
    runs. Resolving the CLI's route must not run a second OAuth exchange per launch."""
    shell, mints = _vertex_cli_shell(monkeypatch, tmp_path, loadable=True)

    assert shell._ensure_runtime_credentials() is True
    assert mints == [], "the CLI process minted a Vertex token the gateway mints again"
    assert shell.provider == "vertex"
    assert shell.base_url == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/sa-proj/locations/us-central1/endpoints/openapi")
    assert callable(shell.api_key)
    # A caller that really sends from this process still gets a token, minted once and reused.
    assert shell.api_key() == shell.api_key() == "ya29.minted-1"
    assert mints == [1]


def test_gateway_backed_cli_keeps_vertex_credential_failure_copy(monkeypatch, tmp_path, capsys):
    """Unloadable Vertex credentials still fail in the CLI with main's actionable copy."""
    shell, mints = _vertex_cli_shell(monkeypatch, tmp_path, loadable=False)

    assert shell._ensure_runtime_credentials() is False
    err = capsys.readouterr().err
    assert "Vertex AI credentials could not be resolved" in err and "VERTEX_CREDENTIALS_PATH" in err, err
    assert mints == []
