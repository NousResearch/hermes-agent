"""Regression: ``resolve_provider_client("minimax-oauth", ...)`` must build a
refresh-capable Anthropic auxiliary client, not silently return (None, None).

The auxiliary router dispatches on ``PROVIDER_REGISTRY[*].auth_type``.
``minimax-oauth`` registers ``auth_type == "oauth_minimax"``; without an
explicit arm the resolver falls through the ``oauth_device_code`` /
``oauth_external`` branch, logs a one-time warning, and returns (None, None).
Every aux task pinned to ``provider: minimax-oauth`` (compression,
title_generation, …) then silently re-routes to the Step-2 fallback chain —
the operator's explicit configuration never reaches the wire.

The fix routes through ``resolve_minimax_oauth_runtime_credentials(as_token_provider=True)``
and wraps the resulting Anthropic SDK client in ``AnthropicAuxiliaryClient``. The callable
bearer mints a fresh access token per outbound request because MiniMax's tokens live
~15 minutes and a static string would 401 mid-session.

``is_oauth`` is derived via ``anthropic_route_is_oauth(base_url, token_provider)``: the
MiniMax host is a third-party Anthropic-protocol endpoint, so the wrapper must NOT carry
the Claude Code OAuth identity (mcp__ tool-name wire transforms, system-prompt rewrites,
response prefix stripping) — those are native api.anthropic.com-only (#114967).
The arm lives in ``agent/auxiliary_client_registry.py`` (one new module, not a
trampoline into a second sibling); absent/not-logged-in credentials follow the
``_log_once_debug`` contract (``test_auxiliary_client_resolve_dedup.py``) so a logged-out
user gets one debug line, not a WARNING-with-traceback per aux resolution (#21521).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.auth_constants import AuthError

INFERENCE_BASE_URL = "https://api.minimax.io/anthropic"


def _runtime_creds():
    """Return what resolve_minimax_oauth_runtime_credentials(as_token_provider=True) yields."""
    return {
        "provider": "minimax-oauth",
        "api_key": MagicMock(name="minimax_token_provider", return_value="fresh-bearer"),
        "base_url": INFERENCE_BASE_URL,
        "source": "oauth",
    }


def test_resolve_minimax_oauth_builds_anthropic_wrapper_with_third_party_wire():
    """Happy path: token-provider + base_url → AnthropicAuxiliaryClient with the
    third-party is_oauth invariant (False for api.minimax.io), Anthropic SDK built
    with the callable bearer (not a static string), and the resolved model passed
    through verbatim.
    """
    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient,
        resolve_provider_client,
    )

    fake_anthropic_client = MagicMock(name="anthropic_sdk_client")
    creds = _runtime_creds()

    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        return_value=creds,
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=fake_anthropic_client,
    ) as mock_build:
        client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")

    assert client is not None, (
        "minimax-oauth must produce a configured client when credentials are "
        "present, but the resolver returned (None, None). The oauth_minimax "
        "arm in the registry auth-type dispatch table is missing."
    )
    assert isinstance(client, AnthropicAuxiliaryClient), (
        f"minimax-oauth must build an AnthropicAuxiliaryClient (the inference "
        f"endpoint is /anthropic). Got {type(client).__name__}."
    )
    assert client.chat.completions._is_oauth is False, (
        "MiniMax is a third-party Anthropic-protocol endpoint: is_oauth enables "
        "Claude Code-native transforms (mcp__ tool-name wire prefixing, identity "
        "rewrites, response prefix stripping) that 401/403 or corrupt tool calls "
        "there. anthropic_route_is_oauth('https://api.minimax.io/anthropic', …) "
        "is False by contract (tests/agent/test_anthropic_route_oauth_identity.py)."
    )
    # The callable token provider — not a string — must reach the Anthropic SDK so
    # the SDK mints a fresh access token per outbound request (MiniMax tokens
    # are short-lived; a static bearer 401s mid-session).
    positional, _kwargs = mock_build.call_args[0], mock_build.call_args[1]
    assert positional[0] is creds["api_key"], (
        "build_anthropic_client must receive the callable token provider, "
        "not a stringified snapshot of the current bearer."
    )
    assert positional[1] == INFERENCE_BASE_URL
    assert model == "MiniMax-M3"


def test_resolve_minimax_oauth_tool_names_unprefixed_on_wire():
    """Witness for the user-visible contract: a tool list sent through the
    minimax-oauth auxiliary wrapper must reach the wire with its registry names
    verbatim. ``is_oauth=True`` would rename ``read_file`` to ``mcp__read_file``
    (and alias session_search/memory) on a host that never round-trips those
    names — every tool call against MiniMax would 400 or name a nonexistent tool.
    """
    from agent.auxiliary_client import resolve_provider_client

    captured = {}

    def _fake_create(client, api_kwargs, **kwargs):
        captured["tools"] = api_kwargs.get("tools")
        return SimpleNamespace(
            content=[],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        return_value=_runtime_creds(),
    ), patch(
        "agent.anthropic_adapter.build_anthropic_client",
        return_value=MagicMock(name="anthropic_sdk_client"),
    ), patch(
        "agent.anthropic_adapter.create_anthropic_message",
        side_effect=_fake_create,
    ):
        client, _model = resolve_provider_client("minimax-oauth", "MiniMax-M3")
        assert client is not None
        client.chat.completions.create(
            model="MiniMax-M3",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {"type": "function", "function": {"name": "read_file", "description": "x", "parameters": {}}},
                {"type": "function", "function": {"name": "session_search", "description": "y", "parameters": {}}},
            ],
        )
    wire_names = sorted(t["name"] for t in captured["tools"])
    assert wire_names == ["read_file", "session_search"], (
        "Third-party Anthropic-protocol endpoints must see unprefixed tool names; "
        "the Claude Code OAuth wire renamer must not run for api.minimax.io."
    )


def test_resolve_minimax_oauth_missing_credentials_log_once_debug(caplog):
    """Absent / not-logged-in / AuthError → (None, None), no exception, and one
    DEBUG record per process — never a per-call WARNING with traceback.

    The resolver contract is "absent → call_llm's fallback chain", never
    "absent → exception"; the compression step must fall through to its
    Step-2 providers instead of crashing the turn. And because every aux task
    (compression, title generation, background review) re-resolves on each
    call, a per-call WARNING would recreate the exact log-spam complaint
    #21521 was filed about — worse once a quarantine wipes the tokens.
    """
    import agent.auxiliary_client_registry as acr
    from agent.auxiliary_client import resolve_provider_client

    acr._LOGGED_MINIMAX_ABSENT_KEYS.clear()
    acr._LOGGED_MINIMAX_UNEXPECTED_KEYS.clear()
    resolved: tuple = (None, None)
    with patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        side_effect=AuthError(
            "Not logged into MiniMax OAuth.", provider="minimax-oauth",
            code="not_logged_in", relogin_required=True),
    ):
        with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
            for _ in range(5):
                resolved = resolve_provider_client("minimax-oauth", "MiniMax-M3")

    client, model = resolved

    assert client is None
    assert model is None
    recs = [r for r in caplog.records if "minimax-oauth" in r.getMessage()]
    # Five resolutions → exactly one debug record, zero warnings, zero tracebacks.
    assert len(recs) == 1, (
        f"expected exactly one deduped log record, got {len(recs)}: "
        f"{[r.getMessage() for r in recs]}"
    )
    assert recs[0].levelno == logging.DEBUG, (
        "logged-out minimax-oauth must be debug-once, not a warning"
    )
    assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
        "logged-out minimax-oauth must not warn per aux resolution (#21521 spam)"
    )
    assert not any(r.exc_info for r in caplog.records), (
        "an expected not-logged-in state never carries a traceback"
    )


def test_resolve_minimax_oauth_probe_reads_raw_state_no_refresh():
    """``aux_probe_mode()`` answers "resolvable?" for availability gates and must
    not touch the network: the arm reads ``get_provider_auth_state("minimax-oauth")``
    raw and stubs when access_token + inference_base_url are present, skipping the
    refresh that an expired token would otherwise trigger (mirrors
    ``credential_pool.py::_seed_minimax_singleton``).
    """
    import agent.auxiliary_client as aux
    from agent.auxiliary_client import resolve_provider_client

    expired_state = {
        "access_token": "expired-but-present",
        "refresh_token": "rt",
        "expires_at": "2001-01-01T00:00:00+00:00",  # long expired
        "inference_base_url": INFERENCE_BASE_URL,
    }

    with patch(
        "hermes_cli.auth.get_provider_auth_state", return_value=expired_state,
    ) as mock_state, patch(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
    ) as mock_resolve:
        with aux.aux_probe_mode():
            client, model = resolve_provider_client("minimax-oauth", "MiniMax-M3")

    assert model == "MiniMax-M3"
    assert client is not None and isinstance(client, aux._AuxProbeClientStub), (
        "probe mode must stub minimax-oauth from the raw persisted state"
    )
    assert client.base_url == INFERENCE_BASE_URL
    mock_resolve.assert_not_called()
    mock_state.assert_called_once_with("minimax-oauth")
