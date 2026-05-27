"""Tests for the OAuth 401 actionable-guidance branch in
``agent.conversation_loop.run_conversation`` for qwen-oauth,
google-gemini-cli, and minimax-oauth.

Source-inspection style (matches ``test_nous_oauth_401_guidance.py``): we
assert that the guidance strings exist in the function body so that the
user-facing hint cannot be silently removed by a future refactor.

Regression context: the guidance branch only covered openai-codex, xai-oauth,
and nous.  qwen-oauth, google-gemini-cli, and minimax-oauth — all OAuth-only
providers with no API key path — fell through to the generic
"Your API key was rejected... run hermes setup" message, which is wrong advice
for pure-OAuth providers.
"""
from __future__ import annotations

import inspect

from agent import conversation_loop


def test_oauth_providers_in_401_gate():
    """All OAuth-only providers must appear in the 401-guidance gating set."""
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "\"qwen-oauth\"" in source
    assert "\"google-gemini-cli\"" in source
    assert "\"minimax-oauth\"" in source


def test_qwen_oauth_401_guidance_strings_present():
    """User-facing remediation strings for qwen-oauth 401s must exist."""
    source = inspect.getsource(conversation_loop.run_conversation)

    # Must identify this as an OAuth token problem, not an API key problem.
    assert "Qwen OAuth token was rejected" in source

    # Must give the exact re-auth command used by hermes status and docs.
    assert "qwen auth qwen-oauth" in source


def test_google_gemini_cli_401_guidance_strings_present():
    """User-facing remediation strings for google-gemini-cli 401s must exist."""
    source = inspect.getsource(conversation_loop.run_conversation)

    # Must identify this as an OAuth token problem.
    assert "Google Gemini OAuth token was rejected" in source

    # Must give the exact re-auth command from auth_commands.py.
    assert "hermes auth add google-gemini-cli --type oauth" in source


def test_minimax_oauth_401_guidance_strings_present():
    """User-facing remediation strings for minimax-oauth 401s must exist."""
    source = inspect.getsource(conversation_loop.run_conversation)

    # Must identify this as an OAuth token problem.
    assert "MiniMax OAuth token was rejected" in source

    # Must give the exact re-auth command shown by hermes status.
    assert "hermes auth add minimax-oauth" in source
