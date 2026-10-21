"""Codex fallback exhaustion recovery uses its native positive quota probe."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.chat_completion_helpers import _candidate_pool_exhausted


class Pool:
    provider = "openai-codex"

    def __init__(self, available=False):
        self.available = available

    def has_credentials(self):
        return True

    def has_available(self, model=None):
        return self.available

    def next_available_at(self, model=None):
        return None


def test_codex_stale_cooldown_cleared_after_positive_native_probe():
    agent = SimpleNamespace(_credential_pool=Pool())
    with patch("hermes_cli.auth.resolve_codex_runtime_credentials") as resolver, \
         patch("agent.credential_pool.load_pool", return_value=Pool(available=True)):
        assert _candidate_pool_exhausted(agent, "openai-codex", "gpt-6.1-sol") is False
    resolver.assert_called_once_with()


def test_codex_actual_exhaustion_remains_blocked():
    agent = SimpleNamespace(_credential_pool=Pool())
    with patch("hermes_cli.auth.resolve_codex_runtime_credentials", side_effect=RuntimeError("quota exhausted")), \
         patch("agent.credential_pool.load_pool", return_value=Pool()):
        assert _candidate_pool_exhausted(agent, "openai-codex", "gpt-6.1-sol") is True


def test_codex_recovery_error_fails_safe():
    agent = SimpleNamespace(_credential_pool=Pool())
    with patch("hermes_cli.auth.resolve_codex_runtime_credentials", side_effect=OSError("probe failed")), \
         patch("agent.credential_pool.load_pool", return_value=Pool()):
        assert _candidate_pool_exhausted(agent, "openai-codex", "gpt-6.1-sol") is True


def test_other_provider_does_not_invoke_codex_recovery():
    agent = SimpleNamespace(_credential_pool=Pool())
    with patch("hermes_cli.auth.resolve_codex_runtime_credentials") as resolver, \
         patch("agent.credential_pool.load_pool", return_value=Pool()):
        assert _candidate_pool_exhausted(agent, "openrouter", "model") is True
    resolver.assert_not_called()
