"""Regression: _try_anthropic() must fall back to the legacy token resolver
when the credential pool is present but has no usable entry.

Root cause (observed 2026-07-05): the pooled Anthropic OAuth entry expired and
its refresh_token was stale, so `_select_pool_entry("anthropic")` returned
`(True, None)` — pool exists, no selectable entry. The old `_try_anthropic`
hard-failed on that branch (`return None, None`), even though a perfectly
valid `ANTHROPIC_TOKEN` / credentials-file token was available. This wedged
every auxiliary task routed to Anthropic (goal judge → "no auxiliary client
configured"), while the MAIN session stayed healthy because it resolves the
env token directly.

openrouter (test_try_openrouter_pool_exhausted_falls_back_to_env) and codex
(TestBuildCodexClient.test_pool_without_selected_entry_falls_back_to_auth_store)
already fall through to their standalone credential on `(True, None)`. This
test pins the same invariant for anthropic so the three paths stay symmetric:
a temporarily dead pool entry must never hard-fail when a valid standalone
credential exists.
"""

from unittest.mock import MagicMock, patch


class TestAnthropicPoolExhaustedFallsBackToEnv:
    def test_pool_present_no_entry_falls_back_to_resolve_token(self, monkeypatch):
        """pool=(True, None) but a valid env token exists → client is built."""
        monkeypatch.setenv("ANTHROPIC_TOKEN", "«redacted:sk-…»-oauth-token")
        with patch(
            "agent.auxiliary_client._select_pool_entry", return_value=(True, None)
        ), patch(
            "agent.anthropic_adapter.build_anthropic_client"
        ) as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient

            client, model = _try_anthropic()

        assert client is not None, (
            "_try_anthropic must fall back to resolve_anthropic_token() when the "
            "pool is present but has no usable entry (parity with openrouter/codex)"
        )
        assert isinstance(client, AnthropicAuxiliaryClient)
        # Default aux model when none configured.
        assert model == "claude-haiku-4-5-20251001"
        # Must have used the env/legacy token, not a pooled entry.
        assert mock_build.call_args.args[0] == "«redacted:sk-…»-oauth-token"

    def test_pool_present_no_entry_and_no_token_still_returns_none(self, monkeypatch):
        """No pooled entry AND no resolvable token → clean (None, None), no crash."""
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        with patch(
            "agent.auxiliary_client._select_pool_entry", return_value=(True, None)
        ), patch(
            "agent.anthropic_adapter.resolve_anthropic_token", return_value=None
        ):
            from agent.auxiliary_client import _try_anthropic

            client, model = _try_anthropic()

        assert client is None
        assert model is None

    def test_base_url_defaults_when_pool_present_but_no_entry(self, monkeypatch):
        """Falling through with pool_present=True must not crash on base_url
        resolution (previously guarded by `if pool_present`)."""
        monkeypatch.setenv("ANTHROPIC_TOKEN", "«redacted:sk-…»-oauth-token")
        captured = {}

        def _fake_build(token, base_url):
            captured["base_url"] = base_url
            return MagicMock()

        with patch(
            "agent.auxiliary_client._select_pool_entry", return_value=(True, None)
        ), patch(
            "agent.anthropic_adapter.build_anthropic_client", side_effect=_fake_build
        ):
            from agent.auxiliary_client import _try_anthropic

            client, _model = _try_anthropic()

        assert client is not None
        assert captured["base_url"] == "https://api.anthropic.com"
