"""Regression test: auxiliary client must not clobber the auto-chain's
resolved model with the user's main model when the auto chain successfully
resolves via the fallback chain to a different provider.

Background
----------
Bug in ``resolve_provider_client()`` at line 3692:

The function pre-clobbers ``model`` with ``_read_main_model()`` for any
provider whose ``_get_aux_model_for_provider()`` returns empty (lines
3626-3627).  When ``provider == "auto"`` and the auto chain successfully
resolves to a fallback provider (e.g. ``opencode-go`` with
``deepseek-v4-flash``), the returned model is ``model or resolved`` — but
``model`` is already the user's main chat model (``MiniMax-M3``), so it
wins and the call goes out with the wrong model name.  The fallback
endpoint rejects it with HTTP 401.

Reproduction config (per the user's actual report, 2026-06-24):

    model:
      default: MiniMax-M3
      provider: minimax-oauth
      base_url: https://opencode.ai/zen/go/v1
    fallback_providers:
      - model: deepseek-v4-flash
        provider: opencode-go

Without the fix:

    resolve_provider_client("auto", None, task="title_generation")
        → (client_for_opencode_go, "MiniMax-M3")    # WRONG — 401 from OpenCode Go

With the fix:

    resolve_provider_client("auto", None, task="title_generation")
        → (client_for_opencode_go, "deepseek-v4-flash")    # CORRECT

Note: a related bug exists for OAuth-gated direct provider calls
(e.g. ``flush_memories.provider = openai-codex, model = ''`` with the
user's main chat being a MiniMax model — Codex returns 400 because
``MiniMax-M3`` is not in Codex's allowed-model list).  That bug class is
NOT fixed here; it is intentionally out of scope because the fix would
regress the behavior pinned in ``test_auxiliary_client.py`` at lines
573-627 (the universal main-model fallback for OAuth-gated providers,
introduced in #47235).  The two failure modes (auto-chain clobber and
direct OAuth-gated clobber) need separate, scope-limited fixes.

Reproduces the issue tracked by the operator on 2026-06-24: title
generation fails with HTTP 401 ``Model MiniMax-M3 is not supported``
because the auxiliary call goes to OpenCode Go with the user's main
chat model name instead of the fallback chain's correctly-resolved
``deepseek-v4-flash``.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestAuxiliaryModelClobber:
    """The pre-clobber of ``model`` at lines 3626-3639 must not corrupt the
    auto chain's resolution when the chain succeeds via fallback."""

    def test_auto_chain_resolved_model_wins_over_main_model_clobber(
        self, tmp_path, monkeypatch
    ):
        """Reproduces the title_generation failure: ``resolve_provider_client("auto", None)``
        on a minimax-oauth user with ``fallback_providers[0]=opencode-go/deepseek-v4-flash``
        must return ``deepseek-v4-flash``, not the user's main ``MiniMax-M3``.

        Buggy behavior: pre-fix code returns ``MiniMax-M3`` (clobbered from main
        runtime), causing OpenCode Go to reject the call with HTTP 401.
        """
        import agent.auxiliary_client as mod

        # Hermetic: no aggregator creds, no stale OPENAI_BASE_URL.
        for var in ("OPENROUTER_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY",
                    "OPENAI_BASE_URL", "HERMES_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        # Empty config — runtime globals carry the user's setup.
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model:\n  default: ''\n  provider: ''\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # User's actual config: minimax-oauth main provider (the OAuth auth_type
        # the resolver doesn't handle → falls through to fallback chain).
        mod.clear_runtime_main()
        try:
            mod.set_runtime_main(
                "minimax-oauth", "MiniMax-M3",
                base_url="https://opencode.ai/zen/go/v1",
                api_key="",
                api_mode="chat_completions",
            )

            # Patch the fallback chain to return a known-good (client, model)
            # pair — simulating the user's fallback_providers[0].
            fake_client = MagicMock()
            fake_client.base_url = "https://opencode.ai/zen/go/v1"
            with patch.object(
                mod, "_try_main_fallback_chain",
                return_value=(fake_client, "deepseek-v4-flash", "fallback_providers[0](opencode-go)"),
            ):
                # Also short-circuit the unhealthy-provider check so the chain
                # is actually exercised.
                with patch.object(mod, "_is_provider_unhealthy", return_value=False):
                    client, resolved = mod.resolve_provider_client(
                        "auto", None, task="title_generation"
                    )

            assert client is fake_client, (
                "auto chain returned a different client than the fallback chain did"
            )
            assert resolved == "deepseek-v4-flash", (
                f"resolved model was {resolved!r} — expected 'deepseek-v4-flash'. "
                "The pre-clobber of ``model`` at lines 3626-3639 overwrote the "
                "fallback chain's correct resolution with the user's main model."
            )
            # Explicit regression assertion: MiniMax-M3 must NEVER come back
            # from the auto path when the fallback chain resolved successfully.
            assert resolved != "MiniMax-M3", (
                "regression: auto path returned the user's main chat model "
                "('MiniMax-M3') instead of the fallback chain's resolved model. "
                "This produces HTTP 401 from OpenCode Go because the endpoint "
                "does not serve 'MiniMax-M3' (its catalog is minimax-m2.7, glm-5.1, etc)."
            )
        finally:
            mod.clear_runtime_main()

    def test_configured_fallback_chain_also_preserves_resolved_model(
        self, tmp_path, monkeypatch
    ):
        """The auto chain has two fallback paths: the user-configured
        ``auxiliary.<task>.fallback_chain`` (task-specific) and the top-level
        ``fallback_providers`` (global). Both return to ``_resolve_auto`` at
        the same place and both benefit from the fix at line 3692. This test
        pins the configured-fallback path specifically so a future refactor
        that splits the two paths would not silently regress one of them."""
        import agent.auxiliary_client as mod

        for var in ("OPENROUTER_API_KEY", "NOUS_API_KEY", "OPENAI_API_KEY",
                    "OPENAI_BASE_URL", "HERMES_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("model:\n  default: ''\n  provider: ''\n")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        mod.clear_runtime_main()
        try:
            mod.set_runtime_main(
                "minimax-oauth", "MiniMax-M3",
                base_url="https://opencode.ai/zen/go/v1",
                api_key="",
                api_mode="chat_completions",
            )

            fake_client = MagicMock()
            fake_client.base_url = "https://opencode.ai/zen/go/v1"
            with patch.object(
                mod, "_try_configured_fallback_chain",
                return_value=(fake_client, "kimi-k2.6", "configured_chain[0]"),
            ) as mock_configured, patch.object(
                mod, "_try_main_fallback_chain",
            ) as mock_main:
                with patch.object(mod, "_is_provider_unhealthy", return_value=False):
                    client, resolved = mod.resolve_provider_client(
                        "auto", None, task="title_generation"  # type: ignore[arg-type]
                    )

            assert client is fake_client, "configured chain returned a different client"
            assert resolved == "kimi-k2.6", (
                f"configured-chain path returned {resolved!r}; expected 'kimi-k2.6'. "
                "The configured-fallback chain's model must win the same way the "
                "main-fallback chain's model does."
            )
            assert not mock_main.called, (
                "configured-fallback chain should have short-circuited the main "
                "fallback chain — if main was called, the configured path was bypassed"
            )
        finally:
            mod.clear_runtime_main()