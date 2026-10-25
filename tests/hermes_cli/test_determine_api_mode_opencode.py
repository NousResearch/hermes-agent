"""``determine_api_mode`` must route an OpenCode model on THAT model's wire.

OpenCode Zen/Go serve several wire formats behind one endpoint: the Responses models
(``muse-spark*``/``gpt-*``/``grok-*``) ride ``/v1/responses``, the Anthropic-shaped ones
(``claude-*``/``minimax-*``) ride ``/v1/messages``, and everything else rides
``/v1/chat/completions``. The provider overlay declares ``openai_chat`` for all of them, so a
resolver that stops at the provider's transport pins EVERY OpenCode model on chat/completions.
A Responses model then reaches the relay over the wrong wire and is refused with
``ModelProtocolUnsupported`` (#141861).

``hermes_cli.models.opencode_model_api_mode`` is the single owner of that per-model table (the
same owner ``model_derived_api_mode`` and ``runtime_provider`` consult); this resolver must
consult it too, the way it already consults ``nous_api_mode`` for the other dual-wire provider.
"""

from __future__ import annotations

from hermes_cli.providers import determine_api_mode

GO_BASE_URL = "https://opencode.ai/zen/go/v1"
ZEN_BASE_URL = "https://opencode.ai/zen/v1"


class TestOpenCodePerModelWire:
    """The wire follows the MODEL, not the provider slug."""

    def test_responses_model_uses_the_responses_wire(self):
        assert determine_api_mode("opencode-go", GO_BASE_URL, "muse-spark-1.3-contributor") == "codex_responses"

    def test_zen_responses_model_uses_the_responses_wire(self):
        assert determine_api_mode("opencode-zen", ZEN_BASE_URL, "gpt-5") == "codex_responses"

    def test_anthropic_shaped_model_uses_the_messages_wire(self):
        assert determine_api_mode("opencode-go", GO_BASE_URL, "claude-haiku-4-5-20251001") == "anthropic_messages"

    def test_chat_model_stays_on_the_chat_wire(self):
        # A model the relay serves over chat/completions must not be promoted to a stricter wire.
        assert determine_api_mode("opencode-go", GO_BASE_URL, "deepseek-v4") == "chat_completions"

    def test_custom_provider_extending_a_family_slug_follows_the_family_table(self):
        # opencode_provider_family() is the single owner of family membership (#85589).
        assert determine_api_mode("opencode-go-bridge", GO_BASE_URL, "muse-spark-1.3-contributor") == "codex_responses"


class TestNoModelStillResolves:
    """A caller that cannot name a model keeps the historical provider-level answer."""

    def test_missing_model_falls_back_to_the_chat_wire(self):
        assert determine_api_mode("opencode-go", GO_BASE_URL) == "chat_completions"
        assert determine_api_mode("opencode-go", GO_BASE_URL, "") == "chat_completions"


class TestOtherProvidersUnchanged:
    """The new branch is scoped to the OpenCode families."""

    def test_nous_dual_wire_is_unchanged(self):
        assert determine_api_mode("nous", "https://inference-api.nousresearch.com/v1", "anthropic/claude-haiku-4-5") == "chat_completions"

    def test_anthropic_provider_is_unchanged(self):
        assert determine_api_mode("anthropic", "https://api.anthropic.com/v1", "claude-haiku-4-5-20251001") == "anthropic_messages"

    def test_bedrock_is_unchanged(self):
        assert determine_api_mode("bedrock", "https://bedrock-runtime.us-east-1.amazonaws.com", "anthropic.claude") == "bedrock_converse"

    def test_openai_provider_is_unchanged(self):
        assert determine_api_mode("openai", "https://api.openai.com/v1", "gpt-5") == "codex_responses"

    def test_unknown_provider_is_unchanged(self):
        assert determine_api_mode("some-unknown-relay", "https://relay.example.com/v1", "some-model") == "chat_completions"


class TestSwitchPathResolvesTheWire:
    """The sibling path: an in-place model switch that carries no api_mode.

    ``_resolve_switch_destination`` asks this resolver for the wire when the caller hands it
    none, so a restore/resume that lands on an OpenCode model resolves the wire the same way the
    managed-worker path does (#141861).
    """

    def test_switch_to_an_opencode_responses_model_resolves_the_responses_wire(self):
        from agent.agent_runtime_helpers import _resolve_switch_destination

        class _Agent:
            base_url = GO_BASE_URL
            api_mode = "codex_responses"

        api_mode, base_url, _ = _resolve_switch_destination(
            _Agent(), "muse-spark-1.3-contributor", "opencode-go", GO_BASE_URL, "", None, "openai", "openai"
        )
        assert api_mode == "codex_responses"
        # The Responses wire keeps the /v1 tail.
        assert base_url == GO_BASE_URL
