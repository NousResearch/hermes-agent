"""opencode-go serves its Claude models on the Messages wire, not chat/completions.

The Go catalog carries ``claude-haiku-5-5``. Measured against
``https://opencode.ai/zen/go/v1`` with one key and one model, varying only the
endpoint:

* ``/v1/chat/completions`` -> HTTP 400
  ``{"type":"error","error":{"type":"ModelProtocolUnsupported",
  "message":"Model does not support this protocol."}}``
* ``/v1/responses`` (what ``codex_responses`` would send) -> the same 400
* ``/v1/messages`` -> 200, streaming and ``tools`` payloads included

Hermes classifies that 400 as a non-retryable client error, so a turn on this
model dies on its first API call: no retry, no fallback, nothing in the
transcript. The relay is reached through the third-party ``x-api-key`` branch,
so the only missing half was the routing table — without a ``claude-`` entry for
``opencode-go``, ``opencode_model_api_mode()`` fell through to
``chat_completions`` and posted the request to the wire the relay rejects.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model
from hermes_cli.models import normalize_opencode_base_url, opencode_model_api_mode


class TestOpenCodeGoClaudeWire:
    """``claude-*`` on opencode-go must resolve to anthropic_messages."""

    def test_claude_haiku_5_5_uses_the_messages_wire(self):
        assert opencode_model_api_mode("opencode-go", "claude-haiku-5-5") == "anthropic_messages"

    def test_namespaced_model_id_still_routes(self):
        assert opencode_model_api_mode("opencode-go", "opencode-go/claude-haiku-5-5") == "anthropic_messages"

    def test_messages_wire_strips_the_trailing_v1(self):
        """The anthropic client appends its own /v1/messages — a kept /v1 double-prefixes it."""
        assert (
            normalize_opencode_base_url("opencode-go", "anthropic_messages", "https://opencode.ai/zen/go/v1")
            == "https://opencode.ai/zen/go"
        )


class TestOpenCodeGoSiblingsUnchanged:
    """Adding ``claude-`` must not move models that were already routed correctly."""

    def test_chat_models_stay_on_chat_completions(self):
        for model in ("deepseek-v4.1-flash", "glm-5.3", "kimi-k2.7-code", "hy3"):
            assert opencode_model_api_mode("opencode-go", model) == "chat_completions", model

    def test_responses_models_stay_on_codex_responses(self):
        for model in ("gpt-5.6-luna", "grok-4.6", "muse-spark-1.3-contributor"):
            assert opencode_model_api_mode("opencode-go", model) == "codex_responses", model

    def test_existing_messages_models_are_unchanged(self):
        for model in ("minimax-m2.7", "qwen3.7-plus"):
            assert opencode_model_api_mode("opencode-go", model) == "anthropic_messages", model

    def test_zen_claude_routing_is_untouched(self):
        assert opencode_model_api_mode("opencode-zen", "claude-haiku-5-5") == "anthropic_messages"


class TestSwitchToGoClaudePicksTheMessagesWire:
    """``/model claude-haiku-5-5`` on opencode-go must land on the stripped URL."""

    _MOCK_VALIDATION = {"accepted": True, "persist": True, "recognized": True, "message": None}

    def _run_switch(self, raw_input: str):
        with (
            patch("hermes_cli.model_switch.resolve_alias", return_value=None),
            patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "api_key": "test-key",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "api_mode": "chat_completions",
                },
            ),
            patch("hermes_cli.models_validate.validate_requested_model", return_value=self._MOCK_VALIDATION),
            patch("hermes_cli.model_switch.get_model_info", return_value=None),
            patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
            patch("hermes_cli.models.detect_provider_for_model", return_value=None),
        ):
            return switch_model(
                raw_input=raw_input,
                current_provider="opencode-go",
                current_model="glm-5.3",
                current_base_url="https://opencode.ai/zen/go/v1",
                current_api_key="test-key",
            )

    def test_switch_resolves_messages_wire_and_stripped_url(self):
        result = self._run_switch("claude-haiku-5-5")

        assert result.success, f"switch_model failed: {result.error_message}"
        assert result.api_mode == "anthropic_messages"
        assert result.base_url == "https://opencode.ai/zen/go"
