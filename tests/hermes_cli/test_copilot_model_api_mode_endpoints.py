"""Copilot API-mode routing must follow the catalog, not just the model name.

``copilot_model_api_mode`` classifies a Copilot model as ``chat_completions``
or ``codex_responses``. The name-pattern check (``^gpt-N``) only recognises
GPT-family models, but Copilot also ships responses-only models under other
vendor prefixes — ``grok-4.5`` advertises exactly ``["/responses"]``. Routing
those to ``/chat/completions`` returns HTTP 400.

The catalog's ``supported_endpoints`` is the authoritative signal, so it is
consulted for models the name pattern can't classify. The upgrade is
deliberately conservative: only when ``/responses`` is present *and*
``/chat/completions`` is absent, so dual-endpoint models (Claude, gpt-5-mini)
keep their existing chat_completions routing.
"""

from unittest.mock import patch

from hermes_cli.models import copilot_model_api_mode


def _catalog(*entries):
    return [{"id": mid, "supported_endpoints": eps} for mid, eps in entries]


class TestCopilotModelApiMode:
    def test_responses_only_non_gpt_model_uses_responses_api(self):
        """grok-4.5 advertises only /responses — the name pattern misses it."""
        catalog = _catalog(("grok-4.5", ["/responses"]))
        with patch("hermes_cli.models.normalize_copilot_model_id",
                   return_value="grok-4.5"):
            assert copilot_model_api_mode("grok-4.5", catalog=catalog) == "codex_responses"

    def test_dual_endpoint_model_stays_on_chat_completions(self):
        """Claude advertises both endpoints — must not be upgraded."""
        catalog = _catalog(("claude-opus-5", ["/v1/messages", "/chat/completions"]))
        with patch("hermes_cli.models.normalize_copilot_model_id",
                   return_value="claude-opus-5"):
            assert copilot_model_api_mode("claude-opus-5", catalog=catalog) == "chat_completions"

    def test_gpt_model_uses_responses_api_without_catalog(self):
        """The name pattern still short-circuits, so no catalog is required."""
        with patch("hermes_cli.models.normalize_copilot_model_id",
                   return_value="gpt-5.6-sol"):
            assert copilot_model_api_mode("gpt-5.6-sol", catalog=[]) == "codex_responses"

    def test_chat_only_model_stays_on_chat_completions(self):
        catalog = _catalog(("some-chat-model", ["/chat/completions"]))
        with patch("hermes_cli.models.normalize_copilot_model_id",
                   return_value="some-chat-model"):
            assert copilot_model_api_mode("some-chat-model", catalog=catalog) == "chat_completions"

    def test_model_absent_from_catalog_stays_on_chat_completions(self):
        """Unknown models keep the safe default rather than guessing."""
        catalog = _catalog(("other-model", ["/responses"]))
        with patch("hermes_cli.models.normalize_copilot_model_id",
                   return_value="mystery-model"):
            assert copilot_model_api_mode("mystery-model", catalog=catalog) == "chat_completions"

    def test_empty_model_id_stays_on_chat_completions(self):
        assert copilot_model_api_mode(None, catalog=[]) == "chat_completions"
        assert copilot_model_api_mode("", catalog=[]) == "chat_completions"
