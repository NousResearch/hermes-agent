"""Regression tests for ``_normalize_main_model_assignment`` (POST /api/model/set).

Named custom providers are represented as ``custom:<name>`` slugs everywhere
else in the codebase (``runtime_provider.py``, ``model_switch.py``), but
``_KNOWN_PROVIDER_NAMES`` only lists the bare ``"custom"`` bucket. Before this
fix, persisting a main-slot assignment for a named custom provider (e.g. a
LiteLLM proxy fronting Ollama, registered as ``custom:litellm``) together with
a slash-bearing model id (``ollama/glm-5.2``) was indistinguishable from the
"vendor prefix posing as a provider" analytics-fallback case, and got silently
rewritten to ``provider: openrouter`` in ``config.yaml`` -- reassigning the
provider entirely, not just mangling the model id.
"""

from hermes_cli.web_server import _normalize_main_model_assignment


class TestNamedCustomProviderIsNotTreatedAsStrayVendorPrefix:
    def test_named_custom_provider_slug_is_preserved(self):
        assert _normalize_main_model_assignment("custom:litellm", "ollama/glm-5.2") == (
            "custom:litellm",
            "ollama/glm-5.2",
        )

    def test_bare_custom_bucket_is_preserved(self):
        assert _normalize_main_model_assignment("custom", "ollama/glm-5.2") == (
            "custom",
            "ollama/glm-5.2",
        )


class TestStrayVendorPrefixFallbackStillWorks:
    """The original bug this function fixes: an analytics row with no
    ``billing_provider`` falls back to the model's vendor prefix as the
    "provider" (e.g. ``provider="anthropic"`` from
    ``modelVendor("anthropic/claude-opus-4.6")``). That must still resolve
    to the native provider with its model normalized -- unaffected by the
    ``custom:`` exclusion above.
    """

    def test_known_native_provider_still_normalizes_model(self):
        assert _normalize_main_model_assignment(
            "anthropic", "anthropic/claude-opus-4.6"
        ) == ("anthropic", "claude-opus-4-6")
