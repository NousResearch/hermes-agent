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

A properly *configured* ``custom:<name>`` provider is now resolved earlier,
via ``resolve_custom_provider`` against ``custom_providers``/``providers`` in
config -- that's the primary, intended path and isn't this module's concern
to re-test. What's tested here is the fallback safety net for when that
resolution comes up empty (typo, config drift, entry removed) -- a
``custom:<name>`` slug must still not be misread as a stray analytics vendor
prefix and reassigned to openrouter, even though it isn't in
``_KNOWN_PROVIDER_NAMES`` either.
"""

from unittest.mock import patch

from hermes_cli.web_server import _normalize_main_model_assignment


def _no_custom_providers_configured():
    """Patch load_config so resolve_user_provider/resolve_custom_provider
    both come up empty, forcing execution into the fallback path under
    test -- independent of whatever config.yaml happens to be on disk."""
    return patch("hermes_cli.web_server.load_config", return_value={})


class TestUnresolvedNamedCustomProviderIsNotTreatedAsStrayVendorPrefix:
    """Covers the case where ``resolve_custom_provider`` finds no match --
    e.g. ``custom:litellm`` was configured once, then the entry was renamed
    or dropped from ``custom_providers``, but old sessions/config still
    reference the old slug.
    """

    def test_unresolved_named_custom_provider_slug_is_preserved(self):
        with _no_custom_providers_configured():
            assert _normalize_main_model_assignment("custom:litellm", "ollama/glm-5.2") == (
                "custom:litellm",
                "ollama/glm-5.2",
            )

    def test_bare_custom_bucket_is_preserved(self):
        with _no_custom_providers_configured():
            assert _normalize_main_model_assignment("custom", "ollama/glm-5.2") == (
                "custom",
                "ollama/glm-5.2",
            )

    def test_unconfigured_non_custom_vendor_name_still_falls_back(self):
        """A name that merely starts with the substring "custom" but isn't
        the durable ``custom:<name>`` syntax (no colon) is NOT exempted --
        it's just another unknown vendor label and should still hit the
        openrouter fallback like any other unrecognized provider string.
        """
        with _no_custom_providers_configured():
            assert _normalize_main_model_assignment(
                "customproxy", "anthropic/claude-opus-4.6"
            ) == ("openrouter", "anthropic/claude-opus-4.6")


class TestConfiguredNamedCustomProviderResolvesViaPrimaryPath:
    """The primary, intended path: a ``custom:<name>`` slug that IS present
    in ``custom_providers`` resolves through ``resolve_custom_provider``
    before the fallback under test above is ever reached.
    """

    def test_configured_named_custom_provider_resolves(self):
        cfg = {
            "custom_providers": [
                {
                    "name": "litellm",
                    "base_url": "http://localhost:4000/v1",
                    "key_env": "LITELLM_API_KEY",
                }
            ]
        }
        with patch("hermes_cli.web_server.load_config", return_value=cfg):
            assert _normalize_main_model_assignment(
                "custom:litellm", "ollama/glm-5.2"
            ) == ("custom:litellm", "ollama/glm-5.2")


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
