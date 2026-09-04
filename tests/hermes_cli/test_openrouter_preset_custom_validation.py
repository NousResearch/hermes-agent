"""OpenRouter dashboard presets on custom endpoints must not be catalog-validated (#97907).

``@preset/<slug>`` identifiers reference server-side OpenRouter dashboard
objects. They never appear in ``/models`` listings, so the custom-provider
validation branch used to soft-accept them with a false "not found in this
custom endpoint's model listing" warning that embedded the raw catalog URL —
which Telegram's link preview then fetched and rendered as a ``models.json``
document on every preset switch.

These tests pin the contract from the issue:

* a bare ``@preset/<slug>`` is accepted structurally, with NO probe call and a
  URL-free message (nothing for a link preview to fetch);
* a combined ``<model>@preset/<slug>`` validates its BASE model against the
  live listing and preserves the preset suffix through auto-correction
  (mirroring the ``:nitro`` routing-variant rule);
* a malformed slug is rejected with the slug contract, not a catalog warning;
* non-preset identifiers keep the existing custom-branch behavior.
"""

from unittest.mock import patch

from hermes_cli.models import validate_requested_model


FAKE_CUSTOM_MODELS = [
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.4",
]

BASE_URL = "https://openrouter.example-proxy.dev/v1"


def _probe_payload(models=None):
    return {
        "models": FAKE_CUSTOM_MODELS if models is None else models,
        "probed_url": f"{BASE_URL}/models",
        "resolved_base_url": BASE_URL,
        "suggested_base_url": None,
        "used_fallback": False,
    }


def _validate(model, *, models=None, provider="custom", base_url=BASE_URL, api_mode=None):
    """Call validate_requested_model against a mocked custom-endpoint probe."""
    with patch(
        "hermes_cli.models.probe_api_models", return_value=_probe_payload(models)
    ) as probe:
        result = validate_requested_model(
            model,
            provider,
            api_key="sk-test",
            base_url=base_url,
            api_mode=api_mode,
        )
    return result, probe


class TestBarePresetCustomEndpoint:
    """Layer 1 — bare ``@preset/<slug>`` on a custom (OpenRouter-compatible) endpoint."""

    def test_bare_preset_accepted_structurally_without_probe(self):
        result, probe = _validate("@preset/my-team-config")
        assert result["accepted"] is True
        assert result["persist"] is True
        # OpenRouter validates the slug at request time; there is nothing a
        # catalog probe could confirm for a server-side dashboard object.
        probe.assert_not_called()

    def test_bare_preset_message_carries_no_url(self):
        result, _ = _validate("@preset/my-team-config")
        message = result.get("message") or ""
        # The Telegram document bug is triggered by URLs in the confirmation
        # text; the preset path must not emit any.
        assert "http" not in message
        assert "/models" not in message
        assert "not found" not in message

    def test_bare_preset_on_named_custom_provider(self):
        result, probe = _validate("@preset/fast-tier", provider="custom:my-openrouter")
        assert result["accepted"] is True
        probe.assert_not_called()

    def test_bare_preset_reachable_and_unreachable_endpoint_agree(self):
        # A preset reference is server-side: endpoint reachability must not
        # change the verdict (no probe happens on this path at all).
        reachable, _ = _validate("@preset/a")
        with patch(
            "hermes_cli.models.probe_api_models", return_value=_probe_payload(None)
        ) as probe_none:
            unreachable = validate_requested_model(
                "@preset/a", "custom", api_key="sk-test", base_url=BASE_URL
            )
            probe_none.assert_not_called()
        assert reachable["accepted"] == unreachable["accepted"] is True


class TestCombinedPresetCustomEndpoint:
    """Layer 2 — ``<model>@preset/<slug>`` validates the base, keeps the suffix."""

    def test_combined_preset_base_listed_accepts_with_suffix(self):
        result, probe = _validate("anthropic/claude-opus-4.6@preset/cheap")
        assert result["accepted"] is True
        assert result["persist"] is True
        probe.assert_called_once()
        # No false warning: the base was found in the listing.
        assert result.get("message") is None

    def test_combined_preset_autocorrect_preserves_suffix(self):
        # Base is a near-typo of a listed model: the corrector must fix the
        # base AND keep the preset suffix — not strip it.
        result, _ = _validate("anthropic/claude-opus-4.5@preset/cheap")
        assert result["accepted"] is True
        assert result["corrected_model"] == "anthropic/claude-opus-4.6@preset/cheap"
        assert "preset/cheap" in (result.get("message") or "")

    def test_combined_preset_unknown_base_warns_about_base(self):
        # The suffix alone cannot make an unknown base valid; the warning
        # names the base model, and remains a true positive.
        result, _ = _validate("totally-unknown-model@preset/cheap")
        assert result["accepted"] is True  # custom endpoints soft-accept
        message = result.get("message") or ""
        assert "totally-unknown-model@preset/cheap" in message
        assert "preset" in message


class TestMalformedPresetSlug:
    """Layer 3 — malformed preset references are rejected with the slug contract."""

    def test_empty_slug_rejected(self):
        result, probe = _validate("@preset/")
        assert result["accepted"] is False
        assert result["persist"] is False
        probe.assert_not_called()
        assert "preset" in (result.get("message") or "").lower()

    def test_invalid_slug_charset_rejected(self):
        # A charset-only violation (no whitespace) exercises this branch
        # specifically rather than the shared spaces guard.
        result, probe = _validate("@preset/bad!slug")
        assert result["accepted"] is False
        assert result["persist"] is False
        probe.assert_not_called()
        assert "preset" in (result.get("message") or "").lower()

    def test_multiple_markers_not_a_preset_reference(self):
        # Two markers is not a defined OpenRouter form: fall back to the
        # existing custom-branch behavior for an ordinary (unknown) id.
        result, probe = _validate("a@preset/x@preset/y")
        assert result["accepted"] is True  # soft-accept, as today
        probe.assert_called_once()


class TestNonPresetUnchanged:
    """Layer 4 — non-preset identifiers keep today's custom-branch behavior."""

    def test_plain_unknown_model_still_soft_accepts_with_warning(self):
        result, probe = _validate("some-hidden-model")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is False
        assert "not found" in (result.get("message") or "")
        probe.assert_called_once()

    def test_plain_listed_model_still_recognized(self):
        result, probe = _validate("openai/gpt-5.4")
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result.get("message") is None
        probe.assert_called_once()

    def test_plain_typo_still_autocorrects_without_suffix_logic(self):
        result, _ = _validate("anthropic/claude-opus-4.5")
        assert result["accepted"] is True
        assert result["corrected_model"] == "anthropic/claude-opus-4.6"
