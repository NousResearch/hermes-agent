"""#64 — failure copy must not name a vendor that never answered.

``provider_label("")`` answers ``"OpenRouter"``, which is right for ``hermes status`` ("what would
be used if nothing is configured") and wrong for a failed turn ("who answered this call"). The copy
layer reused it, so a turn routed to an endpoint with no OpenRouter key anywhere was reported as
"isn't available on OpenRouter" — pointing the user at the wrong service and hiding the routing
fault. Nobody has to touch ``provider_label`` for this: the distinction belongs to the caller.
"""

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason
from agent.error_surface import _provider_label
from agent.turn_failure_copy import (
    PROVIDER_UNKNOWN_LABEL,
    _sentence,
    content_policy_copy,
    exhausted_copy,
    nonretryable_copy,
    provider_label_for,
    site_copy,
)


class TestProviderLabelFor:
    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_is_neutral(self, blank):
        assert provider_label_for(blank) == PROVIDER_UNKNOWN_LABEL

    def test_the_neutral_label_names_no_vendor(self):
        assert "openrouter" not in PROVIDER_UNKNOWN_LABEL.lower()

    @pytest.mark.parametrize("provider,expected", [("deepseek", "DeepSeek"), ("openrouter", "OpenRouter")])
    def test_a_recorded_provider_keeps_its_real_name(self, provider, expected):
        assert provider_label_for(provider) == expected

    def test_status_semantics_are_untouched(self):
        """``hermes status`` depends on this default; the fix must not migrate into it."""
        from hermes_cli.models import provider_label

        assert provider_label("") == "OpenRouter"
        # The two surfaces must stay deliberately different — that difference IS the fix.
        assert provider_label_for("") != provider_label("")

    def test_a_custom_entry_key_is_not_rendered_as_a_vendor(self):
        """``custom:lab`` is a config identifier; the copy should read like "lab", not like a key."""
        assert provider_label_for("custom:lab") == "lab"


class TestErrorSurface:
    def test_blank_is_neutral(self):
        assert _provider_label("") == PROVIDER_UNKNOWN_LABEL

    def test_recorded_provider_is_labelled(self):
        assert _provider_label("deepseek") == "DeepSeek"


class TestFailureCopyEndToEnd:
    def _copy(self, provider):
        classified = SimpleNamespace(reason=FailoverReason.model_not_found, is_auth=False)
        return nonretryable_copy(classified, provider=provider, model="qwen38-27b", summary="nope")

    def test_unknown_provider_sentence_names_no_vendor(self):
        text = self._copy(None)
        assert "OpenRouter" not in text
        assert PROVIDER_UNKNOWN_LABEL in text

    def test_known_provider_sentence_still_names_it(self):
        text = self._copy("deepseek")
        assert "DeepSeek" in text
        assert "OpenRouter" not in text


class TestSentenceInitialForms:
    """The label can open the sentence, so it has to be capitalised there (#64 review, major).

    ``provider_label("")`` used to answer a capitalised proper noun ("OpenRouter"), and nine
    templates put ``{label}`` first — so swapping in a lowercase phrase made the majority of failure
    copy start with a lowercase "the".
    """

    def test_sentence_form_capitalises_and_leaves_names_alone(self):
        assert _sentence(provider_label_for("")) == "The configured provider"
        assert _sentence("OpenRouter") == "OpenRouter"
        assert _sentence("") == ""

    @pytest.mark.parametrize("reason", [FailoverReason.format_error, FailoverReason.role_alternation,
                                        FailoverReason.provider_policy_blocked])
    def test_sentence_initial_templates_start_with_the_capital(self, reason):
        text = nonretryable_copy(SimpleNamespace(reason=reason, is_auth=False),
                                 provider=None, model="m", summary="x")
        assert text.startswith("The configured provider")

    def test_the_default_nonretryable_template_starts_with_the_capital(self):
        text = nonretryable_copy(SimpleNamespace(reason=FailoverReason.server_error, is_auth=False),
                                 provider=None, model="m", summary="x")
        assert text.startswith("The configured provider")

    def test_exhausted_copy_starts_with_the_capital(self):
        text = exhausted_copy(FailoverReason.rate_limit.value, label=provider_label_for(""),
                              attempts=3, summary="x")
        assert text.startswith("The configured provider")

    def test_exhausted_first_attempt_copy_starts_with_the_capital(self):
        text = exhausted_copy(FailoverReason.rate_limit.value, label=provider_label_for(""),
                              attempts=1, summary="x")
        assert text.startswith("The configured provider")

    def test_site_copy_invalid_response_starts_with_the_capital(self):
        text = site_copy("invalid_response", label=provider_label_for(""), attempts=3, detail="d")
        assert text.startswith("The configured provider")

    def test_site_copy_stream_closed_starts_with_the_capital(self):
        text = site_copy("stream_closed_tool_call", label=provider_label_for(""))
        assert text.startswith("The configured provider")

    def test_content_policy_copy_capitalises_the_label(self):
        text = content_policy_copy(label=provider_label_for(""), summary="x")
        assert text.startswith("The configured provider")

    def test_mid_sentence_templates_keep_the_lowercase_form(self):
        text = nonretryable_copy(SimpleNamespace(reason=FailoverReason.model_not_found, is_auth=False),
                                 provider=None, model="m", summary="x")
        assert "isn't available on the configured provider" in text
        assert "The configured provider" not in text

    def test_no_rendered_failure_copy_starts_lowercase(self):
        """Sweep of every sentence-initial path, rendered with an unrecorded provider."""
        label = provider_label_for("")
        texts = [content_policy_copy(label=label, summary="x")]
        for reason in FailoverReason:
            texts.append(nonretryable_copy(SimpleNamespace(reason=reason, is_auth=False),
                                           provider=None, model="m", summary="x"))
            texts.append(exhausted_copy(reason.value, label=label, attempts=3, summary="x"))
        for code in ("invalid_response", "stream_closed_tool_call", "stream_dropped_tool_call"):
            texts.append(site_copy(code, label=label, attempts=3, detail="d"))
        lowercase = [t[:60] for t in texts if t[:1].islower()]
        assert not lowercase, lowercase

    def test_an_unrecorded_auth_failure_does_not_guess_the_credential_kind(self):
        """``auth_kind("")`` says "api_key"; telling the user to replace a key nobody proved exists
        is the same mistake as naming a vendor that never answered."""
        text = nonretryable_copy(SimpleNamespace(reason=FailoverReason.auth, is_auth=True),
                                 provider=None, model="m", summary="x")
        assert "rejected your API key" not in text
        assert "credentials for this session" in text
        assert text.startswith("The configured provider")

    def test_a_recorded_auth_failure_still_names_the_credential_kind(self):
        text = nonretryable_copy(SimpleNamespace(reason=FailoverReason.auth, is_auth=True),
                                 provider="deepseek", model="m", summary="x")
        assert "rejected your API key" in text
        assert text.startswith("DeepSeek")
