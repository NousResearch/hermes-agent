"""Copilot catalog policy filtering — models the account has not opted into.

GitHub's ``/models`` catalog carries a per-user ``policy`` block:

    {"id": "gpt-6-astra", "policy": {"state": "disabled", "terms": "Enable access to ..."}}

``disabled`` means the account has not accepted that model's terms on github.com. Every chat
request for one of those fails with HTTP 400 ``model_not_supported``, so listing them puts
choices in the picker that cannot serve a single turn. Verified against a live Copilot account:
all 12 ``disabled`` models returned 400, all 12 policy-less/enabled-and-serving models returned
200.

The filter is deliberately narrow — only an explicit ``disabled`` excludes. An absent policy,
``enabled``, or an unrecognised state stays listed, because entitlement has other causes the
catalog does not express and Hermes must not hide a model it cannot prove is broken.
"""


def _item(model_id: str, **extra):
    """A minimal chat-shaped catalog row."""
    return {"id": model_id, "model_picker_enabled": True, **extra}


class TestCopilotPolicyIsDisabled:
    """The predicate itself."""

    def test_disabled_state_is_detected(self):
        from hermes_cli.models import _copilot_policy_is_disabled
        assert _copilot_policy_is_disabled(
            _item("gpt-6-astra", policy={"state": "disabled", "terms": "Enable access ..."})) is True

    def test_enabled_state_is_not_disabled(self):
        from hermes_cli.models import _copilot_policy_is_disabled
        assert _copilot_policy_is_disabled(
            _item("gpt-4.1", policy={"state": "enabled", "terms": "..."})) is False

    def test_absent_policy_is_not_disabled(self):
        """gpt-4o and friends ship no policy block at all — they must stay listed."""
        from hermes_cli.models import _copilot_policy_is_disabled
        assert _copilot_policy_is_disabled(_item("gpt-4o")) is False

    def test_state_matching_is_case_and_whitespace_insensitive(self):
        from hermes_cli.models import _copilot_policy_is_disabled
        assert _copilot_policy_is_disabled(_item("x", policy={"state": "  DISABLED "})) is True

    def test_unknown_state_stays_listed(self):
        """Only a definite 'disabled' excludes; an unrecognised value must not hide a model."""
        from hermes_cli.models import _copilot_policy_is_disabled
        assert _copilot_policy_is_disabled(_item("x", policy={"state": "pending"})) is False

    def test_non_dict_policy_is_ignored(self):
        from hermes_cli.models import _copilot_policy_is_disabled
        assert _copilot_policy_is_disabled(_item("x", policy="disabled")) is False
        assert _copilot_policy_is_disabled(_item("x", policy=None)) is False


class TestCopilotCatalogFiltersDisabledPolicy:
    """The catalog filter that feeds every picker surface."""

    def test_disabled_model_is_dropped(self):
        from hermes_cli.models import _copilot_text_models
        models = _copilot_text_models([
            _item("gpt-4o"),
            _item("gpt-6-astra", policy={"state": "disabled", "terms": "Enable access ..."}),
        ])
        assert [m["id"] for m in models] == ["gpt-4o"]

    def test_enabled_and_policyless_models_survive(self):
        from hermes_cli.models import _copilot_text_models
        models = _copilot_text_models([
            _item("gpt-4o"),
            _item("gpt-4.1", policy={"state": "enabled"}),
        ])
        assert [m["id"] for m in models] == ["gpt-4o", "gpt-4.1"]

    def test_picker_flag_retry_does_not_resurrect_disabled_models(self):
        """The ``model_picker_enabled: false`` rescue path must not undo the policy filter.

        GitHub has been observed returning ``model_picker_enabled: false`` for every model on
        some accounts; Hermes retries with ``ignore_picker_flag=True`` so the picker isn't
        stranded. That rescue must still exclude models the account cannot call.
        """
        from hermes_cli.models import _copilot_text_models
        items = [
            {"id": "gpt-4o", "model_picker_enabled": False},
            {"id": "gpt-6-astra", "model_picker_enabled": False,
             "policy": {"state": "disabled", "terms": "Enable access ..."}},
        ]
        assert _copilot_text_models(items) == []
        rescued = _copilot_text_models(items, ignore_picker_flag=True)
        assert [m["id"] for m in rescued] == ["gpt-4o"]

    def test_policy_filter_composes_with_the_other_checks(self):
        """A disabled model is dropped regardless of which other check would have kept it."""
        from hermes_cli.models import _copilot_text_models
        items = [
            _item("embed", capabilities={"type": "embeddings"}),
            _item("wrong-endpoint", supported_endpoints=["/embeddings"]),
            _item("disabled-but-chat", capabilities={"type": "chat"},
                  supported_endpoints=["/chat/completions"],
                  policy={"state": "disabled"}),
            _item("keeper", capabilities={"type": "chat"},
                  supported_endpoints=["/chat/completions"]),
        ]
        assert [m["id"] for m in _copilot_text_models(items)] == ["keeper"]
