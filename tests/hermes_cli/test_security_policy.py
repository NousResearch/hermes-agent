from __future__ import annotations

from hermes_cli.security_policy import (
    RiskClass,
    classify_command,
    classify_tool_action,
    typed_confirmation_phrase,
    validate_typed_confirmation,
)


def test_read_only_tool_does_not_require_typed_confirmation():
    decision = classify_tool_action("read_file")

    assert decision.risk_class == RiskClass.READ_ONLY
    assert decision.approval_policy == "allow"
    assert decision.requires_typed_confirmation is False


def test_external_side_effect_tool_requires_typed_confirmation():
    decision = classify_tool_action("send_message")

    assert decision.risk_class == RiskClass.EXTERNAL_SIDE_EFFECT
    assert decision.approval_policy == "typed_confirm"
    assert decision.requires_typed_confirmation is True


def test_private_memory_tool_defaults_to_typed_confirmation():
    decision = classify_tool_action("memory", action="save user preference")

    assert decision.risk_class == RiskClass.PRIVATE_DATA_ACCESS
    assert decision.approval_policy == "typed_confirm"
    assert decision.requires_typed_confirmation is True


def test_provider_token_destination_is_credential_sensitive():
    decision = classify_command("printf '%s' placeholder > ~/.hermes/provider-token")

    assert decision.risk_class == RiskClass.CREDENTIAL_SENSITIVE
    assert decision.approval_policy == "typed_confirm"
    assert decision.requires_typed_confirmation is True


def test_unmapped_tool_defaults_restricted():
    decision = classify_tool_action("experimental_provider_mutator")

    assert decision.risk_class == RiskClass.UNKNOWN_RESTRICTED
    assert decision.approval_policy == "typed_confirm"
    assert decision.default_restricted is True


def test_high_risk_commands_get_named_risk_classes():
    assert classify_command("git reset --hard").risk_class == RiskClass.DESTRUCTIVE
    assert classify_command("git push --force").risk_class == RiskClass.EXTERNAL_SIDE_EFFECT
    assert classify_command("echo TOKEN=value > .env").risk_class == RiskClass.CREDENTIAL_SENSITIVE
    assert classify_command("buy BTC with account balance").risk_class == RiskClass.FINANCIAL_OR_ACCOUNT_ACTION


def test_typed_confirmation_requires_exact_phrase():
    decision = classify_command("git reset --hard")
    phrase = typed_confirmation_phrase(decision)

    assert validate_typed_confirmation(phrase, phrase) is True
    assert validate_typed_confirmation(phrase.lower(), phrase) is False
    assert validate_typed_confirmation(f" {phrase}", phrase) is False
    assert validate_typed_confirmation("", phrase) is False
