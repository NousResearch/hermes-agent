"""Tests for Hermes Code Mode execution policy."""

from hermes_cli.code.execution_policy import (
    ExecutionPolicyEngine,
    RiskClass,
    classify_command,
    redact_secrets,
)


def test_safe_command_classification():
    assert classify_command("git status") == RiskClass.SAFE_READONLY


def test_destructive_command_classification():
    assert classify_command("rm -rf /tmp/demo") == RiskClass.DESTRUCTIVE


def test_secret_sensitive_command_classification():
    assert classify_command("cat .env") == RiskClass.SECRET_SENSITIVE


def test_redaction_hides_sensitive_values():
    text = "Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    redacted = redact_secrets(text)
    assert "ghp_" not in redacted
    assert "[REDACTED]" in redacted


def test_assess_command_contract():
    engine = ExecutionPolicyEngine()
    assessment = engine.assess_command("git commit -m test")
    assert assessment["risk_class"] == RiskClass.GIT_WRITE
    assert assessment["requires_approval"] is True
    assert assessment["blocked"] is False
