"""Deterministic policy contracts for the Dobby package skeleton.

The samples in this file are intentionally local and synthetic. They should
fail before a sellable package accepts unsafe defaults, leaked secret-shaped
template values, or write-capable tools without explicit operator approval.
"""

import re

import pytest


ALLOW_ALL_VALUES = {"*", "all", "any", "everyone", "@everyone", "public"}


def _has_allow_all_value(value):
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ALLOW_ALL_VALUES
    if isinstance(value, list):
        return any(_has_allow_all_value(item) for item in value)
    return False


def config_policy_errors(config):
    errors = []
    if config.get("allow_all_users") is True or _has_allow_all_value(config.get("allowed_users")):
        errors.append("users must be explicitly allowlisted")
    if config.get("allow_all_channels") is True or _has_allow_all_value(config.get("allowed_channels")):
        errors.append("channels must be explicitly allowlisted")
    return errors


def test_config_policy_accepts_explicit_user_and_channel_allowlists():
    sample_config = {
        "allow_all_users": False,
        "allow_all_channels": False,
        "allowed_users": ["111111111111111111"],
        "allowed_channels": ["222222222222222222"],
    }

    assert config_policy_errors(sample_config) == []


@pytest.mark.parametrize(
    ("sample_config", "expected_error"),
    [
        (
            {"allow_all_users": True, "allowed_channels": ["222222222222222222"]},
            "users must be explicitly allowlisted",
        ),
        (
            {"allowed_users": ["*"], "allowed_channels": ["222222222222222222"]},
            "users must be explicitly allowlisted",
        ),
        (
            {"allowed_users": ["111111111111111111"], "allow_all_channels": True},
            "channels must be explicitly allowlisted",
        ),
        (
            {"allowed_users": ["111111111111111111"], "allowed_channels": ["public"]},
            "channels must be explicitly allowlisted",
        ),
    ],
)
def test_config_policy_rejects_allow_all_users_and_channels(sample_config, expected_error):
    assert expected_error in config_policy_errors(sample_config)


SECRET_VALUE_PATTERNS = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "discord_token": re.compile(
        r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b"
    ),
}


def secret_like_template_findings(template_text):
    return [
        label
        for label, pattern in SECRET_VALUE_PATTERNS.items()
        if pattern.search(template_text)
    ]


def test_templates_accept_placeholders_but_no_secret_shaped_values():
    safe_template = "\n".join(
        [
            "DISCORD_BOT_TOKEN=<DISCORD_BOT_TOKEN>",
            "DISCORD_HOME_CHANNEL=<DISCORD_CHANNEL_ID>",
            "OPENAI_API_KEY=<MODEL_API_KEY>",
            "WEBHOOK_SECRET=<WEBHOOK_HMAC_SECRET>",
        ]
    )

    assert secret_like_template_findings(safe_template) == []


@pytest.mark.parametrize(
    "unsafe_template",
    [
        "OPENAI_API_KEY=" + "sk-" + ("A" * 32),
        "GITHUB_TOKEN=" + "ghp_" + ("B" * 36),
        "SLACK_BOT_TOKEN=" + "xoxb-" + ("1234-" * 6) + "abcd",
        "DISCORD_BOT_TOKEN=" + ("C" * 24) + "." + ("D" * 6) + "." + ("E" * 32),
    ],
)
def test_templates_reject_secret_shaped_values(unsafe_template):
    assert secret_like_template_findings(unsafe_template)


CONFIRMATION_REQUIRED_RISKS = {"write", "execute", "external"}


def tool_policy_errors(tool_policy):
    risk = tool_policy["risk"]
    requires_confirmation = tool_policy.get("requires_confirmation", False)
    if risk in CONFIRMATION_REQUIRED_RISKS and not requires_confirmation:
        return [f"{risk} tools require explicit confirmation"]
    return []


@pytest.mark.parametrize("risk", sorted(CONFIRMATION_REQUIRED_RISKS))
def test_tool_risk_policy_requires_confirmation_for_write_execute_and_external_actions(risk):
    unsafe_tool_policy = {
        "name": f"sample_{risk}_tool",
        "risk": risk,
        "requires_confirmation": False,
    }

    assert f"{risk} tools require explicit confirmation" in tool_policy_errors(unsafe_tool_policy)


def test_tool_risk_policy_allows_read_only_tools_without_confirmation():
    read_only_tool_policy = {
        "name": "repo_status",
        "risk": "read",
        "requires_confirmation": False,
    }

    assert tool_policy_errors(read_only_tool_policy) == []


REQUIRED_MEMORY_FIELDS = {"consent", "export", "delete", "forget"}


def memory_policy_errors(memory_policy):
    errors = []
    missing = sorted(REQUIRED_MEMORY_FIELDS - memory_policy.keys())
    if missing:
        errors.append(f"memory policy missing fields: {', '.join(missing)}")
        return errors

    if not memory_policy["consent"].get("required"):
        errors.append("durable memory writes require consent")
    if not memory_policy["export"].get("supported"):
        errors.append("memory export must be supported")
    for field in ("delete", "forget"):
        if not memory_policy[field].get("supported"):
            errors.append(f"memory {field} must be supported")
        if not memory_policy[field].get("requires_confirmation"):
            errors.append(f"memory {field} requires confirmation")
    return errors


def test_memory_policy_requires_consent_export_delete_and_forget_fields():
    valid_memory_policy = {
        "consent": {"required": True, "default": "off"},
        "export": {"supported": True},
        "delete": {"supported": True, "requires_confirmation": True},
        "forget": {"supported": True, "requires_confirmation": True},
    }

    assert memory_policy_errors(valid_memory_policy) == []


@pytest.mark.parametrize(
    ("memory_policy", "expected_error"),
    [
        (
            {
                "consent": {"required": True},
                "export": {"supported": True},
                "delete": {"supported": True, "requires_confirmation": True},
            },
            "memory policy missing fields: forget",
        ),
        (
            {
                "consent": {"required": False},
                "export": {"supported": True},
                "delete": {"supported": True, "requires_confirmation": True},
                "forget": {"supported": True, "requires_confirmation": True},
            },
            "durable memory writes require consent",
        ),
        (
            {
                "consent": {"required": True},
                "export": {"supported": True},
                "delete": {"supported": True, "requires_confirmation": False},
                "forget": {"supported": True, "requires_confirmation": True},
            },
            "memory delete requires confirmation",
        ),
    ],
)
def test_memory_policy_rejects_missing_or_unsafe_privacy_controls(memory_policy, expected_error):
    assert expected_error in memory_policy_errors(memory_policy)


REQUIRED_WEBHOOK_FIELDS = {"signature_header", "timestamp_header", "replay_window_seconds"}


def webhook_policy_errors(webhook_policy):
    errors = []
    missing = sorted(REQUIRED_WEBHOOK_FIELDS - webhook_policy.keys())
    if missing:
        errors.append(f"webhook policy missing fields: {', '.join(missing)}")
        return errors

    if not webhook_policy.get("require_signature"):
        errors.append("webhooks require signatures")
    if webhook_policy.get("signature_algorithm") != "hmac-sha256":
        errors.append("webhooks require hmac-sha256 signatures")
    replay_window_seconds = webhook_policy["replay_window_seconds"]
    if replay_window_seconds <= 0 or replay_window_seconds > 300:
        errors.append("webhook replay window must be between 1 and 300 seconds")
    return errors


def test_webhook_policy_requires_signature_timestamp_and_replay_window_fields():
    valid_webhook_policy = {
        "require_signature": True,
        "signature_algorithm": "hmac-sha256",
        "signature_header": "X-Dobby-Signature",
        "timestamp_header": "X-Dobby-Timestamp",
        "replay_window_seconds": 300,
    }

    assert webhook_policy_errors(valid_webhook_policy) == []


@pytest.mark.parametrize(
    ("webhook_policy", "expected_error"),
    [
        (
            {
                "require_signature": True,
                "signature_algorithm": "hmac-sha256",
                "timestamp_header": "X-Dobby-Timestamp",
                "replay_window_seconds": 300,
            },
            "webhook policy missing fields: signature_header",
        ),
        (
            {
                "require_signature": False,
                "signature_algorithm": "hmac-sha256",
                "signature_header": "X-Dobby-Signature",
                "timestamp_header": "X-Dobby-Timestamp",
                "replay_window_seconds": 300,
            },
            "webhooks require signatures",
        ),
        (
            {
                "require_signature": True,
                "signature_algorithm": "hmac-sha256",
                "signature_header": "X-Dobby-Signature",
                "timestamp_header": "X-Dobby-Timestamp",
                "replay_window_seconds": 900,
            },
            "webhook replay window must be between 1 and 300 seconds",
        ),
    ],
)
def test_webhook_policy_rejects_missing_or_unsafe_authentication_controls(
    webhook_policy, expected_error
):
    assert expected_error in webhook_policy_errors(webhook_policy)
