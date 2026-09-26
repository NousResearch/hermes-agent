"""Contract tests for opt-in Gemini delegation routing configuration."""

from copy import deepcopy

import pytest
import yaml

from hermes_cli.config import DEFAULT_CONFIG, load_config, validate_config_structure


EXPECTED_GEMINI_ROUTING_DEFAULTS = {
    "enabled": False,
    "profiles": [],
    "default_route": "gemini",
    "default_data_classification": "restricted",
    "command": "agy",
    "model": "gemini-3.8-flash-low",
    "effort": "low",
    "timeout_seconds": 120,
    "max_input_bytes": 262_144,
    "max_output_bytes": 131_072,
    "fallback_to_delegation_model": True,
    "receipt_db": "routing/gemini-routing.sqlite3",
    "retention": {"raw_days": 30, "aggregate_days": 180},
    "extra_args": [],
    "review": {
        "enabled": False,
        "timezone": "America/Los_Angeles",
        "sample_size": 5,
        "not_before_local": "00:15",
        "review_provider": "openai-codex",
        "review_model": "gpt-5.6-sol",
        "alert_target": "slack:C0AEMP1AG0H",
        "alert_workspace_id": "",
    },
}


def _routing_config(**overrides):
    config = deepcopy(DEFAULT_CONFIG)
    routing = deepcopy(EXPECTED_GEMINI_ROUTING_DEFAULTS)
    routing.update(overrides)
    config["delegation"]["gemini_routing"] = routing
    return config


def _error_messages(config):
    return [
        issue.message
        for issue in validate_config_structure(config)
        if issue.severity == "error"
    ]


def test_gemini_routing_is_disabled_by_default_without_changing_delegation_model():
    delegation = DEFAULT_CONFIG["delegation"]

    assert delegation["gemini_routing"] == EXPECTED_GEMINI_ROUTING_DEFAULTS
    assert delegation["gemini_routing"]["enabled"] is False
    assert delegation["model"] == ""
    assert delegation["provider"] == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_provider", "arbitrary-provider"),
        ("review_model", "arbitrary-model"),
    ],
)
def test_review_identity_must_match_canonical_sol_reviewer(field, value):
    review = deepcopy(EXPECTED_GEMINI_ROUTING_DEFAULTS["review"])
    review[field] = value

    errors = _error_messages(_routing_config(review=review))

    assert any(f"review.{field}" in message for message in errors)


def test_enabling_without_profile_scope_stays_inactive(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"delegation": {"gemini_routing": {"enabled": True}}}),
        encoding="utf-8",
    )

    routing = load_config()["delegation"]["gemini_routing"]

    assert routing["enabled"] is True
    assert routing["profiles"] == []
    assert not (routing["enabled"] and "default" in routing["profiles"])


def test_explicit_profile_scope_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "delegation": {
                    "provider": "openai-codex",
                    "model": "gpt-5.6-sol",
                    "gemini_routing": {"enabled": True, "profiles": ["default"]},
                }
            }
        ),
        encoding="utf-8",
    )

    delegation = load_config()["delegation"]

    assert delegation["gemini_routing"]["profiles"] == ["default"]
    assert delegation["provider"] == "openai-codex"
    assert delegation["model"] == "gpt-5.6-sol"


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"enabled": "yes"}, "enabled"),
        ({"profiles": "default"}, "profiles"),
        ({"profiles": ["default", 7]}, "profiles[1]"),
        ({"default_route": []}, "default_route"),
        ({"command": ""}, "command"),
        ({"model": 7}, "model"),
        ({"timeout_seconds": True}, "timeout_seconds"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_input_bytes": 0}, "max_input_bytes"),
        ({"max_output_bytes": -1}, "max_output_bytes"),
        ({"fallback_to_delegation_model": 1}, "fallback_to_delegation_model"),
        ({"retention": []}, "retention"),
        ({"retention": {"raw_days": 0, "aggregate_days": 180}}, "retention.raw_days"),
        ({"retention": {"raw_days": 30, "aggregate_days": False}}, "retention.aggregate_days"),
        ({"extra_args": "--sandbox"}, "extra_args"),
        ({"review": {**EXPECTED_GEMINI_ROUTING_DEFAULTS["review"], "sample_size": 0}}, "review.sample_size"),
        ({"review": {**EXPECTED_GEMINI_ROUTING_DEFAULTS["review"], "sample_size": 4}}, "review.sample_size"),
        ({"review": {**EXPECTED_GEMINI_ROUTING_DEFAULTS["review"], "timezone": "UTC"}}, "review.timezone"),
    ],
)
def test_gemini_routing_rejects_invalid_types_and_bounds(overrides, field):
    errors = _error_messages(_routing_config(**overrides))

    assert any(field in message for message in errors), errors


@pytest.mark.parametrize(
    ("field", "allowed"),
    [
        ("default_route", ("auto", "gemini", "sol")),
        ("default_data_classification", ("standard", "restricted")),
        ("output_contract", ("text", "json")),
    ],
)
def test_gemini_routing_accepts_supported_route_data_and_output_enums(field, allowed):
    for value in allowed:
        assert not _error_messages(_routing_config(**{field: value}))


def test_gemini_routing_rejects_unsupported_route_data_and_output_enums():
    config = _routing_config(
        default_route="other",
        default_data_classification="secret",
        output_contract="yaml",
    )

    errors = _error_messages(config)

    assert any("default_route" in message for message in errors)
    assert any("default_data_classification" in message for message in errors)
    assert any("output_contract" in message for message in errors)


@pytest.mark.parametrize(
    "receipt_db",
    ["/tmp/gemini.sqlite3", "../gemini.sqlite3", "routing/../../gemini.sqlite3"],
)
def test_gemini_routing_requires_safe_relative_receipt_path(receipt_db):
    errors = _error_messages(_routing_config(receipt_db=receipt_db))

    assert any("receipt_db" in message for message in errors), errors


def test_gemini_routing_accepts_profile_relative_receipt_path():
    assert not _error_messages(
        _routing_config(receipt_db="routing/gemini-routing.sqlite3")
    )


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--dangerously-skip-permissions"],
        ["--dangerously-skip-permissions=true"],
        ["--print"],
        ["--print=false"],
        ["--add-dir=/tmp/escape"],
        ["--continue"],
        ["--continue=recent"],
        ["--conversation=other"],
        ["--model=other"],
        ["--effort=high"],
        ["--mode=agent"],
        ["--sandbox=false"],
        ["--no-sandbox"],
        ["--output-format=text"],
        ["--print-timeout=999s"],
        ["--json-schema=other.json"],
        ["--disable-slash-commands=false"],
        ["--no-disable-slash-commands"],
        None,
    ],
)
def test_gemini_routing_refuses_dangerous_permission_bypass_extra_args(extra_args):
    errors = _error_messages(_routing_config(extra_args=extra_args))

    assert any("extra_args" in message for message in errors)


def test_enabled_review_requires_pinned_workspace_identity():
    review = {
        **EXPECTED_GEMINI_ROUTING_DEFAULTS["review"],
        "enabled": True,
        "alert_workspace_id": "",
    }

    errors = _error_messages(_routing_config(review=review))

    assert any("review.alert_workspace_id" in message for message in errors)


@pytest.mark.parametrize(
    "alert_target",
    ["email:operator@example.com", "slack:", "slack:C:extra", "slack: C0A12345678"],
)
def test_enabled_review_requires_one_exact_slack_alert_target(alert_target):
    review = {
        **EXPECTED_GEMINI_ROUTING_DEFAULTS["review"],
        "enabled": True,
        "alert_target": alert_target,
        "alert_workspace_id": "T0A12345678",
    }

    errors = _error_messages(_routing_config(review=review))

    assert any("review.alert_target" in message for message in errors)


@pytest.mark.parametrize("receipt_db", ["/tmp/other-profile.sqlite3", "../escape.sqlite3"])
def test_receipt_db_must_remain_profile_local(receipt_db):
    errors = _error_messages(_routing_config(receipt_db=receipt_db))

    assert any("receipt_db" in message for message in errors)
