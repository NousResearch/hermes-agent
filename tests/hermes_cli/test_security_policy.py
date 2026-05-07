"""Tests for the OpenShell-inspired Hermes security posture surface."""

from __future__ import annotations

from argparse import Namespace

import pytest

from hermes_cli.security_policy import (
    build_effective_policy,
    evaluate_posture,
    security_command,
    validate_policy_mapping,
)


def test_effective_policy_prefers_env_safe_root():
    cfg = {
        "security": {
            "policy": {"filesystem": {"write_safe_root": "/from-config"}},
            "redact_secrets": True,
        }
    }
    policy = build_effective_policy(cfg, {"HERMES_WRITE_SAFE_ROOT": "/from-env"})

    assert policy["filesystem"]["write_safe_root"] == "/from-env"
    assert policy["filesystem"]["write_safe_root_source"] == "HERMES_WRITE_SAFE_ROOT"
    assert policy["inference"]["redact_secrets"] is True


def test_effective_policy_reports_provider_env_passthrough():
    cfg = {"terminal": {"env_passthrough": ["OPENAI_API_KEY", "NOTION_TOKEN"]}}
    policy = build_effective_policy(cfg, {})

    passthrough = policy["environment"]["provider_credentials_in_passthrough"]
    assert "OPENAI_API_KEY" in passthrough
    assert "NOTION_TOKEN" not in passthrough


def test_effective_policy_maps_openshell_domains():
    cfg = {
        "security": {
            "policy": {
                "version": 1,
                "filesystem_policy": {
                    "read_only": ["/usr"],
                    "read_write": ["/workspace", "/tmp/cache"],
                    "deny_hermes_control_plane": True,
                },
                "network_policies": {
                    "github_api": {
                        "endpoints": [
                            {
                                "host": "api.github.com",
                                "port": 443,
                                "protocol": "rest",
                                "access": "read-only",
                                "enforcement": "enforce",
                            }
                        ],
                        "binaries": [{"path": "/usr/bin/curl"}],
                    }
                },
                "permissions": {"tools_deny": ["terminal"]},
                "gateway": {"require_user_allowlist": True},
                "inference": {"managed_provider_routing": True},
            },
            "redact_secrets": True,
        },
        "approvals": {"mcp_reload_confirm": True},
    }
    policy = build_effective_policy(cfg, {"GATEWAY_ALLOWED_USERS": "123"})

    assert policy["filesystem"]["read_write_paths"] == ["/workspace", "/tmp/cache"]
    assert policy["network"]["policy_entry_count"] == 1
    assert policy["permissions"]["tools_deny"] == ["terminal"]
    assert policy["gateway"]["allowlist_sources"] == ["GATEWAY_ALLOWED_USERS"]
    assert policy["inference"]["managed_provider_routing"] is True


def test_posture_strict_flags_enterprise_gaps():
    policy = build_effective_policy(
        {
            "approvals": {"mode": "off"},
            "security": {
                "redact_secrets": False,
                "tirith_enabled": False,
                "tirith_fail_open": True,
                "allow_private_urls": True,
                "policy": {"filesystem": {"deny_hermes_control_plane": False}},
            },
            "terminal": {"env_passthrough": ["OPENAI_API_KEY"]},
        },
        {},
    )
    checks = evaluate_posture(policy)
    failing = {(check.domain, check.name) for check in checks if check.status == "fail"}
    warning = {(check.domain, check.name) for check in checks if check.status == "warn"}

    assert ("filesystem", "control_plane_write_deny") in failing
    assert ("environment", "provider_env_passthrough") in failing
    assert ("process", "dangerous_command_approvals") in failing
    assert ("inference", "secret_redaction") in warning


def test_validate_policy_mapping_accepts_legacy_supported_shape():
    errors, warnings = validate_policy_mapping(
        {
            "security": {
                "policy": {
                    "filesystem": {
                        "write_safe_root": "/workspace",
                        "deny_hermes_control_plane": True,
                    },
                    "process": {
                        "approvals_mode": "manual",
                        "tirith_enabled": True,
                        "tirith_fail_open": False,
                    },
                }
            }
        }
    )

    assert errors == []
    assert warnings == []


def test_validate_policy_mapping_accepts_openshell_shape():
    errors, warnings = validate_policy_mapping(
        {
            "version": 1,
            "filesystem_policy": {
                "read_only": ["/usr", "/lib"],
                "read_write": ["/workspace"],
                "include_workdir": True,
            },
            "landlock": {"compatibility": "best_effort"},
            "process": {"run_as_user": "sandbox", "run_as_group": "sandbox"},
            "network_policies": {
                "github_api": {
                    "name": "github-api",
                    "endpoints": [
                        {
                            "host": "api.github.com",
                            "port": 443,
                            "protocol": "rest",
                            "access": "read-only",
                            "enforcement": "enforce",
                        }
                    ],
                    "binaries": [{"path": "/usr/bin/curl"}],
                }
            },
        }
    )

    assert errors == []
    assert warnings == []


def test_validate_policy_mapping_rejects_unknown_and_wrong_types():
    errors, _warnings = validate_policy_mapping(
        {
            "filesystem": {
                "write_safe_root": 123,
                "unknown": True,
            },
            "kernel": {"landlock": True},
        }
    )

    assert "filesystem.write_safe_root must be a string" in errors
    assert "unsupported policy key: filesystem.unknown" in errors
    assert "unsupported policy domain: kernel" in errors


def test_validate_policy_mapping_rejects_invalid_openshell_shape():
    errors, _warnings = validate_policy_mapping(
        {
            "version": 2,
            "process": {"run_as_user": "root"},
            "network_policies": {
                "bad": {
                    "endpoints": [{"host": "api.github.com", "port": "443"}],
                    "binaries": [{"path": 123}],
                }
            },
        }
    )

    assert "version must be 1" in errors
    assert "process.run_as_user must not be root or 0" in errors
    assert "network_policies.bad.endpoints[0].port must be an integer" in errors
    assert "network_policies.bad.binaries[0].path must be a string" in errors


def test_security_doctor_strict_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.security_policy.build_effective_policy",
        lambda: build_effective_policy({"approvals": {"mode": "off"}}, {}),
    )

    with pytest.raises(SystemExit) as exc:
        security_command(Namespace(security_action="doctor", strict=True, json=False))

    assert exc.value.code == 1
    assert "dangerous_command_approvals" in capsys.readouterr().out
