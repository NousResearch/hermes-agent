import argparse
import subprocess

import pytest


def test_keychain_copy_never_returns_secret(monkeypatch):
    from hermes_cli import credentials

    calls = []
    monkeypatch.setattr(credentials.sys, "platform", "darwin")

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["security", "find-generic-password"]:
            return subprocess.CompletedProcess(argv, 0, stdout="super-secret-token\n", stderr="")
        if argv[:2] == ["security", "add-generic-password"]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    result = credentials.copy_keychain_secret(
        source_service="chief-of-staff",
        source_account="JIRA_API_TOKEN",
        destination_service="hermes-agent",
        destination_account="JIRA_API_TOKEN",
    )

    assert result == {
        "copied": True,
        "source_service": "chief-of-staff",
        "source_account": "JIRA_API_TOKEN",
        "destination_service": "hermes-agent",
        "destination_account": "JIRA_API_TOKEN",
    }
    assert "super-secret-token" not in repr(result)
    assert calls[-1] == [
        "security",
        "add-generic-password",
        "-U",
        "-s",
        "hermes-agent",
        "-a",
        "JIRA_API_TOKEN",
        "-w",
        "super-secret-token",
    ]


def test_import_chief_of_staff_uses_known_fallbacks(monkeypatch):
    from hermes_cli import credentials

    source_values = {
        ("chief-of-staff", "JIRA_API_TOKEN"): "jira-token",
        ("chief-of-staff", "ATLASSIAN_API_TOKEN"): "atlassian-token",
        ("chief-of-staff", "ZENDESK_API_TOKEN"): "zendesk-token",
    }
    written = []

    def fake_read(account, service=None):
        return source_values.get((service, account), "")

    def fake_write(account, password, service=None):
        written.append((service, account, password))

    monkeypatch.setattr(credentials, "get_keychain_password", fake_read)
    monkeypatch.setattr(credentials, "set_keychain_password", fake_write)

    result = credentials.import_chief_of_staff_credentials(destination_service="hermes-agent")

    assert result["destination_service"] == "hermes-agent"
    assert result["copied"] == [
        {"name": "jira.api_token", "account": "JIRA_API_TOKEN", "source_service": "chief-of-staff", "source_account": "JIRA_API_TOKEN"},
        {"name": "atlassian.api_token", "account": "ATLASSIAN_API_TOKEN", "source_service": "chief-of-staff", "source_account": "ATLASSIAN_API_TOKEN"},
        {"name": "confluence.api_token", "account": "CONFLUENCE_API_TOKEN", "source_service": "chief-of-staff", "source_account": "ATLASSIAN_API_TOKEN"},
        {"name": "zendesk.api_token", "account": "ZENDESK_API_TOKEN", "source_service": "chief-of-staff", "source_account": "ZENDESK_API_TOKEN"},
    ]
    assert result["missing"] == [{"name": "zendesk.oauth_token", "account": "ZENDESK_OAUTH_TOKEN"}]
    assert ("hermes-agent", "CONFLUENCE_API_TOKEN", "atlassian-token") in written
    assert "jira-token" not in repr(result)
    assert "atlassian-token" not in repr(result)
    assert "zendesk-token" not in repr(result)


def test_register_credentials_parser_wires_safe_commands():
    from hermes_cli.credentials import register_credentials_subparser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_credentials_subparser(subparsers)

    args = parser.parse_args(["credentials", "import-chief-of-staff", "--service", "hermes-agent"])

    assert args.command == "credentials"
    assert args.credentials_action == "import-chief-of-staff"
    assert callable(args.func)


def test_get_keychain_password_returns_empty_on_missing(monkeypatch):
    from hermes_cli import credentials

    monkeypatch.setattr(credentials.sys, "platform", "darwin")

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 44, stdout="", stderr="not found")

    monkeypatch.setattr(credentials.subprocess, "run", fake_run)

    assert credentials.get_keychain_password("NOPE", service="hermes-agent") == ""
