"""Hermetic tests for the macOS Keychain secret-source integration."""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import keychain as kc  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("TELEGRAM_BOT_TOKEN", "OPENROUTER_API_KEY", "EXISTING_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_normalise_env_map_accepts_mapping_list_and_shorthand():
    assert kc._normalise_env_map(  # noqa: SLF001 - private parser is the contract under test
        {
            "TELEGRAM_BOT_TOKEN": {
                "service": "halvo-shared",
                "account": "HERMES_MBP_TELEGRAM_BOT_TOKEN",
                "keychain": "/Users/alice/Library/Keychains/login.keychain-db",
            }
        }
    ) == {
        "TELEGRAM_BOT_TOKEN": {
            "service": "halvo-shared",
            "account": "HERMES_MBP_TELEGRAM_BOT_TOKEN",
            "keychain": "/Users/alice/Library/Keychains/login.keychain-db",
        }
    }

    assert kc._normalise_env_map(  # noqa: SLF001
        [{"env": "OPENROUTER_API_KEY", "service": "svc", "username": "acct"}]
    ) == {"OPENROUTER_API_KEY": {"service": "svc", "account": "acct"}}

    assert kc._normalise_env_map(  # noqa: SLF001
        {"EXISTING_API_KEY": "svc/acct"}
    ) == {"EXISTING_API_KEY": {"service": "svc", "account": "acct"}}


def test_read_keychain_secret_invokes_security_and_strips_newline(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="secret-value\n", stderr="")

    monkeypatch.setattr(kc.subprocess, "run", fake_run)

    value = kc.read_keychain_secret(
        service="halvo-shared",
        account="HERMES_MBP_TELEGRAM_BOT_TOKEN",
        keychain="/Users/alice/Library/Keychains/login.keychain-db",
        binary=Path("/usr/bin/security"),
        timeout_seconds=4,
    )

    assert value == "secret-value"
    assert captured["cmd"] == [
        "/usr/bin/security",
        "find-generic-password",
        "-s",
        "halvo-shared",
        "-a",
        "HERMES_MBP_TELEGRAM_BOT_TOKEN",
        "-w",
        "/Users/alice/Library/Keychains/login.keychain-db",
    ]
    assert captured["kwargs"]["timeout"] == 4


def test_read_keychain_secret_timeout_is_reported(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 0)

    monkeypatch.setattr(kc.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 2s"):
        kc.read_keychain_secret(
            service="svc",
            account="acct",
            binary=Path("/usr/bin/security"),
            timeout_seconds=2,
        )


def test_apply_resolves_keychain_specs_and_skips_existing(monkeypatch):
    monkeypatch.setenv("EXISTING_API_KEY", "already-real")
    monkeypatch.setattr(kc, "find_security_binary", lambda path="": Path("/usr/bin/security"))

    calls = []

    def fake_read(**kwargs):
        calls.append(kwargs)
        return f"resolved:{kwargs['account']}"

    monkeypatch.setattr(kc, "read_keychain_secret", fake_read)

    result = kc.apply_keychain_secrets(
        enabled=True,
        env={
            "TELEGRAM_BOT_TOKEN": {
                "service": "halvo-shared",
                "account": "HERMES_MBP_TELEGRAM_BOT_TOKEN",
            },
            "EXISTING_API_KEY": {"service": "svc", "account": "existing"},
        },
        override_existing=False,
    )

    assert result.ok
    assert result.applied == ["TELEGRAM_BOT_TOKEN"]
    assert result.skipped == ["EXISTING_API_KEY"]
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "resolved:HERMES_MBP_TELEGRAM_BOT_TOKEN"
    assert os.environ["EXISTING_API_KEY"] == "already-real"
    assert [call["account"] for call in calls] == ["HERMES_MBP_TELEGRAM_BOT_TOKEN"]


def test_apply_rejects_invalid_names_before_security_lookup(monkeypatch):
    monkeypatch.setattr(
        kc,
        "find_security_binary",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("security lookup not needed")),
    )

    result = kc.apply_keychain_secrets(
        enabled=True,
        env={"1BAD": {"service": "svc", "account": "acct"}},
    )

    assert result.ok
    assert not result.applied
    assert any("invalid environment variable" in warning for warning in result.warnings)


def test_apply_reports_missing_security_binary(monkeypatch):
    monkeypatch.setattr(kc, "find_security_binary", lambda path="": None)

    result = kc.apply_keychain_secrets(
        enabled=True,
        env={"TELEGRAM_BOT_TOKEN": {"service": "svc", "account": "acct"}},
    )

    assert not result.ok
    assert "security CLI not found" in (result.error or "")
