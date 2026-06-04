"""Hermetic tests for the 1Password secret-source integration.

No real 1Password account, Service Account token, or network access is used.
The `op` CLI subprocess is mocked in every fetch/apply test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import onepassword as opsec  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cache():
    opsec._reset_cache_for_tests()
    yield
    opsec._reset_cache_for_tests()


def test_normalize_mapping_rejects_invalid_entries():
    mapping, warnings = opsec._normalize_mapping(
        {
            "VALID_KEY": "op://Hermes/Valid/password",
            "1BAD": "op://Hermes/Bad/password",
            "NOT_OP": "secret-value",
        }
    )

    assert mapping == {"VALID_KEY": "op://Hermes/Valid/password"}
    assert len(warnings) == 2


def test_fetch_happy_path_uses_op_read_and_token_env(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    mapping = {
        "OPENAI_API_KEY": "op://Hermes/OpenAI/password",
        "ANTHROPIC_API_KEY": "op://Hermes/Anthropic/password",
    }
    values = {
        "op://Hermes/OpenAI/password": "sk-openai-test",
        "op://Hermes/Anthropic/password": "sk-anthropic-test\n",
    }
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs["env"].copy()))
        assert cmd[0] == str(fake_binary)
        assert cmd[1] == "read"
        assert kwargs["env"]["OP_SERVICE_ACCOUNT_TOKEN"] == "ops_test_token"
        return mock.Mock(returncode=0, stdout=values[cmd[2]], stderr="")

    monkeypatch.setattr(opsec.subprocess, "run", fake_run)

    secrets, warnings = opsec.fetch_onepassword_secrets(
        service_account_token="ops_test_token",
        mapping=mapping,
        binary=fake_binary,
        use_cache=False,
    )

    assert secrets == {
        "OPENAI_API_KEY": "sk-openai-test",
        "ANTHROPIC_API_KEY": "sk-anthropic-test",
    }
    assert warnings == []
    assert [call[0][2] for call in calls] == list(mapping.values())


def test_fetch_op_failure_surfaces_reference_not_secret(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")

    monkeypatch.setattr(
        opsec.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=1, stdout="", stderr="permission denied"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        opsec.fetch_onepassword_secrets(
            service_account_token="ops_test_token",
            mapping={"API_KEY": "op://Hermes/API/password"},
            binary=fake_binary,
            use_cache=False,
        )

    msg = str(excinfo.value)
    assert "op read failed for API_KEY" in msg
    assert "op://Hermes/API/password" not in msg
    assert "permission denied" in msg
    assert "ops_test_token" not in msg


def test_fetch_timeout(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")

    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="op", timeout=20)

    monkeypatch.setattr(opsec.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        opsec.fetch_onepassword_secrets(
            service_account_token="ops_test_token",
            mapping={"API_KEY": "op://Hermes/API/password"},
            binary=fake_binary,
            use_cache=False,
        )


def test_fetch_cache_hits(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    call_count = {"n": 0}

    def fake_run(*a, **kw):
        call_count["n"] += 1
        return mock.Mock(returncode=0, stdout="secret-value", stderr="")

    monkeypatch.setattr(opsec.subprocess, "run", fake_run)

    kwargs = {
        "service_account_token": "ops_test_token",
        "mapping": {"API_KEY": "op://Hermes/API/password"},
        "binary": fake_binary,
        "cache_ttl_seconds": 60,
    }
    opsec.fetch_onepassword_secrets(**kwargs)
    opsec.fetch_onepassword_secrets(**kwargs)

    assert call_count["n"] == 1


def test_apply_missing_token_does_not_raise(monkeypatch):
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)

    result = opsec.apply_onepassword_secrets(
        enabled=True,
        mapping={"API_KEY": "op://Hermes/API/password"},
    )

    assert not result.ok
    assert result.error is not None
    assert "OP_SERVICE_ACCOUNT_TOKEN" in result.error
    assert not result.applied


def test_apply_does_not_override_existing(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_test_token")
    monkeypatch.setenv("API_KEY", "existing")
    monkeypatch.setattr(opsec, "find_op", lambda: fake_binary)
    monkeypatch.setattr(
        opsec.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout="from-1password", stderr=""),
    )

    result = opsec.apply_onepassword_secrets(
        enabled=True,
        mapping={"API_KEY": "op://Hermes/API/password"},
        override_existing=False,
    )

    assert result.ok
    assert "API_KEY" in result.skipped
    assert os.environ["API_KEY"] == "existing"


def test_apply_override_existing_and_skip_bootstrap(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_original")
    monkeypatch.setenv("API_KEY", "stale")
    monkeypatch.setattr(opsec, "find_op", lambda: fake_binary)

    def fake_run(cmd, **kwargs):
        ref = cmd[2]
        if ref.endswith("/token"):
            return mock.Mock(returncode=0, stdout="ops_replacement", stderr="")
        return mock.Mock(returncode=0, stdout="fresh", stderr="")

    monkeypatch.setattr(opsec.subprocess, "run", fake_run)

    result = opsec.apply_onepassword_secrets(
        enabled=True,
        mapping={
            "API_KEY": "op://Hermes/API/password",
            "OP_SERVICE_ACCOUNT_TOKEN": "op://Hermes/Bootstrap/token",
        },
        override_existing=True,
    )

    assert result.ok
    assert os.environ["API_KEY"] == "fresh"
    assert os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "ops_original"
    assert "API_KEY" in result.applied
    assert "OP_SERVICE_ACCOUNT_TOKEN" in result.skipped
