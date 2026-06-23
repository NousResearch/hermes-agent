"""Hermetic tests for the 1Password CLI secret-source integration."""

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

from agent.secret_sources import onepassword as op  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("TELEGRAM_BOT_TOKEN", "OPENROUTER_API_KEY", "EXISTING_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_normalise_env_map_accepts_mapping_and_list():
    assert op._normalise_env_map(  # noqa: SLF001 - private parser is the contract under test
        {" TELEGRAM_BOT_TOKEN ": " op://Vault/Bot/token "}
    ) == {"TELEGRAM_BOT_TOKEN": "op://Vault/Bot/token"}

    assert op._normalise_env_map(  # noqa: SLF001
        [{"env_var": "OPENROUTER_API_KEY", "reference": "op://Vault/OpenRouter/key"}]
    ) == {"OPENROUTER_API_KEY": "op://Vault/OpenRouter/key"}


def test_read_onepassword_reference_uses_op_read_and_strips_newline(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="secret-value\n", stderr="")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    value = op.read_onepassword_reference(
        "op://Vault/Item/field",
        binary=Path("/usr/local/bin/op"),
        timeout_seconds=2,
        account="empire",
    )

    assert value == "secret-value"
    assert captured["cmd"] == [
        "/usr/local/bin/op",
        "read",
        "op://Vault/Item/field",
        "--account",
        "empire",
    ]
    assert captured["kwargs"]["timeout"] == 2
    assert captured["kwargs"]["env"]["NO_COLOR"] == "1"


def test_read_onepassword_reference_redacts_reference_on_failure(monkeypatch):
    reference = "op://Vault/Item/field"

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=f"could not read {reference}: item not found",
        )

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        op.read_onepassword_reference(reference, binary=Path("/usr/local/bin/op"))

    message = str(exc_info.value)
    assert reference not in message
    assert "[1Password reference]" in message


def test_read_onepassword_reference_timeout_is_reported(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 0)

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 3s"):
        op.read_onepassword_reference(
            "op://Vault/Item/field",
            binary=Path("/usr/local/bin/op"),
            timeout_seconds=3,
        )


def test_apply_resolves_config_refs_and_env_refs(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "op://Vault/Bot/token")
    monkeypatch.setattr(op, "find_op", lambda op_path="": Path("/usr/local/bin/op"))

    calls = []

    def fake_read(reference, **kwargs):
        calls.append((reference, kwargs))
        return f"resolved:{reference.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(op, "read_onepassword_reference", fake_read)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"OPENROUTER_API_KEY": "op://Vault/OpenRouter/key"},
        resolve_env_references=True,
    )

    assert result.ok
    assert sorted(result.applied) == ["OPENROUTER_API_KEY", "TELEGRAM_BOT_TOKEN"]
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "resolved:token"
    assert os.environ["OPENROUTER_API_KEY"] == "resolved:key"
    assert [call[0] for call in calls] == [
        "op://Vault/OpenRouter/key",
        "op://Vault/Bot/token",
    ]


def test_apply_skips_existing_real_value_unless_override(monkeypatch):
    monkeypatch.setenv("EXISTING_API_KEY", "already-real")
    monkeypatch.setattr(op, "find_op", lambda op_path="": Path("/usr/local/bin/op"))
    monkeypatch.setattr(
        op,
        "read_onepassword_reference",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not read skipped ref")),
    )

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"EXISTING_API_KEY": "op://Vault/Existing/key"},
        override_existing=False,
    )

    assert result.ok
    assert result.skipped == ["EXISTING_API_KEY"]
    assert os.environ["EXISTING_API_KEY"] == "already-real"


def test_apply_rejects_invalid_names_and_non_refs_before_find_op(monkeypatch):
    monkeypatch.setattr(
        op,
        "find_op",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("op lookup not needed")),
    )

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"1BAD": "op://Vault/Bad/key", "OPENROUTER_API_KEY": "not-a-reference"},
    )

    assert result.ok
    assert not result.applied
    assert any("invalid environment variable" in warning for warning in result.warnings)
    assert any("not an op:// reference" in warning for warning in result.warnings)


def test_apply_reports_missing_op(monkeypatch):
    monkeypatch.setattr(op, "find_op", lambda op_path="": None)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"OPENROUTER_API_KEY": "op://Vault/OpenRouter/key"},
    )

    assert not result.ok
    assert "op` not found" in (result.error or "")
