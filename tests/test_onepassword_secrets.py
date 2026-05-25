"""Hermetic tests for the 1Password secret-reference integration."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import onepassword as op_secret  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_caches():
    op_secret._reset_cache_for_tests()
    yield
    op_secret._reset_cache_for_tests()


def test_fetch_happy_path(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert cmd[:2] == [str(fake_binary), "read"]
        assert kwargs["env"]["NO_COLOR"] == "1"
        return mock.Mock(returncode=0, stdout="secret-value\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    secrets, warnings = op_secret.fetch_onepassword_secrets(
        references={"OPENAI_API_KEY": "op://Private/OpenAI/credential"},
        binary=fake_binary,
        use_cache=False,
    )

    assert secrets == {"OPENAI_API_KEY": "secret-value"}
    assert warnings == []
    assert calls[0][0] == [str(fake_binary), "read", "op://Private/OpenAI/credential"]


def test_fetch_account_flag(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return mock.Mock(returncode=0, stdout="value\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    op_secret.fetch_onepassword_secrets(
        references={"KEY": "op://Vault/Item/field"},
        binary=fake_binary,
        account="my.1password.com",
        use_cache=False,
    )

    assert captured["cmd"] == [
        str(fake_binary),
        "read",
        "op://Vault/Item/field",
        "--account",
        "my.1password.com",
    ]


def test_fetch_uses_custom_service_account_token_env(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    captured_env = {}
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "custom-token")

    def fake_run(_cmd, **kwargs):
        captured_env.update(kwargs["env"])
        return mock.Mock(returncode=0, stdout="value\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    op_secret.fetch_onepassword_secrets(
        references={"KEY": "op://Vault/Item/field"},
        binary=fake_binary,
        service_account_token_env="CUSTOM_OP_TOKEN",
        use_cache=False,
    )

    assert captured_env["OP_SERVICE_ACCOUNT_TOKEN"] == "custom-token"


def test_fetch_skips_invalid_mappings(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    run_called = False

    def fake_run(*_args, **_kwargs):
        nonlocal run_called
        run_called = True
        return mock.Mock(returncode=0, stdout="value\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    secrets, warnings = op_secret.fetch_onepassword_secrets(
        references={
            "1BAD": "op://Vault/Item/field",
            "SPACES BAD": "op://Vault/Item/field",
            "EMPTY": "",
            "NOT_OP": "https://example.com/secret",
        },
        binary=fake_binary,
        use_cache=False,
    )

    assert secrets == {}
    assert len(warnings) == 4
    assert not run_called


def test_fetch_op_failure(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")

    monkeypatch.setattr(
        op_secret.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=1, stdout="", stderr="not signed in"),
    )

    with pytest.raises(RuntimeError, match="not signed in"):
        op_secret.fetch_onepassword_secrets(
            references={"KEY": "op://Vault/Item/field"},
            binary=fake_binary,
            use_cache=False,
        )


def test_fetch_partial_failure_returns_successes_and_warnings(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")

    def fake_run(cmd, **_kwargs):
        if "bad" in cmd[2]:
            return mock.Mock(returncode=1, stdout="", stderr="not signed in")
        return mock.Mock(returncode=0, stdout="good-value\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    secrets, warnings = op_secret.fetch_onepassword_secrets(
        references={
            "GOOD": "op://Vault/Item/good",
            "BAD": "op://Vault/Item/bad",
        },
        binary=fake_binary,
        use_cache=False,
    )

    assert secrets == {"GOOD": "good-value"}
    assert len(warnings) == 1
    assert "BAD" in warnings[0]
    assert "not signed in" in warnings[0]


def test_fetch_timeout(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")

    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="op", timeout=30)

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        op_secret.fetch_onepassword_secrets(
            references={"KEY": "op://Vault/Item/field"},
            binary=fake_binary,
            use_cache=False,
        )


def test_fetch_cache_hits(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    call_count = {"n": 0}

    def fake_run(*a, **kw):
        call_count["n"] += 1
        return mock.Mock(returncode=0, stdout="value\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    kwargs = {
        "references": {"KEY": "op://Vault/Item/field"},
        "binary": fake_binary,
        "cache_ttl_seconds": 60,
    }
    op_secret.fetch_onepassword_secrets(**kwargs)
    op_secret.fetch_onepassword_secrets(**kwargs)

    assert call_count["n"] == 1


def test_fetch_cache_hit_recomputes_validation_warnings(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    call_count = {"n": 0}

    def fake_run(*a, **kw):
        call_count["n"] += 1
        return mock.Mock(returncode=0, stdout="value\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    kwargs = {
        "binary": fake_binary,
        "cache_ttl_seconds": 60,
    }
    op_secret.fetch_onepassword_secrets(
        references={"KEY": "op://Vault/Item/field", "BAD": "not-op"},
        **kwargs,
    )
    _secrets, warnings = op_secret.fetch_onepassword_secrets(
        references={"KEY": "op://Vault/Item/field", "ALSO_BAD": ""},
        **kwargs,
    )

    assert call_count["n"] == 1
    assert len(warnings) == 1
    assert "ALSO_BAD" in warnings[0]


def test_apply_disabled_returns_empty():
    result = op_secret.apply_onepassword_secrets(enabled=False)
    assert result.ok
    assert not result.applied
    assert not result.error


def test_apply_missing_references():
    result = op_secret.apply_onepassword_secrets(enabled=True, references={})
    assert not result.ok
    assert result.error is not None
    assert "no references" in result.error


def test_apply_all_invalid_references_is_error(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    monkeypatch.setattr(op_secret, "find_op", lambda: fake_binary)

    result = op_secret.apply_onepassword_secrets(
        enabled=True,
        references={"BAD": "not-op"},
    )

    assert not result.ok
    assert result.error is not None
    assert "No valid" in result.error
    assert result.warnings


def test_apply_missing_op(monkeypatch):
    monkeypatch.setattr(op_secret, "find_op", lambda: None)
    result = op_secret.apply_onepassword_secrets(
        enabled=True,
        references={"KEY": "op://Vault/Item/field"},
    )
    assert not result.ok
    assert result.error is not None
    assert "op" in result.error


def test_apply_does_not_override_existing(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    monkeypatch.setattr(op_secret, "find_op", lambda: fake_binary)
    monkeypatch.setenv("OPENAI_API_KEY", "existing")
    monkeypatch.setattr(
        op_secret.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout="from-1password\n", stderr=""),
    )

    result = op_secret.apply_onepassword_secrets(
        enabled=True,
        references={"OPENAI_API_KEY": "op://Vault/Item/field"},
        override_existing=False,
    )

    assert result.ok
    assert result.skipped == ["OPENAI_API_KEY"]
    assert os.environ["OPENAI_API_KEY"] == "existing"


def test_apply_override_existing(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    monkeypatch.setattr(op_secret, "find_op", lambda: fake_binary)
    monkeypatch.setenv("OPENAI_API_KEY", "existing")
    monkeypatch.setattr(
        op_secret.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout="from-1password\n", stderr=""),
    )

    result = op_secret.apply_onepassword_secrets(
        enabled=True,
        references={"OPENAI_API_KEY": "op://Vault/Item/field"},
        override_existing=True,
    )

    assert result.ok
    assert result.applied == ["OPENAI_API_KEY"]
    assert os.environ["OPENAI_API_KEY"] == "from-1password"


def test_apply_skips_service_account_token_env(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    monkeypatch.setattr(op_secret, "find_op", lambda: fake_binary)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "bootstrap")
    monkeypatch.setattr(
        op_secret.subprocess,
        "run",
        lambda *a, **kw: mock.Mock(returncode=0, stdout="replacement\n", stderr=""),
    )

    result = op_secret.apply_onepassword_secrets(
        enabled=True,
        references={"OP_SERVICE_ACCOUNT_TOKEN": "op://Vault/Token/credential"},
        override_existing=True,
    )

    assert result.ok
    assert result.skipped == ["OP_SERVICE_ACCOUNT_TOKEN"]
    assert os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "bootstrap"


def test_fetch_uses_fresh_disk_cache_across_processes(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    calls = {"n": 0}

    def fake_run(*_args, **_kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout="value-from-op\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)
    kwargs = {
        "references": {"KEY": "op://Vault/Item/field"},
        "binary": fake_binary,
        "cache_ttl_seconds": 60,
        "home_path": tmp_path,
    }

    secrets, warnings = op_secret.fetch_onepassword_secrets(**kwargs)
    assert secrets == {"KEY": "value-from-op"}
    assert warnings == []
    assert calls["n"] == 1

    op_secret._reset_cache_for_tests()
    cached, cached_warnings = op_secret.fetch_onepassword_secrets(**kwargs)

    assert cached == {"KEY": "value-from-op"}
    assert cached_warnings == []
    assert calls["n"] == 1


def test_fetch_disk_cache_is_written_0600_without_auth_material(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_secret_token")
    monkeypatch.setattr(
        op_secret.subprocess,
        "run",
        lambda *_args, **_kwargs: mock.Mock(returncode=0, stdout="value\n", stderr=""),
    )

    op_secret.fetch_onepassword_secrets(
        references={"KEY": "op://Vault/Item/field"},
        binary=fake_binary,
        cache_ttl_seconds=60,
        home_path=tmp_path,
    )

    cache_path = tmp_path / "cache" / "op_cache.json"
    payload = json.loads(cache_path.read_text())
    mode = stat.S_IMODE(cache_path.stat().st_mode)

    assert mode == 0o600
    assert payload["secrets"] == {"KEY": "value"}
    assert "ops_secret_token" not in cache_path.read_text()


def test_fetch_ignores_stale_disk_cache(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "op_cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "key": op_secret._cache_key_str(
                    (
                        op_secret._auth_fingerprint(),
                        "",
                        (("KEY", "op://Vault/Item/field"),),
                    )
                ),
                "secrets": {"KEY": "stale"},
                "fetched_at": 1,
            }
        )
    )
    calls = {"n": 0}

    def fake_run(*_args, **_kwargs):
        calls["n"] += 1
        return mock.Mock(returncode=0, stdout="fresh\n", stderr="")

    monkeypatch.setattr(op_secret.subprocess, "run", fake_run)

    secrets, _warnings = op_secret.fetch_onepassword_secrets(
        references={"KEY": "op://Vault/Item/field"},
        binary=fake_binary,
        cache_ttl_seconds=0.001,
        home_path=tmp_path,
    )

    assert secrets == {"KEY": "fresh"}
    assert calls["n"] == 1


def test_apply_skips_existing_without_contacting_op_when_not_overriding(monkeypatch, tmp_path):
    fake_binary = tmp_path / "op"
    fake_binary.write_text("")
    monkeypatch.setattr(op_secret, "find_op", lambda: fake_binary)
    monkeypatch.setenv("OPENAI_API_KEY", "existing")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("op should not be invoked for an env var that will be skipped")

    monkeypatch.setattr(op_secret.subprocess, "run", fail_run)

    result = op_secret.apply_onepassword_secrets(
        enabled=True,
        references={"OPENAI_API_KEY": "op://Vault/Item/field"},
        override_existing=False,
        home_path=tmp_path,
    )

    assert result.ok
    assert result.applied == []
    assert result.skipped == ["OPENAI_API_KEY"]
    assert os.environ["OPENAI_API_KEY"] == "existing"
