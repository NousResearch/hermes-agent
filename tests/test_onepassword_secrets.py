"""Hermetic tests for the 1Password (`op` CLI) secret source.

We never invoke the real ``op`` binary: ``subprocess.run`` is mocked so the
suite stays fast and offline-safe.  A live resolve is exercised manually via
``hermes secrets onepassword sync`` outside of pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest


# Make the worktree importable without depending on the installed wheel.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import onepassword as op  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_caches():
    op._reset_cache_for_tests()
    yield
    op._reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _clean_op_env(monkeypatch):
    """Start every test from a known 1Password auth state."""
    for key in list(os.environ):
        if key.startswith("OP_SESSION_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.delenv("OP_ACCOUNT", raising=False)
    monkeypatch.delenv("OP_CONNECT_HOST", raising=False)
    monkeypatch.delenv("OP_CONNECT_TOKEN", raising=False)
    yield


def _ok(value: str):
    return mock.Mock(returncode=0, stdout=value, stderr="")


def _err(code: int, stderr: str):
    return mock.Mock(returncode=code, stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


def test_validate_references_filters_bad_names_and_refs():
    refs = {
        "OPENAI_API_KEY": "op://Private/OpenAI/api key",
        "1BAD_NAME": "op://Private/x/y",          # bad env name
        "HAS SPACE": "op://Private/x/y",          # bad env name
        "NOT_A_REF": "https://example.com",        # not op://
        "WHITESPACE": "  op://Private/z/field  ",  # stripped + kept
    }
    valid, warnings = op._validate_references(refs)
    assert valid == {
        "OPENAI_API_KEY": "op://Private/OpenAI/api key",
        "WHITESPACE": "op://Private/z/field",
    }
    assert len(warnings) == 3


# ---------------------------------------------------------------------------
# fetch_onepassword_secrets
# ---------------------------------------------------------------------------


def test_fetch_happy_path(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    values = {
        "op://Private/OpenAI/api key": "sk-abc\n",
        "op://Private/Anthropic/credential": "sk-ant-xyz",
    }

    def fake_run(cmd, **kwargs):
        # argv list, never shell=True; reference passed after `--`.
        assert "--" in cmd
        ref = cmd[cmd.index("--") + 1]
        return _ok(values[ref])

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={
            "OPENAI_API_KEY": "op://Private/OpenAI/api key",
            "ANTHROPIC_API_KEY": "op://Private/Anthropic/credential",
        },
        binary=fake_op,
        use_cache=False,
    )
    assert secrets == {"OPENAI_API_KEY": "sk-abc", "ANTHROPIC_API_KEY": "sk-ant-xyz"}
    assert warnings == []


def test_fetch_uses_option_terminator_and_account(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _ok("value")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"},
        account="my.1password.com",
        binary=fake_op,
        use_cache=False,
    )
    cmd = captured["cmd"]
    assert cmd[:2] == [str(fake_op), "read"]
    assert "--account" in cmd and "my.1password.com" in cmd
    # `--` must precede the positional reference.
    assert cmd[-2:] == ["--", "op://V/I/F"]


def test_fetch_empty_rc0_does_not_clobber(monkeypatch, tmp_path):
    """returncode 0 with empty stdout must surface as a warning, not a value."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: _ok("   \n"))

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, binary=fake_op, use_cache=False
    )
    assert secrets == {}
    assert any("empty value" in w for w in warnings)


def test_fetch_read_failure_becomes_warning(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(
        op.subprocess, "run", lambda *a, **k: _err(1, "\x1b[31m[ERROR] not signed in\x1b[0m")
    )

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, binary=fake_op, use_cache=False
    )
    assert secrets == {}
    assert len(warnings) == 1
    # ANSI control sequences are fully scrubbed from the surfaced message.
    assert "\x1b" not in warnings[0]
    assert "[31m" not in warnings[0]
    assert "not signed in" in warnings[0]


def test_fetch_one_bad_one_good(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")

    def fake_run(cmd, **kwargs):
        ref = cmd[cmd.index("--") + 1]
        if ref == "op://V/good/f":
            return _ok("good-value")
        return _err(1, "no access")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"GOOD": "op://V/good/f", "BAD": "op://V/bad/f"},
        binary=fake_op,
        use_cache=False,
    )
    assert secrets == {"GOOD": "good-value"}
    assert len(warnings) == 1


def test_fetch_missing_binary_raises(monkeypatch):
    monkeypatch.setattr(op, "find_op", lambda binary_path="": None)
    with pytest.raises(RuntimeError, match="op CLI not found"):
        op.fetch_onepassword_secrets(
            references={"K": "op://V/I/F"}, use_cache=False
        )


def test_fetch_child_env_is_allowlisted(monkeypatch, tmp_path):
    """The op child must NOT inherit unrelated provider credentials."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setenv("OPENAI_API_KEY", "leak-me")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_tok")
    monkeypatch.setenv("OP_SESSION_myacct", "sess123")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _ok("v")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, binary=fake_op, use_cache=False
    )
    env = captured["env"]
    assert "OPENAI_API_KEY" not in env          # not inherited
    assert env["OP_SERVICE_ACCOUNT_TOKEN"] == "ops_tok"
    assert env["OP_SESSION_myacct"] == "sess123"
    assert env.get("NO_COLOR") == "1"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_inprocess_cache_hit(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op._reset_cache_for_tests(tmp_path)
    for _ in range(2):
        op.fetch_onepassword_secrets(
            references={"K": "op://V/I/F"}, cache_ttl_seconds=60,
            binary=fake_op, home_path=tmp_path,
        )
    assert calls["n"] == 1  # second call served from L1 cache


def test_disk_cache_roundtrip_and_no_token_on_disk(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_supersecret")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("resolved")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op._reset_cache_for_tests(tmp_path)

    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=300,
        binary=fake_op, home_path=tmp_path,
    )
    assert calls["n"] == 1

    cache_path = op._disk_cache_path(tmp_path)
    assert cache_path.exists()
    assert (os.stat(cache_path).st_mode & 0o777) == 0o600
    text = cache_path.read_text()
    assert "ops_supersecret" not in text            # token never on disk
    payload = json.loads(text)
    assert payload["secrets"] == {"K": "resolved"}

    # Simulate a fresh process: clear only the in-process cache.
    op._CACHE.clear()
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=300,
        binary=fake_op, home_path=tmp_path,
    )
    assert calls["n"] == 1  # served from disk, op not re-invoked


def test_ttl_zero_disables_both_layers(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op._reset_cache_for_tests(tmp_path)

    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=0,
        binary=fake_op, home_path=tmp_path,
    )
    # No disk file written when TTL is 0.
    assert not op._disk_cache_path(tmp_path).exists()
    op._CACHE.clear()
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=0,
        binary=fake_op, home_path=tmp_path,
    )
    assert calls["n"] == 2  # never cached


def test_session_change_invalidates_cache(monkeypatch, tmp_path):
    """A different OP_SESSION_* identity must not reuse a cached value."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op._reset_cache_for_tests(tmp_path)

    monkeypatch.setenv("OP_SESSION_acctA", "sessA")
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=300,
        binary=fake_op, home_path=tmp_path,
    )
    # Switch identity.
    monkeypatch.delenv("OP_SESSION_acctA", raising=False)
    monkeypatch.setenv("OP_SESSION_acctB", "sessB")
    op._CACHE.clear()
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=300,
        binary=fake_op, home_path=tmp_path,
    )
    assert calls["n"] == 2  # cache key changed → refetch


def test_connect_credential_change_invalidates_cache(monkeypatch, tmp_path):
    """A different 1Password Connect identity must not reuse a cached value."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("v")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op._reset_cache_for_tests(tmp_path)

    monkeypatch.setenv("OP_CONNECT_HOST", "https://connect.example.com")
    monkeypatch.setenv("OP_CONNECT_TOKEN", "tokenA")
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=300,
        binary=fake_op, home_path=tmp_path,
    )
    # Rotate the Connect token → new identity.
    monkeypatch.setenv("OP_CONNECT_TOKEN", "tokenB")
    op._CACHE.clear()
    op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, cache_ttl_seconds=300,
        binary=fake_op, home_path=tmp_path,
    )
    assert calls["n"] == 2  # cache key changed → refetch


def test_partial_failure_not_cached(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")

    def fake_run(cmd, **kwargs):
        ref = cmd[cmd.index("--") + 1]
        return _ok("v") if ref == "op://V/good/f" else _err(1, "fail")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    op._reset_cache_for_tests(tmp_path)
    op.fetch_onepassword_secrets(
        references={"G": "op://V/good/f", "B": "op://V/bad/f"},
        cache_ttl_seconds=300, binary=fake_op, home_path=tmp_path,
    )
    # A pull with any read error must not be persisted.
    assert not op._disk_cache_path(tmp_path).exists()


def test_reset_cache_clears_disk(tmp_path):
    cache_path = op._disk_cache_path(tmp_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{}")
    assert cache_path.exists()
    op._reset_cache_for_tests(tmp_path)
    assert not cache_path.exists()
    op._reset_cache_for_tests(tmp_path)  # idempotent


# ---------------------------------------------------------------------------
# find_op
# ---------------------------------------------------------------------------


def test_find_op_pinned_path_not_on_path(tmp_path, monkeypatch):
    pinned = tmp_path / "op"
    pinned.write_text("")
    pinned.chmod(0o755)
    # PATH lookup must NOT be consulted when a binary_path is pinned.
    monkeypatch.setattr(op.shutil, "which", lambda name: "/usr/bin/op")
    assert op.find_op(str(pinned)) == pinned


def test_find_op_pinned_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(op.shutil, "which", lambda name: "/usr/bin/op")
    assert op.find_op(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------------------
# apply_onepassword_secrets
# ---------------------------------------------------------------------------


def test_apply_disabled_returns_empty():
    result = op.apply_onepassword_secrets(enabled=False, env={"K": "op://V/I/F"})
    assert result.ok
    assert not result.applied


def test_apply_missing_binary_sets_error(monkeypatch):
    monkeypatch.setattr(op, "find_op", lambda binary_path="": None)
    result = op.apply_onepassword_secrets(
        enabled=True, env={"K": "op://V/I/F"}
    )
    assert not result.ok
    assert "op CLI" in result.error


def test_apply_sets_env(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: _ok("resolved-val"))
    monkeypatch.delenv("MY_OP_KEY", raising=False)

    result = op.apply_onepassword_secrets(
        enabled=True, env={"MY_OP_KEY": "op://V/I/F"}, cache_ttl_seconds=0,
    )
    assert result.ok
    assert result.applied == ["MY_OP_KEY"]
    assert os.environ["MY_OP_KEY"] == "resolved-val"


def test_apply_skips_before_fetch_when_not_overriding(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setenv("MY_OP_KEY", "from-env")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("from-1password")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    result = op.apply_onepassword_secrets(
        enabled=True, env={"MY_OP_KEY": "op://V/I/F"},
        override_existing=False, cache_ttl_seconds=0,
    )
    assert "MY_OP_KEY" in result.skipped
    assert os.environ["MY_OP_KEY"] == "from-env"
    assert calls["n"] == 0  # never even called op for a value we'd discard


def test_apply_never_overrides_token_var(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "original")
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        return _ok("malicious")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    result = op.apply_onepassword_secrets(
        enabled=True,
        env={"OP_SERVICE_ACCOUNT_TOKEN": "op://V/I/F"},
        override_existing=True, cache_ttl_seconds=0,
    )
    assert "OP_SERVICE_ACCOUNT_TOKEN" in result.skipped
    assert os.environ["OP_SERVICE_ACCOUNT_TOKEN"] == "original"
    assert calls["n"] == 0


def test_apply_never_raises_on_read_failure(monkeypatch, tmp_path):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setattr(op, "find_op", lambda binary_path="": fake_op)
    monkeypatch.setattr(op.subprocess, "run", lambda *a, **k: _err(1, "locked"))
    monkeypatch.delenv("MY_OP_KEY", raising=False)

    result = op.apply_onepassword_secrets(
        enabled=True, env={"MY_OP_KEY": "op://V/I/F"}, cache_ttl_seconds=0,
    )
    # Fail-open: warnings, nothing applied, no fatal error, no exception.
    assert result.ok
    assert result.applied == []
    assert result.warnings


def test_apply_no_valid_refs_is_noop(monkeypatch):
    # find_op must never be reached when there's nothing to fetch.
    monkeypatch.setattr(
        op, "find_op",
        lambda binary_path="": (_ for _ in ()).throw(AssertionError("should not resolve op")),
    )
    result = op.apply_onepassword_secrets(enabled=True, env={"BAD NAME": "op://V/I/F"})
    assert result.ok
    assert result.applied == []
    assert result.warnings  # the bad mapping warned
