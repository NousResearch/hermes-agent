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
ROOT = Path(__file__).resolve().parents[2]
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
    """Multi-reference fetch resolves through one batched `op inject` call."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    values = {
        "op://Private/OpenAI/api key": "sk-abc",
        "op://Private/Anthropic/credential": "sk-ant-xyz",
    }
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        # argv list, never shell=True; one invocation carries the whole template.
        calls["n"] += 1
        assert "inject" in cmd
        template = Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8")
        out = Path(cmd[cmd.index("-o") + 1])
        for ref, value in values.items():
            template = template.replace(f"{{{{ {ref} }}}}", value)
        out.write_text(template, encoding="utf-8")
        return _ok("")

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
    # The point of batching: two references cost ONE `op` invocation, not two.
    assert calls["n"] == 1


def test_single_reference_uses_op_read(monkeypatch, tmp_path):
    """One reference gains nothing from a template, so it takes the direct read path."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")

    def fake_run(cmd, **kwargs):
        assert "inject" not in cmd
        assert "--" in cmd
        assert cmd[cmd.index("--") + 1] == "op://Private/OpenAI/api key"
        return _ok("sk-abc\n")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"OPENAI_API_KEY": "op://Private/OpenAI/api key"},
        binary=fake_op,
        use_cache=False,
    )
    assert secrets == {"OPENAI_API_KEY": "sk-abc"}
    assert warnings == []


def test_batch_failure_falls_back_to_per_reference_reads(monkeypatch, tmp_path):
    """`op inject` is all-or-nothing: one bad reference must not lose the good ones."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    values = {"op://V/I/good": "good-value"}
    seen = {"inject": False}

    def fake_run(cmd, **kwargs):
        if "inject" in cmd:  # batch fails wholesale, as op does for an unknown field
            seen["inject"] = True
            return _err(1, "[ERROR] item 'V/I' does not have a field 'bad'")
        ref = cmd[cmd.index("--") + 1]
        if ref not in values:
            return _err(1, "[ERROR] item 'V/I' does not have a field 'bad'")
        return _ok(values[ref])

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"GOOD": "op://V/I/good", "BAD": "op://V/I/bad"},
        binary=fake_op,
        use_cache=False,
    )
    # The batch was attempted, and its wholesale failure did not cost the good reference.
    assert seen["inject"]
    assert secrets == {"GOOD": "good-value"}
    assert len(warnings) == 1
    assert "op://V/I/bad" in warnings[0]


def test_inject_values_survive_special_characters(monkeypatch, tmp_path):
    """Sentinel framing keeps '=', '#', quotes and newlines inside a value intact."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    values = {
        "op://V/I/a": "pa=ss#word 'quoted' \"too\"",
        "op://V/I/b": "-----BEGIN KEY-----\nline2\n-----END KEY-----",
    }

    def fake_run(cmd, **kwargs):
        template = Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8")
        out = Path(cmd[cmd.index("-o") + 1])
        for ref, value in values.items():
            template = template.replace(f"{{{{ {ref} }}}}", value)
        out.write_text(template, encoding="utf-8")
        return _ok("")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    secrets, _ = op.fetch_onepassword_secrets(
        references={"A": "op://V/I/a", "B": "op://V/I/b"}, binary=fake_op, use_cache=False
    )
    assert secrets == {"A": values["op://V/I/a"], "B": values["op://V/I/b"]}


def test_blank_inject_value_falls_back_instead_of_applying_empty(monkeypatch, tmp_path):
    """A blank value must be unusable through BOTH paths, not just `op read`.

    `_run_op_read` rejects `not value.strip()` so an exit-0 empty value never clobbers a good
    credential with "". The batch path has to agree: if it accepted whitespace, the same
    reference would resolve to a usable secret or not depending only on how many references
    happened to be mapped alongside it.
    """
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    reads: list[str] = []

    def fake_run(cmd, **kwargs):
        if "inject" in cmd:  # batch "succeeds" but hands back a whitespace-only value for B
            template = Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8")
            out = Path(cmd[cmd.index("-o") + 1])
            template = template.replace("{{ op://V/I/a }}", "good-value")
            template = template.replace("{{ op://V/I/b }}", "   ")
            out.write_text(template, encoding="utf-8")
            return _ok("")
        ref = cmd[cmd.index("--") + 1]
        reads.append(ref)
        if ref == "op://V/I/b":  # op read applies the same rule and refuses it
            return _ok("   ")
        return _ok("good-value")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"A": "op://V/I/a", "B": "op://V/I/b"}, binary=fake_op, use_cache=False
    )
    # The blank value was not accepted as authoritative; the batch was rejected wholesale and
    # every reference was re-read individually.
    assert reads == ["op://V/I/a", "op://V/I/b"]
    assert secrets == {"A": "good-value"}
    assert "B" not in secrets
    assert len(warnings) == 1 and "op://V/I/b" in warnings[0]






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




def test_op_child_env_forwards_config_directory(monkeypatch):
    """The op child must retain an explicit 1Password config location."""
    monkeypatch.setenv("OP_CONFIG_DIR", "/tmp/op-config")
    monkeypatch.setenv("UNRELATED_PROVIDER_TOKEN", "must-not-leak")

    env = op._op_child_env("")

    assert env["OP_CONFIG_DIR"] == "/tmp/op-config"
    assert "UNRELATED_PROVIDER_TOKEN" not in env


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




