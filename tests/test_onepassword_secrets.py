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
from agent.secret_sources.base import redact_provider_output  # noqa: E402


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


def test_provider_redaction_long_esc_nonmatch_is_bounded():
    """A malformed provider escape payload cannot cause regex blow-up."""
    text = "provider: \x1b" + ("x" * 50_000) + "; ordinary diagnostic"
    started = time.perf_counter()

    result = redact_provider_output(text, ("not-present",))

    assert time.perf_counter() - started < 1.0
    assert "ordinary diagnostic" in result


def test_provider_redaction_bridges_arbitrarily_long_malformed_esc_payload():
    """A long malformed escape cannot split a known credential past a bound."""
    secret = "ops.synthetic-long-esc-token-77468"
    text = f"provider rejected {secret[:8]}\x1b[" + ("A" * 512) + secret[8:]

    result = redact_provider_output(text, (secret,))

    assert secret not in result
    assert "<redacted>" in result


def test_provider_redaction_repeated_malformed_esc_fragments_are_bounded():
    """Repeated malformed fragments stay finite without regex backtracking."""
    secret = ("A" * 40) + "Z"
    text = "provider: " + (("\x1b" + ("A" * 65)) * 8)
    started = time.perf_counter()

    result = redact_provider_output(text, (secret,))

    assert time.perf_counter() - started < 1.0
    assert secret not in result


def test_provider_redaction_does_not_rescan_replacement_marker():
    """A one-character credential must not corrupt the replacement marker."""
    assert redact_provider_output("prefix c suffix", ("c",)) == (
        "prefix <redacted> suffix"
    )


def test_provider_redaction_preserves_private_use_diagnostic_text():
    """Provider-controlled private-use characters are not redaction markers."""
    secret = "ops.synthetic-private-use-77468"
    private_use = "".join(chr(codepoint) for codepoint in range(0xE000, 0xF900))

    result = redact_provider_output(
        f"before {private_use} after {secret}", (secret,)
    )

    assert result.count("<redacted>") == 1
    assert "\ue000" in result
    assert secret not in result


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


def test_op_child_env_preserves_auth_but_not_unrelated_credentials(
    monkeypatch, tmp_path
):
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    token = "ops.synthetic-op-env-77468"
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", token)
    monkeypatch.setenv("OP_SESSION_example", "session-value")
    network_values = {
        "HTTP_PROXY": "http://proxy.example:8080",
        "http_proxy": "http://proxy.lower.example:8080",
        "NO_PROXY": "localhost,.internal",
        "REQUESTS_CA_BUNDLE": "/etc/hermes/custom-ca.pem",
    }
    for key, value in network_values.items():
        monkeypatch.setenv(key, value)
    for key in ("OPENAI_API_KEY", "GH_TOKEN", "AUXILIARY_WEB_API_KEY"):
        monkeypatch.setenv(key, f"sentinel-{key}")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs["env"])
        return _ok("resolved-value")

    monkeypatch.setattr(op.subprocess, "run", fake_run)

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"},
        binary=fake_op,
        use_cache=False,
    )

    assert secrets == {"K": "resolved-value"}
    assert warnings == []
    assert captured["OP_SERVICE_ACCOUNT_TOKEN"] == token
    assert captured["OP_SESSION_example"] == "session-value"
    for key, value in network_values.items():
        assert captured[key] == value
    for key in ("OPENAI_API_KEY", "GH_TOKEN", "AUXILIARY_WEB_API_KEY"):
        assert key not in captured


def test_op_child_env_preserves_session_over_stale_default_token(monkeypatch):
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "stale-service-token")
    monkeypatch.setenv("OP_SESSION_example", "active-session")

    session_env = op._op_child_env("")
    explicit_env = op._op_child_env("configured-service-token")

    assert "OP_SERVICE_ACCOUNT_TOKEN" not in session_env
    assert session_env["OP_SESSION_example"] == "active-session"
    assert explicit_env["OP_SERVICE_ACCOUNT_TOKEN"] == "configured-service-token"






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


def test_fetch_read_failure_redacts_service_account_token(monkeypatch, tmp_path):
    token = "ops.synthetic-op-read-77468"
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", token)
    monkeypatch.setattr(
        op.subprocess,
        "run",
        lambda *a, **k: _err(1, f"provider rejected {token}; invalid token"),
    )

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"},
        binary=fake_op,
        use_cache=False,
    )

    assert secrets == {}
    assert len(warnings) == 1
    assert token not in warnings[0]
    assert "provider rejected <redacted>" in warnings[0]
    assert "invalid token" in warnings[0]


def test_fetch_read_failure_redacts_c1_csi_split_service_account_token(
    monkeypatch, tmp_path
):
    token = "ops.synthetic-op-c1-read-77468"
    split_token = f"{token[:8]}\x9b31m{token[8:]}\x9b0m"
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", token)
    monkeypatch.setattr(
        op.subprocess,
        "run",
        lambda *a, **k: _err(1, f"provider rejected {split_token}; invalid token"),
    )

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, binary=fake_op, use_cache=False
    )

    assert secrets == {}
    assert len(warnings) == 1
    assert token not in warnings[0]
    assert "provider rejected <redacted>" in warnings[0]
    assert "\x9b" not in warnings[0]


@pytest.mark.parametrize(
    "auth_env",
    ["OP_SERVICE_ACCOUNT_TOKEN", "OP_SESSION_demo", "OP_CONNECT_TOKEN"],
)
@pytest.mark.parametrize(
    "control", ["\x00", "\x09", "\x0d", "\x1b", "\x1b["]
)
def test_fetch_read_failure_redacts_every_op_auth_value_split_by_controls(
    monkeypatch, tmp_path, auth_env, control
):
    """Every auth value passed to op is redacted before read warnings surface."""
    auth = f"ops.synthetic-{auth_env.lower()}-77468"
    monkeypatch.setenv(auth_env, auth)
    if auth_env == "OP_CONNECT_TOKEN":
        monkeypatch.setenv("OP_CONNECT_HOST", "https://connect.example")
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    split_auth = f"{auth[:8]}{control}{auth[8:]}"
    monkeypatch.setattr(
        op.subprocess,
        "run",
        lambda *a, **k: _err(
            1,
            f"provider rejected {split_auth}; account=acct host=https://connect.example",
        ),
    )

    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, binary=fake_op, use_cache=False
    )

    assert secrets == {}
    assert len(warnings) == 1
    assert auth not in warnings[0]
    assert "provider rejected <redacted>" in warnings[0]
    assert "account=acct host=https://connect.example" in warnings[0]
    assert "\x1b" not in warnings[0]










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
