"""Tests for ``agent.secret_sources.protonpass.session``.

Session establishment (login -> info, with a logout/relogin recovery), the
minimal/scrubbed child env (no inherited secrets, A3), the token-fingerprinted
isolated session dir (A7), the ANSI/CSI stream cleaner, and token redaction.
The token is NEVER logged, stored, or surfaced in a warning/raised message.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests._protonpass_helpers import (  # noqa: F401
    _fail,
    _ok,
    _reset_caches,
    _session_runner,
    hermes_home,
    pp_session,
)


# ---------------------------------------------------------------------------
# Session establishment helpers (login → info; logout+retry; redaction)
# ---------------------------------------------------------------------------


def test_session_login_then_info_success(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    calls = []
    _session_runner(monkeypatch, {"login": [_ok()], "info": [_ok()]}, calls=calls)

    warnings = pp_session._establish_session("svc-token", binary)
    assert warnings == []
    verbs = [c[0][1] for c in calls]
    assert verbs == ["login", "info"]
    # Child env carries the token + isolated session dir, never logs it.
    env = calls[0][1]
    assert env["PROTON_PASS_PERSONAL_ACCESS_TOKEN"] == "svc-token"
    # A7: the session dir is suffixed with the token fingerprint, so the path is
    # ``.../protonpass-session-<fp>`` (NOT the bare base name).
    assert "protonpass-session" in env["PROTON_PASS_SESSION_DIR"]
    # The raw token must never leak into the session-dir path.
    assert "svc-token" not in env["PROTON_PASS_SESSION_DIR"]


def test_session_recovery_logout_retry(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    calls = []
    # First login fails; after logout, second login + info succeed.
    _session_runner(
        monkeypatch,
        {
            "login": [_fail("auth bad"), _ok()],
            "info": [_ok()],
            "logout": [_ok()],
        },
        calls=calls,
    )

    warnings = pp_session._establish_session("svc-token", binary)
    assert any("recovered" in w for w in warnings)
    verbs = [c[0][1] for c in calls]
    assert "logout" in verbs


def test_session_failure_redacts_token(hermes_home, monkeypatch, tmp_path):
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    token = "super-secret-token-value"
    _session_runner(
        monkeypatch,
        {"login": [_fail(token), _fail(token)], "logout": [_ok()]},
    )

    with pytest.raises(RuntimeError) as exc:
        pp_session._establish_session(token, binary)
    # The token value must never appear in the raised message.
    assert token not in str(exc.value)


def test_session_failure_surfaces_redacted_stderr(hermes_home, monkeypatch, tmp_path):
    """A8: the final RuntimeError carries a redacted, ANSI-stripped login stderr
    (the token is scrubbed) to aid debugging."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    token = "tok-value"
    _session_runner(
        monkeypatch,
        {
            "login": [_fail("\x1b[31minvalid credentials\x1b[0m"),
                      _fail("\x1b[31minvalid credentials\x1b[0m")],
            "logout": [_ok()],
        },
    )

    with pytest.raises(RuntimeError) as exc:
        pp_session._establish_session(token, binary)
    msg = str(exc.value)
    assert "invalid credentials" in msg  # surfaced
    assert "\x1b[" not in msg  # ANSI stripped
    assert token not in msg  # token never present


def test_session_login_failure_detail_uses_stderr_only(hermes_home, monkeypatch, tmp_path):
    """C7: ``_try_login_and_verify`` surfaces ONLY login stderr on failure (the
    ``or login.stdout`` fallback was dropped for consistency with the
    stderr-only secret-command rule)."""
    binary = tmp_path / "pass-cli"
    binary.write_text("", encoding="utf-8")
    # login fails with content ONLY on stdout; stderr is empty.
    login_fail = __import__("unittest").mock.Mock(
        returncode=1, stdout="DETAIL-ON-STDOUT", stderr=""
    )

    def fake_run(cmd, env):
        verb = cmd[1] if len(cmd) > 1 else cmd[-1]
        if verb == "login":
            return login_fail
        return _ok()

    monkeypatch.setattr(pp_session, "_run_pass_cli", fake_run)

    ok, detail = pp_session._try_login_and_verify(binary, {"E": "1"})
    assert ok is False
    # stdout is NOT used as the detail fallback.
    assert detail == ""
    assert "DETAIL-ON-STDOUT" not in detail


def test_run_pass_cli_uses_utf8_and_errors_replace(monkeypatch):
    """C7: ``_run_pass_cli`` decodes pass-cli output as UTF-8 (not the locale
    codepage) and passes ``errors='replace'`` so invalid UTF-8 can't raise."""
    captured = {}

    def fake_run(cmd, *, env, capture_output, text, encoding, errors, timeout):
        captured["errors"] = errors
        captured["text"] = text
        captured["encoding"] = encoding
        return __import__("unittest").mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pp_session.subprocess, "run", fake_run)
    pp_session._run_pass_cli(["pass-cli", "info"], {"E": "1"})
    assert captured["errors"] == "replace"
    assert captured["text"] is True
    assert captured["encoding"] == "utf-8"


# ---------------------------------------------------------------------------
# minimal/scrubbed child env (A3): no inherited secrets
# ---------------------------------------------------------------------------


def test_minimal_env_has_no_token_or_secrets(monkeypatch):
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", "leak-me")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = pp_session._minimal_env()
    assert "PROTON_PASS_PERSONAL_ACCESS_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert env.get("NO_COLOR") == "1"
    assert env.get("PATH") == "/usr/bin"  # ambient PATH is carried


def test_child_env_adds_only_protonpass_vars(hermes_home, monkeypatch):
    monkeypatch.delenv("PROTON_PASS_KEY_PROVIDER", raising=False)
    monkeypatch.delenv("PROTON_PASS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
    env = pp_session._child_env("svc-token")
    assert env["PROTON_PASS_PERSONAL_ACCESS_TOKEN"] == "svc-token"
    assert env["PROTON_PASS_SESSION_DIR"]
    assert env["PROTON_PASS_AGENT_REASON"]
    # Headless safety: default to the filesystem key provider (no OS keyring on
    # servers/containers) and suppress telemetry.
    assert env["PROTON_PASS_KEY_PROVIDER"] == "fs"
    assert env["PROTON_PASS_DISABLE_TELEMETRY"] == "1"
    # No other secret made it into the child env.
    assert "OPENAI_API_KEY" not in env


def test_child_env_key_provider_override_is_honored(hermes_home, monkeypatch):
    """A user who explicitly wants the OS keyring can override the fs default."""
    monkeypatch.setenv("PROTON_PASS_KEY_PROVIDER", "keyring")
    env = pp_session._child_env("svc-token")
    assert env["PROTON_PASS_KEY_PROVIDER"] == "keyring"


def test_child_env_empty_override_falls_back_to_secure_default(hermes_home, monkeypatch):
    """An empty/whitespace override must NOT silently disable the secure default."""
    monkeypatch.setenv("PROTON_PASS_KEY_PROVIDER", "   ")
    monkeypatch.setenv("PROTON_PASS_DISABLE_TELEMETRY", "")
    env = pp_session._child_env("svc-token")
    assert env["PROTON_PASS_KEY_PROVIDER"] == "fs"
    assert env["PROTON_PASS_DISABLE_TELEMETRY"] == "1"


def test_child_env_rejects_symlinked_session_dir(hermes_home, tmp_path):
    """A7/hardening (real filesystem): a session dir that is actually a symlink
    is rejected BEFORE it is chmodded or PROTON_PASS_SESSION_DIR is set — we do
    not follow it to the (attacker-chosen) target."""
    session_dir = pp_session._session_dir("svc-token")
    session_dir.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "evil-target"
    target.mkdir()
    target_mode_before = target.stat().st_mode
    session_dir.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        pp_session._child_env("svc-token")
    # The check fires before chmod, so the symlink target is left untouched.
    assert target.stat().st_mode == target_mode_before


# ---------------------------------------------------------------------------
# A7: session-dir creation/chmod failure -> skip Proton Pass (RuntimeError)
# ---------------------------------------------------------------------------


def test_child_env_raises_when_session_dir_cannot_be_secured(hermes_home, monkeypatch):
    """A7: if the isolated session dir can't be created/locked to 0o700, we
    RAISE (don't continue with unverifiable session storage)."""
    def boom_chmod(path, mode):
        raise OSError("cannot chmod")

    monkeypatch.setattr(pp_session.os, "chmod", boom_chmod)

    with pytest.raises(RuntimeError, match="session directory"):
        pp_session._child_env("svc-token")


def test_session_dir_is_token_fingerprinted(hermes_home):
    """A7 (nice): two different tokens map to different session dirs.

    C7: the token-fingerprint suffix is now UNCONDITIONAL (the dead bare-path
    branch was removed), so every session dir carries a ``protonpass-session-<fp>``
    suffix and never the raw token."""
    a = pp_session._session_dir("token-A")
    b = pp_session._session_dir("token-B")
    assert a != b
    # The fingerprint, never the token, appears in the path.
    assert "token-A" not in str(a)
    assert pp_session._token_fingerprint("token-A") in str(a)
    assert a.name == f"protonpass-session-{pp_session._token_fingerprint('token-A')}"


def test_session_dir_suffix_unconditional(hermes_home):
    """C7: even a bare ``_session_dir()`` (empty token, not used by any real
    caller) carries the fingerprint suffix — the unsuffixed base path branch is
    gone, so the 'session isolation' claim always holds."""
    bare = pp_session._session_dir()
    assert bare.name == (
        f"protonpass-session-{pp_session._token_fingerprint('')}"
    )


def test_session_dir_created_0700(hermes_home):
    """A7: the per-token session dir is created and locked to 0o700."""
    env = pp_session._child_env("svc-token")
    session_dir = Path(env["PROTON_PASS_SESSION_DIR"])
    assert session_dir.exists()
    assert (os.stat(session_dir).st_mode & 0o777) == 0o700


# ---------------------------------------------------------------------------
# token fingerprint + stream cleaning + redaction
# ---------------------------------------------------------------------------


def test_token_fingerprint_is_stable_and_not_the_token():
    fp = pp_session._token_fingerprint("svc-token")
    assert fp == pp_session._token_fingerprint("svc-token")  # stable
    assert "svc-token" not in fp
    assert len(fp) == 16


def test_clean_stream_full_csi_strip():
    # A full CSI sequence (colour codes) must be stripped, not just the ESC byte.
    raw = "\x1b[31merror\x1b[0m: \x1b[1;33mboom\x1b[0m"
    assert pp_session._clean_stream(raw) == "error: boom"


def test_clean_stream_handles_empty():
    assert pp_session._clean_stream("") == ""
    assert pp_session._clean_stream(None) == ""


def test_redact_token_replaces_all_occurrences():
    out = pp_session._redact_token("a tok b tok c", "tok")
    assert "tok" not in out.replace("***REDACTED***", "")
    assert pp_session._redact_token("anything", "") == "anything"
