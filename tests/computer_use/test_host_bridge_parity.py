"""Parity and plumbing tests for the host-bridge launchers.

Covers the AI-review findings on PR #103653:
1. `--session-idle-timeout` must actually reach `run_host_bridge` (dead flag).
2. The launchers' duplicated policy gates (plaintext-bind TLS gate) must stay
   in sync between `host_bridge_cli.py` and `host_bridge_standalone.py`, so a
   future fix to one cannot silently desync the other.
3. `--allowed-hosts`/`--allowed-origins` entries are whitespace-stripped and
   empty entries (trailing commas) are dropped.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect

import pytest

from hermes_cli.subcommands import computer_use as cu_cli
from tools.computer_use import host_bridge_cli
from tools.computer_use import host_bridge_standalone as standalone
from tools.computer_use.host_validation import validate_security_allowlists


# ── 1. --session-idle-timeout plumbing ──────────────────────────────────────


def test_run_host_bridge_accepts_session_idle_timeout():
    """run_host_bridge must accept session_idle_timeout (CLI flag forwards it)."""
    sig = inspect.signature(host_bridge_cli.run_host_bridge)
    assert "session_idle_timeout" in sig.parameters, (
        "--session-idle-timeout is registered on the host-bridge CLI parser but "
        "run_host_bridge cannot receive it — the flag would be dead"
    )
    assert sig.parameters["session_idle_timeout"].default == 1800


def test_cli_handler_forwards_session_idle_timeout(monkeypatch):
    """The host-bridge argparse handler must forward session_idle_timeout.

    Behavior test: invoke the parsed CLI handler with a nondefault
    --session-idle-timeout and capture what ``run_host_bridge`` actually
    receives, so the flag can't silently go dead without the test noticing.
    """
    handler = cu_cli._cu_host_bridge if hasattr(cu_cli, "_cu_host_bridge") else cu_cli._cu_bridge
    assert handler is not None, "no host-bridge handler in the computer-use subcommand"

    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    import argparse
    monkeypatch.setattr(host_bridge_cli, "run_host_bridge", _fake_run)
    # _cu_host_bridge reads the token from the env; populate it so the env
    # gate passes and we reach the run_host_bridge call.
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "t" * 40)
    # Skip the interactive/bind guards so the handler reaches the patched
    # run_host_bridge.
    monkeypatch.setattr(host_bridge_cli, "_ensure_interactive_session", lambda: None)
    monkeypatch.setattr(host_bridge_cli, "_validate_standard_permission_environment", lambda: None)
    monkeypatch.setattr(host_bridge_cli, "_ensure_bind_security", lambda bind: None)
    monkeypatch.setattr(host_bridge_cli, "_build_child_session_context",
                        lambda: contextlib.nullcontext())
    monkeypatch.setattr(host_bridge_cli, "_serve_app", lambda app, bind, port: None)

    args = argparse.Namespace(
        port=8765, bind="127.0.0.1",
        allowed_hosts="localhost:8765",
        allowed_origins="",
        session_idle_timeout=600,
    )
    handler(args)

    assert captured.get("session_idle_timeout") == 600, (
        f"--session-idle-timeout did not reach run_host_bridge (got {captured})"
    )


# ── 2. launcher policy parity ───────────────────────────────────────────────


@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0", "::1"])
@pytest.mark.parametrize("allow_plaintext", [None, "0", "1"])
def test_bind_gate_parity_between_launchers(bind, allow_plaintext, monkeypatch):
    """Both launchers must accept/refuse the same (bind, allow-plaintext) matrix."""
    if allow_plaintext is None:
        monkeypatch.delenv("HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT", raising=False)
    else:
        monkeypatch.setenv("HERMES_CUA_BRIDGE_ALLOW_PLAINTEXT", allow_plaintext)

    results = {}
    for label, gate in (("cli", host_bridge_cli._ensure_bind_security),
                        ("standalone", standalone._ensure_bind_security)):
        try:
            gate(bind)
            results[label] = "accept"
        except RuntimeError:
            results[label] = "refuse"

    assert results["cli"] == results["standalone"], (
        f"launcher bind-gate desync for bind={bind!r} allow_plaintext={allow_plaintext!r}: {results}"
    )
    # And the expected outcome: loopback or acknowledged plaintext accepted,
    # everything else refused.
    expected = "accept" if (bind in ("127.0.0.1", "::1") or allow_plaintext == "1") else "refuse"
    assert results["cli"] == expected, f"unexpected gate outcome for {bind!r}/{allow_plaintext!r}"


def test_loopback_bind_sets_match():
    """The two launchers must agree on what counts as a loopback bind."""
    assert host_bridge_cli._LOOPBACK_BINDS == standalone._LOOPBACK_BINDS


def test_child_env_sanitizer_parity(monkeypatch):
    """Both launchers must strip the same secrets from the cua-driver child env.

    Behavior test: the standalone path is covered by the direct
    ``_sanitize_standalone_env`` mapping assertion below.  For the CLI path we
    invoke ``host_bridge_cli._build_child_session_context`` and capture the
    env it passes to ``_cua_driver_session_context``, asserting the same
    secret names are stripped and the explicit assignments are present —
    so a no-op handler or a renamed variable would still be caught.
    """
    strip_vars = ["HERMES_CUA_REMOTE_TOKEN", "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS",
                  "ANTHROPIC_API_KEY"]
    for v in strip_vars:
        monkeypatch.setenv(v, "secret")
    base = {v: "secret" for v in strip_vars}

    # standalone: _sanitize_standalone_env is a pure mapping → mapping filter
    cleaned = standalone._sanitize_standalone_env(dict(base))
    leaked = [v for v in strip_vars if v in cleaned]
    assert not leaked, f"standalone child env leaks {leaked}"

    # cli: capture the env that _build_child_session_context actually builds,
    # by intercepting the inner _cua_driver_session_context the same way the
    # standalone driver-env test does.
    import tools.computer_use.cua_backend_driver as cua_backend_driver
    import tools.computer_use.host_bridge_cli as cli_mod
    # The builder resolves the driver from PATH and asks it for its MCP
    # invocation via a subprocess; CI runners have no cua-driver installed, so
    # pin the resolution (the established pattern across the computer-use
    # tests). _resolve_mcp_invocation swallows spawn failures and falls back to
    # (driver_cmd, ["mcp"]), so no real process is spawned here either way.
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda override=None: "cua-driver")
    real_ctx = cli_mod._cua_driver_session_context
    captured: dict[str, str] = {}

    def _capture_ctx(*, command, args, env):
        captured.update(env)
        return real_ctx(command=command, args=args, env=env)

    monkeypatch.setattr(cli_mod, "_cua_driver_session_context", _capture_ctx)
    cli_mod._build_child_session_context()

    # The same secrets stripped by the standalone sanitizer must not survive
    # in the CLI builder's child env either.
    cli_leaked = [v for v in strip_vars if v in captured]
    assert not cli_leaked, f"cli child env leaks {cli_leaked}"
    # The explicit assignments the CLI builder sets must be present.
    assert captured.get("CUA_DRIVER_PERMISSION_MODE") == "standard"
    assert captured.get("CUA_DRIVER_RS_TELEMETRY_ENABLED") == "0"


def test_standalone_driver_child_env_is_sanitized(monkeypatch):
    """The ACTUAL driver child env builder must not leak parent secrets.

    This guards ``_build_child_session_context`` — the function that builds the
    env handed to the cua-driver stdio child — not just ``_sanitize_standalone_env``
    (which is used for the X stack).  A canary secret in ``os.environ`` must
    never survive into the child env, while the vars the driver genuinely needs
    (DISPLAY, PATH, HOME) and the ones the builder sets explicitly
    (CUA_DRIVER_PERMISSION_MODE, CUA_DRIVER_RS_TELEMETRY_ENABLED) must be present.
    """
    # Seed a realistic parent env with canary secrets + required vars.
    canary_secrets = {
        "ANTHROPIC_API_KEY": "sk-ant-canary",
        "OPENAI_API_KEY": "sk-canary",
        "HERMES_CUA_REMOTE_TOKEN": "t" * 64,
        "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS": "1",
        "AWS_SECRET_ACCESS_KEY": "aws-canary",
    }
    needed = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp/fake-home",
        "DISPLAY": ":99",
        "LANG": "C.UTF-8",
    }
    for k, v in {**canary_secrets, **needed}.items():
        monkeypatch.setenv(k, v)

    # The builder returns an async context manager; we only need the env it
    # captured, so intercept the inner _cua_driver_session_context.
    captured: dict[str, str] = {}
    real_ctx = standalone._cua_driver_session_context

    def _capture_ctx(*, command, args, env):
        captured.update(env)
        return real_ctx(command=command, args=args, env=env)

    monkeypatch.setattr(standalone, "_cua_driver_session_context", _capture_ctx)
    standalone._build_child_session_context("/fake/cua-driver")

    # Secrets from the parent env must be absent.
    leaked = [k for k in canary_secrets if k in captured]
    assert not leaked, f"driver child env leaks parent secrets: {leaked}"

    # Required allowlist vars the driver genuinely needs must survive.
    for k in ("PATH", "HOME", "DISPLAY", "LANG"):
        assert k in captured, f"driver child env dropped required {k}"

    # Explicit CUA_* assignments from the builder must be present and correct.
    assert captured.get("CUA_DRIVER_PERMISSION_MODE") == "standard"
    assert captured.get("CUA_DRIVER_RS_TELEMETRY_ENABLED") == "0"

    # Non-allowlisted, non-secret parent vars must also be dropped (the whole
    # point of building from the allowlist, not raw os.environ).
    assert "AWS_SECRET_ACCESS_KEY" not in captured


# ── 3. whitespace handling in list args ──────────────────────────────────────


def test_split_list_arg_strips_and_drops_empties():
    entries = standalone._split_list_arg(" a:8765 , b:8765 ,,")
    assert entries == ["a:8765", "b:8765"]


def test_cli_handler_strips_allowed_hosts(monkeypatch):
    """The CLI handler must whitespace-strip --allowed-hosts entries.

    Behavior test: invoke the parsed CLI handler with whitespace-padded host
    entries and capture what ``run_host_bridge`` receives — the normalized
    values must have no surrounding whitespace and no empty entries.
    """
    handler = cu_cli._cu_host_bridge if hasattr(cu_cli, "_cu_host_bridge") else cu_cli._cu_bridge
    assert handler is not None, "no host-bridge handler in the computer-use subcommand"

    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(host_bridge_cli, "run_host_bridge", _fake_run)
    monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", "t" * 40)
    monkeypatch.setattr(host_bridge_cli, "_ensure_interactive_session", lambda: None)
    monkeypatch.setattr(host_bridge_cli, "_validate_standard_permission_environment", lambda: None)
    monkeypatch.setattr(host_bridge_cli, "_ensure_bind_security", lambda bind: None)
    monkeypatch.setattr(host_bridge_cli, "_build_child_session_context",
                        lambda: contextlib.nullcontext())
    monkeypatch.setattr(host_bridge_cli, "_serve_app", lambda app, bind, port: None)

    args = argparse.Namespace(
        port=8765, bind="127.0.0.1",
        allowed_hosts="  localhost:8765 ,  127.0.0.1:8765  ,,",
        allowed_origins="",
        session_idle_timeout=1800,
    )
    handler(args)

    assert captured["allowed_hosts"] == ["localhost:8765", "127.0.0.1:8765"], (
        f"--allowed-hosts entries were not whitespace-stripped/dropped (got {captured.get('allowed_hosts')})"
    )


def test_empty_allowed_hosts_rejected():
    """Empty allowlists must fail closed in the shared validator."""
    with pytest.raises(Exception):
        validate_security_allowlists([], [])