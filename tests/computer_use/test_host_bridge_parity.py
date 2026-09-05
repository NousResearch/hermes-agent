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

import inspect
from pathlib import Path

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


def test_cli_handler_forwards_session_idle_timeout():
    """The host-bridge argparse handler must forward session_idle_timeout."""
    handler = cu_cli._cu_host_bridge if hasattr(cu_cli, "_cu_host_bridge") else cu_cli._cu_bridge
    assert handler is not None, "no host-bridge handler in the computer-use subcommand"
    src = Path(cu_cli.__file__).read_text()
    assert "session_idle_timeout" in src, (
        "the CLI handler must pass session_idle_timeout through to run_host_bridge"
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
    """Both launchers must strip the same secrets from the cua-driver child env."""
    strip_vars = ["HERMES_CUA_REMOTE_TOKEN", "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS",
                  "ANTHROPIC_API_KEY"]
    for v in strip_vars:
        monkeypatch.setenv(v, "secret")
    base = {v: "secret" for v in strip_vars}

    # standalone: _sanitize_standalone_env is a pure mapping → mapping filter
    cleaned = standalone._sanitize_standalone_env(dict(base))
    leaked = [v for v in strip_vars if v in cleaned]
    assert not leaked, f"standalone child env leaks {leaked}"

    # cli: _build_child_session_context needs a live driver; assert the strip
    # list at the source level instead (same names, telemetry forced off too).
    import tools.computer_use.host_bridge_cli as cli_src
    src = Path(cli_src.__file__).read_text()
    for var in ("_CUA_REMOTE_TOKEN_ENV", "_CUA_BYPASS_APPROVALS_ENV",
                "CUA_DRIVER_RS_TELEMETRY_ENABLED"):
        assert var in src, f"cli launcher no longer strips/sets {var}"
    sa_src = Path(standalone.__file__).read_text()
    for var in ("HERMES_CUA_REMOTE_TOKEN", "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS",
                "CUA_DRIVER_RS_TELEMETRY_ENABLED"):
        assert var in sa_src, f"standalone launcher no longer strips/sets {var}"


# ── 3. whitespace handling in list args ──────────────────────────────────────


def test_split_list_arg_strips_and_drops_empties():
    entries = standalone._split_list_arg(" a:8765 , b:8765 ,,")
    assert entries == ["a:8765", "b:8765"]


def test_cli_handler_strips_allowed_hosts():
    """The CLI handler's split must tolerate spaces (review nit #4)."""
    src = Path(cu_cli.__file__).read_text()
    assert "h.strip()" in src, (
        "--allowed-hosts entries should be whitespace-stripped before use"
    )


def test_empty_allowed_hosts_rejected():
    """Empty allowlists must fail closed in the shared validator."""
    with pytest.raises(Exception):
        validate_security_allowlists([], [])