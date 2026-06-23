"""Tests for the MCP secret-propagation check in hermes doctor.

Exercises ``_check_mcp_secrets`` against a real temp HERMES_HOME layout
(no mocked I/O) per the AGENTS.md E2E validation rubric.

Resolution-order contract (must be tested):
  1. profile <profile_dir>/.env  — highest priority
  2. shared ~/.hermes/.env
  3. process environment         — lowest priority
"""
from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    profiles_dir: Path,
    name: str,
    mcp_servers: dict,
    env: dict | None = None,
) -> Path:
    """Create a minimal profile dir with config.yaml and optional .env."""
    pdir = profiles_dir / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "config.yaml").write_text(
        yaml.dump({"mcp_servers": mcp_servers}), encoding="utf-8"
    )
    if env:
        (pdir / ".env").write_text(
            "\n".join(f"{k}={v}" for k, v in env.items()), encoding="utf-8"
        )
    return pdir


def _make_shared_env(hermes_root: Path, env: dict) -> None:
    (hermes_root / ".env").write_text(
        "\n".join(f"{k}={v}" for k, v in env.items()), encoding="utf-8"
    )


def _run_check(
    hermes_root: Path,
    extra_env: dict | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Run _check_mcp_secrets and return (printed_lines, issues, manual_issues).

    Patches:
    - hermes_constants.get_default_hermes_root → hermes_root
    - builtins.print → capture list
    - optional extra os.environ entries (cleaned up after)
    """
    import hermes_cli.doctor as doctor_mod
    import hermes_constants

    issues: list[str] = []
    manual_issues: list[str] = []
    captured: list[str] = []

    original_root = hermes_constants.get_default_hermes_root
    cleanup_keys: list[str] = []

    if extra_env:
        for k, v in extra_env.items():
            os.environ[k] = v
            cleanup_keys.append(k)

    real_print = builtins.print

    def fake_print(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    builtins.print = fake_print
    try:
        hermes_constants.get_default_hermes_root = lambda: hermes_root
        doctor_mod._check_mcp_secrets(issues, manual_issues)
    finally:
        builtins.print = real_print
        hermes_constants.get_default_hermes_root = original_root
        for k in cleanup_keys:
            os.environ.pop(k, None)

    return captured, issues, manual_issues


# ---------------------------------------------------------------------------
# Test: all profiles clean → no warnings
# ---------------------------------------------------------------------------

def test_all_profiles_clean(tmp_path):
    """When every ${VAR} is resolved the check is silent (no manual_issues)."""
    profiles = tmp_path / "profiles"
    _make_profile(
        profiles,
        "alice",
        {"gh": {"command": "npx", "args": ["-y", "@github/mcp"], "env": {"TOKEN": "${GITHUB_TOKEN}"}}},
        env={"GITHUB_TOKEN": "ghp_test"},
    )
    _make_profile(profiles, "bob", {"fs": {"command": "npx", "args": ["fs-mcp"]}})

    lines, issues, manual_issues = _run_check(tmp_path)

    assert not manual_issues, f"Expected no manual_issues, got: {manual_issues}"
    assert not issues, f"Expected no issues, got: {issues}"
    ok_text = " ".join(lines)
    # The OK summary line should appear somewhere
    assert "resolved" in ok_text or "\u2713" in ok_text or "OK" in ok_text.upper()


# ---------------------------------------------------------------------------
# Test: one dirty profile → warning emitted
# ---------------------------------------------------------------------------

def test_one_profile_dirty_warns(tmp_path):
    """A profile with an unresolved ${VAR} yields a manual_issues entry."""
    profiles = tmp_path / "profiles"
    _make_profile(
        profiles,
        "carol",
        {"ghost": {"command": "node", "args": ["ghost-mcp"], "env": {"GHOST_ADMIN_API_KEY": "${GHOST_ADMIN_API_KEY}"}}},
    )
    os.environ.pop("GHOST_ADMIN_API_KEY", None)

    lines, issues, manual_issues = _run_check(tmp_path)

    assert manual_issues, "Expected at least one manual_issue"
    assert any("carol" in m for m in manual_issues)
    assert any("ghost" in m.lower() or "GHOST" in m for m in manual_issues)

    text = " ".join(lines)
    assert "GHOST_ADMIN_API_KEY" in text


# ---------------------------------------------------------------------------
# Test: var resolved via shared .env
# ---------------------------------------------------------------------------

def test_resolved_via_shared_env(tmp_path):
    """${VAR} set in ~/.hermes/.env is resolved — no warning."""
    profiles = tmp_path / "profiles"
    _make_profile(
        profiles,
        "dave",
        {"ghost": {"command": "node", "args": ["ghost-mcp"], "env": {"GHOST_ADMIN_API_KEY": "${GHOST_ADMIN_API_KEY}"}}},
    )
    _make_shared_env(tmp_path, {"GHOST_ADMIN_API_KEY": "abc123:def456"})
    os.environ.pop("GHOST_ADMIN_API_KEY", None)

    lines, issues, manual_issues = _run_check(tmp_path)

    assert not manual_issues, f"Expected no manual_issues, got: {manual_issues}"


# ---------------------------------------------------------------------------
# Test: var resolved via process environment
# ---------------------------------------------------------------------------

def test_resolved_via_process_env(tmp_path):
    """${VAR} present in os.environ is resolved — no warning."""
    profiles = tmp_path / "profiles"
    _make_profile(
        profiles,
        "eve",
        {"ghost": {"command": "node", "args": ["ghost-mcp"], "env": {"GHOST_ADMIN_API_KEY": "${GHOST_ADMIN_API_KEY}"}}},
    )
    os.environ.pop("GHOST_ADMIN_API_KEY", None)

    lines, issues, manual_issues = _run_check(
        tmp_path, extra_env={"GHOST_ADMIN_API_KEY": "xyz"}
    )

    assert not manual_issues, f"Expected no manual_issues, got: {manual_issues}"


# ---------------------------------------------------------------------------
# Test: profile .env wins over shared .env (resolution-order contract)
# ---------------------------------------------------------------------------

def test_profile_env_wins_over_shared(tmp_path):
    """Profile-local .env has higher priority than shared .env; both resolve."""
    profiles = tmp_path / "profiles"
    _make_profile(
        profiles,
        "frank",
        {"svc": {"command": "node", "args": ["svc-mcp"], "env": {"SVC_KEY": "${SVC_KEY}"}}},
        env={"SVC_KEY": "profile_value"},
    )
    _make_shared_env(tmp_path, {"SVC_KEY": "shared_value"})

    lines, issues, manual_issues = _run_check(tmp_path)

    assert not manual_issues


# ---------------------------------------------------------------------------
# Test: partial — only unresolved vars are reported
# ---------------------------------------------------------------------------

def test_partial_only_unresolved_reported(tmp_path):
    """When a server has some resolved and some unresolved vars, only the
    unresolved ones generate a warning."""
    profiles = tmp_path / "profiles"
    _make_profile(
        profiles,
        "grace",
        {
            "mixed": {
                "command": "node",
                "args": ["mixed-mcp"],
                "env": {
                    "RESOLVED_KEY": "${RESOLVED_KEY}",
                    "MISSING_KEY": "${MISSING_KEY}",
                },
            }
        },
        env={"RESOLVED_KEY": "present"},
    )
    os.environ.pop("MISSING_KEY", None)
    os.environ.pop("RESOLVED_KEY", None)

    lines, issues, manual_issues = _run_check(tmp_path)

    assert manual_issues, "Expected warning for MISSING_KEY"
    text = " ".join(lines)
    assert "MISSING_KEY" in text
    assert "RESOLVED_KEY" not in text


# ---------------------------------------------------------------------------
# Test: nested ${VAR} in command field
# ---------------------------------------------------------------------------

def test_nested_var_in_command_field(tmp_path):
    """${VAR} in the 'command' field (not just env block) is detected."""
    profiles = tmp_path / "profiles"
    _make_profile(
        profiles,
        "hank",
        {"custom": {"command": "${MCP_BIN_PATH}", "args": []}},
    )
    os.environ.pop("MCP_BIN_PATH", None)

    lines, issues, manual_issues = _run_check(tmp_path)

    assert manual_issues
    text = " ".join(lines)
    assert "MCP_BIN_PATH" in text


# ---------------------------------------------------------------------------
# Test: nested ${VAR} inside args list
# ---------------------------------------------------------------------------

def test_nested_var_in_args(tmp_path):
    """${VAR} appearing inside the args list is detected."""
    profiles = tmp_path / "profiles"
    _make_profile(
        profiles,
        "ivan",
        {"custom": {"command": "node", "args": ["--token", "${MY_TOKEN}"]}},
    )
    os.environ.pop("MY_TOKEN", None)

    lines, issues, manual_issues = _run_check(tmp_path)

    assert manual_issues
    text = " ".join(lines)
    assert "MY_TOKEN" in text


# ---------------------------------------------------------------------------
# Test: no profiles dir → graceful skip
# ---------------------------------------------------------------------------

def test_no_profiles_dir(tmp_path):
    """When the profiles dir doesn't exist, the check skips gracefully."""
    # tmp_path has no profiles/ subdirectory
    lines, issues, manual_issues = _run_check(tmp_path)
    assert not manual_issues
    assert not issues
