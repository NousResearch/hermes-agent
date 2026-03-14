#!/usr/bin/env python3
"""Diagnostic script for debugging Canvas skill environment variable issues.

Run this to see exactly why the canvas skill thinks your credentials are missing.

Usage:
    python scripts/debug_canvas_env.py
    # or with verbose logging from skills_tool:
    HERMES_DEBUG_SKILLS=1 python scripts/debug_canvas_env.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _redact(value: str) -> str:
    """Show first 4 / last 4 chars of a value, mask the middle."""
    if not value:
        return "(empty)"
    if len(value) <= 10:
        return f"{value[:2]}***"
    return f"{value[:4]}...{value[-4:]}"


def main():
    print("=" * 60)
    print("Canvas Skill Environment Variable Diagnostics")
    print("=" * 60)

    # ── 1. Check os.environ ──
    print("\n[1] Checking os.environ (process environment):")
    for var in ("CANVAS_API_TOKEN", "CANVAS_BASE_URL"):
        val = os.environ.get(var)
        if val:
            print(f"    {var} = {_redact(val)}  ✓")
        else:
            print(f"    {var} = (NOT SET)  ✗")

    # ── 2. Check ~/.hermes/.env file ──
    print("\n[2] Checking ~/.hermes/.env file:")
    from hermes_cli.config import get_env_path, load_env

    env_path = get_env_path()
    print(f"    Path: {env_path}")
    print(f"    Exists: {env_path.exists()}")

    if env_path.exists():
        # Show raw content for canvas-related lines
        print(f"    Canvas-related lines (raw):")
        with open(env_path) as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if "CANVAS" in stripped.upper():
                    # Redact the value part
                    if "=" in stripped and not stripped.startswith("#"):
                        key, _, val = stripped.partition("=")
                        clean_val = val.strip().strip('"').strip("'")
                        print(f"      L{i}: {key}={_redact(clean_val)}")
                    else:
                        print(f"      L{i}: {stripped}")
        if not any("CANVAS" in line.upper() for line in open(env_path)):
            print("      (no lines containing 'CANVAS' found)")

        # Show what load_env() actually parses
        env_snapshot = load_env()
        print(f"\n    Parsed by load_env():")
        for var in ("CANVAS_API_TOKEN", "CANVAS_BASE_URL"):
            val = env_snapshot.get(var)
            if val is not None:
                print(f"      {var} = {_redact(val)}  (truthy={bool(val)})")
            else:
                print(f"      {var} = (NOT IN PARSED RESULT)  ✗")
    else:
        print("    ✗ File does not exist! Canvas vars cannot be persisted.")
        print(f"    Create it: mkdir -p {env_path.parent} && touch {env_path}")

    # ── 3. Check python-dotenv loading ──
    print("\n[3] Checking python-dotenv behavior:")
    try:
        from dotenv import dotenv_values
        if env_path.exists():
            dotenv_parsed = dotenv_values(str(env_path))
            for var in ("CANVAS_API_TOKEN", "CANVAS_BASE_URL"):
                val = dotenv_parsed.get(var)
                if val is not None:
                    print(f"    {var} = {_redact(val)}  (truthy={bool(val)})")
                else:
                    print(f"    {var} = (NOT IN DOTENV RESULT)  ✗")
        else:
            print("    (skipped — no .env file)")
    except ImportError:
        print("    (python-dotenv not installed)")

    # ── 4. Check skill validation logic ──
    print("\n[4] Simulating skill_view validation logic:")
    try:
        from tools.skills_tool import (
            _is_env_var_persisted,
            _get_terminal_backend_name,
            _is_gateway_surface,
            _REMOTE_ENV_BACKENDS,
            _secret_capture_callback,
        )
    except ImportError as e:
        # Direct import may fail due to transitive deps; reimplement inline
        print(f"    (Could not import skills_tool: {e})")
        print("    Reimplementing checks inline...")

        _REMOTE_ENV_BACKENDS = frozenset({"docker", "singularity", "modal", "ssh", "daytona"})

        def _get_terminal_backend_name():
            return str(os.getenv("TERMINAL_ENV", "local")).strip().lower() or "local"

        def _is_gateway_surface():
            return bool(os.getenv("HERMES_GATEWAY_SESSION")) or bool(os.getenv("HERMES_SESSION_PLATFORM"))

        def _is_env_var_persisted(var_name, env_snapshot=None):
            if env_snapshot is None:
                env_snapshot = load_env() if env_path.exists() else {}
            if var_name in env_snapshot:
                return bool(env_snapshot.get(var_name))
            return bool(os.getenv(var_name))

        _secret_capture_callback = None

    backend = _get_terminal_backend_name()
    is_gateway = _is_gateway_surface()
    is_remote = backend in _REMOTE_ENV_BACKENDS

    print(f"    backend = {backend!r}")
    print(f"    is_gateway = {is_gateway}")
    print(f"    is_remote_backend = {is_remote}")
    print(f"    secret_capture_callback set = {_secret_capture_callback is not None}")

    if is_remote:
        print(f"\n    ⚠ PROBLEM FOUND: backend={backend!r} is a remote backend.")
        print(f"    Remote backends ALWAYS report all env vars as missing,")
        print(f"    regardless of whether they are set in ~/.hermes/.env.")
        print(f"    This is the _REMOTE_ENV_BACKENDS check in skills_tool.py:1087.")

    env_snapshot = load_env() if env_path.exists() else {}
    for var in ("CANVAS_API_TOKEN", "CANVAS_BASE_URL"):
        persisted = _is_env_var_persisted(var, env_snapshot)
        in_snapshot = var in env_snapshot
        snapshot_truthy = bool(env_snapshot.get(var))
        in_os = bool(os.getenv(var))

        would_be_missing = is_remote or not persisted

        print(f"\n    {var}:")
        print(f"      in env_snapshot:     {in_snapshot}")
        print(f"      snapshot val truthy: {snapshot_truthy}")
        print(f"      in os.environ:       {in_os}")
        print(f"      _is_env_var_persisted: {persisted}")
        print(f"      would_be_missing:    {would_be_missing}  {'✗ PROBLEM' if would_be_missing else '✓ OK'}")

    # ── 5. Check canvas_api.py runtime ──
    print("\n[5] Checking canvas_api.py runtime environment:")
    print("    The canvas_api.py script reads from os.environ, NOT from ~/.hermes/.env.")
    print("    For the script to work, vars must be in the process environment.")
    for var in ("CANVAS_API_TOKEN", "CANVAS_BASE_URL"):
        val = os.environ.get(var)
        if val:
            print(f"    {var}: ✓ available in os.environ")
        else:
            print(f"    {var}: ✗ NOT in os.environ — script would fail at runtime")

    # ── 6. Environment context ──
    print("\n[6] Environment context:")
    print(f"    HERMES_HOME = {os.getenv('HERMES_HOME', '(not set, defaults to ~/.hermes)')}")
    print(f"    TERMINAL_ENV = {os.getenv('TERMINAL_ENV', '(not set, defaults to local)')}")
    print(f"    HERMES_GATEWAY_SESSION = {os.getenv('HERMES_GATEWAY_SESSION', '(not set)')}")
    print(f"    HERMES_SESSION_PLATFORM = {os.getenv('HERMES_SESSION_PLATFORM', '(not set)')}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    issues = []

    if not env_path.exists():
        issues.append("~/.hermes/.env file does not exist")

    if env_path.exists():
        env_data = load_env()
        for var in ("CANVAS_API_TOKEN", "CANVAS_BASE_URL"):
            if not env_data.get(var):
                issues.append(f"{var} is empty or missing in ~/.hermes/.env")

    if is_remote:
        issues.append(f"Remote backend ({backend}) always reports vars as missing")

    for var in ("CANVAS_API_TOKEN", "CANVAS_BASE_URL"):
        if not os.environ.get(var):
            issues.append(f"{var} not in os.environ (subprocess execution would fail)")

    if not issues:
        print("No issues found! Canvas env vars appear correctly configured.")
    else:
        print(f"Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print()
    print("To enable verbose skill_view debug logging, run hermes with:")
    print("  HERMES_DEBUG_SKILLS=1 python cli.py")
    print("  (or set logging level to DEBUG for 'tools.skills_tool')")


if __name__ == "__main__":
    # Enable debug logging for skills_tool if requested
    if os.getenv("HERMES_DEBUG_SKILLS"):
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(message)s")
    main()
