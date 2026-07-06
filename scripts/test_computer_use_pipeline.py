#!/usr/bin/env python3
"""
Pipeline smoke-test for the Hermes ``computer_use`` toolset.

Idempotent, safe to run on any platform.  Verifies:
  1. Hermes is installed and the tools package importable.
  2. The computer_use tool is registered in the tools registry.
  3. The JSON schema is valid and contains expected fields.
  4. The requirements gate (check_fn) turns cleanly on any host.
  5. The handler code path runs gracefully even without cua-driver.

Output is structured JSON to stderr per-step and a final summary line on stdout.
Exit code 0 when all checks pass, non-zero on failure.

Usage:
    python scripts/test_computer_use_pipeline.py      # from repo root

Set ``COMPUTER_USE_CI_VERBOSE=1`` for richer per-step logs.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys
import traceback


# ── Hermes discovery ────────────────────────────────────────────────────


def _ensure_hermes_on_path() -> None:
    """If ``tools`` isn't importable, locate the Hermes site-packages."""
    try:
        import tools  # noqa: F401
        return
    except ImportError:
        pass

    # 1. pipx venv path
    candidate = os.path.expanduser(
        "~/.local/share/pipx/venvs/hermes-agent/lib/python*/site-packages"
    )
    matches = sorted(glob.glob(candidate))
    if matches:
        sys.path.insert(0, matches[-1])
        return

    # 2. pip show
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "hermes-agent"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                sys.path.insert(0, line.split(":", 1)[1].strip())
                return
    except Exception:
        pass


_ensure_hermes_on_path()


# ── Helpers ──────────────────────────────────────────────────────────────


def log(step: str, status: str, detail: str = "") -> dict:
    """Emit a structured log line to stderr, return it as a dict."""
    entry = {"step": step, "status": status, "detail": detail}
    print(json.dumps(entry), file=sys.stderr, flush=True)
    return entry


# ── Checks ───────────────────────────────────────────────────────────────


def check_hermes_import() -> tuple[bool, str]:
    """Verify the tools package is importable."""
    try:
        import tools  # noqa: F401
        return True, "tools package imported OK"
    except ImportError as exc:
        return False, f"import failed: {exc}"
    except Exception as exc:
        return False, f"unexpected error: {exc}"


def check_computer_use_registered() -> tuple[bool, str]:
    """
    Confirm computer_use is registered.
    The tool uses lazy registration — the shim must be imported first.
    """
    try:
        import tools.computer_use_tool  # noqa: F401 — triggers registration
        from tools.registry import registry

        entry = registry.get_entry("computer_use")
        if entry is None:
            return False, "registry.get_entry('computer_use') returned None"
        info = {
            "name": entry.name,
            "toolset": entry.toolset,
            "description": (entry.description or "")[:120],
        }
        return True, json.dumps(info)
    except Exception as exc:
        return False, f"registry lookup failed: {exc}"


def check_schema_valid() -> tuple[bool, str]:
    """Verify the schema dict has expected top-level keys."""
    try:
        import tools.computer_use_tool  # noqa: F401
        from tools.registry import registry

        entry = registry.get_entry("computer_use")
        schema = entry.schema if entry else None
        if not schema or not isinstance(schema, dict):
            return False, f"schema is not a dict: {type(schema)}"
        required_keys = {"name", "description", "parameters"}
        found = set(schema.keys())
        missing = required_keys - found
        if missing:
            return False, f"schema missing keys: {missing}"
        params = schema.get("parameters", {})
        properties = params.get("properties", {})
        action_prop = properties.get("action", {})
        enum_vals = action_prop.get("enum", [])
        expected_actions = {"capture", "click", "type", "key", "scroll"}
        actual = set(enum_vals)
        if not expected_actions.issubset(actual):
            return False, (
                f"schema action enum missing expected values: "
                f"{expected_actions - actual}"
            )
        return True, (
            f"schema valid; {len(properties)} params, "
            f"{len(enum_vals)} actions"
        )
    except Exception as exc:
        return False, f"schema validation failed: {exc}"


def check_requirements_gate() -> tuple[bool, str]:
    """
    Call the requirements check function.

    On Linux should return False (no cua-driver) without raising,
    proving the check_fn path works end-to-end.
    """
    try:
        import tools.computer_use_tool  # noqa: F401
        from tools.computer_use.tool import check_computer_use_requirements

        ok = check_computer_use_requirements()
        sys_name = platform.system()
        return True, (
            f"requirements check ran on {sys_name} with result={ok} "
            f"({'expected on non-macOS' if not ok and sys_name != 'Darwin' else ''})"
        )
    except Exception as exc:
        return False, f"requirements check raised: {exc}"


def check_dispatch_graceful() -> tuple[bool, str]:
    """
    Attempt a minimal computer_use dispatch.

    On non-macOS it should fail gracefully with an error about an unavailable
    backend, proving the handler entrypoint works.
    """
    try:
        import tools.computer_use_tool  # noqa: F401
        from tools.computer_use.tool import handle_computer_use

        result = handle_computer_use({"action": "capture", "mode": "som"})
        if isinstance(result, dict):
            # Success (rare in CI, possible on a macOS runner)
            return True, "capture succeeded (multimodal result)"
        # Otherwise it's a JSON error string
        parsed = json.loads(result)
        error = parsed.get("error", "")
        if "unavailable" in error.lower():
            return True, (
                f"dispatch gracefully handled: {error[:120]}"
            )
        return True, f"dispatch returned (non-fatal): {parsed}"
    except Exception as exc:
        return False, f"handle_computer_use raised: {exc}"


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    verbose = os.environ.get("COMPUTER_USE_CI_VERBOSE", "") != ""
    checks = [
        ("import_hermes", check_hermes_import),
        ("registered", check_computer_use_registered),
        ("schema_valid", check_schema_valid),
        ("requirements_gate", check_requirements_gate),
        ("dispatch_graceful", check_dispatch_graceful),
    ]

    results: list[dict] = []
    all_ok = True

    print(
        json.dumps(
            {
                "event": "pipeline_start",
                "platform": platform.system(),
                "python": sys.version.split()[0],
            }
        ),
        file=sys.stderr,
        flush=True,
    )

    for step_name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok = False
            detail = f"unexpected exception: {traceback.format_exc()}"
        results.append(log(step_name, "PASS" if ok else "FAIL", detail))
        if verbose and ok:
            log(f"{step_name}_detail", "OK", detail)
        if not ok:
            all_ok = False

    summary = {
        "event": "pipeline_finish",
        "overall": "PASS" if all_ok else "FAIL",
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "total": len(results),
        "checks": results,
    }
    # Final line on stdout = machine-parseable summary.
    print(json.dumps(summary))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
