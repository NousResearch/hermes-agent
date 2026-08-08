"""DDGS search child-process entrypoint (#68096).

Invoked as ``python plugins/web/ddgs/_search_worker.py`` (script path from the
parent provider). Reads one JSON request from stdin, writes one JSON envelope
to stdout, then exits.

Request::
    {"query": str, "safe_limit": int}

Envelope::
    {"ok": true, "results": [...]}
    {"ok": false, "error": str}

Optional test hooks (only when ``HERMES_DDGS_ALLOW_TEST_HOOKS=1``)::
    {"query": ..., "safe_limit": ..., "test_hook": "sleep"|"gil"|"success"|"error"|"empty"}
"""

from __future__ import annotations

import json
import os
import sys
import time


def _hold_gil(secs: int) -> None:
    """Block in a foreign call that keeps the GIL (ctypes.PyDLL).

    Mirrors native ``primp`` holding the interpreter lock. ``PyDLL`` (unlike
    ``CDLL``/``WinDLL``) does not release the GIL around the call.
    """
    import ctypes

    if sys.platform == "win32":
        lib = ctypes.PyDLL("kernel32")
        lib.Sleep.argtypes = [ctypes.c_uint]
        lib.Sleep(int(secs * 1000))
        return

    lib = ctypes.PyDLL(None)
    try:
        sleep = lib.sleep
    except AttributeError:  # pragma: no cover — macOS libSystem fallback
        sleep = ctypes.PyDLL("/usr/lib/libSystem.B.dylib").sleep
    sleep.argtypes = [ctypes.c_uint]
    sleep(int(secs))


def _run_test_hook(hook: str) -> dict:
    if hook == "sleep":
        time.sleep(30)
        return {"ok": False, "error": "sleep hook returned unexpectedly"}
    if hook == "gil":
        _hold_gil(30)
        return {"ok": False, "error": "gil hook returned unexpectedly"}
    if hook == "success":
        return {
            "ok": True,
            "results": [
                {
                    "title": "Hit",
                    "url": "https://example.com",
                    "description": "body",
                    "position": 1,
                }
            ],
        }
    if hook == "empty":
        return {"ok": True, "results": []}
    if hook == "error":
        return {"ok": False, "error": "RuntimeError: boom"}
    return {"ok": False, "error": f"unknown test_hook: {hook!r}"}


def _write_envelope(envelope: dict) -> None:
    json.dump(envelope, sys.stdout)
    sys.stdout.flush()


def _activate_durable_lazy_target() -> None:
    """Put the durable lazy-install target on this worker's ``sys.path``.

    Sealed-venv deployments (the Docker image sets
    ``HERMES_DISABLE_LAZY_INSTALLS=1`` plus ``HERMES_LAZY_INSTALL_TARGET``)
    install ``ddgs`` into a writable directory outside the venv. The gateway
    imports from it because ``hermes_bootstrap`` activates that target at
    startup — but this worker is spawned as a bare script and runs no
    bootstrap, so ``from ddgs import DDGS`` failed with ``ModuleNotFoundError``
    even though the package was installed (#75025).

    Reuses ``tools.lazy_deps.activate_durable_lazy_target`` rather than
    extending ``PYTHONPATH``: that helper *appends* the target so the venv
    keeps precedence on every name collision. A writable directory must never
    be able to shadow a core module. No-ops when the target is unset or
    missing, and never raises — a failure here should degrade to the current
    ModuleNotFoundError, not lose the request envelope.
    """
    try:
        from tools.lazy_deps import activate_durable_lazy_target

        activate_durable_lazy_target()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        _write_envelope({"ok": False, "error": f"invalid request: {exc}"})
        return 2

    hook = request.get("test_hook")
    if hook:
        if os.environ.get("HERMES_DDGS_ALLOW_TEST_HOOKS") != "1":
            _write_envelope(
                {"ok": False, "error": "test_hook refused (hooks not enabled)"}
            )
            return 3
        envelope = _run_test_hook(str(hook))
        _write_envelope(envelope)
        return 0 if envelope.get("ok") else 1

    query = str(request.get("query") or "")
    safe_limit = max(1, int(request.get("safe_limit") or 1))
    try:
        # Before the ddgs import below — on a sealed venv the package lives in
        # the durable target, which nothing else on this path activates.
        _activate_durable_lazy_target()

        # Import inside main so script startup stays light / patchable.
        from plugins.web.ddgs.provider import _run_ddgs_search

        results = _run_ddgs_search(query, safe_limit)
        _write_envelope({"ok": True, "results": results})
        return 0
    except Exception as exc:  # noqa: BLE001
        _write_envelope({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
