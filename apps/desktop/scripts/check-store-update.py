#!/usr/bin/env python3
"""Run a Store check, download, or install with the packaged interpreter.

The desktop keeps the backend alive during download, then stops it before
requesting installation. Each invocation reports one final JSON result.
Exit 0 means success, 2 means a check found updates, and 1 means failure.
"""
from __future__ import annotations

import argparse
import json
import os
import site
import sys
import time


def perform(context, mode, wait, completed_state) -> dict:
    """Use the same update objects for the query and the requested operation."""
    updates = wait(context.get_app_and_optional_store_package_updates_async())
    packages = [update.package.id.full_name for update in updates]
    result = {"available": bool(updates), "packages": packages, "ok": True}
    if mode == "check" or not updates:
        return result
    request = {
        "download": context.request_download_store_package_updates_async,
        "install": context.request_download_and_install_store_package_updates_async,
    }[mode]
    outcome = wait(request(updates))
    if outcome.overall_state != completed_state:
        result.update(ok=False, error=f"Microsoft Store {mode}: {outcome.overall_state.name}")
    return result


def _wait(operation):
    """Keep the STA responsive while Store displays consent and progress UI."""
    import win32con
    import win32event
    import win32gui
    from winrt.windows.foundation import AsyncStatus

    deadline = time.monotonic() + 1800
    while operation.status == AsyncStatus.STARTED:
        if time.monotonic() >= deadline:
            operation.cancel()
            raise TimeoutError("Microsoft Store request timed out")
        win32event.MsgWaitForMultipleObjects((), False, 50, win32con.QS_ALLINPUT)
        if win32gui.PumpWaitingMessages():
            operation.cancel()
            raise RuntimeError("Microsoft Store request was interrupted")
    return operation.get_results()


def _execute(mode: str, window_handle: int | None) -> dict:
    from winrt.runtime.interop import initialize_with_window
    from winrt.windows.services.store import StoreContext, StorePackageUpdateState
    import winrt.windows.applicationmodel  # noqa: F401 — projected Package values
    import winrt.windows.foundation.collections  # noqa: F401 — projected sequence

    if mode != "check" and not window_handle:
        raise ValueError("A desktop window is required for Microsoft Store consent")
    context = StoreContext.get_default()
    if window_handle:
        initialize_with_window(context, window_handle)
    return perform(context, mode, _wait, StorePackageUpdateState.COMPLETED)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "download", "install"), default="check")
    parser.add_argument("--hwnd", type=int)
    args = parser.parse_args(argv)
    # The packaged interpreter receives its dependencies through PYTHONPATH.
    # Process their .pth files so pywin32 can find its extensions and DLLs.
    for directory in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if directory:
            site.addsitedir(directory)
    try:
        from winrt.runtime import ApartmentType, init_apartment, uninit_apartment

        init_apartment(ApartmentType.SINGLE_THREADED)
    except Exception as exc:
        print(json.dumps({"available": None, "ok": False, "error": str(exc)}), flush=True)
        return 1
    try:
        try:
            result = _execute(args.mode, args.hwnd)
        except Exception as exc:
            # Missing Store acquisition is not evidence that the app is current.
            result = {"available": None, "ok": False, "error": f"Microsoft Store request failed: {exc}"}
    finally:
        # _execute's COM objects must be released before apartment shutdown.
        uninit_apartment()
    print(json.dumps(result), flush=True)
    if not result["ok"]:
        return 1
    return 2 if args.mode == "check" and result["available"] else 0


if __name__ == "__main__":
    sys.exit(main())
