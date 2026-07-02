"""The sidecar's node processes must be spawned headless on Windows.

``node.exe`` is a console-subsystem binary. When the gateway runs windowless
(``pythonw`` — the normal Windows service setup), every ``subprocess`` spawn of
node without ``CREATE_NO_WINDOW`` allocates a fresh console: the short-lived
Spectrum patch run flashes a window on every sidecar start, and the persistent
sidecar keeps one open for its whole lifetime.

These are AST invariants pinning ``creationflags=CREATE_NO_WINDOW`` (win32-
gated) onto both spawn sites in ``_start_sidecar`` — they fail if the flag is
dropped from either call.
"""
from __future__ import annotations

import ast
import inspect

from plugins.platforms.photon import adapter as photon_adapter


def _start_sidecar_ast() -> ast.AST:
    src = inspect.getsource(photon_adapter.PhotonAdapter._start_sidecar)
    return ast.parse("class _W:\n" + "\n".join("    " + line for line in src.splitlines()))


def _subprocess_calls(tree: ast.AST, attr: str) -> list[ast.Call]:
    """All ``subprocess.<attr>(...)`` calls under ``tree``."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]


def _passes_no_window_creationflags(call: ast.Call) -> bool:
    """True if the call passes a ``creationflags=`` kwarg referencing
    ``CREATE_NO_WINDOW`` somewhere in its value expression."""
    for kw in call.keywords:
        if kw.arg != "creationflags":
            continue
        return any(
            isinstance(node, ast.Attribute) and node.attr == "CREATE_NO_WINDOW"
            for node in ast.walk(kw.value)
        )
    return False


def test_patch_script_run_is_headless_on_windows():
    tree = _start_sidecar_ast()
    runs = _subprocess_calls(tree, "run")
    assert runs, "expected the Spectrum patch subprocess.run in _start_sidecar"
    assert all(_passes_no_window_creationflags(c) for c in runs), (
        "subprocess.run in _start_sidecar must pass creationflags="
        "CREATE_NO_WINDOW (win32-gated) or it flashes a console window"
    )


def test_persistent_sidecar_popen_is_headless_on_windows():
    tree = _start_sidecar_ast()
    popens = _subprocess_calls(tree, "Popen")
    assert popens, "expected the sidecar subprocess.Popen in _start_sidecar"
    assert all(_passes_no_window_creationflags(c) for c in popens), (
        "subprocess.Popen in _start_sidecar must pass creationflags="
        "CREATE_NO_WINDOW (win32-gated) or it keeps a console window open "
        "for the sidecar's lifetime"
    )
