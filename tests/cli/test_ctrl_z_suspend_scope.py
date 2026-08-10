"""Tests for Ctrl+Z suspend scope (#83006).

The CLI's Ctrl+Z binding used to suspend the *entire process group* via
``os.kill(0, SIGTSTP)``, which also stopped every background job sharing
the group (long-running terminal/OCR tasks spawned from the session) and
turned a stray 0x1A byte in pasted input into an apparent crash. The fix
signals only the current process, matching shell job-control semantics.
"""
import ast
import signal
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = ROOT / "cli.py"


def _load_suspend_helper():
    """Load cli._suspend_cli_process without importing cli.

    AST-loading the exact helper keeps the test tied to production code
    while avoiding cli.py's heavy import side effects. If the helper is
    removed or renamed, this test fails.
    """
    source = CLI_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_suspend_cli_process"
        ),
        None,
    )
    assert helper_node is not None, "cli.py must define _suspend_cli_process()"
    helper_source = ast.get_source_segment(source, helper_node)
    assert helper_source is not None, "failed to extract _suspend_cli_process source"
    import os as _os

    namespace: dict = {"os": _os}
    exec(helper_source, namespace)
    helper = namespace["_suspend_cli_process"]
    assert callable(helper), "extracted _suspend_cli_process must be callable"
    return helper


def test_suspend_targets_only_current_process():
    """Ctrl+Z must signal the current pid, never the process group (0)."""
    _suspend_cli_process = _load_suspend_helper()
    with patch("os.kill") as mock_kill, patch("os.getpid", return_value=4242):
        _suspend_cli_process()
    mock_kill.assert_called_once_with(4242, signal.SIGTSTP)
    called_pids = [call.args[0] for call in mock_kill.call_args_list]
    assert 0 not in called_pids, (
        "Ctrl+Z must not signal the whole process group (os.kill(0, ...))"
    )


def test_suspend_skipped_on_windows():
    """The helper is Unix-only; Windows has no SIGTSTP."""
    if sys.platform == "win32":
        import pytest

        pytest.skip("POSIX-only helper")
