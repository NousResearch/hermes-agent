"""Regression tests for #60328: --yolo must set HERMES_YOLO_MODE in
main() before _prepare_agent_startup() triggers tool imports.

The freeze mechanism in tools.approval (_YOLO_MODE_FROZEN) is correct
by design (PR #7994). The bug was that main() set the env var inside
cmd_chat(), which runs *after* _prepare_agent_startup() has already
imported tools.approval and frozen the constant to False.

These tests verify the ordering in main() itself: the env var must
already be set at the moment _prepare_agent_startup() is called.
If someone moves the assignment back into cmd_chat(), these tests
fail — catching the exact #60328 regression.
"""

import os
import sys


def _run_main_and_capture_yolo_at_startup(monkeypatch, argv):
    """Run main() with *argv*, capturing HERMES_YOLO_MODE at the
    moment _prepare_agent_startup is called.

    Returns the captured env var value (or None if unset).
    """
    yolo_at_startup = {}

    def spy_prepare_startup(args):
        yolo_at_startup["value"] = os.environ.get("HERMES_YOLO_MODE")

    monkeypatch.setattr(
        "hermes_cli.main._prepare_agent_startup", spy_prepare_startup
    )
    # Stub cmd_chat so main() returns cleanly without entering chat.
    monkeypatch.setattr("hermes_cli.main.cmd_chat", lambda args: None)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(sys, "argv", argv)

    from hermes_cli.main import main as cli_main

    cli_main()

    return yolo_at_startup.get("value")


def test_top_level_yolo_flag_sets_env_before_startup(monkeypatch):
    """hermes --yolo must set HERMES_YOLO_MODE before
    _prepare_agent_startup imports tools.approval."""
    result = _run_main_and_capture_yolo_at_startup(
        monkeypatch, ["hermes", "--yolo"]
    )
    assert result == "1", (
        "HERMES_YOLO_MODE was not '1' when _prepare_agent_startup was "
        "called from main() with --yolo. This is the #60328 regression: "
        "the env var is set too late (inside cmd_chat, after tool imports)."
    )


def test_chat_subcommand_yolo_flag_sets_env_before_startup(monkeypatch):
    """hermes chat --yolo must also set HERMES_YOLO_MODE before
    _prepare_agent_startup."""
    result = _run_main_and_capture_yolo_at_startup(
        monkeypatch, ["hermes", "chat", "--yolo"]
    )
    assert result == "1", (
        "HERMES_YOLO_MODE was not '1' when _prepare_agent_startup was "
        "called from main() with 'chat --yolo'."
    )


def test_no_yolo_flag_leaves_env_unset_at_startup(monkeypatch):
    """Without --yolo, HERMES_YOLO_MODE must not be set at startup."""
    result = _run_main_and_capture_yolo_at_startup(
        monkeypatch, ["hermes"]
    )
    assert result is None, (
        "HERMES_YOLO_MODE was unexpectedly set at startup without --yolo."
    )
