"""Regression test: oneshot sets HERMES_YOLO_MODE before plugin discovery.

tools.approval freezes HERMES_YOLO_MODE once at module import, so the env
must be set before anything in run_oneshot can join the background plugin
discovery thread (_validate_explicit_toolsets is the first such call).
Setting it later risks a plugin import chain latching the freeze to False
and silently disabling oneshot's approval auto-bypass (#86526).
"""

import logging
import os

import hermes_cli.oneshot as oneshot
from hermes_cli.oneshot import run_oneshot


def test_yolo_env_set_before_toolset_validation(monkeypatch):
    seen = {}

    def fake_validate(toolsets):
        # Capture the env at the moment toolset validation (which can join
        # plugin discovery) runs.
        seen["yolo"] = os.environ.get("HERMES_YOLO_MODE")
        seen["hooks"] = os.environ.get("HERMES_ACCEPT_HOOKS")
        return None, "boom"  # force the early return; nothing heavy runs

    monkeypatch.setattr(oneshot, "_validate_explicit_toolsets", fake_validate)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    try:
        rc = run_oneshot("hi")
    finally:
        # run_oneshot silences stdlib logging process-wide; restore it so
        # later tests in this worker still log.
        logging.disable(logging.NOTSET)

    assert rc == 2
    assert seen == {"yolo": "1", "hooks": "1"}
