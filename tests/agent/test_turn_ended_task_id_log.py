"""Regression test for the cron job-id ('task=') field in the Turn-ended diag log.

Background: model-fallback detection needs to attribute a logged turn to the cron
job that produced it. The cron scheduler already passes the job id into the agent as
``task_id`` (cron/scheduler.py -> run_conversation(task_id=...)), which becomes
``effective_task_id`` and is threaded into the turn finalizer. This test guards
that the Turn-ended diagnostic log line carries that id as ``task=<id>`` and that
the format string and argument tuple stay in lockstep (a mismatch would raise at
logging time).

This is a source-contract test: the emit block builds a local ``_diag_msg`` /
``_diag_args`` pair in ``agent/turn_finalizer.py`` and passes it to both
logger.info and logger.warning. We assert:
  1. ``task=%s`` is present in the format string.
  2. The number of %-specifiers equals the number of args (so % formatting won't blow up).
"""

import re
from pathlib import Path


TURN_FINALIZER = Path(__file__).resolve().parents[2] / "agent" / "turn_finalizer.py"


def _extract_diag_block(src: str):
    """Return (format_string, n_args) for the Turn-ended _diag_msg/_diag_args block."""
    # _diag_msg = ( "..." "..." )
    m = re.search(r"_diag_msg = \(\s*((?:\s*\"[^\"]*\"\s*)+)\)", src)
    assert m, "could not locate _diag_msg block"
    fmt = "".join(re.findall(r"\"([^\"]*)\"", m.group(1)))

    a = re.search(r"_diag_args = \(\s*(.*?)\)\s*\n", src, re.DOTALL)
    assert a, "could not locate _diag_args block"
    # count top-level comma-separated args (block is flat, no nested parens)
    body = a.group(1).strip().rstrip(",")
    n_args = len([x for x in re.split(r",(?![^()]*\))", body) if x.strip()])
    return fmt, n_args


def test_turn_ended_log_includes_task_field():
    src = TURN_FINALIZER.read_text()
    fmt, _ = _extract_diag_block(src)
    assert "task=%s" in fmt, "Turn-ended log must carry the cron job id as task=%s"
    assert "session=%s" in fmt, "session=%s should still be present (not replaced)"


def test_turn_ended_format_specifiers_match_args():
    src = TURN_FINALIZER.read_text()
    fmt, n_args = _extract_diag_block(src)
    n_specs = len(re.findall(r"%[sd]", fmt))
    assert n_specs == n_args, (
        f"_diag_msg has {n_specs} %-specifiers but _diag_args supplies {n_args}; "
        "these must match or logging will raise TypeError"
    )


def test_effective_task_id_is_the_logged_value():
    """The last _diag_args entry must be effective_task_id (the cron job id)."""
    src = TURN_FINALIZER.read_text()
    a = re.search(r"_diag_args = \(\s*(.*?)\)\s*\n", src, re.DOTALL)
    assert a
    args = [x.strip() for x in re.split(r",(?![^()]*\))", a.group(1)) if x.strip()]
    assert args[-1].startswith("effective_task_id"), (
        f"last diag arg should be effective_task_id, got {args[-1]!r}"
    )
