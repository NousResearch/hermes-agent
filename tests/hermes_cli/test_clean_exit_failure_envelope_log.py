"""Parser-friendly dispatch log for clean-exit failure envelopes (#t_d1e4fcd5).

Background:

Sister cards ``t_8acdc493`` and ``t_a2c1ced7`` populated the envelope
returned by ``run_conversation`` with structured failure metadata
(``failure_reason`` / ``error_class`` / ``provider`` / ``model`` /
``http_status``). The dispatcher, however, only ever saw the worker's
exit code and the free-text ``Error: ...`` stderr print -- no structured
log line. Synthetic ops tests and dashboard scrapers could not grep for
the envelope fields without free-text parsing.

Fix: ``hermes_cli.kanban_db._emit_failure_envelope_log`` emits a single
JSON line on a dedicated logger (``hermes.kanban.failure_envelope``) when
the worker's quiet-query exit path sees a failed result with envelope
fields. The line is JSON so any ``json.loads`` consumer works without
custom parsing. Schema is additive -- new keys may land later, but the
five field names are pinned by this test.

These tests pin:

  * the helper signature (task_id / result / exit_code)
  * the JSON line shape for synthetic 403/429/401 cases
  * the no-op behavior on success / empty / non-mapping inputs
  * the AST-level wiring: cli.py's quiet-query exit must call the helper
"""

from __future__ import annotations

import ast
import io
import json
import logging
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper invocation tests
# ---------------------------------------------------------------------------


def _capture_envelope_log(name: str = "hermes.kanban.failure_envelope"):
    """Attach a memory handler to the envelope logger and return the buffer."""
    log = logging.getLogger(name)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    # Quiet other loggers that might propagate so the buffer stays clean.
    log.propagate = False
    return log, buf


def _lines(buf: io.StringIO) -> list[str]:
    return [ln for ln in buf.getvalue().splitlines() if ln.strip()]


@pytest.fixture(autouse=True)
def _restore_propagate():
    """Restore propagate=True after each test (we set False for isolation)."""
    yield
    logging.getLogger("hermes.kanban.failure_envelope").propagate = True


def _envelope_lines(buf):
    """Pull just the JSON line out of the logger buffer."""
    out = []
    for ln in _lines(buf):
        # Format is "<LEVEL> kanban.failure_envelope {...}" -- split on first "{".
        idx = ln.find("{")
        if idx < 0:
            continue
        try:
            json.loads(ln[idx:])
        except Exception:
            continue
        out.append(ln[idx:])
    return out


def test_emits_json_for_403_billing() -> None:
    """HTTP 403 (billing) must surface all five envelope fields."""
    from hermes_cli.kanban_db import _emit_failure_envelope_log

    log, buf = _capture_envelope_log()
    _emit_failure_envelope_log(
        task_id="t_403_billing",
        result={
            "failed": True,
            "error": "exceeded your current quota",
            "failure_reason": "billing",
            "error_class": "billing",
            "provider": "kimi-coding",
            "model": "kimi-k2.7-code",
            "http_status": 403,
        },
        exit_code=75,
    )
    lines = _envelope_lines(buf)
    assert len(lines) == 1, f"expected one JSON line, got {lines!r}"
    payload = json.loads(lines[0])
    assert payload == {
        "task_id": "t_403_billing",
        "exit_code": 75,
        "failure_reason": "billing",
        "error_class": "billing",
        "provider": "kimi-coding",
        "model": "kimi-k2.7-code",
        "http_status": 403,
    }


def test_emits_json_for_429_rate_limit() -> None:
    """HTTP 429 (rate_limit) must surface all five envelope fields."""
    from hermes_cli.kanban_db import _emit_failure_envelope_log

    log, buf = _capture_envelope_log()
    _emit_failure_envelope_log(
        task_id="t_429_rl",
        result={
            "failed": True,
            "error": "rate limit exceeded",
            "failure_reason": "rate_limit",
            "error_class": "rate_limit",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "http_status": 429,
        },
        exit_code=75,
    )
    payload = json.loads(_envelope_lines(buf)[0])
    assert payload["failure_reason"] == "rate_limit"
    assert payload["http_status"] == 429
    assert payload["exit_code"] == 75


def test_emits_json_for_401_auth() -> None:
    """HTTP 401 (auth) must surface all five envelope fields."""
    from hermes_cli.kanban_db import _emit_failure_envelope_log

    log, buf = _capture_envelope_log()
    _emit_failure_envelope_log(
        task_id="t_401_auth",
        result={
            "failed": True,
            "error": "invalid api key",
            "failure_reason": "auth",
            "error_class": "auth",
            "provider": "anthropic",
            "model": "claude-sonnet-4",
            "http_status": 401,
        },
        exit_code=1,
    )
    payload = json.loads(_envelope_lines(buf)[0])
    assert payload["failure_reason"] == "auth"
    assert payload["http_status"] == 401
    assert payload["exit_code"] == 1


def test_noop_when_not_failed() -> None:
    """Happy-path success must NOT emit an envelope line."""
    from hermes_cli.kanban_db import _emit_failure_envelope_log

    log, buf = _capture_envelope_log()
    _emit_failure_envelope_log(
        task_id="t_happy",
        result={"failed": False, "final_response": "all good"},
        exit_code=0,
    )
    assert _envelope_lines(buf) == [], (
        "successful result must not emit envelope line (non-regressive on "
        f"happy path); got {buf.getvalue()!r}"
    )


def test_noop_when_envelope_fields_absent() -> None:
    """A failed result without envelope fields (legacy shape) must NOT spam
    a half-empty line -- the free-text ``Error: ...`` print already covers
    the user-facing path."""
    from hermes_cli.kanban_db import _emit_failure_envelope_log

    log, buf = _capture_envelope_log()
    _emit_failure_envelope_log(
        task_id="t_legacy",
        result={"failed": True, "error": "something exploded"},
        exit_code=1,
    )
    assert _envelope_lines(buf) == [], (
        "result without envelope fields must not emit (avoids half-empty spam); "
        f"got {buf.getvalue()!r}"
    )


def test_noop_on_non_mapping_result() -> None:
    """Defensive: a string / None / list result must not crash the helper."""
    from hermes_cli.kanban_db import _emit_failure_envelope_log

    log, buf = _capture_envelope_log()
    for bad in (None, "string result", 42, [1, 2, 3]):
        _emit_failure_envelope_log(
            task_id="t_bad", result=bad, exit_code=1,
        )
    assert _envelope_lines(buf) == []


def test_does_not_raise_when_handler_misbehaves() -> None:
    """A broken handler must not propagate -- the worker exit path is sacred."""
    from hermes_cli import kanban_db

    log = logging.getLogger("hermes.kanban.failure_envelope")

    class BrokenHandler(logging.Handler):
        def emit(self, record):  # noqa: D401 -- stdlib override
            raise RuntimeError("handler boom")

    handler = BrokenHandler()
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    try:
        # Must not raise even though the handler explodes.
        kanban_db._emit_failure_envelope_log(
            task_id="t_broken",
            result={
                "failed": True,
                "failure_reason": "rate_limit",
                "error_class": "rate_limit",
                "provider": "x", "model": "y", "http_status": 429,
            },
            exit_code=75,
        )
    finally:
        log.removeHandler(handler)


# ---------------------------------------------------------------------------
# AST-level wiring: cli.py quiet-query exit must call the helper
# ---------------------------------------------------------------------------


def test_cli_py_calls_emit_failure_envelope_log_in_failed_branch() -> None:
    """The cli.py quiet-query exit path must import and call
    ``_emit_failure_envelope_log`` when ``failed=True`` and
    ``HERMES_KANBAN_TASK`` is set. AST check so a future refactor that
    drops the call site is caught immediately. (#t_d1e4fcd5)

    Reads the cli.py source DIRECTLY (no import) so the test does not
    pull in yaml / rich / prompt_toolkit transitively. The repo root is
    resolved via this test file's location (tests/hermes_cli/...).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    cli_path = repo_root / "cli.py"
    src = cli_path.read_text()
    tree = ast.parse(src)

    # 1. Helper must be imported (any local name -- ``as _emit_envelope``
    # is fine).
    helper_local_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "hermes_cli.kanban_db":
            continue
        for alias in node.names:
            if alias.name == "_emit_failure_envelope_log":
                helper_local_names.add(alias.asname or alias.name)
    assert helper_local_names, (
        "cli.py must import _emit_failure_envelope_log from "
        "hermes_cli.kanban_db so the dispatcher log line is wired into "
        "the worker exit path"
    )

    # 2. Helper must be called from inside a branch gated on
    # ``result.get("failed")`` AND ``HERMES_KANBAN_TASK``. We walk every
    # Call node and check that the surrounding source window (30 lines
    # above the call) references both markers.
    found = False
    src_lines = src.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match either the bare name or any of the local aliases.
        local_name = func.id if isinstance(func, ast.Name) else None
        if local_name not in helper_local_names:
            continue
        call_line = node.lineno
        start = max(0, call_line - 30)
        window = "\n".join(src_lines[start:call_line])
        if '"failed"' in window and "HERMES_KANBAN_TASK" in window:
            found = True
            break
    assert found, (
        "cli.py must call _emit_failure_envelope_log from a branch that "
        "is gated on both result.get('failed') and HERMES_KANBAN_TASK"
    )


def test_logger_is_module_level_singleton() -> None:
    """The dedicated logger must be a module-level singleton -- not re-created
    per call -- so operators can route it once at setup time."""
    from hermes_cli import kanban_db
    log1 = logging.getLogger("hermes.kanban.failure_envelope")
    log2 = kanban_db._failure_envelope_log
    assert log1 is log2, (
        "helper must reuse the module-level logger so setup can attach a "
        "dedicated handler without per-call imports"
    )
