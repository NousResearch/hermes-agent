"""Per-handler adapters: map a gateway exec handler's native shape onto the
unified :class:`ExecutionResult`, and render it back to the exact wire dict the
handler returns today.

The contract guarantee for every adapter pair is a round-trip identity:

    X_to_wire(X_to_result(<native fields>)) == <today's _ok result dict>

i.e. the migration is byte-compatible on the wire (additive-only — see
docs/architecture/central-brain-openclaw.md §12). The round-trip is asserted in
tests/action_runtime/. Handlers are migrated one at a time (shell → cli →
slash); only the adapters for already-migrated handlers live here.
"""

from __future__ import annotations

from typing import Optional

from action_runtime.contract import (
    ErrorType,
    ExecError,
    ExecutionResult,
    SideEffect,
    Status,
)


# ── shell.exec ───────────────────────────────────────────────────────
# Native: subprocess.run → {stdout (tail 4000), stderr (tail 2000), code}.
# A non-zero exit is NOT an RPC error — it is an honest FAILED result whose
# wire shape is identical to a success (the client distinguishes via `code`).
# Truncation stays in the handler (it is part of the wire contract); the
# adapter receives already-sliced strings and must not re-truncate.


def shell_to_result(
    stdout: str, stderr: str, code: int, task_id: Optional[str] = None
) -> ExecutionResult:
    if code == 0:
        return ExecutionResult(
            task_id,
            Status.SUCCEEDED,
            outputs={"stdout": stdout, "stderr": stderr, "code": code},
        )
    return ExecutionResult(
        task_id,
        Status.FAILED,
        outputs={"stdout": stdout, "stderr": stderr, "code": code},
        error=ExecError(
            ErrorType.NONZERO_EXIT,
            retryable=False,
            message=f"command exited {code}",
        ),
    )


def shell_to_wire(result: ExecutionResult) -> dict:
    """Byte-identical to the shell.exec _ok payload (tui_gateway/server.py)."""
    return {
        "stdout": result.outputs["stdout"],
        "stderr": result.outputs["stderr"],
        "code": result.outputs["code"],
    }


# ── cli.exec ─────────────────────────────────────────────────────────
# Native: {blocked, code, output} (+ hint when blocked). A pre-exec block gate
# (interactive argv) returns blocked=True/code=-1 WITHOUT spawning; a spawned
# command's non-zero exit is an honest FAILED whose wire shape is unchanged
# (client reads `code`). `output` is already joined+truncated by the handler.


def cli_to_result(
    blocked: bool,
    code: int,
    output: str,
    hint: str = "",
    task_id: Optional[str] = None,
) -> ExecutionResult:
    if blocked:
        return ExecutionResult(
            task_id,
            Status.BLOCKED,
            outputs={"output": output, "code": code, "blocked": True},
            error=ExecError(ErrorType.DENIED, retryable=False, message=hint),
        )
    if code == 0:
        return ExecutionResult(
            task_id,
            Status.SUCCEEDED,
            outputs={"output": output, "code": code, "blocked": False},
        )
    return ExecutionResult(
        task_id,
        Status.FAILED,
        outputs={"output": output, "code": code, "blocked": False},
        error=ExecError(ErrorType.NONZERO_EXIT, retryable=False, message=f"cli exited {code}"),
    )


def cli_to_wire(result: ExecutionResult) -> dict:
    """Byte-identical to the cli.exec _ok payloads (tui_gateway/server.py)."""
    if result.outputs["blocked"]:
        # The blocked-gate payload carries the hint (held in error.message).
        return {
            "blocked": True,
            "hint": result.error.message if result.error else "",
            "code": result.outputs["code"],
            "output": result.outputs["output"],
        }
    return {
        "blocked": False,
        "code": result.outputs["code"],
        "output": result.outputs["output"],
    }


# ── slash.exec: plugin path ──────────────────────────────────────────
# A plugin slash command runs its handler inline. Success → {output}; a raised
# handler → {output, error} where both equal "Plugin command error: <e>"
# (the Phase 1a additive contract — a programmatic client detects the failure
# via result.error without parsing the output prose).


def plugin_to_result(
    output: str = "", exc: Optional[BaseException] = None, task_id: Optional[str] = None
) -> ExecutionResult:
    if exc is None:
        return ExecutionResult(task_id, Status.SUCCEEDED, outputs={"output": output})
    msg = f"Plugin command error: {exc}"
    return ExecutionResult(
        task_id,
        Status.FAILED,
        outputs={"output": msg},
        error=ExecError(ErrorType.PROVIDER_ERROR, retryable=False, message=msg),
    )


def plugin_to_wire(result: ExecutionResult) -> dict:
    """Byte-identical to the slash.exec plugin-path _ok payload. Phase 4
    additive: ``task_id`` is echoed IFF the client supplied one — absent
    task_id leaves the wire dict untouched."""
    out = {"output": result.outputs["output"]}
    if result.error is not None:
        out["error"] = result.error.message  # == output, per the Phase 1a test
    if result.task_id is not None:
        out["task_id"] = result.task_id
    return out


# ── slash.exec: worker path + live-agent side effects ────────────────
# Runs the persistent slash worker, then mirrors live-agent side effects. The
# outcome of the mirror is a _SlashSideEffect(kind, message) (duck-typed here —
# no import, avoids a cycle): kind in {"", "warning", "failure", "busy"}.
# Wire: {output} (+warning when rendered non-empty) (+error == warning when the
# side effect did NOT take — kind in {failure, busy}). The rendered warning
# string is produced by the handler's _slash_side_effect_warning(effect) and
# passed in, so presentation stays where it is.


def slash_to_result(output: str, effect, task_id: Optional[str] = None) -> ExecutionResult:
    """``effect`` is a _SlashSideEffect with .kind / .message attributes."""
    status, err, side = Status.SUCCEEDED, None, []
    if effect.kind in ("failure", "busy"):
        status = Status.FAILED
        etype = ErrorType.TRANSPORT if effect.kind == "busy" else ErrorType.PROVIDER_ERROR
        err = ExecError(etype, retryable=(effect.kind == "busy"), message=effect.message)
    if effect.kind in ("failure", "busy", "warning"):
        side = [
            SideEffect(
                kind="slash_sync",
                detail=effect.message,
                applied=effect.kind == "warning",
            )
        ]
    return ExecutionResult(
        task_id, status, outputs={"output": output}, error=err, side_effects=side
    )


def slash_to_wire(result: ExecutionResult, warning: str) -> dict:
    """Byte-identical to the slash.exec worker-path _ok payload. ``warning`` is
    the rendered _slash_side_effect_warning(effect) string. Phase 4 additive:
    ``task_id`` is echoed IFF the client supplied one — absent task_id leaves
    the wire dict untouched."""
    payload = {"output": result.outputs["output"]}
    if warning:
        payload["warning"] = warning
        # error mirrors warning only when the side effect did NOT take
        # (failure/busy) — exactly when result.error is set.
        if result.error is not None:
            payload["error"] = warning
    if result.task_id is not None:
        payload["task_id"] = result.task_id
    return payload


# ── rich wire renderer (Phase 4: task.submit) ────────────────────────
# The full ExecutionResult, rendered losslessly — no legacy aliasing, no
# omitted-when-empty keys. task.submit returns THIS shape; the legacy
# slash.exec wire above stays byte-compatible and untouched.


def result_to_wire_rich(result: ExecutionResult) -> dict:
    """Render the complete :class:`ExecutionResult` as a wire dict.

    Every key is always present: ``error`` is ``None`` (not omitted) on
    success, ``side_effects`` is ``[]`` when nothing changed, and ``status``
    is the :class:`Status` enum's value string (e.g. ``"succeeded"``).
    Sole exception: ``trace_id`` (§12 trace plumbing) appears ONLY when set —
    absent-when-None keeps every pre-trace wire dict byte-identical (same
    additive pattern as ``AgentTaskRecord.snapshot()``'s ``session_id``).
    """
    wire = {
        "task_id": result.task_id,
        "status": result.status.value,
        "outputs": result.outputs,
        "error": (
            None
            if result.error is None
            else {
                "type": result.error.type.value,
                "retryable": result.error.retryable,
                "message": result.error.message,
            }
        ),
        "side_effects": [
            {
                "kind": s.kind,
                "detail": s.detail,
                "applied": s.applied,
                "target": s.target,
            }
            for s in result.side_effects
        ],
    }
    if result.trace_id is not None:
        wire["trace_id"] = result.trace_id
    return wire
