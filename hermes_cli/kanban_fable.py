"""Selective Fable worker lane for the kanban dispatcher.

Normal kanban workers are spawned by :func:`hermes_cli.kanban_db._default_spawn`
as ``hermes -p <assignee> chat -q ...`` — a full Hermes agent running the
assignee profile's configured model. The Fable lane replaces *only* that argv,
for *only* the tasks whose assignee is the configured lane profile, with an
**external OAuth-only Claude Code launcher** that runs the exact model
``claude-fable-5[1m]`` with a read-only tool surface.

Everything else about dispatch is unchanged: the task is claimed, the workspace
is resolved, the same env pins (``HERMES_KANBAN_DB`` / ``_BOARD`` / ``_TASK`` /
``_RUN_ID``) are injected, the child writes to the same per-task worker log, and
its PID is recorded for crash detection. What differs is who closes the task:
the external launcher has no kanban tools, so the lane runs a thin **supervisor**
(``python -m hermes_cli.kanban_fable run``) that invokes the launcher and then
performs the ``complete`` / ``block`` transition itself.

Design rules (all load-bearing — do not relax without re-reading them):

* **Opt-in and default-disabled.** The lane engages only when the assignee
  profile's own ``config.yaml`` has ``kanban.fable_lane.enabled: true`` *and*
  the task's assignee matches the configured lane name (default ``fable``).
  With no such config every profile — including one literally named ``fable`` —
  keeps the normal Hermes worker path.
* **Fail closed.** Once the lane engages, a missing launcher, a missing
  ``claude`` binary, an auth failure, a nonzero exit, a timeout, or an output
  that does not prove the Fable model ran all end the same way: an error. The
  task is blocked (or the spawn is recorded as a failure). There is **no
  fallback to another model** — silently answering with the profile's normal
  Luna model is the exact outcome this lane exists to prevent.
* **No credential copying, no paid usage.** API-key / Bedrock / Vertex / proxy
  env vars are stripped from the child env (see :data:`BLOCKED_CREDENTIAL_ENV`)
  so the launcher can only use the interactive OAuth session, and the launcher's
  self-reported ``usage_credits_enabled`` must stay false.

Example profile-local ``config.yaml`` (``~/.hermes/profiles/fable/config.yaml``)::

    kanban:
      fable_lane:
        enabled: true
        launcher: ~/.hermes/profiles/coder/fable/fable_launcher.py
        # assignee: fable          # lane name; defaults to "fable"
        # max_turns: 5
        # timeout_seconds: 300

Read-only canary (no DB writes, no task, safe to run by hand)::

    python -m hermes_cli.kanban_fable canary --profile fable --workspace .
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

# The one model this lane is allowed to run. A config that asks for anything
# else is a misconfiguration, not a request to run something else.
FABLE_MODEL = "claude-fable-5[1m]"

DEFAULT_LANE_ASSIGNEE = "fable"
DEFAULT_MAX_TURNS = 5
DEFAULT_TIMEOUT_SECONDS = 300

# The launcher refuses to do anything without this flag. Passing it is the
# explicit local opt-in; it still enables no paid usage credits.
LAUNCHER_OPT_IN_FLAG = "--enable-local-canary"

# Stripped from the launcher's env. The launcher is OAuth-only by design and
# refuses to run when it sees an API-key / Bedrock path; removing these here
# means the dispatcher never hands a credential to the child in the first
# place, and a gateway that happens to export one cannot turn this lane into
# billed API usage.
BLOCKED_CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)

# The child gets only execution/runtime context. Claude Code reads its existing
# OAuth session from the user's home; no token environment variable is needed.
# An allowlist is safer than chasing every possible *_TOKEN / *_SECRET name.
ALLOWED_ENV = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP",
    "LANG", "LC_ALL", "LC_CTYPE", "TERM", "COLORTERM",
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE", "HERMES_LOG_LEVEL",
})

# Appended to every prompt this lane sends. The launcher already pins
# ``--allowedTools Read,Glob,Grep`` for prompt runs; this is the belt to that
# braces, stated in the prompt so the model's own plan matches its tool surface.
READ_ONLY_NOTICE = (
    "\n\n---\n"
    "READ-ONLY LANE: you are running with read-only tools (Read/Glob/Grep). "
    "Do not edit or write files, do not run shell commands, and do not touch "
    "credentials or external systems. Answer with your analysis only."
)

CANARY_PROMPT = "Reply exactly FABLE_LANE_CANARY_OK and nothing else."


class FableLaneUnavailable(RuntimeError):
    """The Fable lane cannot run this task — and must not be substituted.

    ``kind`` maps onto :func:`hermes_cli.kanban_db.block_task` block kinds so a
    timeout (retryable) and a missing launcher (human-only) route differently.
    """

    def __init__(self, message: str, *, kind: str = "capability") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class FableLaneConfig:
    """Resolved ``kanban.fable_lane`` settings for one assignee profile."""

    assignee: str
    launcher: Optional[Path]
    model: str = FABLE_MODEL
    max_turns: int = DEFAULT_MAX_TURNS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _profile_config(profile: Optional[str]) -> dict:
    """Load ``config.yaml`` as seen by *profile* (not by the dispatcher).

    The dispatcher runs from the gateway's root HERMES_HOME, so reading
    ``load_config()`` directly would consult the wrong file. Scope the read to
    the assignee's profile home the same way ``_resolve_worker_cli_toolsets``
    does, and fall back to the ambient config when the profile has no home of
    its own (test fixtures, ``default``).
    """
    from hermes_cli.config import load_config

    if not profile:
        try:
            return load_config() or {}
        except Exception:
            return {}

    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.profiles import resolve_profile_env

        home = resolve_profile_env(profile)
    except Exception:
        try:
            return load_config() or {}
        except Exception:
            return {}

    token = set_hermes_home_override(home)
    try:
        return load_config() or {}
    except Exception:
        return {}
    finally:
        reset_hermes_home_override(token)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def load_lane_config(profile: Optional[str]) -> Optional[FableLaneConfig]:
    """Return the enabled lane config for *profile*, or ``None``.

    ``None`` means "this profile does not use the Fable lane" — the caller
    keeps the normal worker path. It never means "the lane is broken": a lane
    that is enabled but misconfigured returns a config whose problems
    :func:`preflight` reports as :class:`FableLaneUnavailable`.
    """
    section = ((_profile_config(profile).get("kanban") or {}).get("fable_lane") or {})
    if not isinstance(section, dict) or not section.get("enabled"):
        return None

    raw_launcher = str(section.get("launcher") or "").strip()
    launcher = Path(os.path.expanduser(raw_launcher)) if raw_launcher else None
    return FableLaneConfig(
        assignee=str(section.get("assignee") or DEFAULT_LANE_ASSIGNEE).strip().lower(),
        launcher=launcher,
        model=str(section.get("model") or FABLE_MODEL).strip(),
        max_turns=_positive_int(section.get("max_turns"), DEFAULT_MAX_TURNS),
        timeout_seconds=_positive_int(
            section.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS
        ),
    )


def lane_for_assignee(assignee: Optional[str]) -> Optional[FableLaneConfig]:
    """Return the lane config when *assignee* is the configured lane, else None.

    Both halves of the opt-in are checked here: the assignee profile enables
    ``kanban.fable_lane`` in its own config, and the assignee name matches that
    config's lane name. Any other assignee dispatches normally.
    """
    if not assignee:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name

        canon = normalize_profile_name(assignee)
    except Exception:
        canon = str(assignee).strip().lower()
    cfg = load_lane_config(canon)
    if cfg is None or cfg.assignee != canon:
        return None
    return cfg


# ---------------------------------------------------------------------------
# Preflight / env / argv
# ---------------------------------------------------------------------------

def preflight(cfg: FableLaneConfig, *, env: Optional[dict] = None) -> str:
    """Validate that the lane can actually run. Returns the ``claude`` path.

    Raises :class:`FableLaneUnavailable` for every condition that would
    otherwise degrade into "ran something else": no launcher configured, the
    launcher file is gone, ``claude`` is not installed, or the config asks for
    a model this lane is not allowed to run.
    """
    if cfg.model != FABLE_MODEL:
        raise FableLaneUnavailable(
            f"lane model must be {FABLE_MODEL!r}, config asks for {cfg.model!r}"
        )
    if cfg.launcher is None:
        raise FableLaneUnavailable(
            "kanban.fable_lane.enabled is true but kanban.fable_lane.launcher is unset"
        )
    if not cfg.launcher.is_file():
        raise FableLaneUnavailable(f"launcher not found: {cfg.launcher}")
    claude = shutil.which("claude", path=(env or os.environ).get("PATH"))
    if not claude:
        raise FableLaneUnavailable(
            "`claude` executable not found on PATH; the Fable lane needs an "
            "authenticated local Claude Code install (OAuth, no API key)"
        )
    return claude


def sanitize_env(env: Optional[dict] = None) -> dict:
    """Return a minimal runtime env; OAuth is read from the local session."""
    source = dict(os.environ if env is None else env)
    clean = {name: value for name, value in source.items() if name in ALLOWED_ENV}
    dropped = [name for name in source if name not in clean]
    if dropped:
        _log.info(
            "kanban fable lane: stripped %s from worker env (OAuth-only lane)",
            ", ".join(dropped),
        )
    return clean


def supervisor_argv(
    task_id: str,
    workspace: str,
    *,
    board: Optional[str] = None,
    run_id: Optional[int] = None,
) -> list[str]:
    """Argv for the lane supervisor the dispatcher spawns for a task.

    Interpreter-bound (like ``_module_hermes_argv``) so it works from cron /
    systemd / launchd where the ``hermes`` shim may not be on PATH.
    """
    argv = [
        sys.executable, "-m", "hermes_cli.kanban_fable", "run",
        "--task", task_id,
        "--workspace", workspace,
    ]
    if board:
        argv.extend(["--board", board])
    if run_id is not None:
        argv.extend(["--run-id", str(run_id)])
    return argv


def launcher_argv(cfg: FableLaneConfig, prompt: str, workspace: str) -> list[str]:
    """Argv for one external-launcher invocation."""
    if cfg.launcher is None:  # pragma: no cover - preflight rejects this first
        raise FableLaneUnavailable("no launcher configured")
    return [
        sys.executable, str(cfg.launcher),
        LAUNCHER_OPT_IN_FLAG,
        "--prompt", prompt,
        "--cwd", workspace,
        "--max-turns", str(cfg.max_turns),
    ]


# ---------------------------------------------------------------------------
# Running the launcher
# ---------------------------------------------------------------------------

def _model_identifiers(payload: dict) -> list[str]:
    """Collect model ids the underlying Claude Code run reported, if any."""
    found: list[str] = []
    for check in payload.get("checks") or []:
        result = check.get("result")
        if not isinstance(result, dict):
            continue
        model = result.get("model")
        if isinstance(model, str) and model:
            found.append(model)
        usage = result.get("modelUsage")
        if isinstance(usage, dict):
            found.extend(str(k) for k in usage.keys() if k)
    return found


def verify_payload(payload: dict, cfg: FableLaneConfig) -> None:
    """Raise unless the launcher output proves a clean Fable-model run.

    This is the anti-silent-substitution check. A launcher that ran, exited 0,
    and answered with a *different* model is a failure for this lane, not a
    result.
    """
    if payload.get("model_requested") != FABLE_MODEL:
        raise FableLaneUnavailable(
            f"launcher reported model {payload.get('model_requested')!r}, "
            f"expected {FABLE_MODEL!r}"
        )
    if payload.get("oauth_only") is not True:
        raise FableLaneUnavailable("launcher did not prove OAuth-only execution")
    if payload.get("usage_credits_enabled") is not False:
        raise FableLaneUnavailable("launcher did not prove paid usage credits are disabled")
    checks = payload.get("checks") or []
    if not checks:
        raise FableLaneUnavailable("launcher produced no checks")
    for check in checks:
        if check.get("timeout"):
            raise FableLaneUnavailable(
                f"launcher check {check.get('name')!r} timed out", kind="transient",
            )
        if check.get("exit_code") != 0:
            stderr = (check.get("stderr") or "").strip().splitlines()
            tail = stderr[-1] if stderr else ""
            raise FableLaneUnavailable(
                f"launcher check {check.get('name')!r} exited "
                f"{check.get('exit_code')!r}: {tail}"
            )
    reported = _model_identifiers(payload)
    if FABLE_MODEL not in reported:
        raise FableLaneUnavailable(
            f"run did not prove exact model {FABLE_MODEL!r}; reported {reported!r} "
            "(non-Fable or unexpected model)"
        )
    if any(model != FABLE_MODEL for model in reported):
        raise FableLaneUnavailable(
            f"run reported non-Fable model(s) {reported!r}"
        )


def result_text(payload: dict) -> str:
    """Extract the model's answer from the launcher payload."""
    for check in reversed(payload.get("checks") or []):
        result = check.get("result")
        if isinstance(result, dict):
            for key in ("result", "raw_stdout"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(result, str) and result.strip():
            return result.strip()
    return ""


def run_launcher(
    cfg: FableLaneConfig,
    prompt: str,
    workspace: str,
    *,
    env: Optional[dict] = None,
) -> dict:
    """Invoke the external launcher once and return its verified payload."""
    preflight(cfg, env=env)
    argv = launcher_argv(cfg, prompt + READ_ONLY_NOTICE, workspace)
    child_env = sanitize_env(env)
    cwd = workspace if os.path.isdir(workspace) else None
    started = time.monotonic()
    _log.info(
        "kanban fable lane: invoking %s (model=%s, max_turns=%d, timeout=%ds)",
        cfg.launcher, FABLE_MODEL, cfg.max_turns, cfg.timeout_seconds,
    )
    try:
        proc = subprocess.run(  # noqa: S603 -- argv is a fixed list built above
            argv,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=child_env,
            timeout=cfg.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        raise FableLaneUnavailable(
            f"launcher exceeded {cfg.timeout_seconds}s", kind="transient",
        ) from None
    except FileNotFoundError as exc:
        raise FableLaneUnavailable(f"launcher could not be executed: {exc}") from None
    elapsed_ms = round((time.monotonic() - started) * 1000)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise FableLaneUnavailable(
            f"launcher exited {proc.returncode}: {tail[-1] if tail else '(no output)'}"
        )
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except json.JSONDecodeError:
        raise FableLaneUnavailable("launcher output was not JSON") from None
    if not isinstance(payload, dict):
        raise FableLaneUnavailable("launcher output was not a JSON object")
    if payload.get("status") == "disabled":
        raise FableLaneUnavailable(
            f"launcher is disabled: {payload.get('reason') or 'no reason given'}"
        )
    verify_payload(payload, cfg)
    payload["lane_elapsed_ms"] = elapsed_ms
    _log.info("kanban fable lane: launcher run OK in %dms", elapsed_ms)
    return payload


# ---------------------------------------------------------------------------
# Supervisor entry point (spawned by the dispatcher)
# ---------------------------------------------------------------------------

def run_task(
    task_id: str,
    workspace: str,
    *,
    board: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> int:
    """Run one kanban task through the Fable lane and close it out.

    Returns a process exit code. The kanban transition is written here because
    the external launcher has no kanban tools: success → ``complete``, any
    lane failure → ``block`` with the reason. Blocking (rather than leaving the
    claim to expire) is what keeps the failure visible instead of letting the
    task loop back to ``ready`` and get picked up by a different model.
    """
    from hermes_cli import kanban_db as kb

    conn = kb.connect(board=board)
    try:
        task = kb.get_task(conn, task_id)
        if task is None:
            _log.error("kanban fable lane: unknown task %s", task_id)
            return 2
        run_id = task.current_run_id
        if expected_run_id is not None and run_id != expected_run_id:
            _log.error(
                "kanban fable lane: stale supervisor for task %s (expected run %s, current %s)",
                task_id, expected_run_id, run_id,
            )
            return 2
        cfg = lane_for_assignee(task.assignee)
        if cfg is None:
            # Reached only if config changed between spawn and run. Fail
            # closed: this supervisor must never run a non-lane task.
            reason = (
                f"fable lane not enabled for assignee {task.assignee!r}; refusing "
                "to run the task with any other model"
            )
            _log.error("kanban fable lane: %s", reason)
            if not kb.block_task(
                conn, task_id, reason=reason, kind="capability",
                expected_run_id=run_id,
            ):
                return 2
            return 1
        if task.model_override:
            reason = (
                f"task pins model_override={task.model_override!r} but the fable "
                f"lane only runs {FABLE_MODEL}"
            )
            _log.error("kanban fable lane: %s", reason)
            if not kb.block_task(
                conn, task_id, reason=reason, kind="capability",
                expected_run_id=run_id,
            ):
                return 2
            return 1

        prompt = kb.build_worker_context(conn, task_id)
        try:
            payload = run_launcher(cfg, prompt, workspace)
        except FableLaneUnavailable as exc:
            reason = f"fable lane unavailable: {exc}"
            _log.error("kanban fable lane: task %s blocked — %s", task_id, exc)
            if not kb.block_task(
                conn, task_id, reason=reason, kind=exc.kind,
                expected_run_id=run_id,
            ):
                return 2
            return 1

        answer = result_text(payload)
        if not answer:
            reason = "fable lane produced an empty answer"
            _log.error("kanban fable lane: task %s blocked — %s", task_id, reason)
            if not kb.block_task(
                conn, task_id, reason=reason, kind="transient",
                expected_run_id=run_id,
            ):
                return 2
            return 1
        if not kb.complete_task(
            conn, task_id,
            result=answer,
            summary=answer,
            metadata={
                "lane": "fable",
                "model": FABLE_MODEL,
                "read_only": True,
                "launcher": str(cfg.launcher),
                "elapsed_ms": payload.get("lane_elapsed_ms"),
            },
            expected_run_id=run_id,
        ):
            return 2
        _log.info("kanban fable lane: task %s completed", task_id)
        return 0
    finally:
        conn.close()


def run_canary(
    *,
    profile: str = DEFAULT_LANE_ASSIGNEE,
    workspace: Optional[str] = None,
    prompt: Optional[str] = None,
) -> int:
    """Read-only smoke test of the lane. Touches no board and no task.

    Prints a JSON report and returns 0 only when the launcher ran the Fable
    model cleanly. Safe to run by hand at any time: read-only tools, no DB
    writes, no credentials passed to the child.
    """
    cfg = load_lane_config(profile)
    if cfg is None:
        print(json.dumps({
            "ok": False,
            "reason": f"kanban.fable_lane is not enabled for profile {profile!r}",
        }, indent=2))
        return 1
    root = str(Path(workspace or os.getcwd()).expanduser().resolve())
    try:
        claude = preflight(cfg)
        payload = run_launcher(cfg, prompt or CANARY_PROMPT, root)
    except FableLaneUnavailable as exc:
        print(json.dumps({"ok": False, "kind": exc.kind, "reason": str(exc)}, indent=2))
        return 1
    print(json.dumps({
        "ok": True,
        "model": FABLE_MODEL,
        "profile": profile,
        "claude": claude,
        "launcher": str(cfg.launcher),
        "workspace": root,
        "elapsed_ms": payload.get("lane_elapsed_ms"),
        "answer": result_text(payload)[:500],
    }, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("HERMES_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="hermes_cli.kanban_fable")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one kanban task through the Fable lane")
    run_p.add_argument("--task", required=True)
    run_p.add_argument("--workspace", required=True)
    run_p.add_argument("--board", default=None)
    run_p.add_argument("--run-id", type=int, default=None)

    canary_p = sub.add_parser("canary", help="read-only lane smoke test (no DB writes)")
    canary_p.add_argument("--profile", default=DEFAULT_LANE_ASSIGNEE)
    canary_p.add_argument("--workspace", default=None)
    canary_p.add_argument("--prompt", default=None)

    args = parser.parse_args(argv)
    if args.command == "run":
        return run_task(
            args.task, args.workspace, board=args.board, expected_run_id=args.run_id,
        )
    return run_canary(
        profile=args.profile, workspace=args.workspace, prompt=args.prompt,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
