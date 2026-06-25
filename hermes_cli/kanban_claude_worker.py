"""Direct Claude Code worker lane for the Hermes kanban dispatcher.

Hermes kanban is the shared work board; Claude Code is the coding executor.
The normal path routes a card to a Hermes *profile* worker (e.g. ``coder``)
which then invokes Claude Code interactively. This module adds a *direct*
lane: a card assigned to the sentinel assignee ``claude-code`` is dispatched
straight to a non-interactive ``claude -p`` run, with kanban context, and the
card is updated from the *verified* result.

Design constraints (see handoffs/claude-code-direct-kanban-worker.md):

* Print mode only — never the interactive ``/ecc:*`` slash workflows.
* Never ``--dangerously-skip-permissions`` in automatic dispatch.
* The runner — not Claude's self-report — is the source of truth for
  completion: it re-runs the task's verification command deterministically
  and only completes the card when that passes against a real diff.
* No commit / push / deploy / secret edits (enforced in the prompt; the
  runner only ever runs read-only git + the task's verification command).
* Behavioural knobs live in ``config.yaml`` under
  ``kanban.claude_code_worker``; nothing here reads secrets.

The dispatcher spawn hook (``hermes_cli.kanban_db._default_spawn``) launches
this module as a detached ``python -m hermes_cli.kanban_claude_worker``
subprocess; :func:`main` reads the same ``HERMES_KANBAN_*`` env vars a Hermes
worker receives and drives :func:`run_worker`.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Routing sentinel + behavioural defaults
# ---------------------------------------------------------------------------

# The single assignee value that routes a card to the direct Claude Code lane.
# Intentionally NOT a generic external-worker registry — the handoff's
# non-goals forbid speculative framework infrastructure. One sentinel keeps
# the change surgical and the fallback ``coder`` profile path untouched.
CLAUDE_CODE_WORKER_ASSIGNEE = "claude-code"

# Bounded, non-interactive defaults. Overridable via config.yaml
# ``kanban.claude_code_worker``.
DEFAULT_MAX_TURNS = 30
DEFAULT_PERMISSION_MODE = "acceptEdits"
DEFAULT_OUTPUT_FORMAT = "json"
DEFAULT_CLAUDE_BIN = "claude"

# Permission modes that must never reach the CLI in automatic mode. Mapping a
# requested ``bypassPermissions`` (the flag form is --dangerously-skip-
# permissions) silently down to the safe default is a hard guard, not a
# preference: the dispatcher runs unattended and cannot answer a prompt, so a
# bypass posture would let an automatic run take destructive actions with no
# human in the loop.
FORBIDDEN_PERMISSION_MODES = {"bypasspermissions", "bypass", "dangerously-skip-permissions"}

# Heartbeat cadence for long Claude runs so the dispatcher's stale-claim
# reclaim sees liveness orthogonal to the PID check.
HEARTBEAT_INTERVAL_SECONDS = 60


def is_external_claude_worker(assignee: Optional[str]) -> bool:
    """True when ``assignee`` routes to the direct Claude Code lane."""
    if not assignee:
        return False
    return assignee.strip().lower() == CLAUDE_CODE_WORKER_ASSIGNEE


# ---------------------------------------------------------------------------
# Prompt + command construction (pure, trivially testable)
# ---------------------------------------------------------------------------

_PROMPT_RULES = """\
Rules:
- Implement ONLY this kanban task: make the requested code change and nothing else.
- Keep the diff tightly scoped to the acceptance criteria.
- Do NOT commit, push, merge, deploy, publish, or change secrets.
- If the repo has unrelated dirty changes you did not make, stop and report a blocker instead of overwriting them.
- Do NOT run the verification command yourself, and do NOT use Bash just to verify your work.
  The Hermes runner runs the deterministic verification command from the task body
  after you exit and decides the outcome from the real diff — leave verification to it.
  Spending turns trying to run verification (especially when a tool is denied) only
  burns your turn budget; finish your edit and stop.

Final response MUST report:
- files changed
- what was changed (a brief summary)
- any blocker that prevented making the change
"""


def build_claude_code_prompt(task, workspace: str) -> str:
    """Render the non-interactive prompt handed to ``claude -p``.

    Includes the kanban task id/title/body, the absolute workspace path, the
    hard safety constraints, and the required final-response shape. ``task``
    is a :class:`hermes_cli.kanban_db.Task` (only ``.id/.title/.body`` read).
    """
    title = (getattr(task, "title", "") or "").strip()
    body = (getattr(task, "body", "") or "").strip()
    task_id = getattr(task, "id", "") or ""
    body_block = body if body else "(no body provided)"
    return (
        "You are executing one Hermes kanban task as a direct Claude Code worker.\n\n"
        "Kanban task:\n"
        f"- id: {task_id}\n"
        f"- title: {title}\n"
        "- body:\n"
        f"{body_block}\n\n"
        "Workspace:\n"
        f"{workspace}\n\n"
        f"{_PROMPT_RULES}"
    )


def extract_verification_command(body: Optional[str]) -> Optional[str]:
    """Pull the command(s) under a ``## Verification`` heading from a card body.

    Returns the joined non-empty, non-fence lines between the heading and the
    next ``##`` heading, or ``None`` when no verification section exists. The
    runner executes the result deterministically — it is the completion gate,
    so a missing section means "cannot self-verify" rather than "assume pass".
    """
    if not body:
        return None
    lines = body.splitlines()
    collecting = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().lower()
            if not collecting and heading.startswith("verification"):
                collecting = True
                continue
            if collecting:
                # Next heading ends the section.
                break
        if collecting:
            if stripped.startswith("```"):
                continue  # drop code-fence delimiters, keep their contents
            if stripped:
                collected.append(stripped)
    if not collected:
        return None
    return "\n".join(collected)


def build_claude_cmd(prompt: str, cfg: Optional[dict] = None) -> list[str]:
    """Build the bounded, non-interactive ``claude`` argv.

    Always uses print mode (``-p``), a bounded ``--max-turns``, a structured
    ``--output-format``, and a safe ``--permission-mode``. The argv is
    asserted to never contain ``--dangerously-skip-permissions``; a config
    that requests a bypass posture is downgraded to the safe default.
    """
    cfg = cfg or {}
    claude_bin = str(cfg.get("bin") or DEFAULT_CLAUDE_BIN).strip() or DEFAULT_CLAUDE_BIN

    try:
        max_turns = int(cfg.get("max_turns") or DEFAULT_MAX_TURNS)
    except (TypeError, ValueError):
        max_turns = DEFAULT_MAX_TURNS
    if max_turns < 1:
        max_turns = DEFAULT_MAX_TURNS

    output_format = str(cfg.get("output_format") or DEFAULT_OUTPUT_FORMAT).strip() or DEFAULT_OUTPUT_FORMAT

    mode = str(cfg.get("permission_mode") or DEFAULT_PERMISSION_MODE).strip()
    if mode.lower() in FORBIDDEN_PERMISSION_MODES:
        mode = DEFAULT_PERMISSION_MODE

    cmd = [
        claude_bin,
        "-p", prompt,
        "--output-format", output_format,
        "--max-turns", str(max_turns),
        "--permission-mode", mode,
    ]
    model = str(cfg.get("model") or "").strip()
    if model:
        cmd.extend(["--model", model])

    allowed_tools = cfg.get("allowed_tools")
    if isinstance(allowed_tools, (list, tuple)) and allowed_tools:
        cmd.extend(["--allowedTools", ",".join(str(t) for t in allowed_tools if t)])

    # Hard guard: this argv is run unattended. The bypass flag must never
    # appear regardless of how config was shaped above.
    assert "--dangerously-skip-permissions" not in cmd, (
        "refusing to build an automatic Claude Code command with "
        "--dangerously-skip-permissions"
    )
    return cmd


# ---------------------------------------------------------------------------
# Claude invocation + verification (injectable for tests)
# ---------------------------------------------------------------------------


@dataclass
class ClaudeOutcome:
    """Normalised result of one ``claude -p`` run."""
    returncode: int
    stdout: str = ""
    stderr: str = ""
    result_text: str = ""
    is_error: bool = False
    subtype: str = ""
    num_turns: Optional[int] = None
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        if self.timed_out or self.returncode != 0 or self.is_error:
            return False
        if self.subtype and self.subtype.lower().startswith("error"):
            return False
        return True


def parse_claude_json(stdout: str) -> dict:
    """Best-effort parse of ``claude --output-format json`` stdout.

    Tolerates leading/trailing noise by scanning for the last JSON object.
    Returns ``{}`` on failure so callers fall back to the exit code.
    """
    text = (stdout or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # Fallback: last balanced {...} block.
    start = text.rfind("{")
    while start != -1:
        try:
            obj = json.loads(text[start:])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            start = text.rfind("{", 0, start)
    return {}


def _default_run_claude(
    cmd: list[str],
    *,
    cwd: Optional[str],
    timeout: Optional[int],
    heartbeat: Optional[Callable[[], None]] = None,
) -> ClaudeOutcome:
    """Run ``claude`` to completion, emitting heartbeats while it works.

    Captures stdout (the JSON result) and stderr. Kills the run on timeout
    and reports it as a non-success outcome.
    """
    try:
        proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list
            cmd,
            cwd=cwd if (cwd and os.path.isdir(cwd)) else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return ClaudeOutcome(
            returncode=127,
            stderr=(
                f"`{cmd[0]}` executable not found on PATH. Install Claude Code "
                "or activate its environment before running the kanban dispatcher."
            ),
        )

    deadline = (time.time() + timeout) if timeout else None
    while True:
        try:
            remaining = None
            if deadline is not None:
                remaining = max(1, int(min(HEARTBEAT_INTERVAL_SECONDS, deadline - time.time())))
            else:
                remaining = HEARTBEAT_INTERVAL_SECONDS
            stdout, stderr = proc.communicate(timeout=remaining)
            outcome = ClaudeOutcome(returncode=proc.returncode, stdout=stdout or "", stderr=stderr or "")
            break
        except subprocess.TimeoutExpired:
            if heartbeat is not None:
                try:
                    heartbeat()
                except Exception:
                    pass
            if deadline is not None and time.time() >= deadline:
                proc.kill()
                stdout, stderr = proc.communicate()
                outcome = ClaudeOutcome(
                    returncode=proc.returncode if proc.returncode is not None else -1,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    timed_out=True,
                )
                break

    parsed = parse_claude_json(outcome.stdout)
    if parsed:
        outcome.result_text = str(parsed.get("result") or "")
        outcome.is_error = bool(parsed.get("is_error"))
        outcome.subtype = str(parsed.get("subtype") or "")
        nt = parsed.get("num_turns")
        outcome.num_turns = int(nt) if isinstance(nt, int) else None
    return outcome


@dataclass
class VerificationResult:
    ran: bool
    command: Optional[str] = None
    returncode: Optional[int] = None
    output: str = ""

    @property
    def passed(self) -> bool:
        return self.ran and self.returncode == 0


def _default_run_verification(command: str, *, cwd: Optional[str], timeout: int = 600) -> VerificationResult:
    """Run the task's verification command via the shell and capture output.

    ``shell=True`` is intentional: the command is operator-authored kanban
    card content (same trust level as the existing ``coder`` profile that
    runs it), not untrusted external input.
    """
    try:
        proc = subprocess.run(  # noqa: S602 -- operator-authored verification command
            command,
            cwd=cwd if (cwd and os.path.isdir(cwd)) else None,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(ran=True, command=command, returncode=124, output="verification timed out")
    except Exception as exc:  # pragma: no cover - defensive
        return VerificationResult(ran=True, command=command, returncode=1, output=f"verification error: {exc}")
    combined = (proc.stdout or "") + (proc.stderr or "")
    return VerificationResult(ran=True, command=command, returncode=proc.returncode, output=combined)


def git_diff_summary(workspace: str) -> tuple[str, list[str]]:
    """Return (``git diff --stat`` text, changed-file list); empty if not a repo."""
    if not workspace or not os.path.isdir(workspace):
        return "", []
    try:
        stat = subprocess.run(
            ["git", "-C", workspace, "diff", "--stat"],
            capture_output=True, text=True, timeout=30,
        )
        names = subprocess.run(
            ["git", "-C", workspace, "diff", "--name-only"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return "", []
    if stat.returncode != 0:
        return "", []
    changed = [ln.strip() for ln in (names.stdout or "").splitlines() if ln.strip()]
    return (stat.stdout or "").strip(), changed


# ---------------------------------------------------------------------------
# Worker driver
# ---------------------------------------------------------------------------


@dataclass
class WorkerResult:
    status: str  # completed | failed | blocked_manual_review | error
    detail: str = ""
    comment_body: str = ""
    auto_blocked: bool = False
    verification: Optional[VerificationResult] = None
    changed_files: list[str] = field(default_factory=list)


def _excerpt(text: str, limit: int = 4000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def run_worker(
    conn,
    task,
    workspace: str,
    *,
    board: Optional[str] = None,
    cfg: Optional[dict] = None,
    failure_limit: Optional[int] = None,
    run_claude: Optional[Callable[..., ClaudeOutcome]] = None,
    run_verification: Optional[Callable[..., VerificationResult]] = None,
    diff_summary: Optional[Callable[[str], tuple[str, list[str]]]] = None,
    author: str = CLAUDE_CODE_WORKER_ASSIGNEE,
) -> WorkerResult:
    """Drive one direct Claude Code task to a verified kanban transition.

    The runner — not Claude — owns verification and the outcome decision. It
    runs ``claude -p``, then gathers the real git diff and (when there is a
    diff) runs the task's ``## Verification`` command deterministically. The
    transition is decided from that evidence plus Claude's exit status:

    Claude exited cleanly:
      * non-empty diff + verification passes → complete;
      * verification fails → retryable failure (auto-blocks at the limit);
      * empty diff, or no verification command → block for manual review.

    Claude exited abnormally (nonzero / ``is_error`` / ``error_max_turns``):
      * no diff → retryable failure;
      * diff + verification fails → retryable failure;
      * diff + verification passes → block for manual review (never retry
        forever, never auto-complete an abnormal run — a human reviews);
      * diff + no verification command → block for manual review.

    ``run_claude`` / ``run_verification`` / ``diff_summary`` are injectable so
    unit tests never touch a real ``claude`` binary or shell.
    """
    from hermes_cli import kanban_db as kb

    cfg = cfg or {}
    run_claude = run_claude or _default_run_claude
    run_verification = run_verification or _default_run_verification
    diff_summary = diff_summary or git_diff_summary

    task_id = getattr(task, "id", "")
    run_id = getattr(task, "current_run_id", None)

    prompt = build_claude_code_prompt(task, workspace)
    cmd = build_claude_cmd(prompt, cfg)

    max_runtime = getattr(task, "max_runtime_seconds", None)
    try:
        max_runtime = int(max_runtime) if max_runtime else None
    except (TypeError, ValueError):
        max_runtime = None

    def _heartbeat() -> None:
        try:
            kb.heartbeat_worker(conn, task_id, expected_run_id=run_id)
        except Exception:
            pass

    print(f"[claude-worker] task={task_id} running: {shlex.join(cmd[:1] + ['-p', '<prompt>'] + cmd[3:])}", flush=True)
    outcome = run_claude(cmd, cwd=workspace, timeout=max_runtime, heartbeat=_heartbeat)

    claude_summary = (
        f"Claude Code run: exit={outcome.returncode}"
        f"{' (timed out)' if outcome.timed_out else ''}"
        f"{f', subtype={outcome.subtype}' if outcome.subtype else ''}"
        f"{f', turns={outcome.num_turns}' if outcome.num_turns is not None else ''}"
    )

    # ---- Gather deterministic evidence, REGARDLESS of how Claude exited.
    # The runner — not Claude — owns verification. Claude's exit status is one
    # signal, but the real diff plus the runner's own verification run are the
    # source of truth. An abnormal Claude run (e.g. it burned its turns being
    # denied Bash for a verification it was told NOT to run) must not discard a
    # real, verifiable diff. Verification only runs when there is a diff to
    # verify — there is nothing to check against an empty tree.
    stat_text, changed_files = diff_summary(workspace)
    verify_cmd = extract_verification_command(getattr(task, "body", None))
    vr: Optional[VerificationResult] = (
        run_verification(verify_cmd, cwd=workspace) if (verify_cmd and changed_files) else None
    )

    diff_block = f"**git diff --stat:**\n```\n{_excerpt(stat_text, 2000) or '(no changes)'}\n```"
    claude_block = f"**Claude result:**\n```\n{_excerpt(outcome.result_text or outcome.stdout, 2000)}\n```"
    verify_block = (
        f"**verification** (`{vr.command}`): exit={vr.returncode}\n```\n{_excerpt(vr.output, 2000)}\n```"
        if vr is not None
        else "**verification:** no `## Verification` command available to run"
    )

    def _complete() -> WorkerResult:
        body = (
            f"### Direct Claude Code worker — COMPLETE\n\n"
            f"{claude_summary}\n\nChanged files: {', '.join(changed_files)}\n\n"
            f"{verify_block}\n\n{diff_block}\n\n{claude_block}"
        )
        _safe_comment(conn, task_id, author, body)
        kb.complete_task(
            conn, task_id,
            result=f"claude worker: verified ({len(changed_files)} files, verify exit=0)",
            summary=_excerpt(outcome.result_text or "verified by direct Claude Code worker", 4000),
            metadata={
                "worker": CLAUDE_CODE_WORKER_ASSIGNEE,
                "changed_files": changed_files,
                "verification_command": vr.command if vr else None,
                "verification_exit": vr.returncode if vr else None,
                "num_turns": outcome.num_turns,
                "claude_exit": outcome.returncode,
            },
            expected_run_id=run_id,
        )
        return WorkerResult(
            status="completed", detail="verified", comment_body=body,
            verification=vr, changed_files=changed_files,
        )

    def _manual_review(reason: str, note: str) -> WorkerResult:
        body = (
            f"### Direct Claude Code worker — needs review\n\n"
            f"{claude_summary}\n\n{note}\n\n{verify_block}\n\n{diff_block}\n\n{claude_block}"
        )
        _safe_comment(conn, task_id, author, body)
        try:
            kb.block_task(conn, task_id, reason=f"claude worker: {reason}", expected_run_id=run_id)
        except Exception:
            kb.block_task(conn, task_id, reason=f"claude worker: {reason}")
        return WorkerResult(
            status="blocked_manual_review", detail=reason, comment_body=body,
            verification=vr, changed_files=changed_files,
        )

    def _retry(reason: str, header: str) -> WorkerResult:
        body = (
            f"### Direct Claude Code worker — {header}\n\n{claude_summary}\n\n"
            f"{verify_block}\n\n{diff_block}\n\n"
            f"**stderr excerpt:**\n```\n{_excerpt(outcome.stderr, 1500)}\n```\n\n{claude_block}"
        )
        _safe_comment(conn, task_id, author, body)
        auto_blocked = kb._record_task_failure(
            conn, task_id, f"claude worker: {reason}",
            outcome="claude_worker_failed",
            failure_limit=failure_limit,
            release_claim=True,
            end_run=True,
        )
        return WorkerResult(
            status="failed", detail=reason, comment_body=body,
            auto_blocked=bool(auto_blocked), verification=vr, changed_files=changed_files,
        )

    # ---- Claude exited cleanly ----------------------------------------
    if outcome.succeeded:
        # Verification is never optional for auto-completion.
        if not changed_files:
            return _manual_review(
                "Claude reported success but produced an empty diff",
                "Not auto-completing: Claude reported success but produced an empty diff.",
            )
        if vr is None:
            return _manual_review(
                "no verification command in task body",
                "Not auto-completing: a diff exists but the task has no `## Verification` "
                "command to confirm it.",
            )
        if vr.passed:
            return _complete()
        return _retry(f"verification failed (exit={vr.returncode})", "verification FAILED")

    # ---- Claude exited abnormally (nonzero / is_error / error_max_turns)
    # Inspect the diff anyway: the abnormal exit alone is not proof of failure.
    if not changed_files:
        # Nothing was produced -> genuinely a failed run; retry as before.
        return _retry(claude_summary, "FAILED")
    if vr is not None and vr.passed:
        # The decisive fix: Claude ended abnormally, but the files it left
        # behind verify against a real diff. Do NOT retry forever and do NOT
        # auto-complete an abnormal run — hand it to a human.
        return _manual_review(
            "abnormal Claude run but runner verification passed against a real diff",
            "Claude ended abnormally, but runner verification passed against a real diff; "
            "blocked for manual review.",
        )
    if vr is None:
        # A diff exists but there is no verification command to confirm it, and
        # the run was abnormal — retrying will only reproduce the abnormality.
        return _manual_review(
            "abnormal Claude run with a diff but no verification command",
            "Claude ended abnormally and left a diff, but the task has no `## Verification` "
            "command to confirm it; blocked for manual review.",
        )
    # A diff exists but verification fails -> retryable failure as before.
    return _retry(
        f"abnormal Claude run and verification failed (exit={vr.returncode})",
        "verification FAILED",
    )


def _safe_comment(conn, task_id: str, author: str, body: str) -> Optional[int]:
    from hermes_cli import kanban_db as kb
    try:
        return kb.add_comment(conn, task_id, author, body)
    except Exception as exc:  # pragma: no cover - never let commenting abort the transition
        print(f"[claude-worker] failed to add comment to {task_id}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Detached entrypoint
# ---------------------------------------------------------------------------


def _load_worker_cfg() -> dict:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
    except Exception:
        return {}
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    worker_cfg = kanban_cfg.get("claude_code_worker", {}) if isinstance(kanban_cfg, dict) else {}
    return worker_cfg if isinstance(worker_cfg, dict) else {}


def _resolve_failure_limit() -> Optional[int]:
    try:
        from hermes_cli.config import load_config_readonly
        kanban_cfg = (load_config_readonly() or {}).get("kanban", {})
        val = kanban_cfg.get("failure_limit") if isinstance(kanban_cfg, dict) else None
        return int(val) if val is not None else None
    except Exception:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    """Detached entrypoint: ``python -m hermes_cli.kanban_claude_worker``.

    Reads the kanban context the dispatcher pinned into the environment,
    opens the board-scoped DB, loads the task, and runs the worker.
    """
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    board = os.environ.get("HERMES_KANBAN_BOARD", "").strip() or None
    workspace = (
        os.environ.get("HERMES_KANBAN_WORKSPACE", "").strip()
        or os.environ.get("TERMINAL_CWD", "").strip()
        or os.getcwd()
    )
    if not task_id:
        print("[claude-worker] HERMES_KANBAN_TASK not set; nothing to do", file=sys.stderr)
        return 2

    from hermes_cli import kanban_db as kb

    cfg = _load_worker_cfg()
    failure_limit = _resolve_failure_limit()
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            print(f"[claude-worker] task {task_id} not found on board {board!r}", file=sys.stderr)
            return 2
        result = run_worker(
            conn, task, workspace,
            board=board, cfg=cfg, failure_limit=failure_limit,
        )
    print(f"[claude-worker] task={task_id} result={result.status} detail={result.detail}", flush=True)
    # 0 => terminal transition the dispatcher should treat as a clean exit
    # (completed or blocked-for-review). 1 => recorded a retryable failure.
    return 0 if result.status in ("completed", "blocked_manual_review") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
