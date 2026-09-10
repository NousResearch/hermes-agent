"""Gateway slash-command handlers for GatewayRunner: lifted out of ``gateway/run.py`` into a mixin
so ``self._handle_*_command`` keeps resolving via the MRO.  Cohesive clusters live in the sibling
mixins (``slash_commands_model/_session/_status/_goals``); this module keeps the shared helpers plus
the one-off commands.  run.py helpers are imported lazily."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import json
import logging
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from agent.i18n import t
from gateway.config import HomeChannel, Platform, PlatformConfig, persist_home_channel
from gateway.platforms.base import EphemeralReply
from gateway.platforms.event import MessageEvent
from gateway.session import AsyncSessionStore
from gateway.session_transcript import TranscriptReadError
from gateway.slash_commands_goals import GatewayGoalCommandsMixin
from gateway.slash_commands_model import GatewayModelCommandsMixin
from gateway.slash_commands_session import GatewaySessionCommandsMixin
from gateway.slash_commands_status import HISTORY_UNREADABLE, GatewayStatusCommandsMixin
from hermes_cli.config import atomic_config_write, cfg_get
from utils import atomic_json_write, is_truthy_value

logger = logging.getLogger("gateway.run")


# /rollback result keys -> i18n line for files the safe restore left alone.
_ROLLBACK_SKIP_LINES = (("skipped_user_edits", "gateway.rollback.kept_user_edits"),
                        ("skipped_oversize", "gateway.rollback.kept_oversize"),
                        ("failed_deletes", "gateway.rollback.failed_deletes"))

# /busy input modes -> (status-card behavior, set-confirmation behavior).
_BUSY_MODE_BEHAVIOR = {
    "queue": ("queues for next turn", "Messages will be queued for the next turn while Hermes is busy."),
    "steer": ("steers into current run (after next tool call)",
              "Messages will be steered into the current run (after the next tool call)."),
    "interrupt": ("interrupts current run", "Messages will interrupt the current run while Hermes is busy."),
}

# /diff argument -> diff mode (unknown args leave the mode unchanged).
_DIFF_MODE_BY_ARG = {**dict.fromkeys(("staged", "--staged", "cached", "--cached"), "staged"),
                     **dict.fromkeys(("all", "--all", "head"), "all"), "session": "session"}

# /voice subcommand -> stored mode (None = auto-TTS disabled), confirmation i18n key.
_VOICE_MODE_BY_ARG = {
    **dict.fromkeys(("on", "enable"), ("voice_only", "gateway.voice.enabled_voice_only")),
    **dict.fromkeys(("off", "disable"), ("off", "gateway.voice.disabled_text")),
    "tts": ("all", "gateway.voice.tts_enabled")}

# /footer argument -> new enabled state ("" toggles; anything else is a usage error).
_FOOTER_STATE_BY_ARG = {**dict.fromkeys(("on", "enable", "true", "1"), True),
                        **dict.fromkeys(("off", "disable", "false", "0"), False)}

# /approve modifier tokens -> approval choice (default "once").
_APPROVE_CHOICE_BY_ARG = {**dict.fromkeys(("always", "permanent", "permanently"), "always"),
                          **dict.fromkeys(("session", "ses"), "session")}

_PLATFORM_USAGE = ("Usage: /platform <list|pause|resume> [name]\n"
                   "  /platform list — show platform status\n"
                   "  /platform pause <name> — stop retrying a failing platform\n"
                   "  /platform resume <name> — re-queue a paused platform")

_WINDOWS_UPDATE_HELPER = """
import os, subprocess, sys
output_path, exit_code_path, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
env = dict(os.environ, PYTHONUNBUFFERED="1")
with open(output_path, "wb") as f:
    rc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).wait(timeout=3600)
with open(exit_code_path, "w", encoding="utf-8") as f:
    f.write(str(rc))
""".strip()

# Module-level tunables so tests can patch the exact constant instead of
# globally monkeypatching ``asyncio.wait_for``. Production values come from
# the spec: dispatcher waits up to 8.0s for the helper's ack; the outer
# tool waits up to 10.0s for the dispatcher's tuple.
_RESTART_DISPATCH_ACK_TIMEOUT_S = 8.0
_RESTART_OUTER_WORKER_TIMEOUT_S = 10.0


async def _shielded_wait_for_ack(
    ack_future: "asyncio.Future[bool]", timeout: float
) -> None:
    """Wait for ``ack_future`` with an outer CancelledError shield.

    Critical contract (spec section 一):
    * ``asyncio.wait_for`` cancels its inner awaitable when the timeout
      fires OR when the outer caller is cancelled. We do NOT want either
      to cancel ``ack_future`` itself — the helper may still write to it
      after we exit (the dispatcher has already observed the timeout /
      CancelledError and moved on to outcome_unknown).
    * ``asyncio.shield`` wraps the awaitable so a cancellation of the
      outer task does not propagate to the inner future.
    """
    await asyncio.wait_for(asyncio.shield(ack_future), timeout=timeout)


def _nested_dict(root: dict, *keys: str) -> dict:
    """Walk/create ``root[k1][k2]...`` as dicts, replacing any non-dict value on the path."""
    for k in keys:
        if not isinstance(root.get(k), dict):
            root[k] = {}
        root = root[k]
    return root


def _preview(text: str, limit: int = 60) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


def _execute(command: str, **ctx_kwargs):
    """Run *command* through the shared slash executor on the gateway surface."""
    from hermes_cli.slash_exec import CommandContext, execute_command
    return execute_command(command, CommandContext(surface="gateway", **ctx_kwargs))


def _restart_notify_payload(event: MessageEvent) -> dict:
    """Requester routing info so the new gateway process can notify them once back online."""
    source = event.source
    data = {"platform": source.platform.value if source.platform else None,
            "chat_id": source.chat_id, "chat_type": source.chat_type}
    if source.delivered_via_upstream_relay is True:
        data["delivered_via_upstream_relay"] = True
        data.update({k: getattr(source, k) for k in ("user_id", "scope_id") if getattr(source, k)})
    optional = (("thread_id", source.thread_id), ("message_id", event.message_id))
    data.update({k: v for k, v in optional if v})
    return data


def _spawn_detached_update(hermes_cmd, output_path, exit_code_path) -> None:
    """Spawn ``hermes update --gateway`` detached so it survives the gateway restart it may trigger.
    setsid is portable (works where ``systemd-run --user`` lacks a D-Bus session); ``--gateway``
    enables file-based IPC so interactive prompts are forwarded; PYTHONUNBUFFERED lets the gateway
    stream output live.  Windows has no setsid: an inline helper runs the updater as a module under
    this interpreter (not venv\\Scripts\\hermes.exe — that shim holds its own file open, and the
    update must replace it), redirects both outputs to one file and writes the exit code."""
    import shutil
    import subprocess
    if sys.platform == "win32":
        from hermes_cli._subprocess_compat import windows_detach_popen_kwargs
        subprocess.Popen(
            [sys.executable, "-c", _WINDOWS_UPDATE_HELPER, str(output_path), str(exit_code_path),
             sys.executable, "-m", "hermes_cli.main", "update", "--gateway"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **windows_detach_popen_kwargs())
        return
    hermes_cmd_str = " ".join(shlex.quote(part) for part in hermes_cmd)
    update_cmd = (
        f"PYTHONUNBUFFERED=1 {hermes_cmd_str} update --gateway"
        f" > {shlex.quote(str(output_path))} 2>&1; "
        # Avoid `status=$?`: `status` is read-only in zsh and this template is reused in
        # macOS/zsh operator wrappers, so keep it zsh-safe even though bash runs it here.
        f"rc=$?; printf '%s' \"$rc\" > {shlex.quote(str(exit_code_path))}")
    # Preferred: setsid creates a new session, fully detached; fallback start_new_session=True
    # calls os.setsid() in the child.
    setsid_bin = shutil.which("setsid")
    argv = [setsid_bin, "bash", "-c", update_cmd] if setsid_bin else ["bash", "-c", update_cmd]
    subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def _home_thread_from_source(source) -> Optional[str]:
    """The thread id /sethome should persist on the home target, or None.  Slack thread-per-message
    keying stamps a top-level message's own id as ``source.thread_id`` (a session key, not a
    location); persisting it would pin HOME to that ephemeral thread.  A thread id equal to the
    message's own id is synthetic and dropped; a real thread (id = parent's) is kept."""
    thread_id = getattr(source, "thread_id", None)
    if not thread_id:
        return None
    synthetic = (getattr(source, "platform", None) == Platform.SLACK and getattr(source, "message_id", None)
                 and str(thread_id) == str(source.message_id))
    return None if synthetic else str(thread_id)


from enum import Enum


class _RestartStage(Enum):
    PREPARING = "PREPARING"
    # ABORTING: caller has acquired the abort gate. helper MUST NOT enter
    # IN_FLIGHT from this state.
    ABORTING = "ABORTING"
    # IN_FLIGHT: helper has crossed the commit barrier and is about to /
    # has performed the irreversible handoff. Caller cannot rollback.
    IN_FLIGHT = "IN_FLIGHT"
    HANDOFF_COMMITTED = "HANDOFF_COMMITTED"
    # ABORTED: abort gate won, restart task done, marker rolled back.
    # Caller may retry.
    ABORTED = "ABORTED"
    # OUTCOME_UNKNOWN: helper entered IN_FLIGHT but final commit status
    # could not be confirmed (timeout, CancelledError, helper exception).
    # Marker MUST NOT be rolled back; can_retry MUST be False.
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    STOPPING = "STOPPING"


class _RestartStateBackup:
    """Transactional backup/rollback helper for restart state and files."""

    def __init__(self, request_id: str, notify_path: Path, dedup_path: Path, original_source: Any):
        self.request_id = request_id
        self.notify_path = notify_path
        self.dedup_path = dedup_path
        self.notify_existed = notify_path.exists()
        self.notify_content = notify_path.read_bytes() if self.notify_existed else None
        self.dedup_existed = dedup_path.exists()
        self.dedup_content = dedup_path.read_bytes() if self.dedup_existed else None
        self.original_source = original_source

    def rollback(self, runner: Any) -> bool:
        """Roll back flags and restore original marker files if uncommitted.

        CAS-safe: before restoring a previously-existing file, verifies the
        current file still belongs to this request_id. If another request
        has since replaced it, the current file is left intact.
        Returns True if rollback completed safely, False if an error occurred.
        """
        runner._restart_requested = False
        runner._restart_task_started = False
        runner._detached_restart_helper_started = False
        runner._draining = False
        runner._restart_command_source = self.original_source

        rollback_ok = True

        def _current_request_id(path: Path) -> Optional[str]:
            """Read the request_id from the current file, if any."""
            try:
                if path.exists():
                    return json.loads(path.read_text(encoding="utf-8")).get("request_id")
            except Exception:
                return None
            return None

        # Roll back notify file
        if self.notify_existed and self.notify_content is not None:
            # Only restore if the file still belongs to this request
            cur_id = _current_request_id(self.notify_path)
            if cur_id is None or cur_id == self.request_id:
                try:
                    self.notify_path.write_bytes(self.notify_content)
                except Exception as e:
                    logger.error("Failed to restore notify file during rollback: %s", e)
                    rollback_ok = False
        else:
            try:
                if self.notify_path.exists():
                    data = json.loads(self.notify_path.read_text(encoding="utf-8"))
                    if data.get("request_id") == self.request_id:
                        self.notify_path.unlink(missing_ok=True)
            except Exception:
                self.notify_path.unlink(missing_ok=True)

        # Roll back dedup file
        if self.dedup_existed and self.dedup_content is not None:
            cur_id = _current_request_id(self.dedup_path)
            if cur_id is None or cur_id == self.request_id:
                try:
                    self.dedup_path.write_bytes(self.dedup_content)
                except Exception as e:
                    logger.error("Failed to restore dedup file during rollback: %s", e)
                    rollback_ok = False
        else:
            try:
                if self.dedup_path.exists():
                    data = json.loads(self.dedup_path.read_text(encoding="utf-8"))
                    if data.get("request_id") == self.request_id:
                        self.dedup_path.unlink(missing_ok=True)
            except Exception:
                self.dedup_path.unlink(missing_ok=True)

        return rollback_ok


class _RestartFinalOutcome(Enum):
    """Settles what actually happened after the restart task completes."""
    UNKNOWN = "unknown"
    ABORTED = "aborted"
    COMMITTED = "committed"


class _RestartAckOutcome(Enum):
    """Three-state ack from helper to dispatcher (spec section 二).

    * ACCEPTED: the transaction was claimed by the helper — claim_handoff
      succeeded and the helper entered the safe, non-reentrant IN_FLIGHT
      state (P1 #71876). The dispatcher returns so the requesting turn can
      finish. This ack does NOT mean drain/launcher/stop completed; the
      terminal outcome is carried by stage / final_outcome.
    * NOT_STARTED: the launcher returned False early and the contract
      guarantees no side-effect (stage is ABORTED, task is done, marker
      was rolled back). The caller may retry.
    * OUTCOME_UNKNOWN: the launcher raised an exception, or the result
      could not be confirmed. The dispatcher MUST NOT cancel the restart
      task or rollback the marker. can_retry=False.
    """
    ACCEPTED = "accepted"
    NOT_STARTED = "not_started"
    OUTCOME_UNKNOWN = "outcome_unknown"


class _LauncherResult(Enum):
    """Result of _launch_detached_restart_command() — explicit three-state
    mapping per spec section 三.

    * NOT_STARTED: Popen was not invoked (e.g. early validation failed
      before fork). Safe to rollback the marker.
    * STARTED: the detached watcher process is running. The marker
      CANNOT be rolled back; outcome must be outcome_unknown or accepted.
    * UNKNOWN: launch was attempted but the result cannot be confirmed
      (Popen raised after fork, helper raised BaseException before
      recording the result, helper task was cancelled mid-launch). The
      caller MUST treat this as outcome_unknown.
    """
    NOT_STARTED = "not_started"
    STARTED = "started"
    UNKNOWN = "unknown"


class _TransitionResult(Enum):
    """Return value for atomic terminal transitions.

    * TRANSITIONED: the transition was applied successfully.
    * ALREADY_COMPLETE: the transaction was already at the target
      terminal stage (idempotent).
    * LOST_RACE: another competitor reached a different terminal
      stage first (caller must not write ack, rollback, or retry).
    """
    TRANSITIONED = "transitioned"
    ALREADY_COMPLETE = "already_complete"
    LOST_RACE = "lost_race"


class _RestartTransaction:
    """Per-request state holder for a single restart attempt.

    Concurrency contract (spec sections 二 / 三):

    * A single asyncio.Lock (``_state_lock``) protects stage
      transitions. Locks are held only inside a tiny critical section
      (``claim_*`` and ``_complete_atomic``) — NEVER across Popen, detached
      helpers, sleep, stop() or any external / awaiting operation.
    * Two parties race: CALLER (claim_abort) and HELPER (claim_handoff).
      The first transition PREPARING → {ABORTING, IN_FLIGHT} wins; the
      other side's claim_* must return False.
    * Terminal transitions are atomic: ``_complete_atomic`` updates stage,
      final_outcome, launcher_result, writes the ack_future, and sets
      final_outcome_event — all inside a single critical section.
    * ack_outcome is a three-state future (``_RestartAckOutcome``) so the
      dispatcher can distinguish NOT_STARTED from OUTCOME_UNKNOWN.
    """

    def __init__(
        self,
        request_id: str,
        backup: _RestartStateBackup,
        loop: asyncio.AbstractEventLoop,
        detached: bool,
        via_service: bool,
    ) -> None:
        self.request_id = request_id
        self.backup = backup
        self.loop = loop
        self.detached = detached
        self.via_service = via_service
        self.stage: _RestartStage = _RestartStage.PREPARING
        # Three-state ack future. Shielded by the dispatcher's wait_for
        # so an outer CancelledError never cancels the future itself.
        #
        # P1 #71876: the helper writes ACCEPTED here immediately after
        # claim_handoff — that is the *claimed* acknowledgement, meaning
        # "transaction claimed; restart will proceed". It is NOT a terminal
        # signal. The terminal outcome is carried by stage / final_outcome
        # (set by complete_*). A later complete_* transition therefore never
        # rewrites the ack_future (it is already ACCEPTED), and an
        # ACCEPTED-then-UNKNOWN/NOT_STARTED sequence is NOT an ack conflict
        # — it is the normal claimed-then-failed path.
        self.ack_future: asyncio.Future[_RestartAckOutcome] = loop.create_future()
        self.claimed_ack: bool = False
        self.restart_task: Optional[asyncio.Task] = None
        self.final_outcome: _RestartFinalOutcome = _RestartFinalOutcome.UNKNOWN
        self.final_outcome_event: asyncio.Event = asyncio.Event()
        self._state_lock: asyncio.Lock = asyncio.Lock()
        # Mapping: helper-internal launcher outcome. None until recorded.
        self.launcher_result: Optional[_LauncherResult] = None

    def is_committed(self) -> bool:
        return self.stage == _RestartStage.HANDOFF_COMMITTED

    def is_aborted(self) -> bool:
        return self.stage == _RestartStage.ABORTED

    def is_unknown(self) -> bool:
        return self.stage == _RestartStage.OUTCOME_UNKNOWN

    def is_pre_commit(self) -> bool:
        return self.stage in (
            _RestartStage.PREPARING,
            _RestartStage.ABORTING,
            _RestartStage.ABORTED,
        )

    def in_flight(self) -> bool:
        return self.stage in (
            _RestartStage.IN_FLIGHT,
            _RestartStage.HANDOFF_COMMITTED,
            _RestartStage.OUTCOME_UNKNOWN,
            _RestartStage.STOPPING,
        )

    def can_retry(self) -> bool:
        """True only when the caller owns the abort and may safely retry.

        ABORTED means the abort gate won AND the restart task is fully
        done AND the marker was rolled back. Other stages forbid retry.
        """
        return self.stage == _RestartStage.ABORTED

    def set_final_outcome(self, outcome: _RestartFinalOutcome) -> None:
        """Pre-commit rejection path: set outcome without state lock.
        Only used when request_restart returned False and no complete_*
        was called.
        """
        self.final_outcome = outcome
        try:
            self.final_outcome_event.set()
        except Exception:
            pass

    # ---- ack helpers ----------------------------------------------------

    def try_write_ack(self, outcome: _RestartAckOutcome) -> bool:
        """Helper-side: write ack only if the future has not been set or
        cancelled. Returns True on success, False if the future was
        already done (cancelled or previously set). Never raises.
        """
        if self.ack_future.done():
            return False
        try:
            self.ack_future.set_result(outcome)
            if outcome is _RestartAckOutcome.ACCEPTED:
                # P1 #71876: the claimed acknowledgement (written right
                # after claim_handoff) is recorded so a later terminal
                # transition (complete_unknown / complete_not_started)
                # knows the ack already carries the claimed signal and
                # must not raise an "ack conflict".
                self.claimed_ack = True
            return True
        except asyncio.InvalidStateError:
            return False

    # ---- short-critical-section state transitions ---------------------

    async def claim_abort(self) -> bool:
        """Caller-side: try to win the abort race.

        Holds the state lock only for the duration of the stage
        transition. The lock is released before the caller cancels the
        restart task or awaits it. The helper, observing ABORTING after
        this returns, must refuse to launch.
        """
        async with self._state_lock:
            if self.stage is not _RestartStage.PREPARING:
                return False
            self.stage = _RestartStage.ABORTING
            return True

    async def claim_handoff(self) -> bool:
        """Helper-side: try to win the handoff race.

        Holds the lock only for the PREPARING → IN_FLIGHT transition.
        The lock is released BEFORE the helper invokes any external
        operation (Popen, service signal, etc.).
        """
        async with self._state_lock:
            if self.stage is not _RestartStage.PREPARING:
                return False
            self.stage = _RestartStage.IN_FLIGHT
            return True

    # ---- atomic terminal transition (single critical section) -----------

    _TERMINAL_STAGES = frozenset({
        _RestartStage.ABORTED,
        _RestartStage.HANDOFF_COMMITTED,
        _RestartStage.OUTCOME_UNKNOWN,
    })

    async def _complete_atomic(
        self,
        expected_source: _RestartStage,
        target_stage: _RestartStage,
        target_final_outcome: _RestartFinalOutcome,
        target_launcher_result: Optional[_LauncherResult],
        ack_outcome: _RestartAckOutcome,
    ) -> _TransitionResult:
        """Atomic terminal transition — all synchronous operations inside
        a single critical section.

        Inside the lock:
          1. Verifies source stage.
          2. Checks existing ack does not conflict.
          3. Updates stage, final_outcome, launcher_result.
          4. Writes ack_future.
          5. Sets final_outcome_event.

        Returns TRANSITIONED, ALREADY_COMPLETE, or LOST_RACE.
        Raises RuntimeError on impossible source stage.
        """
        async with self._state_lock:
            # Already at the target terminal stage — idempotent.
            if self.stage == target_stage:
                return _TransitionResult.ALREADY_COMPLETE

            # Another terminal stage already won — lost the race.
            if self.stage in self._TERMINAL_STAGES:
                return _TransitionResult.LOST_RACE

            # Impossible source stage.
            if self.stage is not expected_source:
                raise RuntimeError(
                    f"Invalid transition: {self.stage.value} → "
                    f"{target_stage.value} "
                    f"(expected {expected_source.value}) "
                    f"(request_id={self.request_id})"
                )

            # Check ack_future state before modifying state.
            if self.ack_future.cancelled():
                raise RuntimeError(
                    f"ack future cancelled: cannot write {ack_outcome.value} "
                    f"(request_id={self.request_id})"
                )
            ack_already_set = self.ack_future.done()
            if ack_already_set:
                existing = self.ack_future.result()
                if existing is not ack_outcome and not self.claimed_ack:
                    raise RuntimeError(
                        f"ack conflict: existing={existing.value}, "
                        f"attempted={ack_outcome.value} "
                        f"(request_id={self.request_id})"
                    )
                # Same ack already set, OR the claimed acknowledgement
                # (ACCEPTED written right after claim_handoff, P1 #71876).
                # The claimed ack is NOT a terminal signal: we must still
                # apply the terminal state transition below
                # (stage / final_outcome / launcher_result) so the
                # transaction reaches its true terminal stage. The ack write
                # itself is skipped because the future is already done.

            # Update state.
            self.stage = target_stage
            self.final_outcome = target_final_outcome
            self.launcher_result = target_launcher_result

            if not ack_already_set:
                # Write ack (synchronous, inside lock).
                # The future is guaranteed pending at this point (checked above).
                self.ack_future.set_result(ack_outcome)

            # Set event (synchronous, inside lock).
            self.final_outcome_event.set()

            return _TransitionResult.TRANSITIONED

    async def complete_claimed_abort(self) -> _TransitionResult:
        """Caller-side: ABORTING → ABORTED. Publishes ack NOT_STARTED."""
        return await self._complete_atomic(
            expected_source=_RestartStage.ABORTING,
            target_stage=_RestartStage.ABORTED,
            target_final_outcome=_RestartFinalOutcome.ABORTED,
            target_launcher_result=None,
            ack_outcome=_RestartAckOutcome.NOT_STARTED,
        )

    async def complete_not_started(self) -> _TransitionResult:
        """Helper-side: IN_FLIGHT → ABORTED (no side-effect).
        Publishes ack NOT_STARTED.
        """
        return await self._complete_atomic(
            expected_source=_RestartStage.IN_FLIGHT,
            target_stage=_RestartStage.ABORTED,
            target_final_outcome=_RestartFinalOutcome.ABORTED,
            target_launcher_result=_LauncherResult.NOT_STARTED,
            ack_outcome=_RestartAckOutcome.NOT_STARTED,
        )

    async def complete_started(self) -> _TransitionResult:
        """Helper-side: IN_FLIGHT → HANDOFF_COMMITTED.
        Publishes ack ACCEPTED.
        """
        return await self._complete_atomic(
            expected_source=_RestartStage.IN_FLIGHT,
            target_stage=_RestartStage.HANDOFF_COMMITTED,
            target_final_outcome=_RestartFinalOutcome.COMMITTED,
            target_launcher_result=_LauncherResult.STARTED,
            ack_outcome=_RestartAckOutcome.ACCEPTED,
        )

    async def complete_unknown(self) -> _TransitionResult:
        """Helper/caller-side: IN_FLIGHT → OUTCOME_UNKNOWN.
        Publishes ack OUTCOME_UNKNOWN.
        """
        return await self._complete_atomic(
            expected_source=_RestartStage.IN_FLIGHT,
            target_stage=_RestartStage.OUTCOME_UNKNOWN,
            target_final_outcome=_RestartFinalOutcome.UNKNOWN,
            target_launcher_result=_LauncherResult.UNKNOWN,
            ack_outcome=_RestartAckOutcome.OUTCOME_UNKNOWN,
        )

    async def cancel_restart_task_and_await(self) -> bool:
        """Caller-side (must have already won claim_abort): cancel the
        restart task and await its terminal state.

        Returns True if the task is done after awaiting; False if it
        remains running (the helper may have crossed IN_FLIGHT in the
        meantime — in which case the caller MUST treat the outcome as
        unknown).
        """
        task = self.restart_task
        if task is None:
            return True
        if task.done():
            return True
        task.cancel()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        return task.done()

    def __repr__(self) -> str:
        return (
            f"_RestartTransaction(request_id={self.request_id}, "
            f"stage={self.stage})"
        )


def _read_marker_payload(path: Path) -> Optional[dict]:
    """Read and parse a JSON marker file, returning None on any failure."""
    try:
        raw = path.read_bytes()
        return json.loads(raw.rstrip(b" \t\r\n\0"))
    except Exception:
        return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to a file using a tempfile + rename."""
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)


class GatewaySlashCommandsMixin(
    GatewayModelCommandsMixin,
    GatewaySessionCommandsMixin,
    GatewayStatusCommandsMixin,
    GatewayGoalCommandsMixin):
    """In-session slash-command handlers for GatewayRunner (plus the helpers the sibling mixins share)."""
    async_session_store: AsyncSessionStore

    # ------------------------------------------------------------------ shared helpers
    def _cached_agent_for(self, session_key: str, *, lockless_fallback: bool = False):
        """Peek the cached AIAgent for *session_key* without evicting it, or None. Entries are
        ``(agent, signature, ...)`` tuples (bare agents from test doubles accepted). Historical callers
        read the cache ONLY under ``_agent_cache_lock`` and got None when a fixture that skipped
        ``__init__`` had no lock; the manual codex ``/compress`` path was the one exception that read
        lock-free (``lockless_fallback=True``)."""
        cache = getattr(self, "_agent_cache", None)
        lock = getattr(self, "_agent_cache_lock", None)
        if cache is None or (lock is None and not lockless_fallback):
            return None
        try:
            if lock:
                with lock:
                    entry = cache.get(session_key)
            else:
                entry = cache.get(session_key)
        except Exception:
            return None
        return (entry[0] if entry else None) if isinstance(entry, (tuple, list)) else entry or None

    def _resident_agent_for(self, session_key: str):
        """The live running agent for *session_key*, else the cached one, else None. The pending
        sentinel (a run that is starting) never counts as a usable agent."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        agent = self._running_agents.get(session_key)
        if agent is not None and agent is not _AGENT_PENDING_SENTINEL:
            return agent
        return self._cached_agent_for(session_key)

    @staticmethod
    def _session_db_unavailable_reply() -> str:
        from hermes_state import format_session_db_unavailable
        return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

    def _reply_metadata(self, event: MessageEvent):
        """Thread/reply metadata for an outbound send anchored on *event*."""
        return self._thread_metadata_for_source(event.source, self._reply_anchor_for_event(event))

    def _adapter_and_key_for(self, event: MessageEvent):
        """``(adapter, session_key)`` for the event's source, either None when no source."""
        if not event.source:
            return None, None
        return self.adapters.get(event.source.platform), self._session_key_for_source(event.source)

    def _telegramized_command_reply(self, event: MessageEvent, text: str) -> str:
        from gateway.run import _telegramize_command_mentions
        return _telegramize_command_mentions(text, getattr(getattr(event, "source", None), "platform", None))

    def _checkpoint_manager(self):
        """A CheckpointManager from gateway config, or None when checkpoints are disabled."""
        from gateway.run import _checkpoint_agent_kwargs, _load_gateway_config
        from tools.checkpoint_manager import CheckpointManager
        cp = _checkpoint_agent_kwargs(_load_gateway_config())
        if not cp["checkpoints_enabled"]:
            return None
        # AIAgent kwargs are ``checkpoint_<field>``; CheckpointManager takes the bare field names.
        fields = {k[len("checkpoint_"):]: v for k, v in cp.items() if k.startswith("checkpoint_")}
        return CheckpointManager(enabled=True, **fields)

    def _write_approval_setter(self, section: str, event: MessageEvent):
        """``set_mode_fn`` for /memory and /skills: persist ``<section>.write_approval``. Raw read is
        correct for the write-back round-trip (merged defaults must not be persisted back to the
        user's file); the cached agent is dropped so the setting takes effect next message."""
        from gateway.run import _gateway_config_home
        # Persist to config (default) unless --session opted out, mirroring the text /model command path
        # above so a picked model survives across sessions like a typed one (#49066).
        from hermes_cli.config import read_user_config_raw
        config_path = _gateway_config_home() / "config.yaml"
        session_key = self._session_key_for_source(event.source)

        def _set_approval(enabled: bool):
            user_config = read_user_config_raw(config_path)
            user_config.setdefault(section, {})["write_approval"] = bool(enabled)
            atomic_config_write(config_path, user_config)
            # Evict any cached agent for this session so the next message rebuilds with the correct
            # session_id end-to-end — mirrors /branch and /reset. Without this, the cached AIAgent (and its
            # memory provider, which cached `_session_id` during initialize()) keeps writing into the wrong
            # session's record. See #6672.
            self._evict_cached_agent(session_key)
        return _set_approval

    async def _deliver_approval_confirmation(self, event: MessageEvent, confirmation_text: str, verb: str):
        """Return *confirmation_text* for normal delivery, or push it on native-streaming adapters
        (WeCom msgtype:"stream"), which need it sent directly with control-lane metadata (reliable
        proactive send, not the finalized reply stream). ``is not True``: mocks auto-create attrs."""
        source = event.source
        adapter = self.adapters.get(source.platform)
        if adapter:
            adapter.resume_typing_for_chat(source.chat_id)  # agent is about to continue
        if getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False) is not True:
            return confirmation_text
        if adapter:
            try:
                await adapter.send(
                    source.chat_id, confirmation_text, reply_to=event.message_id,
                    metadata={"is_approval_prompt": True, "force_proactive_send": True})
            except Exception as exc:
                logger.warning("Failed to send /%s confirmation to %s: %s", verb, source.chat_id,
                               exc, exc_info=True)
        return None

    def _typed_command_prefix_for(self, platform) -> str:
        """The prefix users can always type to reach Hermes commands (adapter ``typed_command_prefix``,
        default "/"). Slack and Matrix use "!" because typed "/" is blocked/reserved there; their
        adapters rewrite "!command" to "/command"."""
        adapter = self.adapters.get(platform) if getattr(self, "adapters", None) else None
        return getattr(adapter, "typed_command_prefix", "/") if adapter is not None else "/"

    def _terminal_cwd(self) -> str:
        from tools.terminal_scope import terminal_env
        return terminal_env("TERMINAL_CWD", str(Path.home()))

    @staticmethod
    def _display_config_target(event: MessageEvent):
        """``(config.yaml path, platform config key)`` for the per-platform display settings."""
        from gateway.run import _gateway_config_home, _platform_config_key
        return _gateway_config_home() / "config.yaml", _platform_config_key(event.source.platform)

    async def _handle_profile_command(self, event: MessageEvent) -> str:
        """Handle /profile — show the profile serving this source and its home.  On a multiplexed
        gateway the process-level profile is the multiplexer's own ("default" in every chat), so
        with ``multiplex_profiles`` on report ``source.profile`` and resolve home under that
        profile's runtime scope; when off the stamp is ignored, mirroring ``_run_agent``."""
        from hermes_constants import display_hermes_home
        source = getattr(event, "source", None)
        profile_name = display = ""
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            profile_name = (getattr(source, "profile", "") or "").strip()
            try:
                from gateway.run import _profile_runtime_scope
                with _profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                    display = display_hermes_home()
            except Exception:
                display = display_hermes_home()

        # Shared executor resolves process-level fallbacks; the multiplexed per-source overrides
        # (when any) ride in via options.
        reply = _execute("profile", options={"profile_name": profile_name, "home_display": display})
        return "\n".join([t("gateway.profile.header", profile=reply.data["profile"]),
                          t("gateway.profile.home", home=reply.data["home"])])

    async def _handle_whoami_command(self, event: MessageEvent) -> str:
        """Handle /whoami — platform, DM-vs-group scope, tier and runnable commands (always allowed)."""
        from gateway.slash_access import policy_for_source
        source = event.source
        policy = policy_for_source(self.config, source)
        platform = source.platform.value if source and source.platform else "?"
        chat_type = ((source.chat_type if source else "") or "dm").lower()
        scope = "DM" if chat_type in {"dm", "direct", "private", ""} else "group/channel"
        user_id = (source.user_id if source else None) or "?"
        head = f"**You** — {platform} ({scope})\nUser ID: `{user_id}`\n"
        if not policy.enabled:
            return head + "Tier: unrestricted (no admin list configured for this scope)\nSlash commands: all available"
        if policy.is_admin(user_id):
            return head + "Tier: **admin**\nSlash commands: all available"
        # Non-admin: floor first (mirrors slash_access._ALWAYS_ALLOWED_FOR_USERS), then operator
        # additions, deduped in order.
        runnable = list(dict.fromkeys(["help", "whoami"] + sorted(policy.user_allowed_commands)))
        runnable_str = ", ".join(f"/{c}" for c in runnable) if runnable else "(none)"
        return head + f"Tier: user\nSlash commands you can run: {runnable_str}"

    async def _handle_kanban_command(self, event: MessageEvent) -> str:
        """Handle /kanban — delegate to the shared kanban CLI (DB work in a thread pool). Allowed
        while an agent runs: the board is profile-agnostic and never touches agent state."""
        from hermes_cli.kanban import run_slash

        # Strip the leading "/kanban" (with or without slash), leaving args.
        text = (event.text or "").strip().lstrip("/")
        if text.startswith("kanban"):
            text = text[len("kanban"):].lstrip()
        requested_board = action = None
        tokens = iter(shlex.split(text) if text else [])
        for tok in tokens:  # leading --board/--board=<b> options, then the action verb
            if tok == "--board":
                requested_board = next(tokens, requested_board)
            elif tok.startswith("--board="):
                requested_board = tok.split("=", 1)[1]
            else:
                action = tok
                break
        try:
            output = await asyncio.to_thread(run_slash, text)
        except Exception as exc:  # pragma: no cover - defensive
            return t("gateway.kanban.error_prefix", error=exc)

        # Auto-subscribe on create, parsing the task id from the CLI's standard success line
        # ("Created t_abcd  (ready, ...)"). With --json there is no such line, so a scripting user
        # gets no subscription and can call /kanban notify-subscribe explicitly.
        m = re.search(r"Created\s+(t_[0-9a-f]+)\b", output) if action == "create" and output else None
        if m:
            task_id = m.group(1)
            try:
                if await self._kanban_auto_subscribe(event, task_id, requested_board):
                    output = output.rstrip() + "\n" + t("gateway.kanban.subscribed_suffix", task_id=task_id)
            except Exception as exc:
                logger.warning("kanban create auto-subscribe failed: %s", exc)

        # Gateway messages have practical length caps; truncate long listings.
        if len(output) > 3800:
            output = output[:3800] + "\n" + t("gateway.kanban.truncated_suffix")
        return output or t("gateway.kanban.no_output")

    async def _kanban_auto_subscribe(self, event: MessageEvent, task_id: str, requested_board) -> bool:
        """Subscribe the event's chat to *task_id* notifications (notify+wake). False when the
        source has no platform/chat to route back to."""
        source = event.source

        def _field(name: str) -> Optional[str]:
            return str(getattr(source, name, "") or "") or None
        platform = getattr(source, "platform", None)
        platform_str = (platform.value if hasattr(platform, "value") else str(platform or "")).lower()
        chat_id, chat_type = _field("chat_id"), _field("chat_type")
        delivery_metadata = self._reply_metadata(event) or None
        if isinstance(delivery_metadata, dict) and chat_type:
            delivery_metadata.setdefault("chat_type", chat_type)
        if not (platform_str and chat_id):
            return False

        def _sub():
            from hermes_cli import kanban_db as _kb
            from hermes_cli import kanban_db_connect as _kbc
            from hermes_cli import kanban_db_notify as _kbn
            conn = _kbc.connect(board=requested_board)
            try:
                _kbn.add_notify_sub(
                    conn, task_id=task_id, platform=platform_str, chat_id=chat_id, chat_type=chat_type,
                    thread_id=_field("thread_id"), user_id=_field("user_id"),
                    # Also persist the stable alt id (Signal UUID, Feishu union_id): build_session_key
                    # keys the participant on ``user_id_alt or user_id``, so a replayed wake rebuilds
                    # the same session key only when the alt id survives the round-trip.
                    user_id_alt=_field("user_id_alt"),
                    notifier_profile=_field("profile") or getattr(self, "_kanban_notifier_profile", None) or self._active_profile_name(),
                    # Subscribing from chat: deliver the passive message and wake the destination agent.
                    delivery_mode="notify+wake", delivery_metadata=delivery_metadata)
            finally:
                conn.close()
        await asyncio.to_thread(_sub)
        return True

    async def _handle_stop_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /stop command - interrupt a running agent.  A truly hung agent (blocked thread
        never checking _interrupt_requested) is caught by the early intercept in _handle_message();
        this handler runs via normal dispatch or as a fallback, and force-cleans the session lock in
        all cases.  The session is preserved so the user can continue."""
        from gateway.run import _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_key = session_entry.session_key

        async def _stop(key: str, invalidation_reason: str) -> None:
            await self._interrupt_and_clear_session(
                key, source, interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason=invalidation_reason)
        agent = self._running_agents.get(session_key)
        if agent is _AGENT_PENDING_SENTINEL:  # force-clean the sentinel so the session is unlocked
            await _stop(session_key, "stop_command_pending")
            logger.info("STOP (pending) for session %s — sentinel cleared", session_key)
            return EphemeralReply(t("gateway.stop.stopped_pending"))
        if agent:  # force-clean the session lock so a truly hung agent doesn't keep it forever
            await _stop(session_key, "stop_command_handler")
            return EphemeralReply(t("gateway.stop.stopped"))

        # No run under the caller's own key. In a per-user thread (thread_sessions_per_user=True) a
        # run another user started lives under a different key, yet authorized users must still be
        # able to /stop it: fall back to sibling runs in this thread, gated on authorization.
        sibling_keys = self._sibling_thread_run_keys(source, session_key)
        if sibling_keys and self._is_user_authorized(source):
            for sibling_key in sibling_keys:
                await _stop(sibling_key, "stop_command_thread_sibling")
            logger.info("STOP (thread sibling) by %s — interrupted %d run(s) in thread: %s",
                        session_key, len(sibling_keys), ", ".join(sibling_keys))
            return EphemeralReply(t("gateway.stop.stopped"))

        # No running agent anywhere for this scope. A platform status indicator can still be stuck —
        # e.g. Slack's persistent assistant.threads.setStatus survives a gateway restart or a turn
        # that died without a final send.
        # Best-effort clear so /stop always dismisses a phantom "is thinking...". See #32295.
        adapter = getattr(self, "adapters", {}).get(source.platform)
        try:
            if adapter and hasattr(adapter, "_stop_typing_with_metadata"):
                await adapter._stop_typing_with_metadata(source.chat_id, self._reply_metadata(event))
        except Exception:
            logger.debug("Failed to clear typing on /stop with no active agent", exc_info=True)
        return t("gateway.stop.no_active")

    async def _handle_platform_command(self, event: MessageEvent) -> str:
        """Handle ``/platform list|pause|resume [name]`` — inspect and manually control failed/paused
        adapters (pause stops the reconnect watcher; resume re-queues for retry)."""
        # Strip the leading "/platform" (or "/PLATFORM") token if present
        parts = (getattr(event, "content", "") or "").strip().split(maxsplit=2)
        if parts and parts[0].lower().lstrip("/").startswith("platform"):
            parts = parts[1:]
        action = (parts[0] if parts else "list").lower()
        target = parts[1].lower() if len(parts) > 1 else ""
        failed = getattr(self, "_failed_platforms", {}) or {}
        if action == "list":
            connected = ", ".join(sorted(p.value for p in self.adapters)) or "(none)"
            lines = ["**Gateway platforms**", f"Connected: {connected}"]
            for p, info in failed.items():
                if info.get("paused"):
                    reason = info.get("pause_reason") or "paused"
                    lines.append(f"  · {p.value} — PAUSED ({reason}). Resume with `/platform resume {p.value}`.")
                else:
                    lines.append(f"  · {p.value} — retrying (attempt {info.get('attempts', 0)})")
            return "\n".join(lines + ([] if failed else ["Failed/paused: (none)"]))
        if action not in {"pause", "resume"}:
            return _PLATFORM_USAGE
        if not target:
            return f"Usage: /platform {action} <name>"
        # Resolve platform name (case-insensitive, value match)
        platform = next((p for p in Platform.__members__.values() if p.value.lower() == target), None)
        if platform is None:
            return f"Unknown platform: {target}"
        name = platform.value
        queued = platform in failed
        paused = queued and bool(failed[platform].get("paused"))
        if action == "pause":
            if not queued:
                return f"{name} is not in the retry queue (it's either connected or not enabled)."
            if paused:
                return f"{name} is already paused."
            self._pause_failed_platform(platform, reason="paused via /platform pause")
            return f"✓ {name} paused. Resume with `/platform resume {name}` or `hermes gateway restart` to reset."
        if not queued:
            return f"{name} is not in the retry queue — nothing to resume."
        if not paused:
            return f"{name} is already retrying — no resume needed."
        self._resume_paused_platform(platform)
        return f"✓ {name} resumed — retrying on next watcher tick."

    async def dispatch_gateway_restart(
        self,
        source: SessionSource,
        reason: str = "",
        update_id: Optional[int] = None,
        message_id: Optional[str] = None,
        origin: str = "slash_command",
    ) -> tuple[bool, str]:
        """Shared async restart coordinator used by both /restart and Agent tools.

        Returns (success: bool, user_message: str).
        """
        import uuid
        from gateway.restart import is_gateway_supervisor_process
        from gateway.run import _hermes_home

        if self._restart_requested or self._draining:
            count = self._running_agent_count()
            if count:
                return False, t("gateway.draining", count=count)
            return False, t("gateway.restart.in_progress")

        # P1 #71876 lifecycle remediation: after a claimed restart whose
        # launch outcome could not be confirmed, an explicit FINITE retry
        # block prevents an immediate re-trigger. It is a monotonic
        # deadline, NOT a permanent flag: once it expires the gateway
        # accepts a new restart normally.
        retry_blocked_until = getattr(self, "_restart_retry_blocked_until", 0.0)
        if retry_blocked_until:
            try:
                loop_now = asyncio.get_running_loop().time()
            except RuntimeError:
                loop_now = 0.0
            if loop_now < retry_blocked_until:
                remaining = int(retry_blocked_until - loop_now) + 1
                return (
                    False,
                    "Restart outcome could not be confirmed earlier. "
                    f"Please wait ~{remaining}s before retrying.",
                )
            # Window expired: clear the block so later dispatches are
            # unambiguous.
            self._restart_retry_blocked_until = 0.0

        request_id = f"req-{uuid.uuid4().hex[:8]}"
        notify_path = Path(_hermes_home) / ".restart_notify.json"
        dedup_path = Path(_hermes_home) / ".restart_last_processed.json"
        backup = _RestartStateBackup(
            request_id=request_id,
            notify_path=notify_path,
            dedup_path=dedup_path,
            original_source=self._restart_command_source,
        )

        effective_msg_id = message_id or source.message_id

        # 1. Transactionally write notify file
        notify_written = False
        try:
            notify_data: dict = {
                "request_id": request_id,
                "platform": source.platform.value if source.platform else None,
                "chat_id": source.chat_id,
                "chat_type": source.chat_type,
            }
            if getattr(source, "delivered_via_upstream_relay", None) is True:
                notify_data["delivered_via_upstream_relay"] = True
                if source.user_id:
                    notify_data["user_id"] = source.user_id
                if getattr(source, "scope_id", None):
                    notify_data["scope_id"] = source.scope_id
            if source.thread_id:
                notify_data["thread_id"] = source.thread_id
            if effective_msg_id:
                notify_data["message_id"] = str(effective_msg_id)

            try:
                self._restart_command_source = dataclasses.replace(
                    source,
                    message_id=str(effective_msg_id)
                    if effective_msg_id is not None
                    else source.message_id,
                )
            except Exception:
                self._restart_command_source = source

            await asyncio.to_thread(
                atomic_json_write,
                notify_path,
                notify_data,
                indent=None,
            )
            notify_written = True
        except Exception as e:
            logger.error("dispatch_gateway_restart: failed to write notify file: %s", e)
            backup.rollback(self)
            return False, "Failed to persist restart notification file: Gateway remains active."

        # 2. Transactionally write dedup file
        try:
            dedup_data = {
                "request_id": request_id,
                "platform": source.platform.value if source.platform else None,
                "chat_id": source.chat_id,
                "thread_id": source.thread_id,
                "requested_at": time.time(),
                "origin": origin,
            }
            if update_id is not None:
                dedup_data["update_id"] = update_id
            if effective_msg_id:
                dedup_data["message_id"] = str(effective_msg_id)
            if reason:
                dedup_data["reason"] = reason[:200]

            await asyncio.to_thread(
                atomic_json_write,
                dedup_path,
                dedup_data,
                indent=None,
            )
        except Exception as e:
            logger.error("dispatch_gateway_restart: failed to write dedup marker: %s", e)
            backup.rollback(self)
            return False, "Failed to persist restart dedup marker: Gateway remains active."

        _under_service = is_gateway_supervisor_process()
        _in_container = os.path.exists("/.dockerenv") or os.path.exists(
            "/run/.containerenv"
        )
        detached = not (_under_service or _in_container)
        via_service = _under_service or _in_container

        loop = getattr(self, "_gateway_loop", None) or asyncio.get_running_loop()

        # Create request-scoped transaction
        transaction = _RestartTransaction(
            request_id=request_id,
            backup=backup,
            loop=loop,
            detached=detached,
            via_service=via_service,
        )
        self._restart_transaction = transaction

        accepted = self.request_restart(
            detached=detached,
            via_service=via_service,
            transaction=transaction,
        )
        if not accepted:
            transaction.set_final_outcome(_RestartFinalOutcome.ABORTED)
            transaction.backup.rollback(self)
            if self._restart_transaction is transaction:
                self._restart_transaction = None
            return False, t("gateway.restart.in_progress")

        # 3. Await handoff acknowledgement with the dispatcher's ack timeout.
        # The ack_future is shielded: an outer CancelledError or timeout
        # never cancels the future itself — only the wait. The helper may
        # still write the ack after we exit (which becomes no-op via
        # try_write_ack, spec section 一).
        #
        # ack_future now returns _RestartAckOutcome (three-state):
        #   ACCEPTED → handoff confirmed, proceed to success path.
        #   NOT_STARTED → launcher confirmed no side-effect, rollback.
        #   OUTCOME_UNKNOWN → cannot confirm, must not rollback.
        ack_outcome = None
        try:
            await _shielded_wait_for_ack(
                transaction.ack_future, timeout=_RESTART_DISPATCH_ACK_TIMEOUT_S
            )
            ack_outcome = transaction.ack_future.result()
        except asyncio.TimeoutError:
            logger.error("Handoff acknowledgement timed out.")
            await self._handle_timeout_or_cancel(transaction)
        except asyncio.CancelledError:
            logger.error("Handoff cancelled before timeout/cancel handling.")
            await self._handle_timeout_or_cancel(transaction)
            raise  # Caller cancellation must propagate

        # Re-derive ack_outcome from terminal stage if we hit timeout/cancel.
        if ack_outcome is None:
            if transaction.is_committed():
                ack_outcome = _RestartAckOutcome.ACCEPTED
            elif transaction.is_aborted():
                ack_outcome = _RestartAckOutcome.NOT_STARTED
            else:
                ack_outcome = _RestartAckOutcome.OUTCOME_UNKNOWN

        # Map ack_outcome → (success, message, rollback_marker).
        if ack_outcome is _RestartAckOutcome.ACCEPTED:
            # P1 #71876: the accepted ack is now written by the helper
            # immediately after claim_handoff — it means "transaction
            # claimed; restart will proceed", NOT "drain/launcher/stop
            # completed". Do NOT set final_outcome=COMMITTED here: the
            # helper sets it later via complete_started() once the launcher
            # actually started (or complete_not_started / complete_unknown
            # on failure). Dispatch returns so the requesting turn can
            # finish; drain/launcher/stop failures are recorded via
            # transaction final_outcome, marker rollback, and notification.
            active_agents = self._running_agent_count()
            if active_agents:
                return True, t("gateway.draining", count=active_agents)
            return True, t("gateway.restart.restarting")

        if ack_outcome is _RestartAckOutcome.NOT_STARTED:
            # Definite failure: no side-effect, safe to rollback.
            # The helper already called complete_not_started() which set
            # stage=ABORTED and final_outcome=ABORTED.
            logger.info(
                "Handoff ack: NOT_STARTED — rolling back marker "
                "(request_id=%s).",
                transaction.request_id,
            )
            try:
                transaction.backup.rollback(self)
            except Exception:
                pass
            if self._restart_transaction is transaction:
                self._restart_transaction = None
            return False, "Handoff failed to initialize: detached restart helper could not be spawned. Gateway remains active."

        # OUTCOME_UNKNOWN
        logger.info(
            "Handoff ack: OUTCOME_UNKNOWN — marker preserved "
            "(request_id=%s).",
            transaction.request_id,
        )
        if self._restart_transaction is transaction:
            self._restart_transaction = None
        return False, "Restart outcome could not be confirmed. Do not retry immediately."

    async def _handle_timeout_or_cancel(
        self, transaction: "_RestartTransaction"
    ) -> None:
        """Common handler for pre-commit timeout AND caller cancellation.

        Implements spec sections 二 / 三:
          * Caller-side: claim_abort() — if True, we own the abort decision
            and may cancel the task and rollback the marker.
          * If helper already entered IN_FLIGHT (or beyond), claim_abort
            returns False. The dispatcher is a pure observer — it MUST NOT
            modify transaction state, cancel the restart task, rollback the
            marker, or write any ack/event. The helper continues running
            and will eventually complete the transaction.

        observer role when claim_abort fails:
          1. Does NOT call any complete_*
          2. Does NOT modify stage/final_outcome/launcher_result
          3. Does NOT write ack/event
          4. Does NOT cancel restart_task
          5. Does NOT rollback
          6. Only returns to the caller who then returns outcome_unknown

        The state lock is acquired only for the short stage transition;
        we never hold it across Popen / detached helpers / await on the
        restart task.
        """
        aborted_owned = await transaction.claim_abort()
        if not aborted_owned:
            # Helper already won the commit race. We MUST NOT cancel the
            # task, rollback the marker, or modify transaction state.
            # The helper will eventually complete the transaction
            # (STARTED → complete_started → stop, or NOT_STARTED →
            # rollback → complete_not_started, or UNKNOWN →
            # complete_unknown).
            # The dispatcher returns outcome_unknown to the caller.
            return

        # We won the abort gate. Cancel the restart task and await it.
        task_done = await transaction.cancel_restart_task_and_await()

        if not task_done:
            # Task refused to die — helper may have crossed IN_FLIGHT
            # despite ABORTING. This is a contradiction: claim_abort
            # succeeded (stage=ABORTING) so claim_handoff should have
            # failed. Log the inconsistency and return outcome_unknown
            # without modifying transaction state.
            logger.error(
                "Pre-commit cancel: restart task still running after cancel "
                "(request_id=%s). Abort gate was won but task refused to die. "
                "Returning outcome_unknown.",
                transaction.request_id,
            )
            return

        # Definite abort path: rollback marker and finalize.
        try:
            transaction.backup.rollback(self)
        except Exception:
            pass
        tr = await transaction.complete_claimed_abort()
        if tr is _TransitionResult.LOST_RACE:
            # Another terminal state won (cannot happen after successful
            # cancel_restart_task_and_await, but defensive).
            logger.error("complete_claimed_abort returned LOST_RACE after abort win.")
            return

    async def _handle_restart_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /restart command - drain active work, then restart the gateway."""

        if self._is_stale_restart_redelivery(event):
            src = event.source
            logger.info("Ignoring redelivered /restart (platform=%s, update_id=%s) — "
                        "already processed by a previous gateway instance.",
                        src.platform.value if src and src.platform else "?",
                        event.platform_update_id)
            return ""
        success, msg = await self.dispatch_gateway_restart(
            source=event.source,
            update_id=event.platform_update_id,
            message_id=event.message_id,
            origin="slash_command",
        )
        if not success and msg == t("gateway.restart.in_progress"):
            return EphemeralReply(msg)
        return msg if not success else (
            t("gateway.draining", count=self._running_agent_count())
            if self._running_agent_count() else EphemeralReply(msg)
        )
    async def _handle_version_command(self, event: MessageEvent) -> str:
        """Handle /version — show the running Hermes Agent version."""
        return _execute("version").text

    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        return self._telegramized_command_reply(event, _execute("help").text)

    async def _handle_commands_command(self, event: MessageEvent) -> str:
        # Page size is a surface parameter (Telegram messages are shorter).
        page_size = 15 if event.source.platform == Platform.TELEGRAM else 20
        reply = _execute("commands", args=event.get_command_args(), options={"page_size": page_size})
        return self._telegramized_command_reply(event, reply.text)

    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        from gateway.run import _home_target_env_var, _home_thread_env_var
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id
        if source.platform is None:
            return t("gateway.set_home.save_failed", error="Missing logical platform")
        via_relay = getattr(source, "delivered_via_upstream_relay", False) is True
        if via_relay:
            adapter_for_source = getattr(self, "_adapter_for_source", None)
            relay_adapter = adapter_for_source(source) if callable(adapter_for_source) else None
            fronts_platform = getattr(relay_adapter, "fronts_platform", None)
            if (source.platform in {None, Platform.LOCAL, Platform.RELAY}
                    or not getattr(source, "user_id", None)
                    or not callable(fronts_platform) or not fronts_platform(source.platform)):
                return t("gateway.set_home.save_failed",
                         error="Relay does not authenticate this logical home target")
        thread_id = _home_thread_from_source(source)
        home = HomeChannel(
            platform=source.platform, chat_id=str(chat_id), name=chat_name, thread_id=thread_id,
            user_id=str(source.user_id) if getattr(source, "user_id", None) else None,
            scope_id=str(source.scope_id) if getattr(source, "scope_id", None) else None)
        # config.yaml is canonical because it can persist the authenticated logical-target
        # provenance required by Relay after a restart.
        try:
            persist_home_channel(home, enabled_if_new=not via_relay)
        except Exception as e:
            return t("gateway.set_home.save_failed", error=e)
        # Preserve legacy home env vars for existing cron/setup consumers.
        try:
            from hermes_cli.config import save_env_value
            save_env_value(_home_target_env_var(platform_name), str(chat_id))
            save_env_value(_home_thread_env_var(platform_name), str(thread_id or ""))
        except Exception as e:
            logger.warning("Home config saved but legacy env persistence failed: %s", e)
        # Keep the running gateway config in sync too. The pre-restart notification path reads
        # self.config before the process reloads config.
        platform_config = self.config.platforms.setdefault(source.platform, PlatformConfig(enabled=not via_relay))
        platform_config.home_channel = home
        return t("gateway.set_home.success", name=chat_name, chat_id=chat_id)

    async def _handle_voice_command(self, event: MessageEvent) -> str:
        """Handle /voice [on|off|tts|channel|leave|status] command."""
        args = event.get_command_args().strip().lower()
        chat_id = event.source.chat_id
        # Voice state belongs to the (bot, chat) pair: resolve the adapter that received the
        # command and key the mode by its owning profile so two multiplexed bots in one chat keep
        # independent /voice state.
        # See #75198.
        voice_key = self._voice_key_for_source(event.source)
        adapter = self._adapter_for_source(event.source)

        def _set_mode(mode: str) -> None:
            self._voice_mode[voice_key] = mode
            self._save_voice_modes()
            if not adapter:
                return
            if mode == "off":
                self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
            else:
                self._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)

        if args in _VOICE_MODE_BY_ARG:
            mode, reply_key = _VOICE_MODE_BY_ARG[args]
            _set_mode(mode)
            return t(reply_key)
        if args in {"channel", "join"}:
            return await self._handle_voice_channel_join(event)
        if args == "leave":
            return await self._handle_voice_channel_leave(event)
        if args == "status":
            mode = self._voice_mode.get(voice_key, "off")
            label = t(f"gateway.voice.label_{mode}") if mode in ("off", "voice_only", "all") else mode
            lines = [t("gateway.voice.status_mode", label=label)]
            guild_id = self._get_guild_id(event)  # append voice channel info if connected
            info = adapter.get_voice_channel_info(guild_id) if guild_id and hasattr(adapter, "get_voice_channel_info") else None
            if info:
                lines += [t("gateway.voice.status_channel", channel=info['channel_name']),
                          t("gateway.voice.status_participants", count=info['member_count'])]
                for m in info["members"]:
                    status = t("gateway.voice.speaking") if m.get("is_speaking") else ""
                    lines.append(t("gateway.voice.status_member", name=m['display_name'], status=status))
            return "\n".join(lines)

        # Toggle: off → on, on/all → off
        turning_on = self._voice_mode.get(voice_key, "off") == "off"
        _set_mode("voice_only" if turning_on else "off")
        toggle_line = t("gateway.voice.enabled_short" if turning_on else "gateway.voice.disabled_short")
        # Bare /voice still toggles, but append an explainer so users discover the on/off/tts/status
        # subcommands (and, on Discord, live voice-channel join/leave). Toggle result shows first.
        supports_voice_channels = adapter is not None and hasattr(adapter, "join_voice_channel")
        channels = t("gateway.voice.help_channels") if supports_voice_channels else ""
        return t("gateway.voice.help", toggle=toggle_line, channels=channels)

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from tools.checkpoint_manager import format_checkpoint_list
        mgr = self._checkpoint_manager()
        if mgr is None:
            return t("gateway.rollback.not_enabled")
        cwd = self._terminal_cwd()
        # --all / --force: classic full restore, overwriting user edits too.
        tokens = event.get_command_args().strip().split()
        restore_all = any(tok.lower() in ("--all", "--force") for tok in tokens)
        arg = " ".join(tok for tok in tokens if tok.lower() not in ("--all", "--force"))
        checkpoints = mgr.list_checkpoints(cwd)
        if not arg:
            return format_checkpoint_list(checkpoints, cwd)
        if not checkpoints:
            return t("gateway.rollback.none_found", cwd=cwd)

        # Restore by number or hash
        try:
            idx = int(arg) - 1
        except ValueError:
            target_hash = arg
        else:
            if not 0 <= idx < len(checkpoints):
                return t("gateway.rollback.invalid_number", max=len(checkpoints))
            target_hash = checkpoints[idx]["hash"]
        result = mgr.restore(cwd, target_hash, safe=not restore_all)
        if not result["success"]:
            return t("gateway.rollback.restore_failed", error=result["error"])
        msg = t("gateway.rollback.restored", hash=result["restored_to"], reason=result["reason"])
        for result_key, i18n_key in _ROLLBACK_SKIP_LINES:
            files = result.get(result_key) or []
            if files:
                more = f" (+{len(files) - 5})" if len(files) > 5 else ""
                msg += "\n" + t(i18n_key, files=", ".join(files[:5]) + more)
        return msg

    async def _handle_diff_command(self, event: MessageEvent) -> str:
        """Handle /diff — show git changes in the working directory.  Diff body is truncated hard
        here (chat is not a pager); platform senders clamp further."""
        args = [a.lower() for a in event.get_command_args().strip().split()]
        stat_only = bool({"--stat", "stat"} & set(args))
        mode = "working"
        for low in args:
            mode = _DIFF_MODE_BY_ARG.get(low, mode)
        cwd = self._terminal_cwd()
        if mode == "session":
            # Cumulative checkpoint-baseline diff.
            mgr = self._checkpoint_manager()
            if mgr is None:
                return t("gateway.diff.not_enabled")
            result = await asyncio.to_thread(mgr.session_diff, cwd)
        else:
            from tools.working_diff import collect_working_diff
            result = await asyncio.to_thread(collect_working_diff, cwd, mode)
        if not result.get("success"):
            return t("gateway.diff.failed", error=result.get("error", "Could not generate diff"))
        return self._render_diff_result(result, stat_only)

    def _render_diff_result(self, result: dict, stat_only: bool) -> str:
        """Render a working/session diff result: stat block, untracked list, fenced (truncated) diff."""
        stat = result.get("stat", "")
        diff = result.get("diff", "")
        untracked = result.get("untracked", [])
        if result.get("empty") or (not stat and not diff and not untracked):
            return t("gateway.diff.no_changes")
        out: list[str] = []
        if stat:
            out.append(f"```\n{stat}\n```")
        if untracked:
            shown = "\n".join(f"+ {rel}" for rel in untracked[:15])
            more = f"\n... and {len(untracked) - 15} more" if len(untracked) > 15 else ""
            out.append(f"**Untracked:**\n```\n{shown}{more}\n```")
        if not stat_only and diff:
            out.append(self._fenced_truncated_diff(diff))
        return "\n\n".join(out)

    @staticmethod
    def _fenced_truncated_diff(diff: str, max_lines: int = 60, max_chars: int = 3000) -> str:
        """Fence a diff body, truncating to messaging-friendly size."""
        diff_lines = diff.splitlines()
        truncated = len(diff_lines) > max_lines
        if truncated:
            diff = "\n".join(diff_lines[:max_lines])
        if len(diff) > max_chars:
            diff = diff[:max_chars]
            truncated = True
        note = ""
        if truncated:
            note = f"\n... (truncated — {len(diff_lines)} lines total; use /diff --stat for a summary)"
        return f"```diff\n{diff}{note}\n```"

    def _track_background_task(self, coro) -> None:
        """Fire-and-forget *coro*, keeping a strong ref in ``_background_tasks`` until it finishes."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /bg <prompt> — run a prompt in a background thread with its own session; the
        result is sent to the same chat without touching the active session's history."""
        prompt = event.get_command_args().strip()
        if not prompt:
            return t("gateway.background.usage")
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"
        self._track_background_task(self._run_background_task(
            prompt, event.source, task_id, event_message_id=self._reply_anchor_for_event(event),
            # Forward image/audio attachments so the background agent can see them.
            media_urls=list(event.media_urls or []), media_types=list(event.media_types or [])))
        return t("gateway.background.started", preview=_preview(prompt), task_id=task_id)

    async def _handle_btw_command(self, event: MessageEvent) -> str:
        """Handle /btw <question> — one-shot auxiliary LLM call on a transcript snapshot; live history
        is never touched (alternation + prompt cache intact, current turn keeps running). Unlike /bg,
        which spawns a fresh contextless session."""
        question = event.get_command_args().strip()
        if not question:
            return t("gateway.btw.usage")
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        try:
            history = await self.async_session_store.load_transcript(session_entry.session_id)
        except TranscriptReadError:
            return HISTORY_UNREADABLE
        if not history:
            return t("gateway.btw.no_history")
        try:
            model, rt = self._resolve_session_agent_runtime(source=source)
        except Exception:
            model, rt = None, {}
        if not rt.get("api_key"):
            return t("gateway.btw.no_provider")
        main_runtime = {"model": model, **{k: rt.get(k) for k in ("provider", "base_url", "api_key", "api_mode")}}
        history_snapshot = list(history)
        # Prefer the cache-parity fork when a live cached AIAgent exists: it replays the snapshot
        # against the warm provider prefix cache, giving FULL context at cache-read prices. With no
        # cached agent the cache is cold anyway — answer_side_question's digest fallback handles it.
        try:
            parent_agent = self._cached_agent_for(self._session_key_for_source(source))
        except Exception:
            parent_agent = None
        _thread_metadata = self._reply_metadata(event)
        adapter = self._adapter_for_source(source)
        preview = _preview(question)

        async def _run_side_question() -> None:
            from agent.side_question import answer_side_question
            try:
                answer = await asyncio.to_thread(
                    answer_side_question, question, history_snapshot,
                    parent_agent=parent_agent, main_runtime=main_runtime)
                reply = t("gateway.btw.answer", preview=preview, answer=answer or "")
            except Exception as e:
                logger.warning("/btw side question failed: %s", e)
                reply = t("gateway.btw.failed", preview=preview, error=str(e))
            if adapter is not None:
                await adapter.send(source.chat_id, reply, metadata=_thread_metadata)

        self._track_background_task(_run_side_question())
        return t("gateway.btw.started", preview=preview)

    async def _handle_memory_command(self, event: MessageEvent) -> str:
        """Handle /memory — review pending memory writes + toggle the approval gate. Entries are small
        enough to review inline, so the full flow works on every platform."""
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        from tools.memory_tool import load_on_disk_store
        # Apply approved writes against a fresh on-disk store (the gateway has no long-lived agent;
        # the store persists to the same MEMORY/USER.md and honors the configured char limits).
        out = handle_pending_subcommand(
            wa.MEMORY, event.get_command_args().strip().split(), memory_store=load_on_disk_store(),
            set_mode_fn=self._write_approval_setter("memory", event))
        return out if out is not None else (
            "Unknown /memory subcommand. Use: pending, approve <id>, reject <id>, approval <on|off>."
        )

    async def _handle_skills_command(self, event: MessageEvent) -> str:
        """Handle /skills on the gateway — pending skill-write review only (hub stays CLI-only). Gated
        by ``skills.write_approval`` but still answers when staged writes exist after the gate is off
        (never stranded). ``diff`` is truncated for chat."""
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        args = event.get_command_args().strip().split()
        sub = args[0].lower() if args else ""
        gate_off = not wa.write_approval_enabled(wa.SKILLS) and sub not in {"approval", "mode"}
        if gate_off and wa.pending_count(wa.SKILLS) == 0:
            return ("Skill write approval is off (skills.write_approval). "
                    "Enable it with /skills approval on, then review staged "
                    "writes here with /skills pending.")
        out = handle_pending_subcommand(
            wa.SKILLS, args, set_mode_fn=self._write_approval_setter("skills", event))
        if out is None:
            return ("Unknown /skills subcommand on this platform. Use: pending, "
                    "approve <id>, reject <id>, diff <id>, approval <on|off>. "
                    "(Search/install are CLI-only.)")

        # Chat bubbles can't hold a full skill diff — truncate and point at the pending JSON file
        # (NOT `hermes skills diff <name>`, which diffs a bundled skill against its stock version).
        if sub == "diff" and len(out) > 3000:
            pending_id = args[1] if len(args) > 1 else "<id>"
            out = (out[:3000]
                   + "\n… (truncated — full diff in "
                     f"~/.hermes/pending/skills/{pending_id}.json)")
        return out

    async def _handle_approvals_command(self, event: MessageEvent) -> str:
        """Show or persist the profile-wide dangerous-command approval mode."""
        from gateway.slash_access import policy_for_source
        from hermes_cli.approval_mode import run_approval_mode_command
        requested = event.get_command_args().strip() or None
        # This mutates profile-wide security policy. The central slash gate can allow selected
        # commands to non-admin users, so enforce admin again at this side-effect boundary.
        # Unconfigured policies remain unrestricted.
        policy = policy_for_source(self.config, event.source)
        if requested and not policy.is_admin(event.source.user_id):
            return "Only gateway admins can change the persistent approval mode."
        # Approval checks load config dynamically; do not evict the cached agent or alter its
        # system prompt/tool schema (prompt-cache prefix is sacred).
        return run_approval_mode_command(requested).message

    async def _handle_yolo_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /yolo — toggle dangerous command approval bypass for this session only."""
        from tools.approval import disable_session_yolo, enable_session_yolo, is_session_yolo_enabled
        session_key = self._session_key_for_source(event.source)
        if is_session_yolo_enabled(session_key):
            disable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.disabled"))
        enable_session_yolo(session_key)
        return EphemeralReply(t("gateway.yolo.enabled"))

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose — cycle tool progress display mode (off → new → all → verbose → log) per
        *current platform*, saved to ``display.platforms.<platform>.tool_progress``. Gated by
        ``display.tool_progress_command`` (default off)."""
        from gateway.run import _load_gateway_config
        config_path, platform_key = self._display_config_target(event)
        try:
            user_config = _load_gateway_config()
            gate_enabled = is_truthy_value(cfg_get(user_config, "display", "tool_progress_command"),
                                           default=False)
        except Exception:
            gate_enabled = False
        if not gate_enabled:
            return t("gateway.verbose.not_enabled")
        # Cycle mode (per-platform), reading the current effective mode via the resolver.
        from gateway.display_config import resolve_display_setting
        cycle = ["off", "new", "all", "verbose", "log"]
        current = resolve_display_setting(user_config, platform_key, "tool_progress", "all")
        new_mode = cycle[(cycle.index(current if current in cycle else "all") + 1) % len(cycle)]
        description = t(f"gateway.verbose.mode_{new_mode}")
        try:
            _nested_dict(user_config, "display", "platforms", platform_key)["tool_progress"] = new_mode
            atomic_config_write(config_path, user_config)
            return f"{description}\n" + t("gateway.verbose.saved_suffix", platform=platform_key)
        except Exception as e:
            logger.warning("Failed to save tool_progress mode: %s", e)
            return f"{description}\n" + t("gateway.verbose.save_failed", error=e)

    async def _handle_busy_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /busy — control what happens when messaging while Hermes is working."""
        arg = event.get_command_args().strip().lower()
        if not arg or arg == "status":
            mode = self._effective_busy_input_mode(event.source)
            behavior = _BUSY_MODE_BEHAVIOR.get(mode, _BUSY_MODE_BEHAVIOR["interrupt"])[0]
            return EphemeralReply(
                f"**Busy input mode: `{mode}`\nMessages while busy: _{behavior}_\n"
                f"Change with `/busy queue`, `/busy steer`, or `/busy interrupt`.")
        if arg not in _BUSY_MODE_BEHAVIOR:
            return EphemeralReply(
                f"Unknown mode `{arg}`. Use `/busy queue`, `/busy steer`, or `/busy interrupt`.")

        # Persist before mutate
        from cli import save_config_value
        if not save_config_value("display.busy_input_mode", arg):
            return EphemeralReply("Busy input mode could not be saved to config. Mode unchanged.")
        profile_name = self._busy_profile_name_for_source(event.source)
        if profile_name:
            from gateway.run import _load_gateway_runtime_config
            self._snapshot_profile_busy_modes(profile_name, _load_gateway_runtime_config())
        else:
            self._busy_input_mode = arg
            # busy_input_mode is also the source of truth for the text mode — re-derive it so the
            # adapter refresh below doesn't keep a stale value and keep interrupting.
            self._busy_text_mode = self._load_busy_text_mode()

        adapter = self._adapter_for_source(event.source)
        if adapter is not None:
            adapter._busy_text_mode = self._effective_busy_text_mode(event.source)
        return EphemeralReply(
            f"Busy input mode set to **`{arg}`** (saved).\n_{_BUSY_MODE_BEHAVIOR[arg][1]}_")

    async def _handle_footer_command(self, event: MessageEvent) -> str:
        """Handle /footer command — toggle the runtime-metadata footer."""
        from gateway.run import _load_gateway_config, _resolve_gateway_model
        from gateway.runtime_footer import format_runtime_footer, resolve_footer_config
        config_path, platform_key = self._display_config_target(event)
        arg = ""
        try:
            text = (getattr(event, "message", None) or "").strip()
            if text.startswith("/"):
                parts = text.split(None, 1)
                arg = parts[1].strip().lower() if len(parts) > 1 else ""
        except Exception:
            arg = ""
        try:
            user_config: dict = _load_gateway_config()
        except Exception as e:
            return t("gateway.config_read_failed", error=e)
        effective = resolve_footer_config(user_config, platform_key)

        def _state(enabled: bool) -> str:
            return t("gateway.footer.state_on") if enabled else t("gateway.footer.state_off")
        if arg in {"status", "?"}:
            return t("gateway.footer.status", state=_state(effective["enabled"]),
                     fields=", ".join(effective.get("fields") or []), platform=platform_key)
        if arg and arg not in _FOOTER_STATE_BY_ARG:
            return t("gateway.footer.usage")
        new_state = _FOOTER_STATE_BY_ARG[arg] if arg else not effective["enabled"]
        try:
            _nested_dict(user_config, "display", "runtime_footer")["enabled"] = new_state
            atomic_config_write(config_path, user_config)
        except Exception as e:
            logger.warning("Failed to save runtime_footer.enabled: %s", e)
            return t("gateway.config_save_failed", error=e)
        example = ""
        if new_state:
            # Show a preview using current agent state if available.
            preview = format_runtime_footer(
                model=_resolve_gateway_model(user_config) or None, context_tokens=0, context_length=None,
                fields=effective.get("fields") or ["model", "context_pct", "cwd"])
            if preview:
                example = t("gateway.footer.example_line", preview=preview)
        return t("gateway.footer.saved", state=_state(new_state), example=example)

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /reload-mcp — reconnect MCP servers and rebuild the cached agent. Reloading
        invalidates the provider prompt cache (tool schemas live in the system prompt), so it routes
        through slash-confirm; "Always Approve" persists ``approvals.mcp_reload_confirm: false``."""
        session_key = self._session_key_for_source(event.source)
        # Read the gate fresh from disk so a prior "always" click takes effect on the next
        # invocation without restarting the gateway.
        user_config = self._read_user_config()
        approvals = user_config.get("approvals") if isinstance(user_config, dict) else None
        if isinstance(approvals, dict) and not approvals.get("mcp_reload_confirm", True):
            return await self._execute_mcp_reload(event)
        # Route through slash-confirm. The primitive sends the prompt and stores the resume handler;
        # the button/text response triggers ``_resolve_slash_confirm`` which invokes the handler
        # with the chosen outcome.
        async def _on_confirm(choice: str) -> Optional[str]:
            if choice == "cancel":
                return t("gateway.reload_mcp.cancelled")
            if choice == "always":
                # Persist the opt-out and run the reload.
                try:
                    from cli import save_config_value
                    save_config_value("approvals.mcp_reload_confirm", False)
                    logger.info("User opted out of /reload-mcp confirmation (session=%s)", session_key)
                except Exception as exc:
                    logger.warning("Failed to persist mcp_reload_confirm=false: %s", exc)
            # once / always → run the reload
            result = await self._execute_mcp_reload(event)
            if choice == "always":
                return f"{result}\n\n" + t("gateway.reload_mcp.always_followup")
            return result
        return await self._request_slash_confirm(
            event=event, command="reload-mcp", title="/reload-mcp",
            message=t("gateway.reload_mcp.confirm_prompt"), handler=_on_confirm)

    async def _handle_reload_skills_command(self, event: MessageEvent) -> str:
        """Handle /reload-skills — rescan skills dir, queue a note for next turn. Skills are invoked at
        runtime, not baked into the system prompt, so this does NOT clear the prompt cache. The diff
        goes into ``_pending_skills_reload_notes[session_key]``, prepended to the NEXT user message —
        nothing out-of-band, so alternation is preserved."""
        try:
            from agent.skill_commands import reload_skills

            # _run_in_executor_with_context, not a bare hop: the rescan walks
            # get_hermes_home()/skills, a contextvar override under multiplex.
            result = await self._run_in_executor_with_context(reload_skills)
            added, removed = result.get("added", []), result.get("removed", [])  # [{"name", "description"}]
            total = result.get("total", 0)
            # Let adapters refresh platform-side state that cached the skill list at startup (today:
            # Discord /skill autocomplete — otherwise new skills stay invisible and deleted ones
            # error). Adapters without refresh_skill_group are skipped; the in-process reload suffices.
            for adapter in list(self.adapters.values()):
                refresh = getattr(adapter, "refresh_skill_group", None)
                try:
                    maybe = refresh() if callable(refresh) else None
                    if inspect.isawaitable(maybe):
                        await maybe
                except Exception as exc:
                    logger.warning("Adapter %s refresh_skill_group raised: %s",
                                   getattr(adapter, "name", adapter), exc)

            lines = [t("gateway.reload_skills.header")]
            if not added and not removed:
                lines += [t("gateway.reload_skills.no_new"), t("gateway.reload_skills.total", count=total)]
                return "\n".join(lines)

            def _fmt_line(item: dict) -> str:
                nm, desc = item.get("name", ""), item.get("description", "")
                return (t("gateway.reload_skills.item_with_desc", name=nm, desc=desc) if desc
                        else t("gateway.reload_skills.item_no_desc", name=nm))

            # Queue a one-shot note for the next user turn in this session too. Format matches how
            # the system prompt renders pre-existing skills (``    - name: description``) so the
            # model reads the diff in the same shape as its original skill catalog.
            sections = ["[USER INITIATED SKILLS RELOAD:"]
            for i18n_key, note_header, items in (
                ("gateway.reload_skills.added_header", "Added Skills:", added),
                ("gateway.reload_skills.removed_header", "Removed Skills:", removed)):
                if items:
                    formatted = [_fmt_line(item) for item in items]
                    lines += [t(i18n_key)] + formatted
                    sections += ["", note_header] + formatted
            lines.append(t("gateway.reload_skills.total", count=total))
            sections += ["", "Use skills_list to see the updated catalog.]"]
            session_key = self._session_key_for_source(event.source)
            if not hasattr(self, "_pending_skills_reload_notes"):
                self._pending_skills_reload_notes = {}
            if session_key:
                self._pending_skills_reload_notes[session_key] = "\n".join(sections)
            return "\n".join(lines)
        except Exception as e:
            logger.warning("Skills reload failed: %s", e)
            return t("gateway.reload_skills.failed", error=e)

    async def _handle_bundles_command(self, event: MessageEvent) -> str:
        """Handle /bundles — list installed skill bundles (mirrors the CLI handler). Bundles are
        loaded by invoking their own ``/<slug>`` command, not by this one."""
        reply = _execute("bundles")
        if "error" in reply.data:
            logger.warning("Bundles command unavailable: %s", reply.data["error"])
            return reply.text
        bundles = reply.data["bundles"]
        if not bundles:
            return ("No skill bundles installed.\nCreate one on the host with:\n"
                    "  `hermes bundles create <name> --skill <s1> --skill <s2>`\n"
                    f"Directory: `{reply.data['dir']}`")
        lines = [f"**Skill Bundles** ({len(bundles)} installed):", ""]
        for info in bundles:
            skills = info.get("skills", [])
            desc = info.get("description") or f"Load {len(skills)} skills"
            lines += [f"• `/{info['slug']}` — {desc} _({len(skills)} skills)_"] + [f"    · {s}" for s in skills]
        return "\n".join(lines + ["", "Invoke a bundle with `/<slug>` to load all its skills."])

    def _blocking_approval_or_stale(self, event: MessageEvent, stale_key: str, none_key: str):
        """``(session_key, None)`` when an agent thread is blocked on approval, else the reply to send.
        A pending-approvals entry with no blocked thread is a stale prompt: drop it and say so."""
        from tools.approval import has_blocking_approval
        session_key = self._session_key_for_source(event.source)
        if has_blocking_approval(session_key):
            return session_key, None
        if session_key in self._pending_approvals:
            self._pending_approvals.pop(session_key)
            return session_key, t(stale_key)
        return session_key, t(none_key)

    async def _handle_approve_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /approve — unblock waiting agent thread(s). They block inside tools/approval.py;
        signalling the event resumes them so the command executes inline (same flow as the CLI)."""
        from tools.approval import resolve_gateway_approval
        session_key, stale = self._blocking_approval_or_stale(event, "gateway.approval_expired",
                                                              "gateway.approve.no_pending")
        if stale:
            return stale
        # Args: "all", "all session", "all always", "session", "always" ("always" beats "session").
        args = event.get_command_args().strip().lower().split()
        choices = {_APPROVE_CHOICE_BY_ARG[a] for a in args if a in _APPROVE_CHOICE_BY_ARG}
        choice = "always" if "always" in choices else "session" if "session" in choices else "once"
        count = resolve_gateway_approval(session_key, choice, resolve_all="all" in args)
        if not count:
            return t("gateway.approve.no_pending")
        confirmation_text = t(f"gateway.approve.{choice}_{'plural' if count > 1 else 'singular'}", count=count)
        logger.info("User approved %d dangerous command(s) via /approve (%s)", count, choice)
        return await self._deliver_approval_confirmation(event, confirmation_text, "approve")

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny — reject pending dangerous command(s) with a definitive BLOCKED result, as in
        the CLI. ``/deny`` denies the oldest; ``/deny all`` denies everything.

        ``/deny <reason>`` (or ``/deny all <reason>``) attaches a one-line reason that is relayed back to
        the agent so it can adapt instead of only hearing "denied". Ported from qwibitai/nanoclaw#2832.
        """
        from tools.approval import resolve_gateway_approval
        session_key, stale = self._blocking_approval_or_stale(event, "gateway.deny.stale",
                                                              "gateway.deny.no_pending")
        if stale:
            return stale
        # A leading "all" denies every pending command; the rest (or the whole arg string without
        # "all") is the optional deny reason relayed to the agent, capped to a sane one-liner.
        raw_args = event.get_command_args().strip()
        tokens = raw_args.split()
        resolve_all = bool(tokens) and tokens[0].lower() == "all"
        reason = (raw_args[len(tokens[0]):].strip() if resolve_all else raw_args)[:280].strip()
        count = resolve_gateway_approval(session_key, "deny", resolve_all=resolve_all, reason=reason or None)
        if not count:
            return t("gateway.deny.no_pending")
        logger.info("User denied %d dangerous command(s) via /deny%s", count,
                    " (with reason)" if reason else "")
        key = "gateway.deny.denied" + ("_reason" if reason else "") + ("_plural" if count > 1 else "_singular")
        confirmation_text = t(key, count=count, reason=reason)
        return await self._deliver_approval_confirmation(event, confirmation_text, "deny")

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug — upload ONLY the summary (system info + log tails), never full logs, to
        protect privacy; ``hermes debug share`` from the CLI does full uploads."""
        from hermes_cli.debug import (_GATEWAY_PRIVACY_NOTICE, _best_effort_sweep_expired_pastes,
                                      _capture_dump, _schedule_auto_delete, collect_debug_report,
                                      upload_to_pastebin)

        def _collect_and_upload():  # blocking I/O (dump capture, log reads, uploads) -> thread
            _best_effort_sweep_expired_pastes()
            report = collect_debug_report(log_lines=200, dump_text=_capture_dump())
            try:
                urls = {"Report": upload_to_pastebin(report)}
            except Exception as exc:
                return t("gateway.debug.upload_failed", error=exc)
            _schedule_auto_delete(list(urls.values()))  # auto-deletion after 6 hours
            label_width = max(len(k) for k in urls)
            return "\n".join([_GATEWAY_PRIVACY_NOTICE, "", t("gateway.debug.header"), "",
                              *(f"`{label:<{label_width}}`  {url}" for label, url in urls.items()),
                              "", t("gateway.debug.auto_delete"), t("gateway.debug.full_logs_hint"),
                              t("gateway.debug.share_hint")])

        # _run_in_executor_with_context, not a bare hop: this collects the profile's logs/config off
        # ``get_hermes_home()`` and uploads them to a public paste. Losing the contextvar override
        # would publish the DEFAULT profile's diagnostics from another profile's chat.
        return await self._run_in_executor_with_context(_collect_and_upload)

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update — spawn ``hermes update`` detached (``setsid``) so it survives the gateway
        restart it may trigger; marker files let this or the next gateway process notify the user."""
        import json
        from gateway.run import _hermes_home, _resolve_hermes_bin
        from hermes_cli.config import is_managed, format_managed_message
        # Block non-messaging platforms (API server, webhooks, ACP); plugin platforms with
        # allow_update_command=True are also allowed.
        src = event.source
        if src.platform not in self._UPDATE_ALLOWED_PLATFORMS:
            try:
                from gateway.platform_registry import platform_registry
                entry = platform_registry.get(src.platform.value)
                if not entry or not entry.allow_update_command:
                    return t("gateway.update.platform_not_messaging")
            except Exception:
                return t("gateway.update.platform_not_messaging")
        if is_managed():
            return f"✗ {format_managed_message('update Hermes Agent')}"
        if not (Path(__file__).parent.parent.resolve() / '.git').exists():
            return t("gateway.update.not_git_repo")
        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            return t("gateway.update.hermes_cmd_not_found")
        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        pending = {
            "platform": src.platform.value, "chat_id": src.chat_id, "chat_type": src.chat_type,
            "user_id": src.user_id, "session_key": self._session_key_for_source(src),
            "timestamp": datetime.now().isoformat()}
        pending.update({k: v for k, v in (("thread_id", src.thread_id), ("message_id", event.message_id)) if v})
        _tmp_pending = pending_path.with_suffix(".tmp")
        _tmp_pending.write_text(json.dumps(pending), encoding="utf-8")
        _tmp_pending.replace(pending_path)
        exit_code_path.unlink(missing_ok=True)
        try:
            _spawn_detached_update(hermes_cmd, output_path, exit_code_path)
        except Exception as e:
            pending_path.unlink(missing_ok=True)
            exit_code_path.unlink(missing_ok=True)
            return t("gateway.update.start_failed", error=e)
        self._schedule_update_notification_watch()
        return t("gateway.update.starting")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Any  # noqa: F401,E402
import hashlib  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'HISTORY_UNREADABLE': ('gateway.slash_commands_status', 'HISTORY_UNREADABLE'),
    'MessageType': ('gateway.platforms.event', 'MessageType'),
    'SessionSource': ('gateway.session', 'SessionSource'),
    'base_url_host_matches': ('utils', 'base_url_host_matches'),
    'build_session_key': ('gateway.session', 'build_session_key'),
    'clear_model_endpoint_credentials': ('hermes_cli.config', 'clear_model_endpoint_credentials'),
    'extract_api_content_sidecar': ('agent.turn_context', 'extract_api_content_sidecar'),
    'fetch_account_usage': ('agent.account_usage', 'fetch_account_usage'),
    'is_shared_multi_user_session': ('gateway.session', 'is_shared_multi_user_session'),
    'render_account_usage_lines': ('agent.account_usage', 'render_account_usage_lines'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
