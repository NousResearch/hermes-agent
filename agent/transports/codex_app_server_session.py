"""Session adapter for codex app-server runtime.

Owns one Codex thread per Hermes session. Drives `turn/start`, consumes
streaming notifications via CodexEventProjector, handles server-initiated
approval requests (apply_patch, exec command), translates cancellation,
and returns a clean turn result that AIAgent.run_conversation() can splice
into its `messages` list.

Lifecycle:
    session = CodexAppServerSession(cwd="/home/x/proj")
    session.ensure_started()                              # spawns + handshake + thread/start
    result = session.run_turn(user_input="hello")         # blocks until turn/completed
    # result.final_text          → assistant text returned to caller
    # result.projected_messages  → list of {role, content, ...} for messages list
    # result.tool_iterations     → how many tool-shaped items completed (skill nudge counter)
    # result.interrupted         → True if Ctrl+C / interrupt_requested fired mid-turn
    session.close()                                       # tears down subprocess

Threading model: the adapter is single-threaded from the caller's perspective.
The underlying CodexAppServerClient owns its own reader threads but exposes
blocking-with-timeout queues that this adapter polls in a loop, so the run_turn
call is synchronous and behaves like AIAgent's existing chat_completions loop.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.transports.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
)
from agent.transports.codex_event_projector import CodexEventProjector

logger = logging.getLogger(__name__)


# Permission profile mapping mirrors the docstring in PR proposal:
# Hermes' tools.terminal.security_mode → Codex's permissions profile id.
# Defaults if config is missing → workspace-write (matches Codex's own default).
_HERMES_TO_CODEX_PERMISSION_PROFILE = {
    "auto": "workspace-write",
    "approval-required": "read-only-with-approval",
    "unrestricted": "full-access",
    # Backstop alias used by some skills/tests.
    "yolo": "full-access",
}


@dataclass
class TurnResult:
    """Result of one user→assistant→tool turn through the codex app-server."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None  # Set if turn ended in a non-recoverable error
    turn_id: Optional[str] = None
    thread_id: Optional[str] = None


@dataclass
class _ServerRequestRouting:
    """Default policies for codex-side approval requests when no interactive
    callback is wired in. These are only used by tests + cron / non-interactive
    contexts; the live CLI path passes an approval_callback that defers to
    tools.approval.prompt_dangerous_approval()."""

    auto_approve_exec: bool = False
    auto_approve_apply_patch: bool = False


class CodexAppServerSession:
    """One Codex thread per Hermes session, lifetime owned by AIAgent.

    Not thread-safe — one caller drives it at a time, matching how AIAgent's
    run_conversation() loop is structured today. The codex client itself can
    handle interleaved reads/writes via its own threads, but the adapter's
    state (projector, thread_id, turn counter) is owned by the caller thread.
    """

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        codex_bin: str = "codex",
        codex_home: Optional[str] = None,
        permission_profile: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        request_routing: Optional[_ServerRequestRouting] = None,
        client_factory: Optional[Callable[..., CodexAppServerClient]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._codex_bin = codex_bin
        self._codex_home = codex_home
        self._permission_profile = (
            permission_profile or _HERMES_TO_CODEX_PERMISSION_PROFILE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "workspace-write",
            )
        )
        self._approval_callback = approval_callback
        self._on_event = on_event  # Display hook (kawaii spinner ticks etc.)
        self._routing = request_routing or _ServerRequestRouting()
        self._client_factory = client_factory or CodexAppServerClient

        self._client: Optional[CodexAppServerClient] = None
        self._thread_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        # Pending file-change items, keyed by item id. Populated on
        # item/started for fileChange items; consumed by the approval
        # bridge when codex sends item/fileChange/requestApproval. The
        # approval params don't carry the changeset, so we cache here
        # to surface a real summary in the approval prompt (quirk #4).
        self._pending_file_changes: dict[str, str] = {}
        self._closed = False

    # ---------- lifecycle ----------

    def ensure_started(self) -> str:
        """Spawn the subprocess, do the initialize handshake, and start a
        thread. Returns the codex thread id. Idempotent — repeated calls
        return the same thread id."""
        if self._thread_id is not None:
            return self._thread_id
        if self._client is None:
            self._client = self._client_factory(
                codex_bin=self._codex_bin, codex_home=self._codex_home
            )
        self._client.initialize(
            client_name="hermes",
            client_title="Hermes Agent",
            client_version=_get_hermes_version(),
        )
        # Permission selection is intentionally NOT sent on thread/start.
        # Two reasons (live-tested against codex 0.130.0):
        #   1. `thread/start.permissions` is gated behind the experimentalApi
        #      capability on this codex version — we'd have to opt in during
        #      initialize and accept the unstable surface.
        #   2. Even with experimentalApi declared and the correct shape
        #      (`{"type": "profile", "id": "..."}`, not `{"profileId": ...}`),
        #      codex requires a matching `[permissions]` table in
        #      ~/.codex/config.toml or it fails the request with
        #      'default_permissions requires a [permissions] table'.
        # Letting codex pick its default (`:read-only` unless the user has
        # configured otherwise in their codex config.toml) is the standard
        # codex CLI workflow and avoids fighting codex's own validation.
        # Users who want a write-capable profile configure it in their
        # ~/.codex/config.toml the same way they would for any codex usage.
        params: dict[str, Any] = {"cwd": self._cwd}
        result = self._client.request("thread/start", params, timeout=15)
        self._thread_id = result["thread"]["id"]
        logger.info(
            "codex app-server thread started: id=%s profile=%s cwd=%s",
            self._thread_id[:8],
            self._permission_profile,
            self._cwd,
        )
        return self._thread_id

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._client = None
        self._thread_id = None

    def __enter__(self) -> "CodexAppServerSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to issue turn/interrupt
        and unwind. Called by AIAgent's _interrupt_requested path."""
        self._interrupt_event.set()

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: str,
        *,
        turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
    ) -> TurnResult:
        """Send a user message and block until turn/completed, while
        forwarding server-initiated approval requests and projecting items
        into Hermes' messages shape."""
        self.ensure_started()
        assert self._client is not None and self._thread_id is not None

        self._interrupt_event.clear()
        projector = CodexEventProjector()
        result = TurnResult(thread_id=self._thread_id)

        # Send turn/start with the user input. Text-only for now (codex
        # supports rich content but Hermes' text path is the common case).
        try:
            ts = self._client.request(
                "turn/start",
                {
                    "threadId": self._thread_id,
                    "input": [{"type": "text", "text": user_input}],
                },
                timeout=10,
            )
        except CodexAppServerError as exc:
            result.error = f"turn/start failed: {exc}"
            return result

        result.turn_id = (ts.get("turn") or {}).get("id")
        deadline = time.time() + turn_timeout
        turn_complete = False

        while time.time() < deadline and not turn_complete:
            if self._interrupt_event.is_set():
                self._issue_interrupt(result.turn_id)
                result.interrupted = True
                break

            # Drain any server-initiated requests (approvals) before
            # reading notifications, so the codex side isn't blocked.
            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                # Drain any pending notifications first so per-turn state
                # (e.g. _pending_file_changes for fileChange approvals) is
                # up to date when we make the approval decision. Bounded
                # to avoid starving the server-request response.
                for _ in range(8):
                    pending = self._client.take_notification(timeout=0)
                    if pending is None:
                        break
                    self._track_pending_file_change(pending)
                    proj = projector.project(pending)
                    if proj.messages:
                        result.projected_messages.extend(proj.messages)
                    if proj.is_tool_iteration:
                        result.tool_iterations += 1
                    if proj.final_text is not None:
                        result.final_text = proj.final_text
                self._handle_server_request(sreq)
                continue

            note = self._client.take_notification(
                timeout=notification_poll_timeout
            )
            if note is None:
                continue

            method = note.get("method", "")
            if self._on_event is not None:
                try:
                    self._on_event(note)
                except Exception:  # pragma: no cover - display callback
                    logger.debug("on_event callback raised", exc_info=True)

            # Track in-progress fileChange items so the approval bridge
            # can surface a real change summary when codex requests
            # approval (the approval params themselves don't carry the
            # changeset). Quirk #4 fix.
            self._track_pending_file_change(note)

            # Project into messages
            projection = projector.project(note)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
            if projection.final_text is not None:
                # Codex can emit multiple agentMessage items in one turn
                # (e.g. partial then final). Take the last one as canonical.
                result.final_text = projection.final_text

            if method == "turn/completed":
                turn_complete = True
                turn_status = (
                    (note.get("params") or {}).get("turn") or {}
                ).get("status")
                if turn_status and turn_status not in ("completed", "interrupted"):
                    err_obj = (
                        (note.get("params") or {}).get("turn") or {}
                    ).get("error")
                    if err_obj:
                        result.error = (
                            f"turn ended status={turn_status}: "
                            f"{err_obj.get('message') or err_obj}"
                        )

        if not turn_complete and not result.interrupted:
            # Hit the deadline. Issue interrupt to stop wasted compute.
            self._issue_interrupt(result.turn_id)
            result.interrupted = True
            result.error = result.error or f"turn timed out after {turn_timeout}s"

        return result

    # ---------- internals ----------

    def _issue_interrupt(self, turn_id: Optional[str]) -> None:
        if self._client is None or self._thread_id is None or turn_id is None:
            return
        try:
            self._client.request(
                "turn/interrupt",
                {"threadId": self._thread_id, "turnId": turn_id},
                timeout=5,
            )
        except CodexAppServerError as exc:
            # "no active turn to interrupt" is fine — already done.
            logger.debug("turn/interrupt non-fatal: %s", exc)
        except TimeoutError:
            logger.warning("turn/interrupt timed out")

    def _handle_server_request(self, req: dict) -> None:
        """Translate a codex server request (approval) into Hermes' approval
        flow, then send the response.

        Method names verified live against codex 0.130.0 (Apr 2026):
          item/commandExecution/requestApproval — exec approvals
          item/fileChange/requestApproval       — apply_patch approvals
          item/permissions/requestApproval      — permissions changes
                                                  (we decline; user controls
                                                  permission profile in
                                                  ~/.codex/config.toml).
        """
        if self._client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")
        params = req.get("params") or {}

        if method == "item/commandExecution/requestApproval":
            decision = self._decide_exec_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/fileChange/requestApproval":
            decision = self._decide_apply_patch_approval(params)
            self._client.respond(rid, {"decision": decision})
        elif method == "item/permissions/requestApproval":
            # Codex sometimes asks to escalate permissions mid-turn. We
            # always decline — the user already chose their permission
            # profile in ~/.codex/config.toml and surprise escalations
            # shouldn't be silently accepted.
            self._client.respond(rid, {"decision": "decline"})
        elif method == "mcpServer/elicitation/request":
            # Codex's MCP layer asks the user for structured input on
            # behalf of an MCP server (e.g. tool-call confirmation,
            # OAuth, form data). For our own hermes-tools callback we
            # auto-accept — the user already approved Hermes' tools
            # by enabling the runtime, and we never expose anything
            # codex's built-in shell can't already do. For other MCP
            # servers we decline so the user explicitly opts in via
            # codex's own auth flow.
            server_name = params.get("serverName") or ""
            if server_name == "hermes-tools":
                self._client.respond(
                    rid,
                    {"action": "accept", "content": None, "_meta": None},
                )
            else:
                self._client.respond(
                    rid,
                    {"action": "decline", "content": None, "_meta": None},
                )
        else:
            # Unknown server request — codex can extend this surface. Reject
            # cleanly so codex doesn't hang waiting for us.
            logger.warning("Unknown codex server request: %s", method)
            self._client.respond_error(
                rid, code=-32601, message=f"Unsupported method: {method}"
            )

    def _decide_exec_approval(self, params: dict) -> str:
        if self._routing.auto_approve_exec:
            return "accept"
        command = params.get("command") or ""
        # Codex's CommandExecutionRequestApprovalParams has cwd as Optional —
        # fall back to the session's cwd when codex doesn't include it so the
        # approval prompt is never empty (quirk #10 fix).
        cwd = params.get("cwd") or self._cwd or "<unknown>"
        reason = params.get("reason")
        description = f"Codex requests exec in {cwd}"
        if reason:
            description += f" — {reason}"
        if self._approval_callback is not None:
            try:
                choice = self._approval_callback(
                    command, description, allow_permanent=False
                )
                return _approval_choice_to_codex_decision(choice)
            except Exception:
                logger.exception("approval_callback raised on exec request")
                return "decline"
        return "decline"  # fail-closed when no callback wired

    def _decide_apply_patch_approval(self, params: dict) -> str:
        if self._routing.auto_approve_apply_patch:
            return "accept"
        if self._approval_callback is not None:
            # FileChangeRequestApprovalParams gives us reason + grantRoot.
            # The actual changeset lives on the corresponding fileChange
            # item which the projector has already cached for us — look it
            # up by item_id so the user sees what's actually changing.
            reason = params.get("reason")
            grant_root = params.get("grantRoot")
            item_id = params.get("itemId") or ""
            change_summary = self._lookup_pending_file_change(item_id)
            description_parts = []
            if reason:
                description_parts.append(reason)
            if change_summary:
                description_parts.append(change_summary)
            if grant_root:
                description_parts.append(f"grants write to {grant_root}")
            description = (
                "; ".join(description_parts)
                if description_parts
                else "Codex requests to apply a patch"
            )
            command_label = (
                f"apply_patch: {change_summary}" if change_summary
                else f"apply_patch: {reason}" if reason
                else "apply_patch"
            )
            try:
                choice = self._approval_callback(
                    command_label,
                    description,
                    allow_permanent=False,
                )
                return _approval_choice_to_codex_decision(choice)
            except Exception:
                logger.exception("approval_callback raised on apply_patch")
                return "decline"
        return "decline"

    def _track_pending_file_change(self, note: dict) -> None:
        """Maintain self._pending_file_changes from item/started + item/completed
        notifications. Lets the apply_patch approval prompt show what's
        actually changing — codex's approval params don't carry the data."""
        method = note.get("method", "")
        params = note.get("params") or {}
        item = params.get("item") or {}
        if item.get("type") != "fileChange":
            return
        item_id = item.get("id") or ""
        if not item_id:
            return
        if method == "item/started":
            changes = item.get("changes") or []
            if not changes:
                self._pending_file_changes[item_id] = "1 change pending"
                return
            kinds: dict[str, int] = {}
            paths: list[str] = []
            for ch in changes:
                if not isinstance(ch, dict):
                    continue
                kind = (ch.get("kind") or {}).get("type") or "update"
                kinds[kind] = kinds.get(kind, 0) + 1
                p = ch.get("path") or ""
                if p:
                    paths.append(p)
            counts = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            preview = ", ".join(paths[:3])
            if len(paths) > 3:
                preview += f", +{len(paths) - 3} more"
            self._pending_file_changes[item_id] = (
                f"{counts}: {preview}" if preview else counts
            )
        elif method == "item/completed":
            self._pending_file_changes.pop(item_id, None)

    def _lookup_pending_file_change(self, item_id: str) -> Optional[str]:
        """Look up an in-progress fileChange item by id and summarize its
        changes for the approval prompt. Returns None when we don't have
        the item cached (e.g. approval arrived before item/started, or
        fileChange item content not tracked yet)."""
        if not item_id:
            return None
        cached = self._pending_file_changes.get(item_id)
        if not cached:
            return None
        return cached


def _approval_choice_to_codex_decision(choice: str) -> str:
    """Map Hermes approval choices onto codex's CommandExecutionApprovalDecision
    / FileChangeApprovalDecision wire values.

    Hermes returns 'once', 'session', 'always', or 'deny'.
    Codex expects 'accept', 'acceptForSession', 'decline', or 'cancel'
    (verified against codex-rs/app-server-protocol/src/protocol/v2/item.rs
    on codex 0.130.0).
    """
    if choice in ("once",):
        return "accept"
    if choice in ("session", "always"):
        return "acceptForSession"
    return "decline"


def _get_hermes_version() -> str:
    """Best-effort Hermes version string for codex's userAgent line."""
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"
