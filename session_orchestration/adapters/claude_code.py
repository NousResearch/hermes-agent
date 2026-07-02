"""
ClaudeCodeAdapter — session-orchestration adapter for Claude Code CLI.

Lifecycle
---------
1. ``launch()``   — ``tmux new-session`` → ``cd <workdir> && claude --dangerously-skip-permissions``
                    → trust-dialog handler (Enter) → bypass-permissions handler (Down, Enter)
                    → prompt-readiness wait → ``drive()`` with initial prompt.
2. ``drive()``    — prompt-readiness check (❯ visible in pane) → ``load-buffer`` → ``paste-buffer``
                    (NOT ``send-keys``; avoids shell-metacharacter expansion on multi-line prompts).
3. ``detect()``   — capture pane → parse last N lines for ❯ (WAITING_USER), ● (RUNNING),
                    handoff keyword (PAUSED_HANDOFF), session-dead / error markers (ERROR/DONE).
4. ``resume()``   — idempotent: only acts when ``detect()`` returns ``PAUSED_HANDOFF``; sends
                    ``/clear`` then re-injects the new prompt via ``drive()``.

Tmux injection strategy
-----------------------
``drive()`` uses ``load-buffer`` + ``paste-buffer -d`` rather than ``send-keys`` because:
- Multi-line prompts survive intact (``send-keys`` requires ``\n`` escaping for each newline).
- Shell metacharacters are not expanded.
- A pipe character mid-prompt won't be misinterpreted.

The tmux runner is injectable (``TmuxRunner`` protocol) so every non-live behaviour can be unit-
tested without a real tmux session.

Live-only parts
---------------
``launch()`` and the ``_handle_dialogs()`` inner helper perform real ``subprocess.run`` calls and
real timing sleeps; they cannot be verified without a live Claude Code + tmux environment.
``detect()`` and ``drive()`` are fully testable with a stubbed ``TmuxRunner``.

Dialog handling sequence (first launch into a directory)
---------------------------------------------------------
Claude Code may show two sequential dialogs:

  Dialog 1 — Workspace Trust (first visit only)
  ┌─────────────────────────────────────────┐
  │ ❯ 1. Yes, I trust this folder  ← default│
  │   2. No, exit                           │
  └─────────────────────────────────────────┘
  → press Enter (default selection is correct)

  Dialog 2 — Bypass-Permissions Warning (every ``--dangerously-skip-permissions`` launch)
  ┌─────────────────────────────────────────┐
  │ ❯ 1. No, exit                  ← default│
  │   2. Yes, I accept                      │
  └─────────────────────────────────────────┘
  → press Down then Enter (default is wrong!)

Both dialogs render into the same pane within a few seconds of startup.  ``launch()`` polls for
each known dialog fragment and handles them in order.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Protocol

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lines captured from the bottom of the pane when polling/detecting.
_CAPTURE_LINES: int = 60

#: Seconds between each readiness-poll iteration.
_POLL_INTERVAL: float = 0.5

#: Maximum seconds to wait for the ❯ prompt after launching/clearing.
_LAUNCH_READY_TIMEOUT: float = 30.0

#: Maximum seconds to wait between dialog checks during launch.
_DIALOG_POLL_TIMEOUT: float = 20.0

#: Compiled regex that matches an active-work indicator line (●).
#: This is the ``idle_indicator_regex`` declared in ``Capabilities``.
ACTIVITY_REGEX: re.Pattern[str] = re.compile(r"●")

#: The ❯ prompt marker — presence (alone on a line-end boundary) signals WAITING_USER.
PROMPT_MARKER: str = "❯"

#: Keyword fragment that signals the agent reached a handoff checkpoint.
#: Claude Code writes "Hermes handoff" or "/clear checkpoint" style lines when instructed.
HANDOFF_MARKER: str = "HERMES_HANDOFF"

#: Fragment present in the trust dialog.
_TRUST_DIALOG_FRAGMENT: str = "trust this folder"

#: Fragment present in the bypass-permissions dialog.
_BYPASS_DIALOG_FRAGMENT: str = "dangerously"

# ---------------------------------------------------------------------------
# TmuxRunner protocol (injectable for testing)
# ---------------------------------------------------------------------------


class TmuxRunner(Protocol):
    """Minimal interface over the ``tmux`` CLI used by ``ClaudeCodeAdapter``.

    Production code uses ``_SubprocessTmuxRunner``; tests inject a fake.
    """

    def run(self, args: list[str], check: bool = True) -> str:
        """Run ``tmux <args>`` and return stdout (stripped).

        Parameters
        ----------
        args:
            Arguments to pass after ``tmux`` (do NOT include ``"tmux"`` itself).
        check:
            If True, raise on nonzero returncode.

        Returns
        -------
        str
            Stdout of the command (stripped).
        """
        ...


class _SubprocessTmuxRunner:
    """Production ``TmuxRunner`` — delegates to ``subprocess.run``."""

    def run(self, args: list[str], check: bool = True) -> str:
        result = subprocess.run(
            ["tmux"] + args,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.stdout.strip()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ClaudeCodeAdapter(AgentAdapter):
    """Adapter that orchestrates an interactive Claude Code CLI session via tmux.

    Parameters
    ----------
    tmux_runner:
        Override the tmux execution back-end.  Defaults to the real
        ``subprocess``-backed runner.  Pass a stub in tests.
    session_prefix:
        Prefix for generated tmux session names.  Default ``"hermes-cc"``.
    pane_width:
        tmux pane width (columns).  Default 200.
    pane_height:
        tmux pane height (rows).  Default 50.
    """

    def __init__(
        self,
        tmux_runner: TmuxRunner | None = None,
        session_prefix: str = "hermes-cc",
        pane_width: int = 200,
        pane_height: int = 50,
    ) -> None:
        self._tmux: TmuxRunner = tmux_runner or _SubprocessTmuxRunner()
        self._session_prefix = session_prefix
        self._pane_width = pane_width
        self._pane_height = pane_height

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        """Declare Claude Code adapter capabilities.

        ``has_hooks=True`` — Claude Code supports ``--hook`` lifecycle callbacks
        (``Stop``, ``PostToolUse``, etc.) which can serve as a positive-liveness
        accelerant for the watcher.

        ``idle_indicator_regex`` — matches the ``●`` active-tool indicator that
        appears while Claude Code is reading/writing/running commands.  The watcher
        uses this to distinguish genuine work from pane-hash-unchanged stalls.
        """
        return Capabilities(
            supports_print_mode=True,
            has_hooks=True,
            rpc_mode=False,
            json_mode=True,
            idle_indicator_regex=ACTIVITY_REGEX,
            dialog_handlers={
                _TRUST_DIALOG_FRAGMENT: self._handle_trust_dialog,
                _BYPASS_DIALOG_FRAGMENT: self._handle_bypass_dialog,
            },
        )

    # ------------------------------------------------------------------
    # launch()  [LIVE — requires real tmux + claude binary]
    # ------------------------------------------------------------------

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        """Spawn a tmux session and start Claude Code inside it.

        Sequence (live):
        1. Generate a unique tmux session name.
        2. ``tmux new-session -d`` with the configured dimensions.
        3. ``cd <workdir> && claude --dangerously-skip-permissions``.
        4. Wait for the trust dialog → press Enter.
        5. Wait for the bypass-permissions dialog → press Down + Enter.
        6. Wait for the ❯ prompt (ready state).
        7. Call ``drive()`` with the initial prompt.

        Parameters
        ----------
        workdir:
            Absolute path to the working directory.  Must exist.
        prompt:
            Initial prompt to inject once Claude is ready.

        Returns
        -------
        SessionHandle
            Populated handle; store in the registry immediately.

        Notes
        -----
        This method is LIVE-ONLY.  It cannot be unit-tested without a real tmux
        session.  The dialog poll loops use ``time.sleep`` which is not injectable
        in the current implementation.  Extract ``_sleep`` if you need to test
        launch() in isolation (the current design trades simplicity for live-correctness).
        """
        session_id = str(uuid.uuid4())
        tmux_session = f"{self._session_prefix}-{session_id[:8]}"

        # Compute the marker file path and ensure its parent directory exists.
        marker_path = f"{workdir}/.hermes/sessions/{session_id}.jsonl"
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)

        # 1. Create a detached tmux session, injecting the marker file env var.
        self._tmux.run([
            "new-session", "-d",
            "-s", tmux_session,
            "-x", str(self._pane_width),
            "-y", str(self._pane_height),
            "-e", f"HERMES_MARKER_FILE={marker_path}",
        ])

        # Resolve the REAL pane id rather than assuming ':0.0'. A hardcoded
        # ':0.0' breaks under `set -g base-index 1` / `pane-base-index 1`
        # (the first pane is then ':1.1'). The stable pane id (e.g. '%12') is
        # immune to base-index and to later window/pane renumbering.
        pane = self._tmux.run([
            "list-panes", "-t", tmux_session, "-F", "#{pane_id}",
        ]).splitlines()[0].strip()

        # 2. Launch Claude Code inside the session.
        cmd = f"cd {_shell_quote(workdir)} && claude --dangerously-skip-permissions"
        self._tmux.run(["send-keys", "-t", pane, cmd, "Enter"])

        # 3. Handle the two startup dialogs.
        self._handle_dialogs(pane)

        # 4. Wait for the ❯ prompt (readiness).
        self._wait_for_prompt(pane, timeout=_LAUNCH_READY_TIMEOUT)

        handle = SessionHandle(
            session_id=session_id,
            tmux_session=tmux_session,
            pane=pane,
            launch_ts=datetime.now(tz=timezone.utc),
            marker_file=marker_path,
        )

        # 5. Inject the initial prompt.
        self.drive(handle, prompt)
        return handle

    # ------------------------------------------------------------------
    # drive()  [testable with stubbed TmuxRunner]
    # ------------------------------------------------------------------

    def drive(
        self,
        handle: SessionHandle,
        message: str,
        *,
        pre_keys: list[str] | None = None,
    ) -> None:
        """Deliver ``message`` to the Claude Code session via load-buffer/paste-buffer.

        Steps
        -----
        0. Send any ``pre_keys`` (e.g. ``Escape`` to cancel a selection menu).
        1. Poll pane for ❯ prompt (readiness).  Raises ``TimeoutError`` after
           ``_LAUNCH_READY_TIMEOUT`` seconds if the pane never reaches readiness.
        2. Write ``message`` to a named tmux buffer.
        3. ``paste-buffer -d`` into the pane (flushes and deletes the buffer).
        4. Send ``Enter`` to submit.

        The load-buffer/paste-buffer mechanism avoids shell-metacharacter
        expansion that ``send-keys`` would perform on strings containing ``$``,
        backticks, pipes, etc.  It also handles multi-line prompts reliably.

        Parameters
        ----------
        handle:
            The ``SessionHandle`` returned by ``launch()``.
        message:
            Text to send.  May contain newlines; they are preserved verbatim.
        pre_keys:
            Optional tmux key names sent before the readiness poll (e.g.
            ``Escape`` to cancel a selection menu).
        """
        for key in pre_keys or []:
            self._tmux.run(["send-keys", "-t", handle.pane, key])
        self._wait_for_prompt(handle.pane, timeout=_LAUNCH_READY_TIMEOUT)

        buf_name = f"hermes-{handle.session_id[:8]}"
        # Write message into a named tmux buffer via subprocess stdin pipe.
        # We bypass TmuxRunner here because stdin delivery requires subprocess.PIPE.
        # In tests, monkeypatch _load_buffer to capture the call.
        self._load_buffer(buf_name, message)
        # Paste the buffer into the pane (deletes the buffer after pasting).
        self._tmux.run(["paste-buffer", "-d", "-b", buf_name, "-t", handle.pane])
        # Submit.
        self._tmux.run(["send-keys", "-t", handle.pane, "", "Enter"])

    # ------------------------------------------------------------------
    # detect()  [testable with stubbed TmuxRunner]
    # ------------------------------------------------------------------

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        """Inspect the current pane state and return a lifecycle value.

        Logic (evaluated in order)
        --------------------------
        1. If pane is dead/session gone → ``ERROR``.
        2. If ``HANDOFF_MARKER`` present in last ``_CAPTURE_LINES`` lines → ``PAUSED_HANDOFF``.
        3. If ``❯`` present at end of visible text → ``WAITING_USER``.
        4. If ``●`` present → ``RUNNING`` (active tool use).
        5. Otherwise → ``RUNNING`` (default; pane has content but no clear idle signal).

        Parameters
        ----------
        handle:
            The ``SessionHandle`` for the session to inspect.

        Returns
        -------
        SessionLifecycle
            Exactly one of ``RUNNING | WAITING_USER | PAUSED_HANDOFF | ERROR``.
            ``STALLED`` and ``DONE`` are determined by the watcher based on
            pane-hash staleness and elapsed time, not by this method.
        """
        pane_text = self._capture_pane(handle.pane)
        if pane_text is None:
            return SessionLifecycle.ERROR

        return _parse_lifecycle(pane_text)

    # ------------------------------------------------------------------
    # resume()  [testable with stubbed TmuxRunner]
    # ------------------------------------------------------------------

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        """Perform a ``/clear`` + re-inject cycle for a handoff session.

        This method is idempotent: if the session is NOT in ``PAUSED_HANDOFF``
        state (i.e., ``detect()`` does not return ``PAUSED_HANDOFF``), it logs
        a warning and returns without taking action.

        Sequence
        --------
        1. Call ``detect()``.  If not ``PAUSED_HANDOFF``, warn and return.
        2. Send ``/clear`` + Enter via ``send-keys`` (slash-command, safe with send-keys).
        3. Wait for the ❯ prompt to reappear.
        4. Call ``drive()`` with the new prompt.

        Parameters
        ----------
        handle:
            The ``SessionHandle`` for the session to resume.
        prompt:
            Prompt to inject after clearing.
        """
        current = self.detect(handle)
        if current != SessionLifecycle.PAUSED_HANDOFF:
            import logging
            logging.getLogger(__name__).warning(
                "resume() called on session %s in state %s (expected PAUSED_HANDOFF); no-op.",
                handle.session_id,
                current.value,
            )
            return

        # /clear is a slash-command; send-keys is safe here (single token, no metacharacters).
        self._tmux.run(["send-keys", "-t", handle.pane, "/clear", "Enter"])
        self._wait_for_prompt(handle.pane, timeout=_LAUNCH_READY_TIMEOUT)
        self.drive(handle, prompt)

    # ------------------------------------------------------------------
    # terminate()  [testable with stubbed TmuxRunner]
    # ------------------------------------------------------------------

    def terminate(self, handle: SessionHandle) -> None:
        """Kill the tmux session associated with ``handle``.

        Sends ``tmux kill-session -t <tmux_session>`` with ``check=False`` so a
        nonzero exit (e.g. session already gone) is silently ignored.
        ``CalledProcessError`` is also swallowed for belt-and-suspenders safety
        when a custom ``TmuxRunner`` implementation raises on failure.

        Parameters
        ----------
        handle:
            The ``SessionHandle`` whose tmux session should be destroyed.
        """
        try:
            self._tmux.run(["kill-session", "-t", handle.tmux_session], check=False)
        except subprocess.CalledProcessError:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_pane(self, pane: str) -> str | None:
        """Capture the last ``_CAPTURE_LINES`` lines of the pane.

        Returns ``None`` if the pane does not exist (session dead / error).
        """
        try:
            text = self._tmux.run([
                "capture-pane", "-t", pane,
                "-p",               # print to stdout
                "-S", f"-{_CAPTURE_LINES}",  # start N lines from bottom
            ], check=True)
            return text
        except subprocess.CalledProcessError:
            return None

    def _wait_for_prompt(self, pane: str, timeout: float) -> None:
        """Poll until ❯ is visible in the pane or ``timeout`` seconds elapse.

        Raises
        ------
        TimeoutError
            If the prompt does not appear within ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self._capture_pane(pane)
            if text is not None and PROMPT_MARKER in text:
                return
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError(
            f"Claude Code prompt (❯) not visible in pane {pane!r} "
            f"after {timeout:.0f}s"
        )

    def _handle_dialogs(self, pane: str) -> None:
        """Drive Claude Code through the startup dialogs.

        LIVE-ONLY.  Polls for each dialog fragment, handles in order, then
        returns.  Uses ``time.sleep`` internally; not injectable in tests.

        Dialog 1 (trust): default is "Yes, trust folder" → press Enter.
        Dialog 2 (bypass permissions): default is "No, exit" → press Down, Enter.
        """
        deadline = time.monotonic() + _DIALOG_POLL_TIMEOUT
        trust_handled = False
        bypass_handled = False

        while time.monotonic() < deadline:
            text = self._capture_pane(pane) or ""

            if not trust_handled and _TRUST_DIALOG_FRAGMENT in text:
                self._handle_trust_dialog(pane)
                trust_handled = True
                time.sleep(0.5)

            if not bypass_handled and _BYPASS_DIALOG_FRAGMENT in text:
                self._handle_bypass_dialog(pane)
                bypass_handled = True
                time.sleep(0.5)

            # Once both are handled (or the prompt is visible), done.
            if (trust_handled and bypass_handled) or PROMPT_MARKER in text:
                return

            time.sleep(_POLL_INTERVAL)

    def _handle_trust_dialog(self, pane: str) -> None:
        """Handle Dialog 1 (Workspace Trust): press Enter for default "Yes"."""
        self._tmux.run(["send-keys", "-t", pane, "", "Enter"])

    def _handle_bypass_dialog(self, pane: str) -> None:
        """Handle Dialog 2 (Bypass-Permissions): press Down then Enter."""
        self._tmux.run(["send-keys", "-t", pane, "Down", ""])
        self._tmux.run(["send-keys", "-t", pane, "", "Enter"])

    def _load_buffer(self, buf_name: str, content: str) -> None:
        """Write ``content`` into a named tmux buffer via subprocess stdin pipe.

        This bypasses the ``TmuxRunner`` abstraction because ``load-buffer``
        requires data on stdin; the TmuxRunner protocol's ``run()`` method does
        not expose stdin.  Tests that stub the TmuxRunner should also stub this
        method (monkeypatch) if they need to assert buffer content.
        """
        subprocess.run(
            ["tmux", "load-buffer", "-b", buf_name, "-"],
            input=content,
            text=True,
            check=True,
        )


# ---------------------------------------------------------------------------
# Pure-function helpers (fully testable, no tmux dependency)
# ---------------------------------------------------------------------------


def _parse_lifecycle(pane_text: str) -> SessionLifecycle:
    """Parse captured pane text and return the inferred ``SessionLifecycle``.

    This is a pure function — no subprocess calls, no side-effects.  All
    detect() logic lives here so unit tests can exercise it without a real pane.

    Decision order
    --------------
    1. ``HANDOFF_MARKER`` anywhere in text → ``PAUSED_HANDOFF``
    2. ``❯`` anywhere in text → ``WAITING_USER``
    3. ``●`` anywhere in text → ``RUNNING``
    4. Otherwise → ``RUNNING`` (Claude is between outputs; watcher tracks staleness)

    Parameters
    ----------
    pane_text:
        Raw text from ``tmux capture-pane -p``.

    Returns
    -------
    SessionLifecycle
        One of ``RUNNING | WAITING_USER | PAUSED_HANDOFF``.
    """
    if HANDOFF_MARKER in pane_text:
        return SessionLifecycle.PAUSED_HANDOFF
    if PROMPT_MARKER in pane_text:
        return SessionLifecycle.WAITING_USER
    if ACTIVITY_REGEX.search(pane_text):
        return SessionLifecycle.RUNNING
    return SessionLifecycle.RUNNING


def _shell_quote(s: str) -> str:
    """Minimal shell-quoting for a directory path in a tmux send-keys argument.

    Wraps ``s`` in single quotes and escapes any literal single-quote characters
    within it.  Only used for the ``cd`` command in ``launch()``.
    """
    return "'" + s.replace("'", "'\\''") + "'"
