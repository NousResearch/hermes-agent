"""
CodexAdapter — session-orchestration adapter for the Codex CLI.

Managed Codex sessions run as an interactive TUI in tmux. The adapter boots
Codex without an initial prompt, waits for the composer, then the generic
spawn flow seeds the first prompt through ``drive()``.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Protocol

from session_orchestration.adapters.base import AgentAdapter, TuiNotReadyError
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle

_CAPTURE_LINES: int = 80
_POLL_INTERVAL: float = 0.5
_LAUNCH_READY_TIMEOUT: float = 30.0
_DIALOG_POLL_TIMEOUT: float = 20.0

_APP_CODEX_BINARY: str = "/Applications/Codex.app/Contents/Resources/codex"
_CODEX_BINARY: str = _APP_CODEX_BINARY if os.path.exists(_APP_CODEX_BINARY) else "codex"
HANDOFF_MARKER: str = "HERMES_HANDOFF"
PROMPT_FRAGMENT: str = "› Implement {feature}"
PROMPT_PATTERN: re.Pattern[str] = re.compile(r"^\s*›\s+(?!\d+\.)(.+)$", re.MULTILINE)
TRUST_DIALOG_FRAGMENT: str = "Do you trust the contents of this directory?"
UPDATE_DIALOG_FRAGMENT: str = "Update available"

ACTIVITY_REGEX: re.Pattern[str] = re.compile(
    r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]|thinking|working|running|executing|applying",
    re.IGNORECASE,
)


class TmuxRunner(Protocol):
    """Minimal interface over the ``tmux`` CLI used by ``CodexAdapter``."""

    def run(self, args: list[str], check: bool = True) -> str:
        """Run ``tmux <args>`` and return stripped stdout."""
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


def build_interactive_argv(
    *,
    workdir: str,
    model: str | None = None,
) -> list[str]:
    """Build argv for an interactive Codex TUI session."""
    args: list[str] = ["-C", workdir]
    if model:
        args += ["--model", model]
    return args


def parse_pane_lifecycle(pane_text: str) -> SessionLifecycle:
    """Infer a lifecycle state from captured Codex tmux pane text."""
    if HANDOFF_MARKER in pane_text:
        return SessionLifecycle.PAUSED_HANDOFF
    if _has_startup_dialog(pane_text) or _is_ready_for_input(pane_text):
        return SessionLifecycle.WAITING_USER
    if ACTIVITY_REGEX.search(pane_text):
        return SessionLifecycle.RUNNING
    return SessionLifecycle.RUNNING


def _has_startup_dialog(pane_text: str) -> bool:
    return TRUST_DIALOG_FRAGMENT in pane_text or UPDATE_DIALOG_FRAGMENT in pane_text


def _is_ready_for_input(pane_text: str) -> bool:
    return bool(PROMPT_PATTERN.search(pane_text)) and not _has_startup_dialog(pane_text)


class CodexAdapter(AgentAdapter):
    """Adapter that orchestrates an interactive Codex CLI session via tmux."""

    def __init__(
        self,
        tmux_runner: TmuxRunner | None = None,
        session_prefix: str = "hermes-codex",
        binary: str = _CODEX_BINARY,
        pane_width: int = 200,
        pane_height: int = 50,
    ) -> None:
        self._tmux: TmuxRunner = tmux_runner or _SubprocessTmuxRunner()
        self._session_prefix = session_prefix
        self._binary = binary
        self._pane_width = pane_width
        self._pane_height = pane_height

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_print_mode=True,
            has_hooks=False,
            rpc_mode=False,
            json_mode=False,
            idle_indicator_regex=ACTIVITY_REGEX,
            dialog_handlers={
                TRUST_DIALOG_FRAGMENT: self._handle_trust_dialog,
                UPDATE_DIALOG_FRAGMENT: self._handle_update_dialog,
            },
        )

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        session_id = str(uuid.uuid4())
        tmux_session = f"{self._session_prefix}-{session_id[:8]}"
        marker_file = f"{workdir}/.hermes/sessions/{session_id}.jsonl"
        os.makedirs(os.path.dirname(marker_file), exist_ok=True)

        self._tmux.run([
            "new-session", "-d",
            "-s", tmux_session,
            "-x", str(self._pane_width),
            "-y", str(self._pane_height),
            "-e", f"HERMES_MARKER_FILE={marker_file}",
        ])

        pane = self._tmux.run([
            "list-panes", "-t", tmux_session, "-F", "#{pane_id}",
        ]).splitlines()[0].strip()

        argv = build_interactive_argv(workdir=workdir)
        cmd = shlex.join([self._binary] + argv)
        self._tmux.run(["send-keys", "-t", pane, cmd, "Enter"])

        try:
            self._handle_dialogs(pane)
            self._wait_for_ready(pane, timeout=_LAUNCH_READY_TIMEOUT)
        except TimeoutError:
            try:
                self._tmux.run(["kill-session", "-t", tmux_session], check=False)
            except Exception:
                pass
            raise

        return SessionHandle(
            session_id=session_id,
            tmux_session=tmux_session,
            pane=pane,
            launch_ts=datetime.now(tz=timezone.utc),
            marker_file=marker_file,
        )

    def drive(
        self,
        handle: SessionHandle,
        message: str,
        *,
        pre_keys: list[str] | None = None,
        type_ahead: bool = False,
    ) -> None:
        for key in pre_keys or []:
            self._tmux.run(["send-keys", "-t", handle.pane, key])
        try:
            self._wait_for_ready(handle.pane, timeout=_LAUNCH_READY_TIMEOUT)
        except TimeoutError as exc:
            if type_ahead:
                raise TuiNotReadyError(str(exc)) from exc
            raise

        buf_name = f"hermes-codex-{handle.session_id[:8]}"
        self._load_buffer(buf_name, message)
        self._tmux.run(["paste-buffer", "-d", "-b", buf_name, "-t", handle.pane])
        self._tmux.run(["send-keys", "-t", handle.pane, "", "Enter"])

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        pane_text = self._capture_pane(handle.pane)
        if pane_text is None:
            return SessionLifecycle.ERROR
        return parse_pane_lifecycle(pane_text)

    def resume(self, handle: SessionHandle, prompt: str, *, force: bool = False) -> None:
        if not force and self.detect(handle) != SessionLifecycle.PAUSED_HANDOFF:
            import logging
            logging.getLogger(__name__).warning(
                "resume() called on codex session %s outside PAUSED_HANDOFF; no-op.",
                handle.session_id,
            )
            return

        self._tmux.run(["send-keys", "-t", handle.pane, "/clear", "Enter"])
        if prompt:
            self.drive(handle, prompt)

    def terminate(self, handle: SessionHandle) -> None:
        try:
            self._tmux.run(["kill-session", "-t", handle.tmux_session], check=False)
        except subprocess.CalledProcessError:
            pass

    def _capture_pane(self, pane: str) -> str | None:
        try:
            return self._tmux.run([
                "capture-pane", "-t", pane,
                "-p",
                "-S", f"-{_CAPTURE_LINES}",
            ], check=True)
        except subprocess.CalledProcessError:
            return None

    def _wait_for_ready(self, pane: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self._capture_pane(pane)
            if text is not None and _is_ready_for_input(text):
                return
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError(
            f"Codex prompt not visible in pane {pane!r} after {timeout:.0f}s"
        )

    def _handle_dialogs(self, pane: str) -> None:
        deadline = time.monotonic() + _DIALOG_POLL_TIMEOUT
        trust_handled = False
        update_handled = False

        while time.monotonic() < deadline:
            text = self._capture_pane(pane) or ""
            if _is_ready_for_input(text):
                return
            if not trust_handled and TRUST_DIALOG_FRAGMENT in text:
                self._handle_trust_dialog(pane)
                trust_handled = True
                time.sleep(1.0)
                continue
            if not update_handled and UPDATE_DIALOG_FRAGMENT in text:
                self._handle_update_dialog(pane)
                update_handled = True
                time.sleep(1.0)
                continue
            time.sleep(_POLL_INTERVAL)

    def _handle_trust_dialog(self, pane: str) -> None:
        self._tmux.run(["send-keys", "-t", pane, "", "Enter"])

    def _handle_update_dialog(self, pane: str) -> None:
        self._tmux.run(["send-keys", "-t", pane, "Down", "Enter"])

    def _load_buffer(self, buf_name: str, content: str) -> None:
        subprocess.run(
            ["tmux", "load-buffer", "-b", buf_name, "-"],
            input=content,
            text=True,
            check=True,
        )
