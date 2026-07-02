"""
OmpAdapter — session-orchestration adapter for oh-my-pi (omp) CLI.

Oh-my-pi (omp) v16 supports two structured transports that make it the
"cleaner" agent from an orchestration standpoint:

  --mode=json   One-shot NDJSON streaming output when combined with -p.
                Preferred for detect/drive when the interaction allows
                a subprocess-call pattern (no persistent session needed).

  --mode=rpc    Interactive RPC session over stdio.  Emits {"type":"ready"}
                on startup then accepts JSON commands.  Useful for a managed
                interactive session but requires a running process or tmux.

The adapter prefers --mode=json for one-shot (print) mode and falls back
to the tmux + --auto-approve pattern for interactive TUI driving.

Probed --mode=json NDJSON schema (omp v16.1.15)
------------------------------------------------
Each line is a JSON object with a ``"type"`` discriminator:

  {"type":"session","version":3,"id":"<uuid>","timestamp":"<ISO8601>","cwd":"<path>"}
  {"type":"agent_start"}
  {"type":"turn_start"}
  {"type":"message_start","message":{"role":"user","content":[{"type":"text","text":"..."}], ...}}
  {"type":"message_end","message":{"role":"user","content":[...], ...}}
  {"type":"message_start","message":{"role":"assistant","content":[{"type":"text","text":"..."}], ...}}
  {"type":"message_update","assistantMessageEvent":{...},"message":{...}}
  {"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"<answer>"}],
      "model":"<model>","usage":{...},"stopReason":"stop","timestamp":<ms>,...}}
  {"type":"turn_end","message":{...}}
  {"type":"agent_end","messages":[<all messages>]}

To extract the final assistant text:
  Parse each line; find ``agent_end``; last entry in ``messages`` with
  ``role == "assistant"``; ``content[0]["text"]``.

Alternatively, stream for lines where ``type == "message_end"`` and
``message.role == "assistant"`` — the last one is the final answer.

--mode=rpc interactive schema (omp v16.1.15)
--------------------------------------------
When launched WITHOUT -p (interactive TUI via --mode=rpc):
  Emits {"type":"ready"} then {"type":"available_commands_update","commands":[...]}
  The rpc session accepts stdin JSON commands for slash-commands and controls.
  NOT used for one-shot; used when omp is left running in a tmux pane.

Detect (interactive tmux session)
----------------------------------
omp TUI uses a prompt marker (">") visible at pane bottom when waiting for
user input.  The adapter captures the pane and looks for:
  - HERMES_HANDOFF keyword → PAUSED_HANDOFF
  - ">" or "❯" at end → WAITING_USER
  - Session process exited → DONE or ERROR
  - Otherwise → RUNNING

Interactive transport choice
-----------------------------
For interactive sessions the adapter uses tmux + --auto-approve.  omp's
--auto-approve skips all tool-approval dialogs so there is NO dialog-dance
required (unlike claude-code).  This means launch() is simpler.

Resume (interactive)
---------------------
omp -c / -r <session-id> continues a previous session.  The adapter stores
the omp session-dir in the handle's metadata so -r can target the right one.
For one-shot continuations, -c re-attaches the most recent session.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Iterable, Protocol

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.markers import MARKER_DONE, MARKER_HEARTBEAT, MARKER_NEEDS_INPUT, MARKER_STATUS, append_marker
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lines captured from the pane bottom for state detection.
_CAPTURE_LINES: int = 60

#: Seconds between each readiness-poll iteration.
_POLL_INTERVAL: float = 0.5

#: Maximum seconds to wait for omp prompt after launching.
_LAUNCH_READY_TIMEOUT: float = 30.0

#: Maximum seconds to wait for omp's TUI to first render on COLD start. Larger
#: than _LAUNCH_READY_TIMEOUT because a cold start does an update check + MCP
#: server connection + heavy splash render before drawing box chrome; on a slow
#: network that legitimately exceeds 30s (observed omp/16.1.15 tripping the old
#: 30s budget even though the TUI came up fine seconds later). drive()/resume()
#: keep the shorter budget since omp is already running by then.
_LAUNCH_START_TIMEOUT: float = 90.0

#: omp prints ">" or "❯" when waiting for user input.
#: Use a pattern that matches either variant.
PROMPT_PATTERN: re.Pattern[str] = re.compile(r"[>❯]\s*$", re.MULTILINE)

#: Handoff keyword written by omp agent instructions.
HANDOFF_MARKER: str = "HERMES_HANDOFF"

#: Active-work indicator pattern for omp (tool-use spinner text).
#: omp shows "⠋" / "⠙" / "⠹" etc. spinner characters and "Running" text.
ACTIVITY_REGEX: re.Pattern[str] = re.compile(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]|Running tool")

#: Interactive input request pattern for omp. Selection menus and prompt
#: dialogs (including z-harness AskUser gates rendered inside omp) end in a
#: navigation footer — "up/down navigate  enter select  esc cancel" — and do
#: NOT use a "❯"/">" input char, so PROMPT_PATTERN misses them. omp also keeps
#: a spinner ("⠹ Requesting task") visible while the menu waits, which would
#: otherwise be misread as RUNNING. This pattern must therefore take precedence
#: over ACTIVITY_REGEX in the lifecycle decision.
INPUT_REQUEST_REGEX: re.Pattern[str] = re.compile(
    r"enter\s+select|esc\s+cancel|↵\s*select|⏎\s*select", re.IGNORECASE
)

#: Default executable path for omp.
_OMP_BINARY: str = "omp"

# ---------------------------------------------------------------------------
# OmpRunner protocol (injectable for testing)
# ---------------------------------------------------------------------------


class OmpRunner(Protocol):
    """Minimal interface over the omp CLI used by ``OmpAdapter``.

    Production code uses ``_SubprocessOmpRunner``; tests inject a fake.
    """

    def run_oneshot(
        self,
        args: list[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run omp as a subprocess and return the completed process.

        Parameters
        ----------
        args:
            Arguments to pass to the omp binary (do NOT include the binary
            itself at index 0 — the runner adds it).
        check:
            If True, raise ``subprocess.CalledProcessError`` on nonzero exit.

        Returns
        -------
        subprocess.CompletedProcess[str]
            The completed process with captured stdout/stderr.
        """
        ...


class TmuxRunner(Protocol):
    """Minimal interface over the tmux CLI used by ``OmpAdapter``."""

    def run(self, args: list[str], check: bool = True) -> str:
        """Run ``tmux <args>`` and return stdout (stripped)."""
        ...


class _SubprocessOmpRunner:
    """Production ``OmpRunner`` — delegates to ``subprocess.run``."""

    def __init__(self, binary: str = _OMP_BINARY) -> None:
        self._binary = binary

    def run_oneshot(
        self,
        args: list[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._binary] + args,
            capture_output=True,
            text=True,
            check=check,
        )


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
# Pure-function helpers (fully testable, no subprocess dependency)
# ---------------------------------------------------------------------------


def parse_oneshot_result(ndjson_output: str) -> str:
    """Extract the final assistant text from an omp --mode=json NDJSON stream.

    Scans the stream for ``agent_end`` (preferred) or the last
    ``message_end`` with ``role == "assistant"`` (fallback).

    Parameters
    ----------
    ndjson_output:
        Raw stdout from ``omp -p --mode=json ...``.  Each line is a JSON object.

    Returns
    -------
    str
        The assistant's final text response.  Empty string if no assistant
        message was found (error / empty run).

    Raises
    ------
    ValueError
        If no line can be parsed as valid JSON (suggests omp failed before
        emitting any structured output).
    """
    parsed_any = False
    last_assistant_text: str = ""

    for raw_line in ndjson_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed_any = True

        event_type = event.get("type", "")

        if event_type == "agent_end":
            messages = event.get("messages", [])
            # Walk in reverse to find the last assistant message.
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if content:
                        return content[0].get("text", "")
            return last_assistant_text

        if event_type == "message_end":
            msg = event.get("message", {})
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if content:
                    last_assistant_text = content[0].get("text", "")

    if not parsed_any:
        raise ValueError("No parseable JSON lines in omp output")

    return last_assistant_text


def parse_pane_lifecycle(pane_text: str) -> SessionLifecycle:
    """Infer a lifecycle state from captured omp tmux pane text.

    Decision order (applied to the last ``_CAPTURE_LINES`` lines)
    -------------------------------------------------------------
    1. ``HERMES_HANDOFF`` anywhere → ``PAUSED_HANDOFF``
    2. Prompt pattern (``>`` or ``❯`` at line end) → ``WAITING_USER``
    3. Activity indicator (spinner or "Running tool") → ``RUNNING``
    4. Otherwise → ``RUNNING`` (default; watcher tracks staleness)

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
    # An interactive menu/prompt takes precedence over the spinner: omp keeps a
    # "⠹ Requesting task" spinner visible WHILE a selection menu waits, so the
    # input-request check must run before ACTIVITY_REGEX or a waiting session
    # would be misread as RUNNING and never surface in the feed.
    if INPUT_REQUEST_REGEX.search(pane_text) or PROMPT_PATTERN.search(pane_text):
        return SessionLifecycle.WAITING_USER
    if ACTIVITY_REGEX.search(pane_text):
        return SessionLifecycle.RUNNING
    return SessionLifecycle.RUNNING


#: omp's composer/status chrome — the rounded box it draws around the input line.
_TUI_BOX_CHARS = ("╭", "╰")


def pane_is_idle_waiting(pane_text: str) -> bool:
    """True when omp has finished its turn and is idling at its composer.

    omp emits no marker and no ``❯`` when it asks a *free-form* question and
    waits (unlike a selection menu, which shows an ``enter select`` footer). The
    only signal is: the TUI composer box is drawn AND no active-work spinner is
    present AND it isn't already a menu/prompt/handoff. This is a *candidate*
    wait — the watcher additionally requires the pane to be stable across a tick
    before promoting RUNNING → WAITING_USER, so a momentary mid-turn idle does
    not flap the session into a spurious notification.
    """
    if not pane_text or not pane_text.strip():
        return False
    if HANDOFF_MARKER in pane_text:
        return False
    # Menus/prompts are already WAITING_USER via parse_pane_lifecycle; a busy
    # spinner means omp is still working.
    if INPUT_REQUEST_REGEX.search(pane_text) or PROMPT_PATTERN.search(pane_text):
        return False
    if ACTIVITY_REGEX.search(pane_text):
        return False
    return any(ch in pane_text for ch in _TUI_BOX_CHARS)


def build_oneshot_argv(
    prompt: str,
    model: str | None = None,
    workdir: str | None = None,
    max_time: int | None = None,
    no_session: bool = True,
) -> list[str]:
    """Build the argument list for a one-shot omp invocation.

    Parameters
    ----------
    prompt:
        The prompt text to send to omp.
    model:
        Optional model override (e.g. ``"openai-codex/gpt-5.5"``).
    workdir:
        Optional ``--cwd`` override.
    max_time:
        Optional ``--max-time`` in seconds.
    no_session:
        If True (default), pass ``--no-session`` to avoid persisting the
        one-shot interaction.

    Returns
    -------
    list[str]
        Argument list to pass to ``OmpRunner.run_oneshot`` (binary not included).
    """
    args: list[str] = ["-p", "--mode=json"]
    if model:
        args += ["--model", model]
    if workdir:
        args += ["--cwd", workdir]
    if max_time is not None:
        args += ["--max-time", str(max_time)]
    if no_session:
        args.append("--no-session")
    args.append(prompt)
    return args


def build_interactive_argv(
    prompt: str | None = None,
    model: str | None = None,
    workdir: str | None = None,
    max_time: int | None = None,
    hook: str | None = None,
    resume_id: str | None = None,
    continue_last: bool = False,
) -> list[str]:
    """Build the argument list for launching an interactive omp TUI session.

    Parameters
    ----------
    prompt:
        Optional initial prompt for the session.
    model:
        Optional model override.
    workdir:
        Optional ``--cwd`` override.
    max_time:
        Optional ``--max-time`` in seconds.
    hook:
        Optional ``--hook`` path for a liveness accelerant hook file.
    resume_id:
        If set, pass ``--resume <id>`` to re-attach a specific session.
    continue_last:
        If True, pass ``-c`` (``--continue``) to continue the most recent session.

    Returns
    -------
    list[str]
        Argument list to pass to the omp binary (binary not included).
    """
    args: list[str] = ["--auto-approve"]
    if model:
        args += ["--model", model]
    if workdir:
        args += ["--cwd", workdir]
    if max_time is not None:
        args += ["--max-time", str(max_time)]
    if hook:
        args += ["--hook", hook]
    if resume_id:
        args += ["--resume", resume_id]
    elif continue_last:
        args.append("--continue")
    if prompt:
        args.append(prompt)
    return args


# ---------------------------------------------------------------------------
# RPC translation helpers (fully testable, no subprocess dependency)
# ---------------------------------------------------------------------------


def translate_rpc_line(raw_line: str) -> "dict | None":
    """Parse one omp ``--mode=rpc`` NDJSON line into a ``{kind, payload}`` dict.

    Maps the omp RPC event vocabulary to the marker vocabulary defined in
    ``markers.py``.  Unknown event types return ``None`` (caller skips them).

    Mapping
    -------
    ``ready``        → ``status{phase="ready", detail=""}``
    ``turn_start``   → ``status{phase="running", detail=""}``
    ``message_end``  with ``role=assistant`` → ``heartbeat{note=content[:120]}``
    ``agent_end``    → ``done{summary=<last assistant text>, artifacts=None}``
    ``needs_input``  → ``needs_input{question=..., options=None, context=None}``

    Parameters
    ----------
    raw_line:
        A single raw line from omp's ``--mode=rpc`` NDJSON stream.

    Returns
    -------
    dict | None
        ``{"kind": <marker_kind>, "payload": {...}}`` on a recognised event,
        or ``None`` for blank lines, non-JSON lines, and unrecognised types.

    Notes
    -----
    The ``turn_start``, ``message_end``, ``agent_end``, and ``needs_input``
    event names are inferred from the ``--mode=json`` schema (see docstring at
    the top of this file).  The ``--mode=rpc`` interactive session is confirmed
    to emit ``ready`` and ``available_commands_update`` on startup; the
    remaining event names are assumed to overlap with ``--mode=json`` and
    should be verified against live omp ``--mode=rpc`` output when wired live.
    """
    line = raw_line.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None

    event_type = event.get("type", "")

    if event_type == "ready":
        return {"kind": MARKER_STATUS, "payload": {"phase": "ready", "detail": ""}}

    if event_type == "turn_start":
        return {"kind": MARKER_STATUS, "payload": {"phase": "running", "detail": ""}}

    if event_type == "message_end":
        msg = event.get("message", {})
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            text = content[0].get("text", "") if content else ""
            return {"kind": MARKER_HEARTBEAT, "payload": {"note": text[:120]}}
        return None

    if event_type == "agent_end":
        messages = event.get("messages", [])
        last_text: str = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                last_text = content[0].get("text", "") if content else ""
                break
        return {"kind": MARKER_DONE, "payload": {"summary": last_text, "artifacts": None}}

    if event_type == "needs_input":
        question = event.get("question", "")
        return {
            "kind": MARKER_NEEDS_INPUT,
            "payload": {"question": question, "options": None, "context": None},
        }

    return None


def tail_rpc_stream(
    stdout_iter: Iterable[str],
    marker_path: str,
    task_id: str,
    append_fn: Callable[..., None] = append_marker,
) -> None:
    """Iterate omp ``--mode=rpc`` NDJSON lines and write markers to *marker_path*.

    For each line in *stdout_iter*, calls ``translate_rpc_line``; if the result
    is not ``None``, calls ``append_fn(marker_path, kind, payload, task_id)``
    to persist the marker.  Lines that produce ``None`` (blank, non-JSON, or
    unrecognised type) are silently skipped.

    Parameters
    ----------
    stdout_iter:
        An iterable of raw text lines (e.g. a file-like object or a list of
        strings) representing omp's ``--mode=rpc`` output stream.
    marker_path:
        Absolute path to the ``.jsonl`` marker file to append to.
    task_id:
        Opaque task/session identifier written into each marker envelope.
    append_fn:
        Callable with signature ``(path, kind, payload, task_id)`` used to
        write each marker.  Defaults to ``append_marker`` from ``markers.py``.
        Inject a stub in tests to capture written markers without touching disk.
    """
    for raw_line in stdout_iter:
        result = translate_rpc_line(raw_line)
        if result is not None:
            append_fn(marker_path, result["kind"], result["payload"], task_id)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OmpAdapter(AgentAdapter):
    """Adapter that orchestrates oh-my-pi (omp) sessions via structured transport.

    One-shot (print) mode
    ---------------------
    For interactions that don't require persistent state, use
    ``run_oneshot(prompt)`` which calls ``omp -p --mode=json`` and parses
    the NDJSON result.  This avoids tmux entirely.

    Interactive TUI mode
    --------------------
    For interactive sessions the adapter spawns omp in a tmux pane with
    ``--auto-approve`` (no dialog dance required — omp skips approval prompts
    automatically).  ``drive()`` uses load-buffer/paste-buffer injection.

    Resume
    ------
    ``resume()`` re-attaches via ``-r <session_id>`` (or ``-c`` if no ID).
    The omp session ID is captured from the first ``session`` event in
    --mode=json output and can be stashed in the ``SessionHandle`` metadata.

    Parameters
    ----------
    omp_runner:
        Override the omp subprocess runner.  Defaults to the real
        subprocess-backed runner.  Pass a stub in tests.
    tmux_runner:
        Override the tmux execution back-end.  Defaults to the real runner.
    session_prefix:
        Prefix for generated tmux session names.  Default ``"hermes-omp"``.
    binary:
        Path to the omp binary.  Default ``"omp"`` (PATH lookup).
    """

    def __init__(
        self,
        omp_runner: OmpRunner | None = None,
        tmux_runner: TmuxRunner | None = None,
        session_prefix: str = "hermes-omp",
        binary: str = _OMP_BINARY,
    ) -> None:
        self._omp: OmpRunner = omp_runner or _SubprocessOmpRunner(binary=binary)
        self._tmux: TmuxRunner = tmux_runner or _SubprocessTmuxRunner()
        self._session_prefix = session_prefix
        self._binary = binary

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        """Declare omp adapter capabilities.

        ``rpc_mode=True``  — omp supports ``--mode=rpc`` for an interactive
        RPC-over-stdio session.

        ``json_mode=True`` — omp supports ``--mode=json`` for NDJSON streaming
        output in one-shot (``-p``) mode.  This is the primary structured
        transport used by this adapter for one-shot calls.

        ``has_hooks=True`` — omp supports ``--hook <file>`` for lifecycle
        callbacks, which the watcher can use as a positive-liveness accelerant.

        ``supports_print_mode=True`` — omp supports ``-p`` / ``--print`` for
        non-interactive (headless) one-shot execution.

        ``idle_indicator_regex`` — matches omp's spinner characters and the
        "Running tool" text that appear while a tool is executing.  The watcher
        uses this to distinguish active work from pane-hash-unchanged stalls.
        """
        return Capabilities(
            supports_print_mode=True,
            has_hooks=True,
            rpc_mode=True,
            json_mode=True,
            idle_indicator_regex=ACTIVITY_REGEX,
            dialog_handlers={},  # omp --auto-approve skips all dialogs
        )

    # ------------------------------------------------------------------
    # One-shot convenience (not part of AgentAdapter ABC)
    # ------------------------------------------------------------------

    def run_oneshot(
        self,
        prompt: str,
        model: str | None = None,
        workdir: str | None = None,
        max_time: int | None = None,
    ) -> str:
        """Run a one-shot omp prompt and return the assistant's text.

        Uses ``omp -p --mode=json`` (non-interactive, NDJSON output).
        No tmux session is created.

        Parameters
        ----------
        prompt:
            The prompt text to send to omp.
        model:
            Optional model override (e.g. ``"openai-codex/gpt-5.5"``).
        workdir:
            Optional working directory (``--cwd``).
        max_time:
            Optional max execution time in seconds.

        Returns
        -------
        str
            The assistant's final text response.

        Raises
        ------
        subprocess.CalledProcessError
            If omp exits with a nonzero return code.
        ValueError
            If the NDJSON output cannot be parsed (e.g. omp crashed early).
        """
        argv = build_oneshot_argv(
            prompt=prompt,
            model=model,
            workdir=workdir,
            max_time=max_time,
        )
        result = self._omp.run_oneshot(argv)
        return parse_oneshot_result(result.stdout)

    # ------------------------------------------------------------------
    # launch()  [LIVE — requires real tmux + omp binary]
    # ------------------------------------------------------------------

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        """Spawn a tmux session and start omp inside it.

        omp does NOT require dialog handling (``--auto-approve`` suppresses
        all approval prompts).  The launch sequence is:

        1. Generate a unique session name.
        2. ``tmux new-session -d`` with standard dimensions.
        3. Run ``omp --auto-approve [--cwd <workdir>] <prompt>``.
        4. Wait for the ``>`` prompt to appear (WAITING_USER).
        5. Return the handle.

        This method is LIVE-ONLY (requires a real tmux and omp binary).

        Parameters
        ----------
        workdir:
            Absolute path to the working directory.
        prompt:
            Initial prompt to inject once omp is ready.

        Returns
        -------
        SessionHandle
            Populated handle; store in the registry immediately.
        """
        session_id = str(uuid.uuid4())
        tmux_session = f"{self._session_prefix}-{session_id[:8]}"

        # Compute the marker file path and ensure its parent directory exists.
        marker_file = f"{workdir}/.hermes/sessions/{session_id}.jsonl"
        os.makedirs(os.path.dirname(marker_file), exist_ok=True)

        # 1. Create a detached tmux session, injecting the marker file env var.
        self._tmux.run([
            "new-session", "-d",
            "-s", tmux_session,
            "-x", "200",
            "-y", "50",
            "-e", f"HERMES_MARKER_FILE={marker_file}",
        ])

        # Resolve the REAL pane id rather than assuming ':0.0'. A hardcoded
        # ':0.0' breaks under `set -g base-index 1` / `pane-base-index 1`
        # (the first pane is then ':1.1', and every send-keys/capture against
        # ':0.0' fails with "can't find window: 0"). The stable pane id (e.g.
        # '%12') is immune to base-index and to later window/pane renumbering.
        pane = self._tmux.run([
            "list-panes", "-t", tmux_session, "-F", "#{pane_id}",
        ]).splitlines()[0].strip()

        # 2. Boot omp BARE (no prompt) so it comes up at an idle prompt. The
        #    first prompt is seeded separately via the relay/``drive()`` after
        #    launch (see spawn_session step 7) — matching the claude adapter.
        #    Passing the prompt here too would (a) deliver it twice and (b) make
        #    omp immediately busy, so the post-launch ``drive()`` wait-for-idle
        #    would time out. ``prompt`` is intentionally unused.
        argv = build_interactive_argv(workdir=workdir)
        cmd = shlex.join([self._binary] + argv)
        self._tmux.run(["send-keys", "-t", pane, cmd, "Enter"])

        # 3. Confirm omp actually started (took over the pane). Do NOT wait for
        #    the idle prompt — a launch prompt makes omp immediately busy, so
        #    the idle prompt won't appear until the first turn completes.
        #    On failure, kill the tmux session so a slow/failed launch doesn't
        #    leave an orphaned, unwatched omp running in a detached pane.
        try:
            self._wait_for_launch(pane, timeout=_LAUNCH_START_TIMEOUT)
        except TimeoutError:
            try:
                self._tmux.run(["kill-session", "-t", tmux_session])
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
        """Deliver ``message`` to the running omp session via load-buffer/paste-buffer.

        Steps
        -----
        0. Send any ``pre_keys`` (e.g. ``Escape`` to cancel a selection menu).
        1. Poll pane for prompt readiness (``>`` or ``❯`` visible).
        2. Write ``message`` to a named tmux buffer via ``_load_buffer``.
        3. ``paste-buffer -d`` into the pane.
        4. Send ``Enter`` to submit.

        The load-buffer/paste-buffer mechanism avoids shell-metacharacter
        expansion that ``send-keys`` would perform on strings containing
        ``$``, backticks, pipes, etc.

        Parameters
        ----------
        handle:
            The ``SessionHandle`` returned by ``launch()``.
        message:
            Text to send.  May contain newlines; they are preserved verbatim.
        pre_keys:
            Optional tmux key names sent (one send-keys each) BEFORE the
            readiness poll — used to ``Escape`` out of a selection menu into
            the freed composer so a natural-language answer can be pasted.
        """
        for key in pre_keys or []:
            self._tmux.run(["send-keys", "-t", handle.pane, key])
        self._wait_for_ready(handle.pane, timeout=_LAUNCH_READY_TIMEOUT)

        buf_name = f"hermes-omp-{handle.session_id[:8]}"
        self._load_buffer(buf_name, message)
        self._tmux.run(["paste-buffer", "-d", "-b", buf_name, "-t", handle.pane])
        self._tmux.run(["send-keys", "-t", handle.pane, "", "Enter"])

    # ------------------------------------------------------------------
    # detect()  [testable with stubbed TmuxRunner]
    # ------------------------------------------------------------------

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        """Inspect the current pane state and return a lifecycle value.

        Uses ``parse_pane_lifecycle`` on captured pane text.
        Returns ``ERROR`` if the pane cannot be captured (session dead).

        Parameters
        ----------
        handle:
            The ``SessionHandle`` for the session to inspect.

        Returns
        -------
        SessionLifecycle
            Exactly one of ``RUNNING | WAITING_USER | PAUSED_HANDOFF | ERROR``.
            ``STALLED`` and ``DONE`` are determined by the watcher based on
            pane-hash staleness and elapsed time.
        """
        pane_text = self._capture_pane(handle.pane)
        if pane_text is None:
            return SessionLifecycle.ERROR
        return parse_pane_lifecycle(pane_text)

    def idle_waiting(self, pane_text: str) -> bool:
        """Candidate free-form wait signal for the watcher (see
        ``pane_is_idle_waiting``). The watcher gates this on pane stability
        before promoting RUNNING → WAITING_USER."""
        return pane_is_idle_waiting(pane_text)

    # ------------------------------------------------------------------
    # resume()  [testable with stubbed TmuxRunner + OmpRunner]
    # ------------------------------------------------------------------

    def resume(self, handle: SessionHandle, prompt: str) -> None:
        """Re-attach a paused omp session and inject a new prompt.

        Idempotent: if ``detect()`` does not return ``PAUSED_HANDOFF``,
        logs a warning and returns without action.

        omp supports ``-c`` (continue last session) or ``-r <id>``
        (resume by ID).  This implementation uses ``-c`` since the
        handle does not currently carry an omp session ID; the watcher
        does not need to differentiate sessions for the v1 handoff case.

        Parameters
        ----------
        handle:
            The ``SessionHandle`` for the session to resume.
        prompt:
            Prompt to inject after re-attaching.
        """
        current = self.detect(handle)
        if current != SessionLifecycle.PAUSED_HANDOFF:
            import logging
            logging.getLogger(__name__).warning(
                "resume() called on omp session %s in state %s "
                "(expected PAUSED_HANDOFF); no-op.",
                handle.session_id,
                current.value,
            )
            return

        # Send omp -c to continue the previous session in this pane.
        argv = build_interactive_argv(prompt=prompt, continue_last=True)
        cmd = shlex.join([self._binary] + argv)
        self._tmux.run(["send-keys", "-t", handle.pane, cmd, "Enter"])
        self._wait_for_prompt(handle.pane, timeout=_LAUNCH_READY_TIMEOUT)

    # ------------------------------------------------------------------
    # terminate()  [testable with stubbed TmuxRunner]
    # ------------------------------------------------------------------

    def terminate(self, handle: SessionHandle) -> None:
        """Kill the tmux session associated with ``handle``.

        Sends ``tmux kill-session -t <tmux_session>`` with ``check=False`` so a
        nonzero exit (e.g. session already gone) is silently ignored.
        ``CalledProcessError`` is also swallowed for belt-and-suspenders safety
        when a custom ``TmuxRunner`` raises on failure.

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
            return self._tmux.run([
                "capture-pane", "-t", pane,
                "-p",
                "-S", f"-{_CAPTURE_LINES}",
            ], check=True)
        except subprocess.CalledProcessError:
            return None

    def _wait_for_prompt(self, pane: str, timeout: float) -> None:
        """Poll until a prompt marker is visible in the pane or timeout elapses.

        Raises
        ------
        TimeoutError
            If the prompt does not appear within ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self._capture_pane(pane)
            if text is not None and PROMPT_PATTERN.search(text):
                return
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError(
            f"omp prompt not visible in pane {pane!r} after {timeout:.0f}s"
        )

    def _wait_for_launch(self, pane: str, timeout: float) -> None:
        """Wait until omp's TUI has actually rendered in the pane.

        A prompt passed at launch makes omp immediately busy, so it will NOT
        return to the idle ``❯`` prompt within the launch window — waiting for
        that (as ``_wait_for_prompt`` does) times out on every real task even
        though omp is running fine. Instead, wait for omp's TUI chrome to draw:
        its rounded box borders (``╭``/``╰``) are TUI-only and never produced by
        the shell, so their presence confirms omp took over the pane. This also
        catches a failed launch (e.g. a shell glob/quoting error) — the shell
        never draws a box, so this raises rather than returning a dead handle.
        Succeed early if the idle prompt is already visible (instant tasks).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self._capture_pane(pane)
            if text is not None:
                # omp TUI chrome (rounded box corners) or an already-idle prompt.
                if "╭" in text or "╰" in text or PROMPT_PATTERN.search(text):
                    return
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError(
            f"omp did not start in pane {pane!r} after {timeout:.0f}s"
        )

    def _wait_for_ready(self, pane: str, timeout: float) -> None:
        """Wait until omp's TUI is up AND idle, so a pasted message lands in the
        input box instead of being dropped or interleaved with a response.

        omp does NOT render a ``>``/``❯`` input char (so ``PROMPT_PATTERN`` never
        matches it). Its idle state is the status-bar box (``╭…╮`` / ``╰…╯``)
        with NO active-work spinner; its busy state matches ``ACTIVITY_REGEX``
        (Braille spinner / "Running tool"). Ready == box present and not busy.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = self._capture_pane(pane)
            if text is not None:
                tui_up = ("╰" in text) or ("╭" in text)
                busy = bool(ACTIVITY_REGEX.search(text))
                if tui_up and not busy:
                    return
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError(
            f"omp not ready for input in pane {pane!r} after {timeout:.0f}s"
        )

    def _load_buffer(self, buf_name: str, content: str) -> None:
        """Write ``content`` into a named tmux buffer via subprocess stdin pipe.

        Bypasses the TmuxRunner abstraction because ``load-buffer`` requires
        data on stdin; the protocol's ``run()`` does not expose stdin.
        Tests should monkeypatch this method if they need to assert buffer content.
        """
        subprocess.run(
            ["tmux", "load-buffer", "-b", buf_name, "-"],
            input=content,
            text=True,
            check=True,
        )
