"""Classic terminal presentation of authority events and fenced controls."""
import asyncio
from contextlib import suppress
import logging
import sys
import uuid

from hermes_cli.gateway_client import GatewayClientError

logger = logging.getLogger(__name__)


_PREVIEW_CHARS = 80


def _pending_preview(text):
    """One terminal-safe line of a pending admission's prompt: escape sequences removed, every
    other non-printable (C0/C1, bidi and zero-width format chars, newlines) folded to a space,
    capped at ``_PREVIEW_CHARS``. Empty when the row carries no text."""
    from tools.ansi_strip import strip_ansi
    if not isinstance(text, str):
        return ""
    line = " ".join("".join(ch if ch.isprintable() else " " for ch in strip_ansi(text)).split())
    if len(line) > _PREVIEW_CHARS:
        line = line[:_PREVIEW_CHARS - 1] + "…"
    return f": {line}" if line else ""


_BLOCKING_CONTROL_EVENTS = frozenset({
    "approval.request", "approval.settled", "clarify.request", "clarify.settled",
    "session.info",
})


class GatewayChatView:
    def __init__(self, client, snapshot, *, quiet=False, emitter=None, usage_file=None):
        self.usage_file = usage_file
        self.client = client
        self.session_id = snapshot["stored_session_id"]
        self.generation = snapshot.get("execution_generation", 0)
        self.prompts = {p["prompt_id"]: p for p in snapshot.get("prompts", [])}
        self.pending = snapshot.get("pending", [])
        self.model = str((snapshot.get("info") or {}).get("model") or "Hermes").split("/")[-1]
        # ``--format stream-json``: stdout belongs to the JSONL protocol, so every human line is
        # replaced by an emitter event and the terminal record carries the exit code.
        self.emitter = emitter
        self.quiet = quiet or emitter is not None
        self.finite = False
        self.unattended = False  # `-z`: the classic one-shot auto-approves; `-q` stays single-query
        self.resume_footer = False  # `chat -q` without -Q: main's "Resume this session with:" block
        self.launch_api_key = None  # `--api-key` of this terminal's launch; never durable (/new reuses it)
        self.finite_admission = None
        self._finite_events = []
        self.streams = {}
        self.completions = {}
        self.changed = asyncio.Event()
        self.failure = None
        # Set once with ``failure``: the interactive composer races its line read against it so a
        # dead owner ends the REPL immediately instead of on the next keypress.
        self.failed = asyncio.Event()
        from hermes_cli.gateway_mutations import PreparedMutations
        self.mutations = PreparedMutations()
        # The interactive composer (set by ``run``); None in one-shot / non-TTY runs, where a
        # guarded model switch stays a refusal instead of a prompt.
        self._composer = None
        # `/undo` puts the removed message back in the composer for the next prompt.
        self.prefill = ""

    def unknown_admissions(self):
        return [row["admission_id"] for row in self.pending if row["status"] == "unknown"]

    def show_pending(self):
        unknown = bool(self.unknown_admissions())
        if unknown:
            print("Execution outcome unknown after restart. Discard acknowledges the lost turn "
                  "without replaying it; queued work may then continue.", file=sys.stderr)
        for row in self.pending:
            admission = row["admission_id"]
            # A queued prompt is not in the transcript yet: show what is about to run (or was lost).
            preview = _pending_preview(row.get("text"))
            if row["status"] == "unknown":
                print(f"Unknown admission: {admission}{preview}\n/discard {admission}", file=sys.stderr)
            elif row["status"] == "queued":
                context = "waiting behind unknown work" if unknown else "waiting to run"
                print(f"Queued admission: {admission} ({context}){preview}", file=sys.stderr)

    def show_prompt(self, prompt):
        print(f"\n{prompt.get('description') or prompt.get('question') or 'Approval required'}", file=sys.stderr)
        if prompt.get("command"):
            print(prompt["command"], file=sys.stderr)
        command = "/approve" if prompt["kind"] == "approval" else "/answer"
        print(f"{command} {prompt['prompt_id']} <{'|'.join(prompt.get('choices', [])) or 'answer'}>", file=sys.stderr)

    async def render(self):
        while True:
            event = await self.client.events.get()
            if isinstance(event, Exception):
                self._fail(event)
                return
            params = event.get("params", {})
            if params.get("session_id") != self.session_id:
                continue
            kind, payload = params.get("type"), params.get("payload", {})
            self.generation = params.get("execution_generation", self.generation)
            if kind == "session.replay_gap":
                self._fail(GatewayClientError("session_replay_gap"))
                return
            admission = params.get("admission_id") or payload.get("admission_id")
            if self.finite and kind in _BLOCKING_CONTROL_EVENTS:
                # Gates and unknown executions block the session FIFO, not merely one
                # admission's output. Track them even when another admission owns the
                # event so a queued one-shot can detach instead of waiting forever.
                self._dispatch_event(kind, admission, payload)
                self.changed.set()
                continue
            if self.finite and admission:
                if self.finite_admission is None:
                    # The owner can publish before prompt.submit's receipt reaches this client.
                    # Hold admission-scoped output until we know which admission this invocation owns.
                    self._finite_events.append((kind, admission, payload))
                    continue
                if admission != self.finite_admission:
                    continue
            self._dispatch_event(kind, admission, payload)
            self.changed.set()

    def _fail(self, error):
        self.failure = error
        self.failed.set()
        self.changed.set()

    async def _read_line(self, prompt, symbol):
        """The next composer line, or ``""`` once the event stream failed first (``self.failure``
        is then set). The pending read is cancelled (prompt_toolkit restores the terminal) so the
        caller reports the unknown outcome without waiting for a keypress."""
        default, self.prefill = self.prefill, ""
        reader = asyncio.ensure_future(prompt.prompt_async(symbol, **({"default": default} if default else {})))
        failed = asyncio.ensure_future(self.failed.wait())
        try:
            await asyncio.wait({reader, failed}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            failed.cancel()
        if reader.done():
            return reader.result()
        reader.cancel()
        with suppress(asyncio.CancelledError):
            await reader
        return ""

    def _dispatch_event(self, kind, admission, payload):
        handler = {
            "message.delta": self._delta, "message.complete": self._complete,
            "tool.start": self._tool_start, "tool.complete": self._tool_complete,
            "approval.request": self._request, "clarify.request": self._request,
            "approval.settled": self._settled, "clarify.settled": self._settled,
            "session.info": self._session_info,
        }.get(kind)
        if handler:
            handler(admission, payload)

    def _session_info(self, admission, payload):
        self.pending = payload.get("pending", self.pending)
        self.generation = payload.get("execution_generation", self.generation)

    def _tool_start(self, admission, payload):
        # Deltas before a tool call are interim commentary the final reply does not repeat;
        # close that segment so `_complete` measures only the final's own stream.
        if self.streams.pop(admission, None) and not self.quiet:
            print(flush=True)
        if self.emitter is not None:
            self.emitter.on_tool_progress("tool.started", payload.get("tool_name"), None, payload.get("args"),
                                          tool_call_id=payload.get("tool_call_id") or None)
        elif not self.quiet:
            # Same line shape the in-process CLI prints: the tool's emoji and its primary argument.
            from agent.display import build_tool_preview, get_tool_emoji
            name = payload.get("tool_name") or payload.get("name") or "tool"
            preview = build_tool_preview(name, payload.get("args") or {}, max_len=0)
            print(f"{get_tool_emoji(name)} {name}{f': {preview}' if preview else ''}", flush=True)

    def _tool_complete(self, admission, payload):
        if self.emitter is not None:
            self.emitter.on_tool_progress("tool.completed", payload.get("tool_name"), None, payload.get("args"),
                                          tool_call_id=payload.get("tool_call_id") or None,
                                          result=payload.get("result"), is_error=payload.get("is_error", False))

    def _delta(self, admission, payload):
        text = payload.get("text") or payload.get("delta") or payload.get("content") or ""
        if not isinstance(text, str):
            return
        if self.emitter is not None:
            self.emitter.on_text_delta(text)
        elif not self.quiet:
            self.streams[admission] = self.streams.get(admission, "") + text
            # No flush: under the live composer, patch_stdout line-buffers this so a redraw
            # (a resize mid-stream) never interleaves with a half-written line. Complete lines
            # still appear as they stream; the last one lands with `_complete`.
            print(text, end="")

    def _complete(self, admission, payload):
        text = payload.get("text") or payload.get("content") or ""
        streamed = self.streams.pop(admission, "")
        if not self.quiet:
            # The final's deltas carry the agent's segment break (leading blank lines after a
            # tool call) that the settled text has trimmed; a match modulo that edge whitespace
            # is the same reply already on screen.
            if not streamed:
                print(text, flush=True)
            elif text.startswith(streamed):
                print(text[len(streamed):], flush=True)
            elif text.strip() == streamed.strip():
                print(flush=True)
            else:
                print("\n" + text, flush=True)
        self.completions[admission] = payload

    def _request(self, admission, payload):
        self.prompts[payload["prompt_id"]] = payload
        # Finite invocations only need the session-blocking state so they can
        # detach cleanly; do not print another admission's control details.
        if not self.finite:
            self.show_prompt(payload)

    def _settled(self, admission, payload):
        self.prompts.pop(payload["prompt_id"], None)

    async def submit(self, text):
        self.mutations.retire(self.session_id)
        return await self.client.rpc("prompt.submit", session_id=self.session_id,
                                     input_id=uuid.uuid4().hex, text=text,
                                     **({"finite": True} if self.finite else {}),
                                     **({"unattended": True} if self.finite and self.unattended else {}))

    async def command(self, text):
        command, _, rest = text.partition(" ")
        if command in {"/quit", "/exit", "/detach"}:
            return False
        if command == "/stop":
            await self.client.rpc("session.interrupt", session_id=self.session_id,
                                  execution_generation=self.generation)
            return True
        if command in {"/approve", "/answer"}:
            prompt_id, _, answer = rest.partition(" ")
            prompt = self.prompts.get(prompt_id)
            expected = "approval" if command == "/approve" else "clarify"
            if not prompt or prompt["kind"] != expected:
                raise GatewayClientError("No matching pending control; resume to refresh")
            await self.client.rpc(expected + ".respond", session_id=self.session_id,
                execution_generation=prompt["execution_generation"], prompt_id=prompt_id,
                **({"choice": answer} if expected == "approval" else {"answer": answer}))
            return True
        if command == "/discard":
            # Acknowledge a turn lost across an owner restart; the resume snapshot
            # is the only source of the generation the authority stamped on it.
            snapshot = await self.client.rpc("session.resume", session_id=self.session_id)
            lost = next((row for row in snapshot.get("pending", [])
                         if row["admission_id"] == rest.strip() and row["status"] == "unknown"), None)
            if lost is None:
                raise GatewayClientError("No unknown (lost) admission with that id; resume to refresh")
            await self.client.rpc("prompt.resolve_unknown", session_id=self.session_id,
                                  admission_id=lost["admission_id"], execution_generation=lost["execution_generation"])
            return True
        if command in {'/branch', '/model', '/compress'}:
            from hermes_cli.gateway_mutations import slash_mutation
            operation, payload = slash_mutation(command, rest.strip())
            original = self.session_id
            confirm = self._confirm_model_switch if operation == 'model' and self._composer is not None else None
            result = await self.mutations.apply(self.client, original, operation, payload, confirm=confirm)
            if result.get('status') == 'cancelled':
                return True
            if result.get('status') == 'preview':
                self.mutations.acknowledge(original, operation, payload)
                print('\n'.join(result['lines']))
                return True
            target = result.get('branched_session_id', original)
            await self.adopt(await self.client.rpc('session.resume', session_id=target))
            self.mutations.acknowledge(original, operation, payload)
            if operation == 'model' and result.get('model'):
                self.model = str(result['model']).split('/')[-1]
            print(f"{operation}: {target}")
            return True
        if command == "/yolo":
            # This session's approval bypass on the owner (same verb as the TUI's /yolo and Desktop).
            word = rest.strip().lower()
            if word not in {"", "on", "off"}:
                raise GatewayClientError("Usage: /yolo [on|off]")
            result = await self.client.rpc("config.set", session_id=self.session_id, key="yolo",
                                           **({"value": "1" if word == "on" else "0"} if word else {}))
            print(f"YOLO {'on' if result.get('value') == '1' else 'off'} for this session")
            return True
        from hermes_cli import gateway_chat_commands
        if command == "/help":
            print(gateway_chat_commands.help_text())
            return True
        return await gateway_chat_commands.run_command(self, command, rest)

    async def adopt(self, snapshot):
        """Attach this view to another session's snapshot (a branch, /new)."""
        self.session_id = snapshot['stored_session_id']
        self.generation = snapshot['execution_generation']
        self.prompts = {p['prompt_id']: p for p in snapshot.get('prompts', [])}
        self.pending = snapshot.get('pending', [])
        self.model = str((snapshot.get('info') or {}).get('model') or self.model).split('/')[-1]
        if not self.finite:
            # Bare `hermes -c` in this terminal continues the session it is now on, as the
            # in-process CLI re-wrote its breadcrumb on every session switch.
            from hermes_cli.terminal_breadcrumbs import write_breadcrumb
            await asyncio.to_thread(write_breadcrumb, self.session_id)

    async def _confirm_model_switch(self, refusal):
        """The owner refused a guarded model target (cost / data policy / large context): ask as
        the in-process CLI does (``_confirm_expensive_model_switch``: switch anyway once, or cancel)
        on this composer. Only an explicit yes applies; anything else keeps the current model."""
        from agent.i18n import t
        from hermes_cli.gateway_mutations import confirm_choice, confirmation_title
        choices = [("once", t("cli.model.choice_switch_anyway"), t("cli.model.desc_switch_anyway")),
                   ("cancel", t("cli.model.choice_cancel"), t("cli.model.desc_keep_current_model"))]
        print(f"\n!!! {confirmation_title(refusal)} !!!\n{refusal['confirm_message']}\n")
        for index, (_, label, detail) in enumerate(choices, 1):
            print(f"  {index}. {label} \u2014 {detail}")
        try:
            raw = await self._read_line(self._composer, t("cli.model.confirm_choice_prompt"))
        except (KeyboardInterrupt, EOFError):
            raw = ""
        if not self.failure and confirm_choice(raw, choices) == "once":
            return True
        print(t("cli.model.switch_cancelled"))
        return False

    def _detach(self, message):
        """One-shot cannot go on without a human: say why on stderr, exit 3, and in stream-json
        mode close the protocol with a failed ``result`` (a consumer parsing stdout must never be
        left without a terminal record)."""
        print(message, file=sys.stderr)
        if self.emitter is not None:
            return self.emitter.emit_result({"failed": True, "error": message}, session_id=self.session_id, exit_code=3)
        return 3

    def _finite_block(self, admission):
        """Why a finite viewer cannot keep waiting for *admission*, or None. Its own lost turn
        settles as the unknown completion would, so the exit code never depends on frame order;
        only a still-queued input behind someone else's unknown turn is retained and detached."""
        own = next((row["status"] for row in self.pending if row["admission_id"] == admission), None)
        if own == "unknown":
            self.completions[admission] = {"outcome": "unknown", "text": (
                "Execution outcome is unknown. Resume this session to inspect the lost turn "
                "and /discard it; do not resend the input.")}
            return None
        if own == "queued" and self.unknown_admissions():
            return ("Unknown execution blocks this session; your accepted input is retained. "
                    "Resume interactively to resolve the lost turn; do not resend the input.")
        if self.prompts:
            return "Input required; detached without cancelling. Resume this session interactively."
        return None

    async def _settled_result(self, admission):
        """The structured result the owner committed with this admission's settlement — the same
        dict the in-process one-shot got from ``run_conversation`` (best-effort, never raises)."""
        from websockets.exceptions import WebSocketException
        try:
            receipt = await self.client.rpc("prompt.receipt", session_id=self.session_id,
                                            admission_id=admission, include_result=True)
            return dict(receipt.get("result") or {})
        # Transport loss / refusal, or a receipt whose ``result`` is not a mapping.
        except (GatewayClientError, OSError, TimeoutError, WebSocketException,
                AttributeError, TypeError, ValueError) as exc:
            # The exit code and ledger still follow the settled outcome; a missing receipt is not fatal.
            logger.debug("settled result for %s unavailable: %s", admission, exc)
            return {}

    def _print_exit_ids(self):
        """The finite run's durable id on stderr (automation wrappers parse it; it names the
        physical row a compaction may have advanced) and, for `chat -q`, main's resume block."""
        print(f"\nsession_id: {self.session_id}", file=sys.stderr, flush=True)
        if self.resume_footer:
            print(resume_footer(self.session_id), flush=True)

    def _write_usage_file(self, result, outcome):
        """``-z --usage-file``: the same JSON ledger the in-process one-shot wrote."""
        from hermes_cli.oneshot import _write_usage_file
        result = {**result, "session_id": result.get("session_id") or self.session_id}
        failure = None if outcome in ("completed", "cancelled") else (result.get("error") or outcome or "failed")
        _write_usage_file(self.usage_file, result, failure=failure)

    async def run(self, query=None, *, oneshot=False):
        self.quiet = self.quiet or oneshot
        self.finite = oneshot
        self.show_pending()
        for prompt in self.prompts.values():
            self.show_prompt(prompt)
        if oneshot and self.unknown_admissions():
            # A new admission would queue behind the unknown row and never run; refuse BEFORE
            # submitting so nothing is left queued for the next interactive resume to find.
            lost = " ".join(self.unknown_admissions())
            return self._detach("Unknown execution blocks this session; nothing was submitted. Resolve it "
                                f"with `hermes sessions discard {self.session_id} --yes` (or /discard {lost} "
                                "interactively; both are prompt.resolve_unknown), then retry.")
        renderer = asyncio.create_task(self.render())
        try:
            receipt = await self.submit(query) if query else None
            if oneshot:
                if receipt is None:
                    raise GatewayClientError("One-shot requires a query")
                admission = receipt["admission_id"]
                self.finite_admission = admission
                for kind, event_admission, payload in self._finite_events:
                    if event_admission == admission:
                        self._dispatch_event(kind, event_admission, payload)
                self._finite_events.clear()
                self.changed.set()
                while admission not in self.completions:
                    self.changed.clear()
                    if self.failure:
                        raise self.failure
                    blocked = self._finite_block(admission)
                    if blocked:
                        return self._detach(blocked)
                    if admission not in self.completions:
                        await self.changed.wait()
                terminal = self.completions[admission]
                outcome = terminal.get("outcome")
                text = terminal.get("text") or terminal.get("content") or ""
                # The outcome only says failed/cancelled; a turn stopped by --max-turns settles
                # 'completed' with ``completed: False`` in its committed result. Judge both with the
                # in-process exit contracts: `-z` 0/2/130 (1 = no text), `chat -q`/`-Q` 0/1/130.
                from hermes_cli.oneshot import _oneshot_exit_code
                from hermes_cli.turn_exit import turn_exit_code
                result = await self._settled_result(admission)
                result.update(failed=bool(result.get("failed")) or outcome not in ("completed", "cancelled"),
                              interrupted=bool(result.get("interrupted")) or outcome == "cancelled")
                exit_code = (_oneshot_exit_code(text, result) if self.unattended
                             else turn_exit_code(result, kanban_worker=False))
                if self.usage_file:
                    self._write_usage_file(result, outcome)
                if self.emitter is not None:
                    return self.emitter.emit_result({**result, "final_response": text},
                                                    session_id=self.session_id, exit_code=exit_code)
                print(text, flush=True)
                # Same stderr exit contract as the legacy -Q path: automation wrappers read the
                # durable id from this line, and it names the physical row (a compaction may have
                # advanced it past the row printed at start).
                self._print_exit_ids()
                return exit_code
            from prompt_toolkit import PromptSession
            from prompt_toolkit.patch_stdout import patch_stdout
            from hermes_cli.skin_engine import get_active_prompt_symbol, get_active_skin
            welcome = "Welcome to Hermes Agent! Type your message or /help for commands."
            print(get_active_skin().get_branding("welcome", welcome), flush=True)
            # The classic status bar's leading segments: model, then the attached session.
            prompt = PromptSession(erase_when_done=True,
                                   bottom_toolbar=lambda: f" \u2624 {self.model} \u2502 {self.session_id} ")
            prompt_symbol = get_active_prompt_symbol("❯ ")
            self._composer = prompt if sys.stdin.isatty() else None
            with patch_stdout():
                while not self.failure:
                    try:
                        # Empty when the stream failed first: the loop guard then raises it.
                        text = (await self._read_line(prompt, prompt_symbol)).strip()
                        if not text:
                            continue
                        if text.startswith("/"):
                            if not await self.command(text):
                                return 0
                        else:
                            # Same scrollback shape as the in-process CLI: the typed prompt line is
                            # erased on submit and the message lands as a `●` preview row.
                            print(f"\n{'─' * 40}\n● {text}", flush=True)
                            await self.submit(text)
                    except KeyboardInterrupt:
                        print("Use /stop to interrupt execution, /quit to detach.")
                    except EOFError:
                        return 0
                    except GatewayClientError as exc:
                        print(f"Error: {exc}", file=sys.stderr)
                raise self.failure
        finally:
            renderer.cancel()
            with suppress(asyncio.CancelledError):
                await renderer


def resume_footer(session_id):
    """The classic single-query exit block (``_print_exit_summary``): how to continue this session.
    A non-default profile needs ``-p`` because session ids are profile-scoped."""
    from agent.i18n import t
    from hermes_cli.profiles import get_active_profile_name
    profile = get_active_profile_name()
    flag = "" if profile in ("default", "custom") else f" -p {profile}"
    return f"\n{t('cli.session.exit_resume_hint')}\n  hermes --resume {session_id}{flag}"
