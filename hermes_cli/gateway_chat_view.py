"""Classic terminal presentation of authority events and fenced controls."""
import asyncio
from contextlib import suppress
import sys
import uuid

from hermes_cli.gateway_client import GatewayClientError


class GatewayChatView:
    def __init__(self, client, snapshot, *, quiet=False):
        self.client = client
        self.session_id = snapshot["stored_session_id"]
        self.generation = snapshot.get("execution_generation", 0)
        self.prompts = {p["prompt_id"]: p for p in snapshot.get("prompts", [])}
        self.pending = snapshot.get("pending", [])
        self.quiet = quiet
        self.finite = False
        self.streams = {}
        self.completions = {}
        self.changed = asyncio.Event()
        self.failure = None
        from hermes_cli.gateway_mutations import PreparedMutations
        self.mutations = PreparedMutations()

    def show_pending(self):
        unknown = any(row["status"] == "unknown" for row in self.pending)
        if unknown:
            print("Execution outcome unknown after restart. Discard acknowledges the lost turn "
                  "without replaying it; queued work may then continue.", file=sys.stderr)
        for row in self.pending:
            admission = row["admission_id"]
            if row["status"] == "unknown":
                print(f"Unknown admission: {admission}\n/discard {admission}", file=sys.stderr)
            elif row["status"] == "queued":
                context = "waiting behind unknown work" if unknown else "waiting to run"
                print(f"Queued admission: {admission} ({context})", file=sys.stderr)

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
                self.failure = event
                self.changed.set()
                return
            params = event.get("params", {})
            if params.get("session_id") != self.session_id:
                continue
            kind, payload = params.get("type"), params.get("payload", {})
            self.generation = params.get("execution_generation", self.generation)
            admission = params.get("admission_id") or payload.get("admission_id")
            handler = {
                "message.delta": self._delta, "message.complete": self._complete,
                "approval.request": self._request, "clarify.request": self._request,
                "approval.settled": self._settled, "clarify.settled": self._settled,
            }.get(kind)
            if handler:
                handler(admission, payload)
            self.changed.set()

    def _delta(self, admission, payload):
        text = payload.get("text") or payload.get("delta") or payload.get("content") or ""
        if isinstance(text, str) and not self.quiet:
            self.streams[admission] = self.streams.get(admission, "") + text
            print(text, end="", flush=True)

    def _complete(self, admission, payload):
        text = payload.get("text") or payload.get("content") or ""
        streamed = self.streams.pop(admission, "")
        if not self.quiet:
            if not streamed:
                print(text, flush=True)
            elif text.startswith(streamed):
                print(text[len(streamed):], flush=True)
            else:
                print("\n" + text, flush=True)
        self.completions[admission] = payload

    def _request(self, admission, payload):
        self.prompts[payload["prompt_id"]] = payload
        self.show_prompt(payload)

    def _settled(self, admission, payload):
        self.prompts.pop(payload["prompt_id"], None)

    async def submit(self, text):
        return await self.client.rpc("prompt.submit", session_id=self.session_id,
                                     input_id=uuid.uuid4().hex, text=text,
                                     **({"finite": True} if self.finite else {}))

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
            result = await self.mutations.apply(self.client, original, operation, payload)
            target = result.get('branched_session_id', original)
            snapshot = await self.client.rpc('session.resume', session_id=target)
            self.session_id = target
            self.generation = snapshot['execution_generation']
            self.prompts = {p['prompt_id']: p for p in snapshot.get('prompts', [])}
            self.mutations.acknowledge(original, operation, payload)
            print(f"{operation}: {target}")
            return True
        if command == "/help":
            print("/stop, /approve <id> <choice>, /answer <id> <text>, /discard <admission_id> (turn lost during restart), /quit (detach). /branch [title], /model <model> [--provider name], /compress [focus].")
            return True
        raise GatewayClientError("Unsupported gateway CLI command; use /help. No local command was run.")

    async def run(self, query=None, *, oneshot=False):
        self.quiet = self.quiet or oneshot
        self.finite = oneshot
        self.show_pending()
        for prompt in self.prompts.values():
            self.show_prompt(prompt)
        renderer = asyncio.create_task(self.render())
        try:
            receipt = await self.submit(query) if query else None
            if oneshot:
                if receipt is None:
                    raise GatewayClientError("One-shot requires a query")
                admission = receipt["admission_id"]
                while admission not in self.completions:
                    self.changed.clear()
                    if self.failure:
                        raise self.failure
                    if self.prompts:
                        print("Input required; detached without cancelling. Resume this session interactively.", file=sys.stderr)
                        return 3
                    await self.changed.wait()
                terminal = self.completions[admission]
                print(terminal.get("text") or terminal.get("content") or "", flush=True)
                return 0 if terminal.get("outcome") == "completed" else 1
            from prompt_toolkit import PromptSession
            from prompt_toolkit.patch_stdout import patch_stdout
            prompt = PromptSession()
            with patch_stdout():
                while not self.failure:
                    try:
                        text = (await prompt.prompt_async("You> ")).strip()
                        if not text:
                            continue
                        if text.startswith("/"):
                            if not await self.command(text):
                                return 0
                        else:
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
