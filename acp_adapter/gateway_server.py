"""ACP projection of the profile gateway; never constructs an execution owner."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import uuid

import acp
from acp.schema import (
    AgentCapabilities, Implementation, InitializeResponse, LoadSessionResponse,
    NewSessionResponse, PromptCapabilities, PromptResponse, ResumeSessionResponse,
    SessionCapabilities, SessionForkCapabilities, SessionListCapabilities, SessionResumeCapabilities,
)

from hermes_cli.gateway_client import GatewayClientError, connect_gateway
from hermes_constants import get_hermes_home
from acp_adapter.session import _translate_acp_cwd, _normalize_cwd_for_compare

logger = logging.getLogger(__name__)


def _stage_user_content(content):
    """Shared-converter output -> ``(text, attachments)`` for ``prompt.submit``.

    Text-only prompts stay a plain string. Media parts (image ``data:`` URLs from
    image blocks, image resource links and embedded blobs) are staged as bytes in
    the profile image cache; the authority commits them at admission. Remote
    image URLs cannot be staged and are kept as text so the model still sees them."""
    if isinstance(content, str):
        return content, []
    import base64
    import binascii
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_ingress_media import _ATTACHMENT_LIMIT, _IMAGE_EXT, sniff_image_mime
    texts, attachments = [], []
    for part in content:
        if part.get('type') == 'text':
            texts.append(part['text'])
            continue
        url = part['image_url']['url']
        if not url.startswith('data:'):
            texts.append(f"[Image attached: {url}]")
            continue
        header, _, data = url.partition(',')
        declared = header[len('data:'):].split(';', 1)[0].strip().lower()
        if len(attachments) >= _ATTACHMENT_LIMIT:
            raise GatewayClientError('acp_content_invalid')
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise GatewayClientError('acp_content_invalid') from exc
        # The staged extension and admitted type come from the sniffed bytes, never the client's
        # text (``text/html`` -> .html, ``image/svg+xml``, or a path-shaped subtype on Windows).
        mime = sniff_image_mime(raw)
        if mime is None or (declared and {'image/jpg': 'image/jpeg'}.get(declared, declared) != mime):
            raise GatewayClientError('acp_content_invalid')
        try:
            path = cache_image_from_bytes(raw, _IMAGE_EXT[mime])
        except ValueError as exc:
            raise GatewayClientError('acp_content_invalid') from exc
        attachments.append({'path': path, 'mime': mime})
    return "\n".join(texts), attachments


def _text(payload):
    """Reply text of a message frame; ``text`` may be present and null (a suppressed delivery)."""
    text = payload.get("text")
    return text if isinstance(text, str) else ""


def _slash_command(text):
    """Canonical Hermes command name ``text`` invokes, or None for ordinary text.

    Only a registered command (or alias) is a command: ``/etc/hosts is wrong`` or ``/tmp/x``
    (a path, the same rule ``MessageEvent.get_command`` applies) and unknown ``/words`` are
    prompt text for the model. A registered command this surface does not carry still raises
    ``unsupported_command`` in ``slash_mutation`` rather than reaching the owner as text."""
    stripped = text.lstrip()
    if not stripped.startswith('/'):
        return None
    name = stripped.split(None, 1)[0][1:]
    if not name or '/' in name:
        return None
    from hermes_cli.commands import resolve_command
    command = resolve_command(name)
    return command.name if command is not None else None


def _submit_fingerprint(session_id, text, attachments):
    """Content identity of one logical prompt: its text and the staged image bytes (staging names
    are disposable; the authority reconciles a retry by the same digests)."""
    import hashlib
    digest = hashlib.sha256(text.encode())
    for item in attachments:
        with open(item['path'], 'rb') as staged:
            digest.update(item['mime'].encode() + b'\0' + hashlib.sha256(staged.read()).digest())
    return session_id, digest.hexdigest()


class GatewayACPAgent(acp.Agent):
    def __init__(self):
        self._conn = None
        self._gateway = None
        self._connection = None
        self._connect_lock = asyncio.Lock()
        self._home = get_hermes_home().resolve()
        self._snapshots = {}
        self._event_task = None
        self._terminals = {}
        self._streamed = {}
        # Admission -> closed reply segments already shown (interim commentary, pre-tool deltas).
        self._segments = {}
        self._changed = asyncio.Condition()
        self._failure = None
        # Connection generation: a waiter whose transport was replaced reports that transport's
        # ambiguous failure (``_lost_failure``) instead of waiting on a connection it never used.
        self._link = 0
        self._lost_failure = None
        # Session -> editor MCP servers a replacement connection must re-attach.
        self._editor_mcp = {}
        self._permissions = {}
        self._admissions = {}
        self._submitting = set()
        # Session -> (content fingerprint, input_id) of a submit whose ack never arrived. The
        # editor's retry (that session's next prompt, same content) reuses it, so the authority
        # returns the same admission.
        self._unacked = {}
        # Admissions whose message.complete the pump is projecting right now.
        self._settling = set()
        self._pending_cancels = set()
        self._tool_args = {}
        # Editor-held compression-tip id -> logical session id (see ``_resume``).
        self._aliases = {}
        self._form_elicitation = False
        from hermes_cli.gateway_mutations import PreparedMutations
        self._mutations = PreparedMutations()

    def on_connect(self, conn):
        self._conn = conn
        add_observer = getattr(getattr(conn, "_conn", None), "add_observer", None)
        if callable(add_observer):
            add_observer(self._observe_client_message)

    def _observe_client_message(self, event):
        """``initialize``'s raw capabilities (the SDK's typed model drops ``elicitation``)."""
        message = getattr(event, "message", None)
        if getattr(event, "direction", None) == "incoming" and isinstance(message, dict) \
                and message.get("method") == "initialize":
            from acp_adapter.elicitation import client_supports_form_elicitation
            self._form_elicitation = client_supports_form_elicitation(message.get("params"))

    async def initialize(self, **kwargs):
        from hermes_cli import __version__
        from acp_adapter.auth import build_auth_methods
        return InitializeResponse(protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=__version__),
            agent_capabilities=AgentCapabilities(load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(fork=SessionForkCapabilities(), list=SessionListCapabilities(),
                                                         resume=SessionResumeCapabilities())),
            auth_methods=build_auth_methods())

    async def authenticate(self, method_id, **kwargs):
        from acp_adapter.server import HermesACPAgent
        return await HermesACPAgent.authenticate(self, method_id, **kwargs)

    async def _client(self):
        async with self._connect_lock:
            if self._failure is not None or self._transport_closed():
                # The event pump died with this transport (gateway restart, socket loss, replay gap).
                # Its waiters hold the ambiguous failure; the next call gets a fresh, re-verified
                # connection instead of the stale error forever.
                await self._teardown()
            if self._gateway is None:
                connection = connect_gateway()
                gateway = await connection.__aenter__()
                try:
                    descriptor = await gateway.rpc("runtime.describe")
                    if descriptor.get("profile_id") != str(self._home):
                        raise GatewayClientError("profile_mismatch")
                    await self._reattach(gateway)
                except BaseException:
                    await connection.__aexit__(None, None, None)
                    raise
                self._connection, self._gateway = connection, gateway
                self._event_task = asyncio.create_task(self._events())
            if self._failure:
                raise self._failure
            return self._gateway

    async def _reattach(self, gateway):
        """Subscribe a replacement connection to every session this editor holds.

        The fresh snapshot carries the new replay epoch/sequence and the authority's current
        pending rows (a turn lost in the restart is ``unknown`` there, never resubmitted); the
        editor's borrowed MCP servers are re-attached because a restarted owner has none."""
        from hermes_cli.gateway_client import GatewayRPCError
        for session_id in list(self._snapshots):
            params = {"editor": self._editor_mcp[session_id]} if session_id in self._editor_mcp else {}
            try:
                snapshot = await gateway.rpc("session.resume", session_id=session_id, **params)
            except GatewayRPCError:
                # A definitive refusal for this session (deleted, retired) must not keep the
                # editor's other sessions offline; its next prompt reports not_found.
                logger.info("ACP session %s not re-attached after reconnect", session_id, exc_info=True)
                self._snapshots.pop(session_id, None)
                continue
            self._snapshots[session_id] = snapshot
            for pending in snapshot.get("prompts", []):
                self._permission(session_id, pending)

    def _transport_closed(self):
        # The socket reader settles every pending RPC before the pump sees its disconnect frame,
        # so an editor retry can arrive while ``_failure`` is still unset.
        reader = getattr(self._gateway, "reader", None)
        return isinstance(reader, asyncio.Future) and reader.done()

    async def _teardown(self):
        task, connection = self._event_task, self._connection
        self._event_task = self._connection = self._gateway = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if connection is not None:
            await connection.__aexit__(None, None, None)
        # Partial reply segments belonged to admissions whose waiters already failed.
        self._streamed.clear()
        self._segments.clear()
        async with self._changed:
            # Every waiter on the old transport wakes to its ambiguous failure, even one that had
            # not yet observed ``_failure`` before it is cleared here.
            self._link += 1
            self._lost_failure = self._failure or GatewayClientError(
                "Gateway disconnected; turn outcome is unknown. Resume the session to check whether "
                "it completed or was interrupted; do not resend the work.")
            self._failure = None
            self._changed.notify_all()

    async def new_session(self, cwd, mcp_servers=None, **kwargs):
        client = await self._client()
        descriptor = await client.rpc("runtime.describe")
        if ("acp" not in descriptor.get("session_create", {}).get("sources", [])
                or "acp-editor-policy-v1" not in descriptor.get("capabilities", [])):
            raise GatewayClientError("acp_policy_unavailable")
        snapshot = await client.rpc("session.create", request_id=uuid.uuid4().hex,
            source="acp", cwd=_translate_acp_cwd(cwd),
            editor={"mcp_servers": [s.model_dump(by_alias=True) for s in mcp_servers or []],
                    "edit_approval_policy": "ask"})
        self._snapshots[snapshot["session_id"]] = snapshot
        if mcp_servers:
            self._editor_mcp[snapshot["session_id"]] = {
                "mcp_servers": [s.model_dump(by_alias=True) for s in mcp_servers]}
        return NewSessionResponse(session_id=snapshot["session_id"])

    def _editor_id(self, session_id):
        return next((held for held, sid in self._aliases.items() if sid == session_id), session_id)

    async def _resume(self, cwd, held_id, mcp_servers):
        client = await self._client()
        from acp_adapter.catalog import logical_session_id
        session_id = await asyncio.to_thread(logical_session_id, self._home / "state.db", held_id)
        # The latest load names the id this editor addresses the conversation by.
        self._aliases = {held: sid for held, sid in self._aliases.items() if sid != session_id}
        if session_id != held_id:
            self._aliases[held_id] = session_id
        info = await client.rpc("session.info", session_id=session_id)
        if "cwd" in info and _normalize_cwd_for_compare(info["cwd"]) != _normalize_cwd_for_compare(_translate_acp_cwd(cwd)):
            raise GatewayClientError("cwd_policy_conflict")
        if "cwd" not in info and self._conn:
            await self._conn.session_update(session_id=held_id, update=acp.update_agent_message_text(
                "Attached to the gateway's existing session policy; editor cwd is not applied.\n"))
        resume_params = {}
        if mcp_servers:
            descriptor = await client.rpc("runtime.describe")
            if "acp-session-mcp-v1" not in descriptor.get("capabilities", []):
                raise GatewayClientError("acp_mcp_policy_unavailable")
            resume_params['editor'] = {'mcp_servers': [s.model_dump(by_alias=True) for s in mcp_servers]}
        snapshot = await client.rpc("session.resume", session_id=session_id, **resume_params)
        self._snapshots[session_id] = snapshot
        if 'editor' in resume_params:
            self._editor_mcp[session_id] = resume_params['editor']
        from acp_adapter.server import _history_replay_updates
        if self._conn:
            for update in _history_replay_updates(snapshot["messages"]):
                await self._conn.session_update(session_id=held_id, update=update)
        for pending in snapshot.get("prompts", []):
            self._permission(session_id, pending)
        if self._has_unknown(session_id) and self._conn:
            # The classic chat's resume notice: say why the next prompt will be refused and how to recover.
            await self._conn.session_update(session_id=held_id, update=acp.update_agent_message_text(
                self._unknown_recovery(session_id, submitted=False).removeprefix("unknown_execution: ") + "\n"))
        return snapshot

    async def load_session(self, cwd, session_id, mcp_servers=None, **kwargs):
        await self._resume(cwd, session_id, mcp_servers)
        return LoadSessionResponse()

    async def resume_session(self, cwd, session_id, mcp_servers=None, **kwargs):
        await self._resume(cwd, session_id, mcp_servers)
        return ResumeSessionResponse()

    async def _cancel_admission(self, session_id, admission_id):
        client = await self._client()
        # Cancel our own admission; another surface's running turn is not ours to
        # interrupt. Only when our admission is the one executing do we interrupt.
        try:
            await client.rpc("prompt.cancel", session_id=session_id, admission_id=admission_id)
        except GatewayClientError as exc:
            if str(exc) != "stale_generation":
                raise
            receipt = await client.rpc("prompt.receipt", session_id=session_id, admission_id=admission_id)
            if receipt["status"] != "started":
                return
            try:
                await client.rpc("session.interrupt", session_id=session_id,
                                 execution_generation=receipt["execution_generation"])
            except GatewayClientError as interrupt_exc:
                if str(interrupt_exc) != "stale_generation":
                    raise
                # The exact admission may have settled between the receipt and interrupt.
                # Re-read that admission only; never substitute the session's current
                # generation, which could already belong to a successor.
                settled = await client.rpc(
                    "prompt.receipt", session_id=session_id, admission_id=admission_id)
                if settled["status"] != "terminal":
                    raise

    async def cancel(self, session_id, **kwargs):
        session_id = self._aliases.get(session_id, session_id)
        if session_id not in self._snapshots:
            raise GatewayClientError("not_found")
        admission_id = self._admissions.get(session_id)
        if admission_id is None:
            if session_id in self._submitting:
                self._pending_cancels.add(session_id)
            return
        await self._cancel_admission(session_id, admission_id)

    async def fork_session(self, cwd, session_id, mcp_servers=None, **kwargs):
        from acp.schema import ForkSessionResponse
        session_id = self._aliases.get(session_id, session_id)
        client = await self._client()
        info = await client.rpc('session.info', session_id=session_id)
        if (mcp_servers or _normalize_cwd_for_compare(info.get('cwd', '')) !=
                _normalize_cwd_for_compare(_translate_acp_cwd(cwd))):
            raise GatewayClientError('cwd_policy_conflict')
        result = await self._mutations.apply(client, session_id, 'branch', {})
        child = result['branched_session_id']
        self._snapshots[child] = await client.rpc('session.resume', session_id=child)
        self._mutations.acknowledge(session_id, 'branch', {})
        return ForkSessionResponse(session_id=child)

    async def set_session_model(self, model_id, session_id, **kwargs):
        from acp.schema import SetSessionModelResponse
        session_id = self._aliases.get(session_id, session_id)
        client = await self._client()
        payload = {'model': model_id}
        result = await self._mutations.apply(client, session_id, 'model', payload,
                                             confirm=self._model_confirmer(session_id))
        if result.get('status') == 'cancelled':
            # Declined: nothing was written; the editor's picker must not show the target as active.
            raise GatewayClientError('model_switch_cancelled')
        self._snapshots[session_id] = await client.rpc('session.resume', session_id=session_id)
        self._mutations.acknowledge(session_id, 'model', payload)
        return SetSessionModelResponse()

    def _model_confirmer(self, session_id):
        """The owner's guarded-model confirmation as an ACP permission request (the mechanism
        approvals use): "Switch anyway" re-sends the mutation once with the owner's token; deny,
        dismissal or a lost editor keeps the current model. None without an editor connection."""
        if self._conn is None:
            return None

        async def confirm(refusal):
            from acp.schema import AllowedOutcome, PermissionOption
            from agent.i18n import t
            from hermes_cli.gateway_mutations import confirmation_title
            title = confirmation_title(refusal)
            tool_call = acp.update_tool_call(
                f"model-confirm-{uuid.uuid4().hex[:12]}", title=f"{title}: {refusal.get('target_model', '')}",
                kind="other", status="pending",
                content=[acp.tool_content(acp.text_block(refusal['confirm_message']))],
                raw_input={'model': refusal.get('target_model'), 'provider': refusal.get('target_provider'),
                           'guards': [w.get('kind') for w in refusal.get('warnings') or []]})
            options = [PermissionOption(option_id="allow_once", kind="allow_once",
                                        name=t("cli.model.choice_switch_anyway")),
                       PermissionOption(option_id="deny", kind="reject_once", name=t("cli.model.choice_cancel"))]
            response = await self._conn.request_permission(session_id=self._editor_id(session_id),
                                                           tool_call=tool_call, options=options)
            return isinstance(response.outcome, AllowedOutcome) and response.outcome.option_id == "allow_once"
        return confirm

    async def set_session_mode(self, mode_id, session_id, **kwargs):
        raise GatewayClientError("acp_edit_policy_mutation_unavailable")

    async def set_config_option(self, config_id, session_id, **kwargs):
        raise GatewayClientError("acp_config_mutation_unavailable")

    async def list_sessions(self, cursor=None, cwd=None, **kwargs):
        from acp.schema import ListSessionsResponse, SessionInfo
        from acp_adapter.catalog import catalog_sessions
        rows = await asyncio.to_thread(catalog_sessions, self._home / "state.db", cwd)
        if cursor:
            index = next((i for i, row in enumerate(rows) if row["session_id"] == cursor), None)
            rows = [] if index is None else rows[index + 1:]
        page = [SessionInfo(session_id=row["session_id"], cwd=row["cwd"], title=row.get("title"),
                            updated_at=row.get("updated_at")) for row in rows[:50]]
        return ListSessionsResponse(sessions=page, next_cursor=page[-1].session_id if len(rows) > 50 else None)

    async def prompt(self, prompt, session_id, **kwargs):
        held_id, session_id = session_id, self._aliases.get(session_id, session_id)
        if session_id not in self._snapshots:
            raise GatewayClientError("not_found")
        if self._has_unknown(session_id):
            raise GatewayClientError(self._unknown_recovery(session_id, submitted=False))
        from acp_adapter.content import _content_blocks_to_openai_user_content
        text, attachments = _stage_user_content(_content_blocks_to_openai_user_content(prompt))
        # Only this session's very next prompt may be the editor's retry of an un-acked submit; any
        # prompt (slash work included) retires the retained identity, so a later repeat is new work.
        retained = self._unacked.pop(session_id, None)
        command = _slash_command(text) if not attachments else None
        if command is not None:
            from hermes_cli.gateway_mutations import slash_mutation
            parts = text.strip().split(None, 1)
            operation, payload = slash_mutation('/' + command, parts[1] if len(parts) > 1 else '')
            if operation == 'branch':
                raise GatewayClientError('use_acp_fork_session')
            client = await self._client()
            result = await self._mutations.apply(client, session_id, operation, payload,
                                                 confirm=self._model_confirmer(session_id) if operation == 'model' else None)
            if result.get('status') == 'cancelled':
                from agent.i18n import t
                await self._conn.session_update(session_id=held_id,
                    update=acp.update_agent_message_text(t('cli.model.switch_cancelled') + '\n'))
            elif result.get('status') == 'preview':
                # Read-only report: nothing changed, so the editor's snapshot is still current.
                await self._conn.session_update(session_id=held_id,
                    update=acp.update_agent_message_text('\n'.join(result['lines']) + '\n'))
            else:
                self._snapshots[session_id] = await client.rpc('session.resume', session_id=session_id)
            self._mutations.acknowledge(session_id, operation, payload)
            return PromptResponse(stop_reason='end_turn')
        client = await self._client()
        link = self._link
        submit = {'text': text}
        if attachments:
            submit['attachments'] = attachments
        retry_key = await asyncio.to_thread(_submit_fingerprint, session_id, text, attachments)
        input_id = retained[1] if retained is not None and retained[0] == retry_key else uuid.uuid4().hex
        self._submitting.add(session_id)
        try:
            receipt = await client.rpc("prompt.submit", session_id=session_id, input_id=input_id, **submit)
        except BaseException as exc:
            from hermes_cli.gateway_client import GatewayRPCError
            if not isinstance(exc, GatewayRPCError):
                # Transport loss, timeout or cancel before the ack: the owner may hold this admission.
                # Keep its identity for the editor's retry of the same prompt (latest 64 sessions).
                self._unacked[session_id] = (retry_key, input_id)
                while len(self._unacked) > 64:
                    self._unacked.pop(next(iter(self._unacked)))
            self._pending_cancels.discard(session_id)
            raise
        finally:
            self._submitting.discard(session_id)
        admission_id = receipt["admission_id"]
        self._admissions[session_id] = admission_id
        if receipt.get("status") == "terminal":
            # A retry of an un-acked submit whose turn already settled: report that turn once.
            await self._settled_receipt(client, held_id, session_id, receipt)
        elif receipt.get("status") == "unknown":
            # A retry of an un-acked submit the owner lost in a restart: never a second turn.
            self._admissions.pop(session_id, None)
            raise GatewayClientError(self._unknown_recovery(session_id, admission_id))
        try:
            if session_id in self._pending_cancels:
                self._pending_cancels.discard(session_id)
                await self._cancel_admission(session_id, admission_id)
            async with self._changed:
                await self._changed.wait_for(lambda: admission_id in self._terminals or self._failure is not None
                                            or self._link != link
                                            or self._blocked_admission(session_id, admission_id))
                # A successor's unknown state cannot replace our committed terminal result.
                if admission_id not in self._terminals:
                    if self._link != link:
                        # Our transport died before this turn's outcome reached us: ambiguous,
                        # never retried on the replacement connection.
                        raise self._lost_failure
                    if self._failure:
                        raise self._failure
                    raise GatewayClientError(self._unknown_recovery(session_id, admission_id))
                terminal = self._terminals.pop(admission_id)
        finally:
            # Do not let one prompt tear down a newer mapping if the session has
            # already advanced while this coroutine unwinds.
            if self._admissions.get(session_id) == admission_id:
                self._admissions.pop(session_id, None)
        outcome = terminal.get("outcome")
        if outcome == "unknown":
            # Our own lost turn: the same refusal whether session.info or this completion woke us.
            raise GatewayClientError(self._unknown_recovery(session_id, admission_id))
        if outcome == "failed":
            raise GatewayClientError("admitted_turn_failed")
        return PromptResponse(stop_reason="cancelled" if outcome == "cancelled" else "end_turn")

    async def _settled_receipt(self, client, held_id, session_id, receipt):
        admission_id = receipt["admission_id"]
        saved = await client.rpc("prompt.receipt", session_id=session_id, admission_id=admission_id,
                                 include_result=True)
        if admission_id in self._terminals or admission_id in self._settling:
            return  # this connection's pump carried the completion itself
        text = (saved.get("result") or {}).get("final_response") or ""
        if text and self._conn:
            await self._conn.session_update(session_id=held_id, update=acp.update_agent_message_text(text))
        outcome = {"interrupted": "cancelled"}.get(receipt.get("outcome"), receipt.get("outcome"))
        async with self._changed:
            self._terminals[admission_id] = {"outcome": outcome}
            self._changed.notify_all()

    def _unknown_recovery(self, session_id, admission_id=None, *, submitted=True):
        """The refusal for a FIFO blocked by a lost turn, naming the existing recovery control.

        ACP has no discard affordance; the classic chat's fenced ``/discard`` (``prompt.resolve_unknown``)
        acknowledges the exact lost admission without replaying it, and the queue then continues."""
        from hermes_constants import profile_name_for_home
        lost = [row["admission_id"] for row in self._snapshots.get(session_id, {}).get("pending", [])
                if row.get("status") == "unknown"] or [admission_id]
        profile = profile_name_for_home(self._home)
        command = "hermes" + (f" -p {profile}" if profile not in (None, "default") else "")
        state = ("your input is accepted and kept" if submitted else "nothing was submitted")
        return (f"unknown_execution: do not resend accepted input; a turn in this session was lost when the "
                f"gateway restarted and its outcome is unknown ({state}). Resolve it in a terminal with "
                f"`{command} chat --cli --resume {session_id}` then "
                + " and ".join(f"`/discard {lost_id}`" for lost_id in lost) + ", then continue here.")

    def _has_unknown(self, session_id):
        return any(row.get("status") == "unknown"
                   for row in self._snapshots[session_id].get("pending", []))

    def _blocked_admission(self, session_id, admission_id):
        return self._has_unknown(session_id) and any(
            row.get("admission_id") == admission_id and row.get("status") in {"queued", "unknown"}
            for row in self._snapshots[session_id].get("pending", []))

    async def _events(self):
        gateway = self._gateway
        try:
            while True:
                frame = await gateway.events.get()
                if isinstance(frame, Exception):
                    raise frame
                event = frame.get("params", {})
                sid = event.get("session_id")
                snapshot = self._snapshots.get(sid)
                if snapshot is None:
                    continue
                epoch, seq = event.get("replay_epoch"), event.get("seq", 0)
                if epoch == snapshot.get("replay_epoch") and seq <= snapshot.get("last_sequence", 0):
                    continue
                snapshot.update(replay_epoch=epoch, last_sequence=seq,
                    execution_generation=event.get("execution_generation", snapshot.get("execution_generation")))
                await self._project(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Boundary: ANY projection failure must reach every waiter (prompt/load re-raise
            # ``_failure``), or they block forever on ``_changed``.
            logger.debug("ACP gateway event projection stopped", exc_info=True)
            async with self._changed:
                self._failure = exc
                self._changed.notify_all()

    async def _project(self, event):
        kind, payload = event.get("type"), event.get("payload", {})
        sid, editor_id = event["session_id"], self._editor_id(event["session_id"])
        aid = event.get("admission_id")
        if kind == "session.info":
            if sid in self._snapshots and "pending" in payload:
                async with self._changed:
                    self._snapshots[sid]["pending"] = payload["pending"]
                    self._changed.notify_all()
            return
        if kind == "session.replay_gap":
            raise GatewayClientError("session_replay_gap")
        if kind in {"approval.request", "clarify.request"}:
            self._permission(sid, payload)
            return
        if kind in {"approval.settled", "clarify.settled"}:
            task = self._permissions.pop((sid, payload["prompt_id"], payload["execution_generation"]), None)
            if task:
                task.cancel()
            return
        # In-process turns publish ``tool_name``; managed workers publish ``name``.
        tool_name = payload.get("tool_name") or payload.get("name") or "tool"
        if kind == "tool.start":
            # Deltas before a tool call are commentary the final reply does not repeat: close
            # that segment so ``message.complete`` measures only the final's own stream.
            self._close_segment(aid)
            from acp_adapter.tools import build_tool_start, coerce_tool_args
            args = coerce_tool_args(payload.get("args"))
            self._tool_args[(sid, payload["tool_call_id"])] = (tool_name, args)
            if self._conn:
                await self._conn.session_update(session_id=editor_id,
                    update=build_tool_start(payload["tool_call_id"], tool_name, args))
            return
        if kind == "tool.complete":
            from acp_adapter.tools import build_tool_complete
            name, args = self._tool_args.pop((sid, payload["tool_call_id"]), (tool_name, {}))
            result = payload.get("result")
            if self._conn:
                await self._conn.session_update(session_id=editor_id, update=build_tool_complete(
                    payload["tool_call_id"], name, result=result if isinstance(result, str) else None,
                    function_args=args))
            return
        if kind == "message.delta":
            text = _text(payload)
            self._streamed[aid] = self._streamed.get(aid, "") + text
            if text and self._conn:
                await self._conn.session_update(session_id=editor_id, update=acp.update_agent_message_text(text))
        elif kind == "message.interim":
            # Commentary the owner could not stream (``already_streamed`` false) reaches viewers
            # only here; streamed commentary is already on screen. Either way it is its own segment.
            self._close_segment(aid)
            text = _text(payload)
            if not payload.get("already_streamed") and text.strip():
                self._segments.setdefault(aid, []).append(text)
                if self._conn:
                    await self._conn.session_update(session_id=editor_id,
                                                    update=acp.update_agent_message_text(text + "\n\n"))
        elif kind == "message.complete":
            self._settling.add(aid)
            try:
                remainder = self._final_remainder(aid, payload)
                if remainder and self._conn:
                    await self._conn.session_update(session_id=editor_id,
                                                    update=acp.update_agent_message_text(remainder))
                async with self._changed:
                    self._terminals[aid] = payload
                    if len(self._terminals) > 256:
                        self._terminals.pop(next(iter(self._terminals)))
                    self._changed.notify_all()
            finally:
                self._settling.discard(aid)

    def _close_segment(self, aid):
        streamed = self._streamed.pop(aid, "")
        if streamed.strip():
            self._segments.setdefault(aid, []).append(streamed)

    def _final_remainder(self, aid, payload):
        """The part of the settled reply the editor has not been shown yet (exactly-once text)."""
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        text = _text(payload)
        streamed = self._streamed.pop(aid, "")
        shown = self._segments.pop(aid, [])
        # Local interrupt status is metadata; ACP carries it in stop_reason.
        if not text or (payload.get("outcome") == "cancelled"
                        and text.startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX)):
            return ""
        if payload.get("response_reused") is True and (streamed or shown):
            # The owner names the final as the reply already painted (never inferred from text).
            return ""
        if text.startswith(streamed):
            remainder = text[len(streamed):]
            # An unstreamed final the owner already published as an interim segment.
            return "" if not streamed and text.strip() in {s.strip() for s in shown} else remainder
        if text.strip() == streamed.strip():
            return ""
        return "\n\n" + text

    def _permission(self, session_id, prompt):
        answer = {"approval": self._answer_permission, "clarify": self._answer_clarify}.get(prompt.get("kind"))
        if answer is None or self._conn is None:
            return
        key = (session_id, prompt["prompt_id"], prompt["execution_generation"])
        if key not in self._permissions:
            task = asyncio.create_task(answer(session_id, prompt))
            self._permissions[key] = task
            task.add_done_callback(lambda done: self._retire_unanswered(key, done))

    def _retire_unanswered(self, key, task):
        """A card whose response the owner did not accept (editor cancel, transport loss, lost
        respond RPC) gives up its key, so the next snapshot that still lists the prompt (reattach,
        ``session/load``) shows a NEW card. An earlier answer is never replayed. An accepted one
        keeps its key until ``*.settled``, so a replayed request raises no second card."""
        if self._permissions.get(key) is task and (task.cancelled() or task.exception() is not None
                                                   or task.result() is not True):
            del self._permissions[key]

    async def _answer_clarify(self, session_id, prompt):
        from acp.schema import AllowedOutcome
        from acp_adapter.elicitation import (
            ELICITATION_METHOD, SKIP_OPTION, build_clarify_elicitation, build_clarify_permission,
            elicited_answer,
        )
        editor_id = self._editor_id(session_id)
        try:
            if self._form_elicitation:
                answer = elicited_answer(await self._conn._conn.send_request(
                    ELICITATION_METHOD, build_clarify_elicitation(editor_id, prompt)))
            else:
                tool_call, options, answers = build_clarify_permission(prompt)
                response = await self._conn.request_permission(
                    session_id=editor_id, tool_call=tool_call, options=options)
                # A dismissed card (also what ACP answers for the editor's own Stop) is the Skip a
                # form decline sends; only an unknown option id is no answer.
                answer = (answers.get(response.outcome.option_id)
                          if isinstance(response.outcome, AllowedOutcome) else answers[SKIP_OPTION])
            # No answer (editor transport loss) leaves the canonical waiter to other viewers.
            if answer is None:
                return False
            await self._gateway.rpc("clarify.respond", session_id=session_id, prompt_id=prompt["prompt_id"],
                execution_generation=prompt["execution_generation"], answer=answer)
            return True
        except Exception:
            # Boundary: as for approvals, a detached editor or expired prompt is not an answer.
            logger.info("ACP clarify viewer detached or control expired", exc_info=True)
            return False

    async def _answer_permission(self, session_id, prompt):
        from acp.schema import AllowedOutcome
        from acp_adapter.permissions import (
            _build_permission_options, _build_permission_tool_call, _OPTION_ID_TO_HERMES,
        )
        choices = prompt["choices"]
        options = [option for option in _build_permission_options(
            allow_permanent="always" in choices, allow_session="session" in choices)
            if _OPTION_ID_TO_HERMES[option.option_id] in choices]
        try:
            if 'edit' in prompt:
                from acp_adapter.edit_approval import EditProposal, build_acp_edit_tool_call
                tool_call = build_acp_edit_tool_call(EditProposal(**prompt['edit']))
            else:
                tool_call = _build_permission_tool_call(prompt.get('command', ''), prompt.get('description', ''))
            response = await self._conn.request_permission(session_id=self._editor_id(session_id),
                tool_call=tool_call, options=options)
            # Transport loss/cancel is not a denial: the canonical waiter belongs
            # to the execution and may still be answered by another viewer.
            if not isinstance(response.outcome, AllowedOutcome):
                return False
            if response.outcome.option_id not in {option.option_id for option in options}:
                return False
            await self._gateway.rpc("approval.respond", session_id=session_id,
                prompt_id=prompt["prompt_id"], execution_generation=prompt["execution_generation"],
                choice=_OPTION_ID_TO_HERMES[response.outcome.option_id])
            return True
        except Exception:
            # Boundary: a detached viewer or expired control is not a denial; the canonical
            # waiter stays answerable by another viewer, so nothing propagates.
            logger.info("ACP permission viewer detached or control expired", exc_info=True)
            return False

    async def aclose(self):
        for task in self._permissions.values():
            task.cancel()
        if self._permissions:
            await asyncio.gather(*self._permissions.values(), return_exceptions=True)
            self._permissions.clear()
        await self._teardown()
