"""ACP projection of the profile gateway; never constructs an execution owner."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import os
import uuid

import acp
from acp.schema import (
    AgentCapabilities, Implementation, InitializeResponse, LoadSessionResponse,
    NewSessionResponse, PromptCapabilities, PromptResponse, ResumeSessionResponse,
    SessionCapabilities, SessionForkCapabilities, SessionListCapabilities, SessionResumeCapabilities,
)

from hermes_cli.gateway_client import GatewayClientError, connect_gateway
from hermes_constants import get_hermes_home
from acp_adapter.gateway_reply import GatewayReplyMixin, _text
from acp_adapter.session import _translate_acp_cwd, _normalize_cwd_for_compare

logger = logging.getLogger(__name__)


def _slash_command(text):
    """Canonical Hermes command name ``text`` invokes, or None for ordinary text.

    Only a registered command (or alias) is a command: ``/etc/hosts is wrong`` or ``/tmp/x``
    (a path, the same rule ``MessageEvent.get_command`` applies) and unknown ``/words`` are
    prompt text for the model. A registered command is never sent as prompt text: it is answered
    by a gateway verb or refused by name (``acp_adapter/gateway_commands.py``)."""
    stripped = text.lstrip()
    if not stripped.startswith('/'):
        return None
    name = stripped.split(None, 1)[0][1:]
    if not name or '/' in name:
        return None
    from hermes_cli.commands import resolve_command
    command = resolve_command(name)
    return command.name if command is not None else None


def _bypass_route():
    """--safe-mode / --ignore-user-config read no profile default route, so the session freezes the
    launch's explicit one (``hermes -m <model> --provider <name> --safe-mode acp``). Both, refused at
    session/new: a model alone is accepted by the owner, then its first turn cannot build the agent."""
    model = (os.environ.get("HERMES_MODEL") or os.environ.get("HERMES_INFERENCE_MODEL") or "").strip()
    if not model:
        raise GatewayClientError("--safe-mode / --ignore-user-config read no profile default model: launch "
                                 "with one, e.g. hermes -m <model> --provider <provider> --safe-mode acp")
    provider = os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip()
    if not provider:
        raise GatewayClientError("--safe-mode / --ignore-user-config read no profile provider: launch with "
                                 "--provider too, e.g. hermes -m <model> --provider <provider> --safe-mode acp")
    return {"model": model, "provider": provider}


def _tools_changed_notice(payload):
    """The owner re-froze this session's editor MCP tools (a changed config or changed schemas)."""
    servers = ", ".join(payload.get("mcp_servers") or []) or "none"
    return f"Editor MCP tools changed (servers: {servers}); the new tool list applies from the next turn.\n"


class GatewayACPAgent(GatewayReplyMixin, acp.Agent):
    def __init__(self):
        self._conn = None
        self._gateway = None
        self._connection = None
        self._connect_lock = asyncio.Lock()
        # A per-task overlay HERMES_HOME attaches to the profile its symlinked lock names (#138949).
        from hermes_cli.gateway_runtime import gateway_profile_home
        self._home = gateway_profile_home(get_hermes_home())
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
        # Session -> (content fingerprint, input_id, staged (text, attachments)) of a submit whose ack
        # never arrived. The editor's retry (that session's next prompt, same content) reuses it, so the authority
        # returns the same admission.
        self._unacked = {}
        # Admissions whose terminal reply one path (the pump's message.complete or a retry's
        # terminal receipt) is delivering right now; the other path yields to the claim.
        self._settling = set()
        # Admissions a terminal receipt already delivered: their late message.complete adds nothing.
        self._receipted = {}
        self._pending_cancels = set()
        # (session, tool_call_id) -> (tool name, args, admission) of a call still open on the editor.
        self._tool_args = {}
        # Session -> AssistantMessageIdAllocator: the ``messageId`` of the assistant message being streamed.
        self._message_ids = {}
        # Editor-held compression-tip id -> logical session id (see ``_resume``).
        self._aliases = {}
        self._form_elicitation = False
        from hermes_cli.gateway_mutations import PreparedMutations
        self._mutations = PreparedMutations()
        # Session -> title the editor was last shown (``session_info_update`` fires on a change).
        self._titles = {}
        # Session -> the owner's latest usage dict (``session.usage`` / ``session.stats``).
        self._usage = {}
        self._background = set()

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
            # Before this session's cards are offered again: until ``_client`` returns, ``self._gateway`` is
            # not this connection, so a card answered while the request is out would lose its answer.
            await self._request_live_reasoning(gateway, session_id)
            for pending in snapshot.get("prompts", []):
                self._permission(session_id, pending)
            await self._notify_unknown(session_id)

    async def _request_live_reasoning(self, client, session_id):
        """Ask the owner to stream this session's reasoning (``show_reasoning``), as the CLI does.

        The in-process server gave every editor turn a reasoning sink (``acp_adapter/server.py``). The
        owner publishes ``reasoning.delta`` only for a live session whose flag a viewer set, and a restarted
        owner's sessions start without it, so every attach and re-attach asks (#134889 ACP-4). The flag is
        the session's: a viewer that hides reasoning later turns it off until the editor attaches again."""
        try:
            await client.rpc("config.set", session_id=session_id, key="show_reasoning", value="1", _timeout=10)
        except Exception:
            # Boundary: a display flag must not fail the attach (a refusal, a lost connection, or no answer
            # within the CLI's 10 s). At worst the session runs without live reasoning; a reconnect's
            # re-attach asks again.
            logger.debug("ACP session %s keeps its reasoning hidden", session_id, exc_info=True)

    async def _notify_unknown(self, session_id):
        """The classic chat's resume notice: say why the next prompt will be refused and how to recover."""
        if self._has_unknown(session_id) and self._conn:
            await self._conn.session_update(session_id=self._editor_id(session_id), update=acp.update_agent_message_text(
                self._unknown_recovery(session_id, submitted=False).removeprefix("unknown_execution: ") + "\n"))

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
        # Partial reply segments belonged to admissions whose waiters already failed. A reply cut off here
        # never gets its ``message.complete`` (re-attach starts at the owner's current sequence), so its
        # ``messageId`` closes too: the next reply must not reuse it.
        self._streamed.clear()
        self._segments.clear()
        self._message_ids.clear()
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
        from hermes_cli.gateway_chat import env_launch_flags
        # `hermes acp --yolo` / `--safe-mode` / ... reach this process as their env equivalents; the
        # owner does not inherit them, so they freeze into this session's policy or not at all.
        flags = env_launch_flags()
        if flags.get("safe_mode") or flags.get("ignore_user_config"):
            flags.update(_bypass_route())
        snapshot = await client.rpc("session.create", request_id=uuid.uuid4().hex,
            source="acp", cwd=_translate_acp_cwd(cwd),
            editor={"mcp_servers": [s.model_dump(by_alias=True) for s in mcp_servers or []],
                    "edit_approval_policy": "ask"}, **flags)
        session_id = snapshot["session_id"]
        self._snapshots[session_id] = snapshot
        if mcp_servers:
            self._editor_mcp[session_id] = {
                "mcp_servers": [s.model_dump(by_alias=True) for s in mcp_servers]}
        info = snapshot.get("info") or await client.rpc("session.info", session_id=session_id)
        await self._request_live_reasoning(client, session_id)
        return NewSessionResponse(session_id=session_id, **await self._session_fields(session_id, info, session_id))

    def _editor_id(self, session_id):
        return next((held for held, sid in self._aliases.items() if sid == session_id), session_id)

    async def _session_fields(self, session_id, info, editor_id):
        """Main's new/load/resume/fork reply fields (model picker, modes, provenance ``_meta``); the
        command list and context indicator follow once the reply is queued, as main scheduled them."""
        from acp_adapter.gateway_session_meta import model_state, provenance_meta, session_modes
        title = info.get("title")
        if isinstance(title, str) and title.strip():
            self._titles[session_id] = title
        models = await asyncio.to_thread(model_state, info)
        meta = await asyncio.to_thread(provenance_meta, self._home / "state.db", editor_id, session_id)
        if self._conn:
            task = asyncio.get_running_loop().create_task(self._after_attach(session_id, editor_id))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        return {"models": models, "modes": session_modes(), "field_meta": meta}

    async def _after_attach(self, session_id, editor_id):
        from acp_adapter.gateway_session_meta import available_commands_update
        await asyncio.sleep(0)  # after the session reply: the editor routes updates by a known id
        try:
            await self._conn.session_update(session_id=editor_id, update=available_commands_update())
            await self._project_usage(session_id, editor_id,
                                      await (await self._client()).rpc("session.stats", session_id=session_id))
        except Exception:
            # Boundary: a detached editor or a refused read is not the attach's failure.
            logger.debug("ACP post-attach notices for %s not delivered", session_id, exc_info=True)

    async def _attach_info(self, client, session_id):
        """``session.info`` for a session to attach. A thread main's in-process server wrote has no
        gateway policy yet; ``session.resume`` is the verb that adopts it (``LEGACY_LOCAL_SOURCES``)."""
        from hermes_cli.gateway_client import GatewayRPCError
        try:
            return await client.rpc("session.info", session_id=session_id)
        except GatewayRPCError as exc:
            if str(exc) != "not_found":
                raise
        adopted = await client.rpc("session.resume", session_id=session_id)
        await client.rpc("session.detach", session_id=session_id, subscription_id=adopted["subscription_id"])
        return await client.rpc("session.info", session_id=session_id)

    async def _resume(self, cwd, held_id, mcp_servers):
        client = await self._client()
        from acp_adapter.catalog import logical_session_id
        from hermes_cli.gateway_client import GatewayRPCError
        session_id = await asyncio.to_thread(logical_session_id, self._home / "state.db", held_id)
        # The latest load names the id this editor addresses the conversation by.
        self._aliases = {held: sid for held, sid in self._aliases.items() if sid != session_id}
        if session_id != held_id:
            self._aliases[held_id] = session_id
        info = await self._attach_info(client, session_id)
        if self._moved(info, cwd):
            info = await self._move_workspace(client, session_id, cwd)
        if "cwd" not in info and self._conn:
            await self._conn.session_update(session_id=held_id, update=acp.update_agent_message_text(
                "Attached to the gateway's existing session policy; editor cwd is not applied.\n"))
        resume_params = {}
        if mcp_servers:
            # A non-empty list is the editor's selection. An empty one (load always carries a list)
            # keeps the session's servers, as fork does: a second editor without MCP config must not
            # strip the first one's tools. An owner without editor MCP bound none.
            descriptor = await client.rpc("runtime.describe")
            if "acp-session-mcp-v1" not in descriptor.get("capabilities", []):
                raise GatewayClientError("acp_mcp_policy_unavailable")
            resume_params['editor'] = {'mcp_servers': [s.model_dump(by_alias=True) for s in mcp_servers]}
        try:
            snapshot = await client.rpc("session.resume", session_id=session_id, **resume_params)
        except GatewayRPCError as exc:
            if str(exc) != "acp_mcp_policy_conflict":
                raise
            # The editor's MCP config changed: the owner re-binds the session's servers (control
            # capability, idle only), as the workspace move above, instead of refusing the thread.
            await self._rebind_editor_mcp(client, session_id, resume_params['editor'])
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
        await self._notify_unknown(session_id)
        await self._request_live_reasoning(client, session_id)
        return await self._session_fields(session_id, {**info, **(snapshot.get("info") or {})}, held_id)

    @staticmethod
    def _moved(info, cwd):
        return "cwd" in info and _normalize_cwd_for_compare(info["cwd"]) != _normalize_cwd_for_compare(_translate_acp_cwd(cwd))

    async def _move_workspace(self, client, session_id, cwd):
        """Main rebound a session opened from another folder to that folder; the owner's ``workspace``
        mutation (Move to project) moves execution cwd and TERMINAL_CWD together, idle sessions only."""
        payload = {'cwd': _translate_acp_cwd(cwd)}
        self._unacked.pop(session_id, None)
        await self._mutations.apply(client, session_id, 'workspace', payload)
        self._mutations.acknowledge(session_id, 'workspace', payload)
        return await client.rpc("session.info", session_id=session_id)

    async def _rebind_editor_mcp(self, client, session_id, editor):
        """The owner's ``editor_mcp`` mutation: rediscovers *editor*'s servers into the session's policy."""
        self._unacked.pop(session_id, None)
        await self._mutations.apply(client, session_id, 'editor_mcp', editor)
        self._mutations.acknowledge(session_id, 'editor_mcp', editor)

    async def load_session(self, cwd, session_id, mcp_servers=None, **kwargs):
        return LoadSessionResponse(**await self._resume(cwd, session_id, mcp_servers))

    async def resume_session(self, cwd, session_id, mcp_servers=None, **kwargs):
        return ResumeSessionResponse(**await self._resume(cwd, session_id, mcp_servers))

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
        # The owner binds the child's editor servers under the child's own scope: these, or else a
        # copy of the parent's.
        payload = {'mcp_servers': [s.model_dump(by_alias=True) for s in mcp_servers]} if mcp_servers else {}
        self._unacked.pop(session_id, None)
        result = await self._mutations.apply(client, session_id, 'branch', payload)
        child = result['branched_session_id']
        if self._moved(info, cwd):
            await self._move_workspace(client, child, cwd)
        self._snapshots[child] = await client.rpc('session.resume', session_id=child)
        if payload or session_id in self._editor_mcp:
            self._editor_mcp[child] = payload or self._editor_mcp[session_id]
        self._mutations.acknowledge(session_id, 'branch', payload)
        info = self._snapshots[child].get('info') or await client.rpc('session.info', session_id=child)
        await self._request_live_reasoning(client, child)
        return ForkSessionResponse(session_id=child, **await self._session_fields(child, info, child))

    async def set_session_model(self, model_id, session_id, **kwargs):
        from acp.schema import SetSessionModelResponse
        session_id = self._aliases.get(session_id, session_id)
        client = await self._client()
        from hermes_cli.models import parse_model_input
        # Picker IDs include the provider, as on the in-process ACP switch path.
        provider, model = await asyncio.to_thread(parse_model_input, model_id, '')
        payload = {'model': model, **({'provider': provider} if provider else {})}
        self._unacked.pop(session_id, None)  # a control between a lost submit and its repeat retires it
        result = await self._mutations.apply(client, session_id, 'model', payload,
                                             confirm=self._model_confirmer(session_id))
        if result.get('status') == 'cancelled':
            # Declined: nothing was written; the editor's picker must not show the target as active.
            raise GatewayClientError('model_switch_cancelled')
        self._snapshots[session_id] = await client.rpc('session.resume', session_id=session_id)
        self._mutations.acknowledge(session_id, 'model', payload)
        if result.get('status') == 'deferred' and self._conn:
            # A pick during a running turn lands for the next one (#139604); say so, as Desktop does.
            from acp_adapter.gateway_commands import deferred_pick_text
            await self._conn.session_update(session_id=self._editor_id(session_id),
                                            update=acp.update_agent_message_text(deferred_pick_text(result) + "\n"))
        return SetSessionModelResponse()

    async def _confirm_deferred_pick(self, session_id, editor_id, ask):
        """The owner's boundary ask for a guarded deferred pick (``confirmation_required``; nothing
        written, the pick stays pending): the same permission request as an idle pick. Yes re-sends
        the same selection once with the owner's token (as Ink/Desktop); no, or no editor, keeps the
        committed model, and the editor is told which outcome it got, never left on the promise."""
        from acp_adapter.gateway_commands import _model_text, deferred_pick_text, refused_pick_text
        from hermes_cli.gateway_mutations import model_refusal_text

        async def say(text):
            if self._conn:
                await self._conn.session_update(session_id=editor_id, update=acp.update_agent_message_text(text + "\n"))
        target = ask.get("target_model") or (ask.get("payload") or {}).get("model")
        committed = {"model": ask.get("committed_model"), "provider": ask.get("committed_provider"),
                     "target_model": target}
        confirm = self._model_confirmer(session_id)
        try:
            if confirm is None or not await confirm(ask):
                await say(refused_pick_text({**committed, "reason": "not confirmed"}))
                return
            client = await self._client()
            snapshot = await client.rpc("session.resume", session_id=session_id)
            result = await client.rpc("session.mutate", session_id=session_id, operation="model",
                                      request_id=uuid.uuid4().hex, expected_revision=snapshot["revision"],
                                      expected_generation=snapshot["execution_generation"],
                                      payload={**(ask.get("payload") or {}), "confirm": ask["confirm"]})
            if result.get("status") == "confirmation_required":
                # Another pick or a turn moved the session first: said once, never re-asked in a loop.
                await say(model_refusal_text(result, confirmed=True))
                return
            if session_id in self._snapshots:
                self._snapshots[session_id] = await client.rpc("session.resume", session_id=session_id)
            # Confirmed while a turn runs: stored again for the next boundary, said as such.
            await say(deferred_pick_text(result) if result.get("status") == "deferred"
                      else _model_text("Model switched to", result))
        except GatewayClientError as exc:
            logger.info("Deferred model pick on %s not confirmed: %s", session_id, exc)
            await say(refused_pick_text({**committed, "reason": str(exc)}))

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

    def _held_session(self, session_id):
        session_id = self._aliases.get(session_id, session_id)
        if session_id not in self._snapshots:
            raise GatewayClientError("not_found")
        return session_id

    async def set_session_mode(self, mode_id, session_id, **kwargs):
        """The advertised mode is the owner's own policy, so selecting it changes nothing; main's
        Accept Edits / Don't Ask need an editor-policy change the gateway has no verb for."""
        from acp.schema import SetSessionModeResponse
        from acp_adapter.gateway_session_meta import DEFAULT_MODE
        self._held_session(session_id)
        if mode_id != DEFAULT_MODE:
            raise GatewayClientError("acp_edit_policy_mutation_unavailable")
        return SetSessionModeResponse()

    async def set_config_option(self, config_id, session_id, value=None, **kwargs):
        """Accepted so editors do not fail on their own config writes, as main did; only the edit
        approval policy has a meaning here, and only the owner's ``ask`` can be selected."""
        from acp.schema import SetSessionConfigOptionResponse
        from acp_adapter.gateway_session_meta import DEFAULT_EDIT_APPROVAL_POLICY, EDIT_APPROVAL_POLICY_CONFIG_ID
        self._held_session(session_id)
        if config_id == EDIT_APPROVAL_POLICY_CONFIG_ID and value != DEFAULT_EDIT_APPROVAL_POLICY:
            raise GatewayClientError("acp_edit_policy_mutation_unavailable")
        return SetSessionConfigOptionResponse(config_options=[])

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
        prepared = await self._prepare_submit(session_id, prompt)
        if prepared is None:
            return PromptResponse(stop_reason='cancelled')
        retry_key, input_id, staged = prepared
        text, attachments = staged
        answered, text = await self._run_slash(held_id, session_id, text) if not attachments else (False, text)
        if answered:
            return PromptResponse(stop_reason='end_turn')
        # ... and every prepared control's (a slash keeps its own key: apply retires the rest).
        self._mutations.retire(session_id)
        submit = {'text': text}
        if attachments:
            submit['attachments'] = attachments
        self._submitting.add(session_id)
        try:
            # Inside the try: a failed reconnect is as ambiguous for a retained identity as a lost ack.
            client = await self._client()
            link = self._link
            receipt = await client.rpc("prompt.submit", session_id=session_id, input_id=input_id, **submit)
        except BaseException as exc:
            from hermes_cli.gateway_client import GatewayRPCError
            if not isinstance(exc, GatewayRPCError):
                # Transport loss, timeout or cancel before the ack: the owner may hold this admission.
                # Keep its identity for the editor's retry of the same prompt (latest 64 sessions).
                self._unacked[session_id] = (retry_key, input_id, staged)
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
        from acp_adapter.gateway_session_meta import turn_usage
        # The turn's last ``session.usage`` precedes its ``message.complete`` on the stream.
        return PromptResponse(stop_reason="cancelled" if outcome == "cancelled" else "end_turn",
                              usage=turn_usage(self._usage.get(session_id)))

    async def _run_slash(self, held_id, session_id, text):
        """Run a slash command's local half: ``(answered, text to submit)``; answered ends the prompt."""
        command = _slash_command(text)
        if command is None:
            return False, text
        from acp_adapter.gateway_commands import CommandRefused, run_command
        parts = text.strip().split(None, 1)
        try:
            reply, text = await run_command(self, session_id, command, parts[1].strip() if len(parts) > 1 else '')
        except CommandRefused as exc:
            # A named refusal is the command's answer, shown to the user; never a failed prompt.
            reply = str(exc)
        if reply is None:
            return False, text
        if self._conn:
            await self._conn.session_update(session_id=held_id, update=acp.update_agent_message_text(reply + '\n'))
        return True, None

    async def _prepare_submit(self, session_id, prompt):
        """``(retry_key, input_id, (text, attachments))`` for one ACP prompt, or None when the editor's Stop
        arrived while it was staged."""
        from acp_adapter.content import _content_blocks_to_openai_user_content
        from acp_adapter.gateway_media import stage_user_content, submit_fingerprint
        content = _content_blocks_to_openai_user_content(prompt)
        retry_key = submit_fingerprint(session_id, content)
        # Only this session's very next prompt may be the editor's retry of an un-acked submit; any
        # prompt (slash work included) retires the retained identity, so a later repeat is new work.
        retained = self._unacked.pop(session_id, None)
        if retained is not None and retained[0] == retry_key:
            # The retry resends what the owner may already hold: the staged content and input_id are
            # reused, never prepared (a remote image fetched) a second time.
            return retained
        # The editor's Stop during staging (a remote image download) is held like one during the
        # submit itself, and the prompt then ends without being sent.
        self._submitting.add(session_id)
        try:
            staged = await stage_user_content(content, self._home)
        finally:
            self._submitting.discard(session_id)
        if session_id in self._pending_cancels:
            self._pending_cancels.discard(session_id)
            return None
        return retry_key, uuid.uuid4().hex, staged

    async def _settled_receipt(self, client, held_id, session_id, receipt):
        admission_id = receipt["admission_id"]
        saved = await client.rpc("prompt.receipt", session_id=session_id, admission_id=admission_id,
                                 include_result=True)
        if admission_id in self._terminals or admission_id in self._settling:
            return  # this connection's pump carried the completion itself
        self._settling.add(admission_id)  # claimed before the first await; the pump yields to it
        self._receipted[admission_id] = None
        while len(self._receipted) > 256:
            self._receipted.pop(next(iter(self._receipted)))
        try:
            outcome = {"interrupted": "cancelled"}.get(receipt.get("outcome"), receipt.get("outcome"))
            text = (saved.get("result") or {}).get("final_response") or ""
            await self._finish_reply(session_id, held_id, admission_id, {"text": text, "outcome": outcome})
            async with self._changed:
                self._terminals[admission_id] = {"outcome": outcome}
                self._changed.notify_all()
        finally:
            self._settling.discard(admission_id)

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
        if kind in {"session.info", "session.updated", "session.usage", "session.tools_changed"}:
            await self._project_session_state(kind, sid, editor_id, payload)
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
        if kind in {"reasoning.delta", "todo.updated"}:
            await self._project_side_update(kind, sid, editor_id, payload)
            return
        if kind in {"tool.start", "tool.complete"}:
            await self._project_tool(kind, sid, editor_id, aid, payload)
            return
        if kind == "message.delta":
            text = _text(payload)
            self._streamed[aid] = self._streamed.get(aid, "") + text
            if text and self._conn:
                await self._conn.session_update(session_id=editor_id, update=self._chunk(sid, text))
        elif kind == "message.interim":
            # Commentary the owner could not stream (``already_streamed`` false) reaches viewers
            # only here; streamed commentary is already on screen. Either way it is its own segment.
            self._close_segment(aid)
            text = _text(payload)
            self._end_message(sid)
            if not payload.get("already_streamed") and text.strip():
                self._segments.setdefault(aid, []).append(text)
                if self._conn:
                    await self._conn.session_update(session_id=editor_id, update=self._chunk(sid, text + "\n\n"))
                self._end_message(sid)
        elif kind == "message.complete":
            if aid in self._settling or aid in self._receipted:
                # A retry's terminal receipt claimed this reply; it is shown once, from there.
                self._streamed.pop(aid, None)
                self._segments.pop(aid, None)
                return
            self._settling.add(aid)
            try:
                await self._finish_reply(sid, editor_id, aid, payload)
                async with self._changed:
                    self._terminals[aid] = payload
                    if len(self._terminals) > 256:
                        self._terminals.pop(next(iter(self._terminals)))
                    self._changed.notify_all()
            finally:
                self._settling.discard(aid)

    async def _project_session_state(self, kind, sid, editor_id, payload):
        if kind == "session.usage":
            await self._project_usage(sid, editor_id, payload)
            return
        if kind == "session.tools_changed":
            if self._conn:
                await self._conn.session_update(session_id=editor_id, update=acp.update_agent_message_text(
                    _tools_changed_notice(payload)))
            return
        if kind == "session.updated" and payload.get("operation") == "model" and payload.get("status") == "refused":
            # A deferred pick that no longer resolved at the turn boundary: dropped, named here.
            if self._conn:
                from acp_adapter.gateway_commands import refused_pick_text
                await self._conn.session_update(session_id=editor_id, update=acp.update_agent_message_text(
                    refused_pick_text(payload) + "\n"))
            return
        if (kind == "session.updated" and payload.get("operation") == "model"
                and payload.get("status") == "confirmation_required" and isinstance(payload.get("confirm"), str)):
            # A deferred pick whose target needs consent at the turn boundary: asked like an idle
            # pick (off the event pump: the answer may take as long as the user does).
            task = asyncio.create_task(self._confirm_deferred_pick(sid, editor_id, payload))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
            return
        if kind == "session.info" and sid in self._snapshots and "pending" in payload:
            async with self._changed:
                self._snapshots[sid]["pending"] = payload["pending"]
                self._changed.notify_all()
        await self._project_title(sid, editor_id, payload.get("title"))

    async def _project_usage(self, sid, editor_id, stats):
        """The owner's usage dict: kept for the turn's ``PromptResponse.usage``, shown as ``usage_update``."""
        from acp_adapter.gateway_session_meta import usage_update
        self._usage[sid] = stats
        if self._conn and (update := usage_update(stats)) is not None:
            await self._conn.session_update(session_id=editor_id, update=update)

    async def _project_title(self, sid, editor_id, title):
        """A new title (auto-title's ``session.info``, a rename's ``session.updated``) as main's
        ``session_info_update``; the owner republishes an unchanged one after every turn."""
        if not isinstance(title, str) or not title.strip() or title == self._titles.get(sid):
            return
        self._titles[sid] = title
        if self._conn:
            from acp_adapter.gateway_session_meta import title_update
            await self._conn.session_update(session_id=editor_id, update=title_update(title))

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
                await self._close_card(editor_id, tool_call, options, response)
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
            await self._close_card(self._editor_id(session_id), tool_call, options, response)
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
        # Post-attach notices first: one paused mid-notice would otherwise resume after the
        # teardown and open a fresh gateway connection through ``_client``.
        background = list(self._background)
        for task in background:
            task.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        for task in self._permissions.values():
            task.cancel()
        if self._permissions:
            await asyncio.gather(*self._permissions.values(), return_exceptions=True)
            self._permissions.clear()
        await self._teardown()
