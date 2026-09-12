"""ACP projection of the profile gateway; never constructs an execution owner."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import uuid

import acp
from acp.schema import (
    AgentCapabilities, Implementation, InitializeResponse, LoadSessionResponse,
    NewSessionResponse, PromptCapabilities, PromptResponse, ResumeSessionResponse,
    SessionCapabilities, SessionResumeCapabilities,
)

from hermes_cli.gateway_client import GatewayClientError, connect_gateway
from hermes_constants import get_hermes_home
from acp_adapter.session import _translate_acp_cwd, _normalize_cwd_for_compare


def _stage_user_content(content):
    """Shared-converter output -> ``(text, attachments)`` for ``prompt.submit``.

    Text-only prompts stay a plain string. Media parts (image ``data:`` URLs from
    image blocks, image resource links and embedded blobs) are staged as bytes in
    the profile image cache; the authority commits them at admission. Remote
    image URLs cannot be staged and are kept as text so the model still sees them."""
    if isinstance(content, str):
        return content, []
    import base64
    from gateway.platforms.base import cache_image_from_bytes
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
        mime = header[len('data:'):].split(';', 1)[0] or 'image/png'
        try:
            path = cache_image_from_bytes(base64.b64decode(data), '.' + mime.split('/', 1)[1])
        except ValueError as exc:
            raise GatewayClientError('acp_content_invalid') from exc
        attachments.append({'path': path, 'mime': mime})
    return "\n".join(texts), attachments


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
        self._changed = asyncio.Condition()
        self._failure = None
        self._permissions = {}
        self._admissions = {}
        self._tool_args = {}
        from hermes_cli.gateway_mutations import PreparedMutations
        self._mutations = PreparedMutations()

    def on_connect(self, conn):
        self._conn = conn

    async def initialize(self, **kwargs):
        from hermes_cli import __version__
        from acp_adapter.auth import build_auth_methods
        return InitializeResponse(protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=__version__),
            agent_capabilities=AgentCapabilities(load_session=True,
                prompt_capabilities=PromptCapabilities(image=True),
                session_capabilities=SessionCapabilities(resume=SessionResumeCapabilities())),
            auth_methods=build_auth_methods())

    async def authenticate(self, method_id, **kwargs):
        from acp_adapter.server import HermesACPAgent
        return await HermesACPAgent.authenticate(self, method_id, **kwargs)

    async def _client(self):
        async with self._connect_lock:
            if self._gateway is None:
                connection = connect_gateway()
                gateway = await connection.__aenter__()
                try:
                    descriptor = await gateway.rpc("runtime.describe")
                    if descriptor.get("profile_id") != str(self._home):
                        raise GatewayClientError("profile_mismatch")
                except BaseException:
                    await connection.__aexit__(None, None, None)
                    raise
                self._connection, self._gateway = connection, gateway
                self._event_task = asyncio.create_task(self._events())
            if self._failure:
                raise self._failure
            return self._gateway

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
        return NewSessionResponse(session_id=snapshot["session_id"])

    async def _resume(self, cwd, session_id, mcp_servers):
        client = await self._client()
        info = await client.rpc("session.info", session_id=session_id)
        if "cwd" in info and _normalize_cwd_for_compare(info["cwd"]) != _normalize_cwd_for_compare(_translate_acp_cwd(cwd)):
            raise GatewayClientError("cwd_policy_conflict")
        if "cwd" not in info and self._conn:
            await self._conn.session_update(session_id=session_id, update=acp.update_agent_message_text(
                "Attached to the gateway's existing session policy; editor cwd is not applied.\n"))
        resume_params = {}
        if mcp_servers:
            descriptor = await client.rpc("runtime.describe")
            if "acp-session-mcp-v1" not in descriptor.get("capabilities", []):
                raise GatewayClientError("acp_mcp_policy_unavailable")
            resume_params['editor'] = {'mcp_servers': [s.model_dump(by_alias=True) for s in mcp_servers]}
        snapshot = await client.rpc("session.resume", session_id=session_id, **resume_params)
        self._snapshots[session_id] = snapshot
        from acp_adapter.server import _history_replay_updates
        if self._conn:
            for update in _history_replay_updates(snapshot["messages"]):
                await self._conn.session_update(session_id=session_id, update=update)
        for pending in snapshot.get("prompts", []):
            self._permission(session_id, pending)
        return snapshot

    async def load_session(self, cwd, session_id, mcp_servers=None, **kwargs):
        await self._resume(cwd, session_id, mcp_servers)
        return LoadSessionResponse()

    async def resume_session(self, cwd, session_id, mcp_servers=None, **kwargs):
        await self._resume(cwd, session_id, mcp_servers)
        return ResumeSessionResponse()

    async def cancel(self, session_id, **kwargs):
        if session_id not in self._snapshots:
            raise GatewayClientError("not_found")
        admission_id = self._admissions.get(session_id)
        if admission_id is None:
            return
        client = await self._client()
        # Cancel our own admission; another surface's running turn is not ours to
        # interrupt. Only when our admission is the one executing do we interrupt.
        try:
            await client.rpc("prompt.cancel", session_id=session_id, admission_id=admission_id)
        except GatewayClientError as exc:
            if str(exc) != "stale_generation":
                raise
            receipt = await client.rpc("prompt.receipt", session_id=session_id, admission_id=admission_id)
            if receipt["status"] == "started":
                await client.rpc("session.interrupt", session_id=session_id,
                                 execution_generation=receipt["execution_generation"])

    async def fork_session(self, cwd, session_id, mcp_servers=None, **kwargs):
        from acp.schema import ForkSessionResponse
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
        client = await self._client()
        payload = {'model': model_id}
        await self._mutations.apply(client, session_id, 'model', payload)
        self._snapshots[session_id] = await client.rpc('session.resume', session_id=session_id)
        self._mutations.acknowledge(session_id, 'model', payload)
        return SetSessionModelResponse()

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
        if session_id not in self._snapshots:
            raise GatewayClientError("not_found")
        from acp_adapter.content import _content_blocks_to_openai_user_content
        text, attachments = _stage_user_content(_content_blocks_to_openai_user_content(prompt))
        if text.lstrip().startswith('/') and not attachments:
            from hermes_cli.gateway_mutations import slash_mutation
            parts = text.strip().split(None, 1)
            operation, payload = slash_mutation(parts[0], parts[1] if len(parts) > 1 else '')
            if operation == 'branch':
                raise GatewayClientError('use_acp_fork_session')
            client = await self._client()
            await self._mutations.apply(client, session_id, operation, payload)
            self._snapshots[session_id] = await client.rpc('session.resume', session_id=session_id)
            self._mutations.acknowledge(session_id, operation, payload)
            return PromptResponse(stop_reason='end_turn')
        client = await self._client()
        submit = {'text': text}
        if attachments:
            submit['attachments'] = attachments
        receipt = await client.rpc("prompt.submit", session_id=session_id,
                                   input_id=uuid.uuid4().hex, **submit)
        admission_id = receipt["admission_id"]
        self._admissions[session_id] = admission_id
        try:
            async with self._changed:
                await self._changed.wait_for(lambda: admission_id in self._terminals or self._failure is not None)
                if self._failure:
                    raise self._failure
                terminal = self._terminals.pop(admission_id)
        finally:
            self._admissions.pop(session_id, None)
        outcome = terminal.get("outcome")
        if outcome == "failed":
            raise GatewayClientError("admitted_turn_failed")
        return PromptResponse(stop_reason="cancelled" if outcome == "cancelled" else "end_turn")

    async def _events(self):
        try:
            while True:
                frame = await self._gateway.events.get()
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
            async with self._changed:
                self._failure = exc
                self._changed.notify_all()

    async def _project(self, event):
        sid, kind, payload = event["session_id"], event.get("type"), event.get("payload", {})
        aid = event.get("admission_id")
        if kind == "approval.request":
            self._permission(sid, payload)
            return
        if kind == "approval.settled":
            task = self._permissions.pop((sid, payload["prompt_id"], payload["execution_generation"]), None)
            if task:
                task.cancel()
            return
        # In-process turns publish ``tool_name``; managed workers publish ``name``.
        tool_name = payload.get("tool_name") or payload.get("name") or "tool"
        if kind == "tool.start":
            from acp_adapter.tools import build_tool_start, coerce_tool_args
            args = coerce_tool_args(payload.get("args"))
            self._tool_args[(sid, payload["tool_call_id"])] = (tool_name, args)
            if self._conn:
                await self._conn.session_update(session_id=sid,
                    update=build_tool_start(payload["tool_call_id"], tool_name, args))
            return
        if kind == "tool.complete":
            from acp_adapter.tools import build_tool_complete
            name, args = self._tool_args.pop((sid, payload["tool_call_id"]), (tool_name, {}))
            result = payload.get("result")
            if self._conn:
                await self._conn.session_update(session_id=sid, update=build_tool_complete(
                    payload["tool_call_id"], name, result=result if isinstance(result, str) else None,
                    function_args=args))
            return
        if kind == "message.delta":
            text = payload.get("text", "")
            self._streamed[aid] = self._streamed.get(aid, "") + text
            if text and self._conn:
                await self._conn.session_update(session_id=sid, update=acp.update_agent_message_text(text))
        elif kind == "message.complete":
            from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

            text = payload.get("text", "")
            # Local interrupt status is metadata; ACP carries it in stop_reason.
            if payload.get("outcome") == "cancelled" and text.startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX):
                text = ""
            prefix = self._streamed.pop(aid, "")
            remainder = text[len(prefix):] if text.startswith(prefix) else text
            if remainder and self._conn:
                await self._conn.session_update(session_id=sid, update=acp.update_agent_message_text(remainder))
            async with self._changed:
                self._terminals[aid] = payload
                if len(self._terminals) > 256:
                    self._terminals.pop(next(iter(self._terminals)))
                self._changed.notify_all()

    def _permission(self, session_id, prompt):
        if prompt.get("kind") != "approval" or self._conn is None:
            return
        key = (session_id, prompt["prompt_id"], prompt["execution_generation"])
        if key not in self._permissions:
            self._permissions[key] = asyncio.create_task(self._answer_permission(session_id, prompt))

    async def _answer_permission(self, session_id, prompt):
        import logging
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
            response = await self._conn.request_permission(session_id=session_id,
                tool_call=tool_call, options=options)
            # Transport loss/cancel is not a denial: the canonical waiter belongs
            # to the execution and may still be answered by another viewer.
            if not isinstance(response.outcome, AllowedOutcome):
                return
            if response.outcome.option_id not in {option.option_id for option in options}:
                return
            await self._gateway.rpc("approval.respond", session_id=session_id,
                prompt_id=prompt["prompt_id"], execution_generation=prompt["execution_generation"],
                choice=_OPTION_ID_TO_HERMES[response.outcome.option_id])
        except Exception:
            logging.getLogger(__name__).info("ACP permission viewer detached or control expired")

    async def aclose(self):
        for task in self._permissions.values():
            task.cancel()
        if self._permissions:
            await asyncio.gather(*self._permissions.values(), return_exceptions=True)
            self._permissions.clear()
        if self._event_task:
            self._event_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._event_task
        if self._connection:
            await self._connection.__aexit__(None, None, None)
            self._connection = None
        self._gateway = None
