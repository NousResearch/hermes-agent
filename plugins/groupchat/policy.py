"""Shared chat ingress/egress policy. No chat SDK imports or wire parsing.

Opt in using the canonical groupchat configuration. Legacy Matrix flags
are converted once by the migration helper, not read at runtime. An enabled relevance policy never bypasses transport ACLs.
"""
from __future__ import annotations

import asyncio
import copy
import contextvars
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_sending = contextvars.ContextVar("conversation_sending", default=None)


def requires_buffered_delivery(config, platform):
    from .config import validate_settings
    settings = validate_settings(config.get("groupchat"))
    return (settings["enabled"] and platform in settings["platforms"]
            and settings["pingpong_guard"]["enabled"])


class GroupchatAddon:
    def __init__(self, adapter, *, scope="", shared_room_transcript=None,
                 shared_passive_context=None, shared_peer_targeted_ids=None):
        self.adapter = adapter
        self._dispatch = adapter._handle_message_after_conversation
        platform = getattr(adapter, "platform", "")
        self.platform = getattr(platform, "value", platform)
        self.last_inbound = {}
        self.relevance = None
        self.guard_enabled = False
        self.guard_settings = {}
        self.scope = scope
        # Relevance queues and outbound context remain isolated per normalized
        # conversation, but every lane in the same adapter observes one room
        # timeline. This lets an agent understand an elliptical top-level
        # message without merging Matrix thread sessions.
        self._shared_room_transcript = (
            shared_room_transcript
            if shared_room_transcript is not None
            else {}
        )
        self._shared_passive_context = (
            shared_passive_context
            if shared_passive_context is not None
            else {}
        )
        self._shared_peer_targeted_ids = (
            shared_peer_targeted_ids
            if shared_peer_targeted_ids is not None
            else {}
        )
        self._workspace_room_transcripts = {"": self._shared_room_transcript}
        self._workspace_passive_contexts = {"": self._shared_passive_context}
        self._workspace_peer_targeted_ids = {"": self._shared_peer_targeted_ids}
        self._scopes = {}
        self._send_locks = {}
        self._closed = False
        self._load()

    def _load(self):
        import yaml
        config_path = get_hermes_home() / "config.yaml"
        config = {}
        if config_path.exists():
            try:
                config = yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                logger.warning("Conversation policy: invalid profile config; using disabled defaults")
        from .config import validate_settings
        raw = dict(config.get("groupchat") or {})
        extra = getattr(getattr(self.adapter, "config", None), "extra", {})
        if isinstance(extra, dict):
            raw.update(extra.get("groupchat") or {})
        settings = validate_settings(raw)
        self._profile_home = get_hermes_home()
        self.filter_model = settings["filter_model"]
        active = settings["enabled"] and self.platform in settings["platforms"]
        relevance = dict(settings["relevance"])
        relevance["_legacy_env"] = False
        relevance["filter_model"] = self.filter_model
        if self.platform != "matrix":
            base = Path(relevance["context_file"])
            suffix = f".{self.platform}" + (f".{self.scope}" if self.scope else "")
            relevance["context_file"] = str(base.with_name(base.stem + suffix + base.suffix))
        self.guard_settings = settings["pingpong_guard"]
        self.guard_enabled = active and self.guard_settings["enabled"]
        if active and relevance["enabled"]:
            from .relevance import IntelligentReactionGate
            self.relevance = IntelligentReactionGate(
                self.adapter, self.adapter.config, settings=relevance,
                platform=self.platform, dispatch=self._dispatch,
                own_user_id=self._own_user_id,
                room_transcript=self._shared_room_transcript,
                passive_context=self._shared_passive_context,
                peer_targeted_event_ids=self._shared_peer_targeted_ids,
            )

    def bind(self, dispatch):
        self._dispatch = dispatch
        if self.relevance is not None:
            self.relevance._dispatch = dispatch

    @property
    def delays_messages(self):
        return self.relevance is not None

    @property
    def buffers_output(self):
        return self.guard_enabled

    def processing(self, event, phase, session_id, outcome=None):
        source = event.source
        policy = self.for_conversation(source.chat_id, event.metadata, source.thread_id, source.scope_id)
        if policy.relevance is None:
            return
        if phase == "on_processing_start":
            policy.relevance.agent_turn_started(event, session_id)
        elif phase == "on_processing_complete":
            policy.relevance.agent_turn_completed(event, session_id, outcome)

    def typing(self, chat_id):
        # Typing is reported by transports at chat/room level, while inbound
        # messages can be handled by a thread-scoped policy. Notify every live
        # lane so a faster agent claiming a voice note cancels delayed voice
        # delivery in the other lanes as well. Each relevance gate still keys
        # its normal buffer by chat_id, so unrelated rooms and thread queues
        # remain isolated.
        policies = (self, *self._scopes.values())
        seen = set()
        for policy in policies:
            relevance = policy.relevance
            if relevance is None or id(relevance) in seen:
                continue
            seen.add(id(relevance))
            relevance.typing_received(chat_id)

    def _own_user_id(self):
        # Identity is supplied by the adapter hook, never inferred from text.
        callback = getattr(self.adapter, "conversation_user_id", None)
        return callback() if callable(callback) else ""

    def for_conversation(self, chat_id, metadata=None, thread_id=None, scope_id=None):
        # All transports isolate workspace/thread lanes using normalized identity.
        if self.scope or not (self.relevance or self.guard_enabled):
            return self
        metadata = metadata or {}
        normalized = metadata.get("conversation") or {}
        thread = thread_id if thread_id is not None else normalized.get("thread_id", metadata.get("thread_id", metadata.get("slack_thread_ts", "")))
        workspace = scope_id or normalized.get("scope_id") or metadata.get("scope_id") or metadata.get("slack_team_id", "")
        if not thread and not workspace:
            return self
        key = json.dumps([str(chat_id), str(thread or ""), str(workspace or "")])
        if key not in self._scopes:
            digest = hashlib.sha256(key.encode()).hexdigest()[:20]
            room_transcript = self._workspace_room_transcripts.setdefault(
                str(workspace or ""), {}
            )
            passive_context = self._workspace_passive_contexts.setdefault(
                str(workspace or ""), {}
            )
            peer_targeted_ids = self._workspace_peer_targeted_ids.setdefault(
                str(workspace or ""), {}
            )
            self._scopes[key] = GroupchatAddon(
                self.adapter,
                scope=digest,
                shared_room_transcript=room_transcript,
                shared_passive_context=passive_context,
                shared_peer_targeted_ids=peer_targeted_ids,
            )
            self._scopes[key].bind(self._dispatch)
        return self._scopes[key]

    async def close(self):
        self._closed = True
        for policy in self._scopes.values():
            await policy.close()
        if self.relevance is not None:
            await self.relevance.close()

    async def receive(self, event, *, is_mentioned=None):
        from gateway.platforms.base import MessageType
        source = event.source
        if self._closed:
            return
        if not (self.relevance or self.guard_enabled):
            return await self._dispatch(event)
        if source is not None:
            scoped = self.for_conversation(source.chat_id, event.metadata, source.thread_id, source.scope_id)
            if scoped is not self:
                return await scoped.receive(event, is_mentioned=is_mentioned)
        if source is not None:
            # Do not send an unauthorized message to the scorer or store it.
            authorized = self.adapter._is_sender_authorized(
                source.user_id or event.user_id, source.chat_type, source.chat_id,
                is_bot=getattr(source, "is_bot", False),
                thread_id=source.thread_id,
            )
            if authorized is False or (authorized is not True and self.platform != "matrix"):
                # Let the existing gateway own rejection/DM feedback.
                return await self._dispatch(event)
            self.last_inbound[source.chat_id] = event.text
            if len(self.last_inbound) > 512:
                self.last_inbound.pop(next(iter(self.last_inbound)))
        # Control commands and internal events must never wait in the scorer.
        if (self.relevance is None or event.internal or event.is_command()
                or event.message_type not in (MessageType.TEXT, MessageType.AUDIO, MessageType.VOICE)):
            return await self._dispatch(event)
        if is_mentioned is None:
            is_mentioned = bool((event.metadata or {}).get("conversation_mentioned"))
        is_mentioned = is_mentioned or event.reply_to_is_own_message
        if event.user_id is None and source is not None:
            import dataclasses
            event = dataclasses.replace(event, user_id=source.user_id, user_name=source.user_name)
        await self.relevance.process(event, is_mentioned)

    def _audit_outbound(self, chat_id, **decision):
        from .audit import record_outbound
        record_outbound(self._profile_home, self.platform, chat_id, self.scope, decision)

    def output_suppressed(self, event, content, reason):
        source = event.source
        scoped = self.for_conversation(
            source.chat_id, event.metadata, source.thread_id, source.scope_id
        )
        if scoped is not self:
            return scoped.output_suppressed(event, content, reason)
        if self.guard_enabled:
            self._audit_outbound(
                source.chat_id, decision="suppress", reason_code=reason
            )

    async def filter_output(self, chat_id, content):
        if self.relevance is not None:
            result = self.relevance.outbound_filter(content, chat_id)
            if result is True:
                self._audit_outbound(chat_id, decision="suppress", reason_code=self.relevance._last_outbound_reason)
                return None
            if isinstance(result, str):
                content = result
        if not self.guard_enabled:
            return content
        process = None
        try:
            from .models import credential
            environment = dict(os.environ)
            for name in ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "MISTRAL_API_KEY"):
                environment.pop(name, None)
                value = credential(name)
                if value:
                    environment[name] = value
            environment["GROUPCHAT_CREDENTIALS_RESOLVED"] = "1"
            environment["HERMES_HOME"] = str(self._profile_home)
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(Path(__file__).with_name("pingpong_guard.py")),
                "--to", f"{self.platform}:{chat_id}", "--payload",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=environment,
            )
            payload = {"text": content, "context": self.last_inbound.get(chat_id, ""),
                       "filter_model": self.filter_model,
                       "min_chars": self.guard_settings.get("min_chars", 60),
                       "silence_patterns": self.guard_settings.get("silence_patterns")}
            output, _ = await asyncio.wait_for(process.communicate(json.dumps(payload).encode()), timeout=25)
            try:
                decision = json.loads(output)
                if not isinstance(decision, dict):
                    raise ValueError("Invalid decision record")
            except (ValueError, TypeError):
                decision = {"reason_code": "worker_suppress" if process.returncode == 10 else "worker_failure_fail_open"}
            decision["decision"] = "suppress" if process.returncode == 10 else "send"
            if process.returncode not in (0, 10):
                decision.update(reason_code="worker_failure_fail_open", fail_open=True)
            self._audit_outbound(chat_id, **decision)
            if process.returncode == 10:
                logger.info("Conversation guard: suppressed reply (%s)", self.platform)
                return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._audit_outbound(chat_id, decision="send", reason_code="worker_timeout_fail_open" if isinstance(exc, asyncio.TimeoutError) else "worker_failure_fail_open", fail_open=True)
            logger.warning("Conversation guard failed open (%s)", type(exc).__name__)
        finally:
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        return content


    async def send(self, chat_id, content, reply_to=None, metadata=None, *, send):
        from gateway.platforms.base import SendResult
        if _sending.get() is self:
            return await send(chat_id, content, reply_to, metadata)
        policy = self.for_conversation(chat_id, metadata)
        if not (policy.relevance or policy.guard_enabled):
            return await send(chat_id, content, reply_to, metadata)
        lock = policy._send_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            gate = policy.relevance
            attributes = ("_recent_outbound_hashes", "_recent_transcriptions", "_pending_voice_transcriptions")
            before = {name: copy.deepcopy(getattr(gate, name).get(chat_id)) for name in attributes} if gate else {}
            committed = False
            token = _sending.set(self)
            try:
                content = await policy.filter_output(chat_id, content)
                if content is None:
                    return SendResult(success=True)
                result = await send(chat_id, content, reply_to, metadata)
                if result.success and result.message_id:
                    committed = True
                    if gate:
                        gate.record_own_message(chat_id, result.message_id, content)
                return result
            finally:
                _sending.reset(token)
                if gate and not committed:
                    for name, value in before.items():
                        target = getattr(gate, name)
                        if value is None:
                            target.pop(chat_id, None)
                        else:
                            target[chat_id] = value
