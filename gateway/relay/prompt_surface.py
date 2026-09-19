"""Relay interactive prompt surface — the prompt lifecycle half of ``gateway/relay/adapter.py``:
minting, sending, consuming and resolving the exec-approval / slash-confirm / clarify prompts, plus
the fire-and-forget in-channel acks they ride. Split out of that module and bound onto
``RelayAdapter`` through ``RelayPromptMixin`` so ``self._mint_prompt`` /
``self._consume_prompt_response`` keep resolving via the MRO. The origin module re-exports every
name defined here."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import Any, Dict, Optional

from gateway.platforms.base import ExecApprovalPrompt, SendResult
from gateway.platforms.base_exec_approval import EA_HEADER_TEXT
from gateway.relay.egress import decline_error

logger = logging.getLogger("gateway.relay.adapter")  # log-record parity with gateway/relay/adapter.py

# Already-answered prompt ids to remember so a duplicate answer (double tap or
# connector redelivery) reads as a repeat, not a stale prompt.
_RESOLVED_PROMPT_MEMORY = 256

# Prompt option id -> in-channel ack label (the option set doubles as the choice allowlist).
_EXEC_APPROVAL_LABELS = {
    "once": "✅ Approved once",
    "session": "✅ Approved for session",
    "always": "✅ Approved permanently",
    "deny": "❌ Denied",
}
_SLASH_CONFIRM_LABELS = {"once": "✅ Approved once", "always": "🔒 Always approve", "cancel": "❌ Cancelled"}


class RelayPromptMixin:
    """Prompt-lifecycle surface of ``RelayAdapter`` (exec approval, slash confirm, clarify)."""

    # ── Phase 3 interactive: prompt + react ──────────────────────────────

    def _mint_prompt(
        self, kind: str, state: Dict[str, Any], timeout_s: float = 3600.0
    ) -> str:
        """Register a pending prompt and return its id (``<owner nonce>.<8 hex>``).
        Expiry is enforced gateway-side on consumption (_pop_prompt); the wire's
        timeout_s is advisory. The nonce marks the minting process so a sibling
        gateway receiving the fanned-out answer stays quiet. Both segments use the
        connector codec's alphabet ([A-Za-z0-9_.-], <=32)."""
        prompt_id = f"{self._prompt_owner_nonce}.{secrets.token_hex(4)}"
        self._pending_prompts[prompt_id] = {**state, "kind": kind, "expires_at": time.time() + timeout_s}
        # Opportunistic sweep so abandoned prompts can't accumulate.
        now = time.time()
        for stale in [k for k, v in self._pending_prompts.items() if v.get("expires_at", 0) < now]:
            self._pending_prompts.pop(stale, None)
        return prompt_id

    def _minted_here(self, prompt_id: str) -> bool:
        """True when this process minted ``prompt_id``. Ids without a ``.`` segment
        predate the owner nonce (in-flight across an in-place upgrade) and are ours."""
        head, sep, _ = str(prompt_id).partition(".")
        return head == self._prompt_owner_nonce if sep else True

    def _pop_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Consume a pending prompt: one answer wins, expired entries miss."""
        state = self._pending_prompts.pop(str(prompt_id), None)
        if not state or state.get("expires_at", 0) < time.time():
            return None
        return state

    def _note_prompt_resolved(self, prompt_id: str) -> None:
        """Remember that this process answered ``prompt_id`` (bounded FIFO: a repeat is
        only interesting while a redelivery/double tap can arrive)."""
        self._resolved_prompts[str(prompt_id)] = time.time()
        while len(self._resolved_prompts) > _RESOLVED_PROMPT_MEMORY:
            self._resolved_prompts.popitem(last=False)

    async def _send_prompt(
        self,
        chat_id: str,
        *,
        prompt_kind: str,
        text: str,
        prompt_id: str,
        options: list,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[int] = None,
    ) -> Optional[SendResult]:
        """Egress one `prompt` op; None when the lane is unavailable (the caller falls
        back to its numbered-text base behaviour). Prompt metadata is forwarded
        VERBATIM: the threading mode is decided in exactly one place — run.py's
        _resolve_progress_thread_id (flat mode suppresses the synthetic self-anchor
        there; thread mode stamps the turn's thread)."""
        action: Dict[str, Any] = {
            "op": "prompt",
            "chat_id": chat_id,
            "content": text,
            "prompt_kind": prompt_kind,
            "prompt_id": prompt_id,
            "options": options,
            "reply_to": self._resolve_reply_to_for_send(chat_id, reply_to, metadata),
            "metadata": self._with_scope(chat_id, metadata),
        }
        if timeout_s is not None:
            action["timeout_s"] = int(timeout_s)
        result = await self._gated_op(chat_id, action, surface_declines=True)
        if result is None:
            return None
        if not result.get("success"):
            # An AUTHORIZATION refusal, surfaced by _gated_op. Returning None
            # would hand the caller back to its fallback — a DIFFERENT op
            # against the SAME destination the connector just refused. Report
            # a failed lane instead, carrying the decline verbatim.
            return SendResult(
                success=False, error=decline_error(result), raw_response=result
            )
        return SendResult(success=True, message_id=result.get("message_id"), raw_response=result)

    async def _mint_and_send_prompt(
        self,
        kind: str,
        state: Dict[str, Any],
        chat_id: str,
        *,
        prompt_kind: str,
        text: str,
        options: list,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[SendResult]:
        """Register + egress a prompt; unregisters and returns None when the lane is unavailable."""
        prompt_id = self._mint_prompt(kind, {**state, "chat_id": str(chat_id)})
        result = await self._send_prompt(
            chat_id, prompt_kind=prompt_kind, text=text, prompt_id=prompt_id, options=options,
            metadata=metadata,
        )
        if result is None or not getattr(result, "success", False):
            # P5(b): a DECLINE now returns a failed SendResult rather than
            # None, and the registration must come down on that path too. A
            # prompt card the connector refused never rendered, so leaving it
            # pending lets the user's next unrelated message be captured as the
            # answer to a prompt they never saw.
            self._pending_prompts.pop(prompt_id, None)
        return result

    _PROMPT_UNAVAILABLE = SendResult(success=False, error="relay prompt op unavailable")

    _EA_HEADER = f"⚠️ **{EA_HEADER_TEXT}**\n\n"
    _EA_SMART_DENY_LINE = "\n\n**Smart DENY:** owner override applies to this one operation only."
    _EA_CMD_BUDGET = 1500

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        """Native-button exec approval over the relay (the press resolves via
        tools.approval.resolve_gateway_approval). When the lane is unavailable the send FAILS
        (success=False) so run.py's button→text fallback runs."""
        options = [{"id": choice, "label": label, **({"style": style} if style else {})}
                   for label, choice, style in prompt.actions]
        result = await self._mint_and_send_prompt(
            "exec_approval", {"session_key": prompt.session_key}, prompt.chat_id, prompt_kind="approval",
            text=prompt.text, options=options, metadata=prompt.metadata,
        )
        return result if result is not None else self._PROMPT_UNAVAILABLE

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Three-button slash-command confirmation over the relay (resolves via
        tools.slash_confirm.resolve; success=False falls back to text-intercept)."""
        options = [
            {"id": "once", "label": "Approve Once", "style": "primary"},
            {"id": "always", "label": "Always Approve"},
            {"id": "cancel", "label": "Cancel", "style": "danger"},
        ]
        result = await self._mint_and_send_prompt(
            "slash_confirm", {"session_key": session_key, "confirm_id": confirm_id}, chat_id,
            prompt_kind="approval", text=f"**{title}**\n\n{message}" if title else message,
            options=options, metadata=metadata,
        )
        return result if result is not None else self._PROMPT_UNAVAILABLE

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Native-button clarify over the relay. A press resolves with the CHOICE TEXT
        (never the option id); "Other" flips to text-capture. Option ids are
        positional (c0..cN / other) — choice text is arbitrary UTF-8 and would blow
        the 64-byte callback budget. Open-ended clarifies and unavailable lanes fall
        back to base."""
        if choices and self.descriptor.supports_op("prompt"):
            options = [{"id": f"c{i}", "label": str(choice)[:75]} for i, choice in enumerate(choices)]
            options.append({"id": "other", "label": "✏️ Other (type your answer)"})
            result = await self._mint_and_send_prompt(
                "clarify",
                {
                    "session_key": session_key,
                    "clarify_id": clarify_id,
                    "choices": [str(c) for c in choices],
                },
                chat_id,
                prompt_kind="clarify",
                text=f"❓ {question}",
                options=options,
                metadata=metadata,
            )
            if result is not None:
                return result
        return await super().send_clarify(
            chat_id, question, choices, clarify_id, session_key, metadata=metadata
        )

    async def _consume_prompt_response(self, event) -> bool:
        """Route an inbound prompt_response to its waiting primitive; True when the
        event was a prompt answer (consumed — never dispatched as chat). EVERY prompt
        answer is consumed, whoever owns it: a sibling's prompt (the connector fans the
        press to every gateway of the tenant; falling through produced a wall of
        "Unknown command"), a repeat answer (first one won), or our own expired/unknown
        prompt (answered with a short expiry notice — option ids are not commands)."""
        pr = getattr(event, "prompt_response", None)
        if not isinstance(pr, dict):
            return False
        prompt_id = str(pr.get("prompt_id") or "")
        option_id = str(pr.get("option_id") or "")
        if not prompt_id or not option_id:
            return False
        if not self._minted_here(prompt_id):
            logger.debug(
                "relay prompt_response %s (option=%s) belongs to another gateway instance — ignoring",
                prompt_id,
                option_id,
            )
            return True
        if prompt_id in self._resolved_prompts:
            logger.debug(
                "relay prompt_response %s (option=%s) already resolved — ignoring repeat", prompt_id, option_id
            )
            return True
        state = self._pop_prompt(prompt_id)
        if state is None:
            logger.info("relay prompt_response for unknown/expired prompt %s (option=%s)", prompt_id, option_id)
            await self._notify_prompt_expired(event)
            return True
        self._note_prompt_resolved(prompt_id)

        kind = state.get("kind")
        chat_id = str(state.get("chat_id") or getattr(event.source, "chat_id", ""))
        handler = _PROMPT_RESOLVERS.get(kind)
        try:
            if handler is None:
                logger.warning("relay prompt_response with unknown kind %r", kind)
            else:
                # Acks are fire-and-forget: we are ON the read loop here (see
                # _send_lifecycle_ack) and awaiting a send would self-deadlock.
                await handler(self, state, option_id, chat_id, self._prompt_reply_metadata(event))
        except Exception:  # noqa: BLE001 - a resolver failure must not kill the reader
            logger.warning("relay prompt_response resolution failed", exc_info=True)
        return True

    async def _resolve_exec_approval(self, state, option_id, chat_id, ack_meta) -> None:
        from tools.approval import resolve_gateway_approval

        choice = option_id if option_id in _EXEC_APPROVAL_LABELS else "deny"
        count = resolve_gateway_approval(str(state.get("session_key") or ""), choice)
        label = _EXEC_APPROVAL_LABELS[choice] if count else "⌛ Approval expired — no command was waiting."
        # In-channel ack preserves the audit trail the native edit gives (the
        # connector's prompt message can't be edited cross-platform yet).
        self._send_lifecycle_ack(chat_id, label, ack_meta)
        if count:
            self.resume_typing_for_chat(chat_id)

    async def _resolve_slash_confirm(self, state, option_id, chat_id, ack_meta) -> None:
        from tools import slash_confirm as slash_confirm_mod

        choice = option_id if option_id in _SLASH_CONFIRM_LABELS else "cancel"
        result_text = await slash_confirm_mod.resolve(
            str(state.get("session_key") or ""), str(state.get("confirm_id") or ""), choice
        )
        self._send_lifecycle_ack(chat_id, _SLASH_CONFIRM_LABELS[choice], ack_meta)
        if result_text:
            self._send_lifecycle_ack(chat_id, str(result_text), ack_meta)

    async def _resolve_clarify(self, state, option_id, chat_id, ack_meta) -> None:
        from tools.clarify_gateway import mark_awaiting_text, resolve_gateway_clarify

        clarify_id = str(state.get("clarify_id") or "")
        if option_id == "other":
            mark_awaiting_text(clarify_id)
            self._send_lifecycle_ack(chat_id, "✏️ Type your answer:", ack_meta)
            return
        choices = state.get("choices") or []
        try:
            idx = int(option_id[1:]) if option_id.startswith("c") else -1
        except ValueError:
            idx = -1
        if 0 <= idx < len(choices):
            resolve_gateway_clarify(clarify_id, str(choices[idx]))
            self._send_lifecycle_ack(chat_id, f"✅ {choices[idx]}", ack_meta)
        else:
            # Unmappable option: flip to text capture (never dead-end a clarify).
            mark_awaiting_text(clarify_id)

    def _send_lifecycle_ack(self, chat_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Fire-and-forget a prompt-lifecycle ack from read-loop context.
        _consume_prompt_response executes ON the transport read loop; an ``await
        self.send(...)`` there is a SELF-DEADLOCK (send() blocks on an outbound_result
        future only the read loop can resolve) — every button tap wedged the transport
        for the full outbound timeout. Acks are cosmetic, so they ride a background
        task; failures log at debug. The task ref is retained (asyncio holds tasks weakly)."""

        async def _ack() -> None:
            try:
                await self.send(chat_id, text, metadata=metadata)
            except Exception:  # noqa: BLE001 - ack is best-effort
                logger.debug("relay lifecycle ack failed", exc_info=True)

        task = asyncio.create_task(_ack(), name="relay-lifecycle-ack")
        self._lifecycle_ack_tasks.add(task)
        task.add_done_callback(self._lifecycle_ack_tasks.discard)

    async def _notify_prompt_expired(self, event) -> None:
        """Tell the presser their prompt is no longer waiting (owning gateway only, best-effort)."""
        chat_id = str(getattr(event.source, "chat_id", "") or "")
        if not chat_id:
            return
        self._send_lifecycle_ack(
            chat_id,
            "⌛ That prompt is no longer waiting for an answer. "
            "Send your reply as a normal message.",
            self._prompt_reply_metadata(event),
        )

    def _prompt_reply_metadata(self, event) -> Dict[str, Any]:
        """Thread metadata so prompt acks land where the prompt lives. Marked INTERIM:
        acks fire while the approval turn's OWN draft stream is open and carry only
        placement metadata, so send()'s single-open-stream fallback sealed the live
        draft with the ack text (frozen stream + duplicate final on every approval turn)."""
        meta: Dict[str, Any] = {"_interim_send": True}
        thread_id = getattr(event.source, "thread_id", None)
        if thread_id:
            meta["thread_id"] = str(thread_id)
        return meta


# prompt kind -> resolver (order-independent: kinds are distinct keys).
_PROMPT_RESOLVERS = {
    "exec_approval": RelayPromptMixin._resolve_exec_approval,
    "slash_confirm": RelayPromptMixin._resolve_slash_confirm,
    "clarify": RelayPromptMixin._resolve_clarify,
}
