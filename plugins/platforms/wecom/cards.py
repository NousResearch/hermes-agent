"""WeCom template-card interactive mixin: exec-approval and model-picker buttons over
``msgtype: template_card`` (DM only by decision — group chats fall back to plain text).

Two interactive surfaces, one event stream:

* **Exec approval** — ``send_exec_approval`` renders a ``button_interaction`` card whose
  buttons (once / session / always / deny) resolve via ``tools.approval.resolve_gateway_approval``
  when the user taps one. Taps arrive as ``aibot_event_callback`` frames with
  ``event.eventtype == "template_card_event"``; the tap is authorized against the DM
  allowlist and the card is updated (``aibot_respond_update_msg``) within WeCom's 5-second
  reply window.
* **Model picker** — ``/model`` without args calls back into ``send_model_picker`` after
  ``_send_model_picker`` (slash_commands_model.py) probes for the adapter method; up to 6
  candidate models render as buttons, a tap invokes ``on_model_selected(chat_id, model_id,
  provider_slug)``.

Card layout follows WeCom's button_interaction schema:
``{"card_type": "button_interaction", "main_title": {...}, "sub_title_text": "...",
"button_list": [{"text": ..., "key": ...}], "task_id": ...}`` — ``button_list`` ≤ 6,
``task_id`` unique per card and echoed back in the tap event.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from gateway.platforms.base import SendResult

from plugins.platforms.wecom.media import APP_CMD_SEND

logger = logging.getLogger("plugins.platforms.wecom.adapter")

APP_CMD_RESPOND_UPDATE = "aibot_respond_update_msg"

# WeCom caps button_list at 6; keep per-chat card state (task_id → session) bounded.
CARD_BUTTON_MAX = 6
CARD_STATE_MAX = 500

# Approval tap labels (mirrors gateway/platforms/base.py _EA_ACTION_LABELS semantics).
_APPROVAL_TAP_LABELS = {
    "once": "✅ 已允许一次",
    "session": "✅ 本次会话已允许",
    "always": "✅ 已永久允许",
    "deny": "❌ 已拒绝",
}


class WeComCardMixin:
    """Template-card interactivity for WeComAdapter (DM-only; groups keep text flows)."""

    # ── state (called from adapter.__init__ via _init_card_state) ──────────────────────

    def _init_card_state(self) -> None:
        """Per-chat interactive-card registries: task_id → state dict."""
        self._approval_state: Dict[str, Dict[str, Any]] = {}  # task_id → {session_key, chat_id, desc}
        self._model_picker_state: Dict[str, Dict[str, Any]] = {}  # task_id → picker state

    @staticmethod
    def _new_card_task_id(prefix: str) -> str:
        """Uniquely identify one interactive card; echoed back in the tap event."""
        return f"{prefix}-{uuid.uuid4().hex[:20]}"

    def _remember_card_state(self, registry: Dict[str, Dict[str, Any]], task_id: str, state: Dict[str, Any]) -> None:
        """Register a card and trim the register to CARD_STATE_MAX (oldest dropped)."""
        registry[task_id] = state
        while len(registry) > CARD_STATE_MAX:
            oldest = next(iter(registry))
            del registry[oldest]

    # ── card builders ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _template_card_body(card: Dict[str, Any]) -> Dict[str, Any]:
        return {"msgtype": "template_card", "template_card": card}

    def _button_interaction_card(
        self, *, title: str, desc: str, sub_title: str, buttons: List[Dict[str, str]], task_id: str,
    ) -> Dict[str, Any]:
        """button_interaction card: main_title (title + desc), sub_title_text, ≤6 buttons."""
        return {
            "card_type": "button_interaction",
            "main_title": {"title": (title or "")[:64], "desc": (desc or "")[:128]},
            "sub_title_text": (sub_title or "")[:112],
            "button_list": [{"text": str(b["text"])[:10], "key": str(b["key"])[:1024]} for b in buttons[:CARD_BUTTON_MAX]],
            "task_id": task_id,
        }

    @staticmethod
    def _text_notice_card(*, title: str, desc: str, task_id: str) -> Dict[str, Any]:
        """Replace an interactive card after a tap: text_notice has no buttons."""
        return {
            "card_type": "text_notice",
            "main_title": {"title": (title or "")[:64], "desc": (desc or "")[:128]},
            "task_id": task_id,  # must match the tapped card's task_id for an update
        }

    # ── outbound ───────────────────────────────────────────────────────────────────────

    async def _send_card(self, chat_id: str, card: Dict[str, Any], *, reply_req_id: Optional[str] = None,
                         is_control: bool = False) -> SendResult:
        """Send a template card through the per-chat queue: passive reply when a req_id is
        usable, else proactive ``aibot_send_msg`` (chat_type=1 → single chat, DM-only)."""
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")
        body = self._template_card_body(card)
        return await self._enqueue_chat_send(
            chat_id, lambda: self._send_card_inner(chat_id, body, reply_req_id), is_control=is_control)

    async def _send_card_inner(self, chat_id: str, body: Dict[str, Any],
                               reply_req_id: Optional[str]) -> SendResult:
        try:
            if reply_req_id:
                response = await self._send_reply_request(reply_req_id, body)
            else:
                response = await self._send_request(APP_CMD_SEND, {"chatid": chat_id, "chat_type": 1, **body})
            self._raise_for_wecom_error(response, "send template card")
        except asyncio.TimeoutError:
            return SendResult(success=False, error="Timeout sending card to WeCom")
        except Exception as exc:
            logger.error("[%s] Card send failed: %s", self.name, exc)
            return SendResult(success=False, error=str(exc))
        if error := self._response_error(response):
            return SendResult(success=False, error=error)
        return SendResult(success=True, message_id=self._payload_req_id(response) or uuid.uuid4().hex[:12],
                          raw_response=response)

    async def _update_card(self, event_req_id: Optional[str], card: Dict[str, Any]) -> None:
        """Best-effort ``aibot_respond_update_msg`` inside WeCom's 5s event reply window.
        Never raises: an expired window (errcode) just leaves the card showing."""
        if not event_req_id:
            return
        try:
            response = await self._request(
                APP_CMD_RESPOND_UPDATE, str(event_req_id),
                {"response_type": "update_template_card", "template_card": card}, timeout=5.0)
            if error := self._response_error(response):
                logger.debug("[%s] Card update declined (window likely expired): %s", self.name, error)
        except Exception as exc:
            logger.debug("[%s] Card update failed (window likely expired): %s", self.name, exc)

    # ── exec approval (override of BasePlatformAdapter template hook) ──────────────────

    async def _send_exec_approval_prompt(self, prompt) -> SendResult:
        """Render the dangerous-command approval as a button_interaction card (DM only).
        Group chats fall back to the plain-text prompt so /approve stays usable there —
        WeCom template cards are DM-only by product decision. Resolves via
        ``tools.approval.resolve_gateway_approval`` on tap, mirroring Telegram's ea: buttons."""
        chat_id = str(prompt.chat_id or "").strip()
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")
        if chat_id in self._group_chat_ids:
            return await self.send(chat_id, prompt.text)
        task_id = self._new_card_task_id("ea")
        self._remember_card_state(self._approval_state, task_id, {
            "session_key": prompt.session_key, "chat_id": chat_id,
            "desc": self._truncate_preview(str(prompt.description or ""), 100),
        })
        buttons = [{"text": label, "key": choice} for label, choice, _ in prompt.actions]
        card = self._button_interaction_card(
            title="⚠️ 命令审批", desc=self._truncate_preview(str(prompt.description or ""), 100),
            sub_title=self._truncate_preview(str(prompt.command or ""), 100),
            buttons=buttons, task_id=task_id)
        reply_req_id = None if self._find_active_turn_for_chat(chat_id) else self._cached_reply_req_id(chat_id, None)
        result = await self._send_card(chat_id, card, reply_req_id=reply_req_id, is_control=True)
        if not result.success:
            self._approval_state.pop(task_id, None)  # don't leave a resolvable dud behind
            # Never lose the approval: fall back to the plain-text prompt.
            fallback = await self.send(chat_id, prompt.text)
            if not fallback.success:
                return result
        return result

    # ── model picker (slash_commands_model.py probes for this method) ─────────────────

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str,
        session_key: str, on_model_selected, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """DM-only model picker card. Candidate models are flattened from ``providers`` with the
        current provider's models first, then deduped and capped at 6 buttons; a tap calls
        ``on_model_selected(chat_id, model_id, provider_slug)``. Groups get Not supported so the
        gateway falls back to the text /model listing."""
        chat_id = str(chat_id or "").strip()
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")
        if chat_id in self._group_chat_ids:
            return SendResult(success=False, error="Not supported")  # text /model listing fallback
        candidates: List[Dict[str, str]] = []
        seen: set = set()
        try:
            from hermes_cli.providers import get_label
        except Exception:
            get_label = None
        for p in providers:  # type: ignore[union-attr]
            slug = str(p.get("slug") or "")
            name = str(p.get("name") or slug)
            if get_label is not None:
                try:
                    name = get_label(slug) or name
                except Exception:
                    pass
            for m in p.get("models", []) or []:
                model_id = str(m)
                if model_id in seen:
                    continue
                seen.add(model_id)
                candidates.append({"model": model_id, "slug": slug, "provider": name})
        if not candidates:
            return SendResult(success=False, error="No providers available")
        # Current-provider models first, then the rest, then current model to the front.
        def _rank(c: Dict[str, str]) -> int:
            return 0 if c["slug"] == str(current_provider or "") else 1
        candidates.sort(key=_rank)
        if current_model and any(c["model"] == current_model for c in candidates):
            idx = next(i for i, c in enumerate(candidates) if c["model"] == current_model)
            candidates.insert(0, candidates.pop(idx))
        top = candidates[:CARD_BUTTON_MAX]
        task_id = self._new_card_task_id("mp")
        self._remember_card_state(self._model_picker_state, task_id, {
            "session_key": session_key, "chat_id": chat_id, "candidates": top,
            "on_model_selected": on_model_selected,
        })
        buttons = [
            {"text": ("✓ " if c["model"] == current_model else "") + c["model"].split("/")[-1][:10],
             "key": f"m:{i}"}
            for i, c in enumerate(top)
        ]
        card = self._button_interaction_card(
            title="⚙️ 模型选择", desc=f"当前：{current_model or 'unknown'}（{str(current_provider or '')}）",
            sub_title="点选模型立即切换", buttons=buttons, task_id=task_id)
        reply_req_id = None if self._find_active_turn_for_chat(chat_id) else self._cached_reply_req_id(chat_id, None)
        result = await self._send_card(chat_id, card, reply_req_id=reply_req_id, is_control=True)
        if not result.success:
            self._model_picker_state.pop(task_id, None)
        return result

    # ── inbound taps (aibot_event_callback → template_card_event) ─────────────────────

    async def _handle_template_card_event(self, payload: Dict[str, Any]) -> None:
        """Dispatch one template-card tap: authorize the sender (DM allowlist), then route to
        the approval or model-picker handler by task_id prefix. Group taps are ignored (DM-only
        cards are never sent there, so a stray tap cannot resolve an approval)."""
        body = payload.get("body") if isinstance(payload.get("body"), dict) else None
        if not body:
            return
        from_hdr = body.get("from") if isinstance(body.get("from"), dict) else {}
        sender_id = str(from_hdr.get("userid") or "").strip()
        chat_id = str(body.get("chatid") or sender_id).strip()
        if str(body.get("chattype") or "").lower() == "group":
            logger.info("[%s] Ignoring template-card tap in group (DM-only cards); chat=%s sender=%s",
                        self.name, body.get("chatid"), sender_id)
            return
        if not sender_id or not self._is_dm_intake_allowed(sender_id):
            logger.info("[%s] Rejecting template-card tap from unauthorized sender %s", self.name, sender_id)
            return
        event = body.get("event") if isinstance(body.get("event"), dict) else {}
        tce = event.get("template_card_event") if isinstance(event.get("template_card_event"), dict) else {}
        task_id = str(tce.get("task_id") or "").strip()
        event_key = str(tce.get("event_key") or "").strip()
        if not task_id or not event_key:
            logger.debug("[%s] template_card_event missing task_id/event_key; body_keys=%s", self.name, list(body.keys()))
            return
        if task_id in self._approval_state:
            await self._resolve_approval_tap(payload, sender_id or chat_id, task_id, event_key)
        elif task_id in self._model_picker_state:
            await self._resolve_model_picker_tap(payload, task_id, event_key, sender_id or chat_id)
        else:
            logger.info("[%s] template-card tap for unknown/expired task_id=%r key=%r sender=%s",
                        self.name, task_id, event_key, sender_id)

    async def _resolve_approval_tap(self, payload: Dict[str, Any], user: str, task_id: str, choice: str) -> None:
        """Resolve a pending exec approval from a card tap, then swap the card for a text_notice
        inside WeCom's 5s window. Pop-first so a repeat tap cannot double-resolve."""
        state = self._approval_state.pop(task_id, None)
        if not state:
            logger.info("[%s] Approval card %s already resolved", self.name, task_id)
            return
        session_key = str(state.get("session_key") or "")
        if not session_key:
            return
        count = 0
        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(session_key, choice)
        except Exception as exc:
            logger.error("[%s] Approval tap resolution failed: %s", self.name, exc)
        label = _APPROVAL_TAP_LABELS.get(choice, "已处理")
        resolved = count > 0
        await self._update_card(self._payload_req_id(payload), self._text_notice_card(
            title=label if resolved else "⌛ 审批已过期",
            desc=(f"决策人：{user}" if resolved else "没有等待的命令：已超时或被其他渠道处理"),
            task_id=task_id))
        logger.info("[%s] Approval card %s resolved by %s: choice=%s count=%d", self.name, task_id, user, choice, count)

    async def _resolve_model_picker_tap(self, payload: Dict[str, Any], task_id: str, event_key: str, user: str) -> None:
        """Apply a model-picker tap: pop the picker state, swap the card to 'switching…', then
        run the stored ``on_model_selected`` callback and forward any result text."""
        state = self._model_picker_state.pop(task_id, None)
        if not state:
            logger.info("[%s] Model picker card %s already resolved", self.name, task_id)
            return
        if not event_key.startswith("m:"):
            return
        try:
            idx = int(event_key[2:])
            candidate = state["candidates"][idx]
        except (ValueError, IndexError, KeyError):
            logger.info("[%s] Bad model-picker tap key=%r", self.name, event_key)
            return
        chat_id = str(state.get("chat_id") or "")
        callback = state.get("on_model_selected")
        await self._update_card(self._payload_req_id(payload), self._text_notice_card(
            title="🔄 正在切换模型…", desc=f"{candidate['model']}（{candidate['provider']}）", task_id=task_id))
        if callback is None:
            return
        try:
            result_text = await callback(chat_id, candidate["model"], candidate["slug"])
        except Exception as exc:
            logger.error("[%s] Model picker switch failed: %s", self.name, exc)
            result_text = f"⚠️ 切换失败：{exc}"
        if result_text and chat_id:
            await self.send(chat_id, str(result_text))
        logger.info("[%s] Model picker %s applied by %s: %s/%s", self.name, task_id, user,
                    candidate["slug"], candidate["model"])