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

    # WeCom caps one card at 6 buttons: model pages use 3 model buttons + Back + page nav,
    # leaving room for the nav row on every page (first/last pages drop one nav button).
    _MODEL_PAGE_SIZE = 3

    @staticmethod
    def _picker_model_label(model_id: str, is_current: bool = False) -> str:
        """Short model label for a WeCom button (≈10 chars incl. the ✓ marker). The most
        distinguishing part of a model id is usually its tail (flash/max/turbo): keep that,
        drop the series prefix. Falls back to a head+tail ellipse when even the tail is long."""
        raw = str(model_id or "").split("/")[-1]
        tail = raw.rsplit("-", 1)[-1] if "-" in raw else raw
        label = f"✓ {tail}" if is_current else tail
        if len(label) <= 10:
            return label
        if len(tail) <= 8:
            return (f"✓ {tail}" if is_current else tail)[:10]
        return (f"✓ {tail[:6]}…" if is_current else f"{tail[:8]}…")[:10]

    async def send_model_picker(
        self, chat_id: str, providers: list, current_model: str, current_provider: str,
        session_key: str, on_model_selected, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """DM-only two-level model picker card: provider page → model page (updated in place,
        same task_id), with Back and page nav. Taps call ``on_model_selected(chat_id, model_id,
        provider_slug)``. Groups get Not supported so the gateway falls back to the text listing."""
        chat_id = str(chat_id or "").strip()
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")
        if chat_id in self._group_chat_ids:
            return SendResult(success=False, error="Not supported")  # text /model listing fallback
        try:
            from hermes_cli.providers import get_label
        except Exception:
            get_label = None
        seen: set = set()
        normalized: List[Dict[str, Any]] = []
        for p in providers or []:  # type: ignore[union-attr]
            slug = str(p.get("slug") or "").strip()
            name = str(p.get("name") or slug)
            if get_label is not None:
                try:
                    name = get_label(slug) or name
                except Exception:
                    pass
            models = [str(m) for m in (p.get("models", []) or []) if str(m) not in seen and not seen.add(str(m))]
            if slug and models:
                normalized.append({"slug": slug, "name": name, "models": models, "is_current": slug == str(current_provider or "")})
        if not normalized:
            return SendResult(success=False, error="No providers available")
        task_id = self._new_card_task_id("mp")
        self._remember_card_state(self._model_picker_state, task_id, {
            "session_key": session_key, "chat_id": chat_id,
            "current_model": str(current_model or ""), "current_provider": str(current_provider or ""),
            "providers": normalized, "stage": "providers", "selected_provider": "",
            "model_page": 0, "on_model_selected": on_model_selected,
        })
        card = self._build_provider_card(task_id)
        reply_req_id = None if self._find_active_turn_for_chat(chat_id) else self._cached_reply_req_id(chat_id, None)
        result = await self._send_card(chat_id, card, reply_req_id=reply_req_id, is_control=True)
        if not result.success:
            self._model_picker_state.pop(task_id, None)
        return result

    # ── picker card builders (pure, driven by the persistent picker state) ──────────────

    def _build_provider_card(self, task_id: str) -> Dict[str, Any]:
        """Provider page: one button per provider (current flagged), up to the 6-button cap."""
        state = self._model_picker_state.get(task_id) or {}
        providers = state.get("providers") or []
        current = str(state.get("current_model") or "unknown")
        cur_provider = str(state.get("current_provider") or "")
        buttons = [{
            "text": self._picker_model_label(p["name"] or p["slug"], p.get("is_current", False)),
            "key": f"p:{p['slug']}",
        } for p in providers[:CARD_BUTTON_MAX]]
        extra = f"（仅列前 {CARD_BUTTON_MAX} 个，可用 /model 直接输名称）" if len(providers) > CARD_BUTTON_MAX else ""
        return self._button_interaction_card(
            title="⚙️ 模型选择", desc=f"当前：{current}（{cur_provider}）",
            sub_title=f"选择提供商{extra}", buttons=buttons, task_id=task_id)

    def _build_model_card(self, task_id: str) -> Dict[str, Any]:
        """Model page for the selected provider: 3 models + Back + page nav (<6 buttons)."""
        state = self._model_picker_state.get(task_id) or {}
        provider = next((p for p in (state.get("providers") or []) if p["slug"] == state.get("selected_provider")), None)
        models = provider.get("models", []) if provider else []
        page = int(state.get("model_page") or 0)
        total_pages = max(1, (len(models) + self._MODEL_PAGE_SIZE - 1) // self._MODEL_PAGE_SIZE)
        page = min(page, total_pages - 1)
        start = page * self._MODEL_PAGE_SIZE
        page_models = models[start:start + self._MODEL_PAGE_SIZE]
        current = str(state.get("current_model") or "")
        buttons = [
            {"text": self._picker_model_label(m, m == current), "key": f"m:{start + i}"}
            for i, m in enumerate(page_models)
        ]
        nav: List[Dict[str, str]] = [{"text": "◀ 返回", "key": "back"}]
        if page > 0:
            nav.append({"text": "◀ 上一页", "key": f"pg:{page - 1}"})
        if page < total_pages - 1:
            nav.append({"text": "下一页 ▶", "key": f"pg:{page + 1}"})
        buttons.extend(nav)
        pname = provider.get("name", state.get("selected_provider") or "") if provider else ""
        page_hint = f" · 第 {page + 1}/{total_pages} 页" if total_pages > 1 else ""
        return self._button_interaction_card(
            title="⚙️ 模型选择", desc=f"{pname} {page_hint}",
            sub_title=f"当前：{current}", buttons=buttons, task_id=task_id)

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
        """Route one model-picker tap within the two-level state machine. Provider/back/page
        taps re-render the same card (5s window); only a model selection pops the state and
        resolves. ``m:<idx>`` indexes into the selected provider's full model list."""
        state = self._model_picker_state.get(task_id)
        if not state:
            logger.info("[%s] Model picker card %s already resolved", self.name, task_id)
            return
        req_id = self._payload_req_id(payload)
        model: Optional[str] = None
        if event_key.startswith("p:") and state.get("stage") == "providers":
            slug = event_key[2:]
            provider = next((p for p in (state.get("providers") or []) if p["slug"] == slug), None)
            if provider:
                state.update(stage="models", selected_provider=slug, model_page=0)
                await self._update_card(req_id, self._build_model_card(task_id))
                return
        if event_key == "back":
            state.update(stage="providers", selected_provider="")
            await self._update_card(req_id, self._build_provider_card(task_id))
            return
        if event_key.startswith("pg:") and state.get("stage") == "models":
            try:
                state["model_page"] = max(0, int(event_key[3:]))
            except ValueError:
                return
            await self._update_card(req_id, self._build_model_card(task_id))
            return
        if event_key.startswith("m:") and state.get("stage") == "models":
            try:
                idx = int(event_key[2:])
            except ValueError:
                return
            provider = next((p for p in (state.get("providers") or []) if p["slug"] == state.get("selected_provider")), None)
            models = provider.get("models", []) if provider else []
            if 0 <= idx < len(models):
                model = models[idx]
        if model is None:
            logger.info("[%s] Model-picker tap not actionable: key=%r stage=%r", self.name, event_key,
                        state.get("stage"))
            return
        # Model chosen — pop the picker (no repeat taps), flip card to switching, then apply.
        self._model_picker_state.pop(task_id, None)
        chat_id = str(state.get("chat_id") or "")
        callback = state.get("on_model_selected")
        provider = next((p for p in (state.get("providers") or []) if p["slug"] == state.get("selected_provider")), None) or {}
        await self._update_card(req_id, self._text_notice_card(
            title="🔄 正在切换模型…", desc=f"{model}（{provider.get('name', '')}）", task_id=task_id))
        if callback is None:
            return
        try:
            result_text = await callback(chat_id, model, state.get("selected_provider") or "")
        except Exception as exc:
            logger.error("[%s] Model picker switch failed: %s", self.name, exc)
            result_text = f"⚠️ 切换失败：{exc}"
        if result_text and chat_id:
            await self.send(chat_id, str(result_text))
        logger.info("[%s] Model picker %s applied by %s: %s/%s", self.name, task_id, user,
                    state.get("selected_provider"), model)