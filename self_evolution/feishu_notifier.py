"""
Self Evolution Plugin — Feishu Notifier
========================================

Pushes evolution proposals to Feishu at 19:00 daily.
Uses interactive card messages with action buttons for approval.

Receives callbacks when user clicks: approve / modify / reject.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from self_evolution import db
from self_evolution.models import Proposal

logger = logging.getLogger(__name__)


class FeishuNotifier:
    """Send evolution proposals via Feishu interactive cards."""

    def __init__(self):
        self.app_id = os.getenv("FEISHU_APP_ID", "")
        self.app_secret = os.getenv("FEISHU_APP_SECRET", "")
        self.enabled = bool(self.app_id and self.app_secret)
        self._client = None
        self._token_cache: Optional[tuple[str, float]] = None  # (token, expire_at)

    def send_daily_report(self):
        """Send pending proposals as a daily Feishu card message.

        Called by the 19:00 cron job.
        """
        if not self.enabled:
            logger.info("Feishu not configured, skipping notification")
            return

        # Load pending proposals
        proposals = db.fetch_all(
            "evolution_proposals",
            where="status = ?",
            params=("pending_approval",),
            order_by="created_at DESC",
        )

        if not proposals:
            logger.info("No pending proposals to send")
            return

        # Load latest reflection report for context
        reports = db.fetch_all(
            "reflection_reports",
            order_by="created_at DESC",
            limit=1,
        )
        report = reports[0] if reports else {}

        # Build card
        card = self._build_card(proposals, report)

        # Send
        self._send_card(card)
        logger.info("Sent %d proposals via Feishu", len(proposals))

    def handle_callback(self, action: str, proposal_id: str, user_input: str = ""):
        """Handle Feishu card button callback.

        Args:
            action: "approve" | "modify" | "reject"
            proposal_id: The proposal ID
            user_input: Optional user modification text

        Returns:
            dict with 'feedback' (str) and 'updated_card' (dict or None).
        """
        result = {"feedback": "", "updated_card": None}

        if action == "approve":
            logger.info("[TRACE] handle_callback: approving proposal %s", proposal_id)
            title = self._approve(proposal_id)
            result["feedback"] = f"✅ 已通过并执行: {title}"
            logger.info("[TRACE] handle_callback: approved '%s'", title)
        elif action == "modify":
            title = self._modify(proposal_id, user_input)
            result["feedback"] = f"✏️ 已修改: {title}"
        elif action == "reject":
            title = self._reject(proposal_id, user_input)
            result["feedback"] = f"❌ 已拒绝: {title}"

        # Build updated card with remaining pending proposals
        logger.info("[TRACE] handle_callback: building updated card")
        result["updated_card"] = self.build_updated_card()
        logger.info("[TRACE] handle_callback: updated_card=%s", "present" if result["updated_card"] else "None (all done)")
        return result

    def build_updated_card(self) -> Optional[dict]:
        """Build a card with remaining pending proposals.

        Returns None if no pending proposals remain (caller can show
        a 'all done' card instead).
        """
        pending = db.fetch_all(
            "evolution_proposals",
            where="status = ?",
            params=("pending_approval",),
            order_by="created_at DESC",
        )

        if not pending:
            return None

        # Load latest report for context
        reports = db.fetch_all("reflection_reports", order_by="created_at DESC", limit=1)
        report = reports[0] if reports else {}

        date_str = time.strftime("%Y-%m-%d", time.localtime())
        elements = []

        # Status bar
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**待审批**: {len(pending)} 个提案"},
        })
        elements.append({"tag": "hr"})

        # Proposals
        for i, p in enumerate(pending):
            type_emoji = {"skill": "🛠️", "strategy": "⚡", "memory": "🧠", "tool_preference": "🔧", "code_improvement": "🏗️"}
            emoji = type_emoji.get(p.get("proposal_type", ""), "📋")

            proposal_text = (
                f"**[{emoji}] {p.get('title', f'提案 {i+1}')}**\n"
                f"{p.get('description', '')[:200]}\n"
                f"预期影响: {p.get('expected_impact', 'N/A')} | "
                f"风险: {p.get('risk_assessment', 'low')}\n"
            )
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": proposal_text},
            })

            # Action buttons
            elements.append({
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "通过"},
                        "type": "primary",
                        "value": {"action": "approve", "proposal_id": p["id"]},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "修改"},
                        "type": "default",
                        "value": {"action": "modify", "proposal_id": p["id"]},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "拒绝"},
                        "type": "danger",
                        "value": {"action": "reject", "proposal_id": p["id"]},
                    },
                ],
            })

        return {
            "header": {
                "title": {"tag": "plain_text", "content": f"Hermes 进化报告 ({date_str})"},
                "template": "blue",
            },
            "elements": elements,
        }

    def send_rollback_notification(self, unit_id: str, reason: str):
        """Notify user that an improvement unit was auto-rolled back."""
        if not self.enabled:
            return
        card = {
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**自动回滚通知**\n\n"
                                   f"改进单元 `{unit_id}` 已自动回滚。\n"
                                   f"原因: {reason}",
                    },
                },
            ],
        }
        self._send_card(card)

    # ── Internal Methods ──────────────────────────────────────────────────

    def _approve(self, proposal_id: str) -> str:
        """Mark proposal as approved and trigger execution. Returns title."""
        row = db.fetch_one("evolution_proposals", where="id = ?", params=(proposal_id,))
        title = row.get("title", proposal_id) if row else proposal_id

        db.update(
            "evolution_proposals",
            {"status": "approved", "resolved_at": time.time()},
            where="id = ?",
            where_params=(proposal_id,),
        )

        # Trigger execution
        if row:
            from self_evolution.evolution_executor import EvolutionExecutor
            executor = EvolutionExecutor()
            proposal = Proposal(
                id=row["id"],
                proposal_type=row["proposal_type"],
                title=row["title"],
                description=row["description"],
                expected_impact=row.get("expected_impact", ""),
                risk_assessment=row.get("risk_assessment", "low"),
                rollback_plan=row.get("rollback_plan", ""),
                status="approved",
            )
            executor.execute(proposal)

        return title

    def _modify(self, proposal_id: str, user_input: str) -> str:
        """Update proposal with user's modification. Returns title."""
        row = db.fetch_one("evolution_proposals", where="id = ?", params=(proposal_id,))
        title = row.get("title", proposal_id) if row else proposal_id

        db.update(
            "evolution_proposals",
            {"user_feedback": user_input, "status": "pending_approval"},
            where="id = ?",
            where_params=(proposal_id,),
        )
        return title

    def _reject(self, proposal_id: str, user_input: str) -> str:
        """Mark proposal as rejected and record reason for learning. Returns title."""
        row = db.fetch_one("evolution_proposals", where="id = ?", params=(proposal_id,))
        title = row.get("title", proposal_id) if row else proposal_id

        db.update(
            "evolution_proposals",
            {"status": "rejected", "user_feedback": user_input, "resolved_at": time.time()},
            where="id = ?",
            where_params=(proposal_id,),
        )
        # Record rejection for the dream engine to learn from
        db.insert("outcome_signals", {
            "session_id": f"evolution_rejection_{proposal_id}",
            "signal_type": "proposal_rejected",
            "signal_value": 0.0,
            "metadata": json.dumps({"proposal_id": proposal_id, "reason": user_input}, ensure_ascii=False),
        })
        return title

    def _build_card(self, proposals: List[dict], report: dict) -> dict:
        """Build Feishu interactive card JSON."""
        # Header
        date_str = time.strftime("%Y-%m-%d", time.localtime())
        elements = []

        # Overview section
        sessions_analyzed = report.get("sessions_analyzed", 0)
        avg_score = report.get("avg_score", 0)
        overview = (
            f"**日期**: {date_str}\n"
            f"**分析Sessions**: {sessions_analyzed}\n"
            f"**平均评分**: {avg_score:.3f}\n"
        )
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": overview},
        })

        # Error summary
        error_summary = report.get("error_summary", "")
        if error_summary:
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**错误分析**\n{error_summary}"},
            })

        # Waste summary
        waste_summary = report.get("waste_summary", "")
        if waste_summary:
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**时间浪费分析**\n{waste_summary}"},
            })

        # Code change summary
        code_change_summary = report.get("code_change_summary", "")
        if code_change_summary:
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": f"**系统代码更新**\n{code_change_summary}"},
            })

        # Separator
        elements.append({"tag": "hr"})

        # Proposals
        for i, p in enumerate(proposals):
            type_emoji = {"skill": "🛠️", "strategy": "⚡", "memory": "🧠", "tool_preference": "🔧", "code_improvement": "🏗️"}
            emoji = type_emoji.get(p.get("proposal_type", ""), "📋")

            proposal_text = (
                f"**[{emoji}] {p.get('title', f'提案 {i+1}')}**\n"
                f"{p.get('description', '')[:200]}\n"
                f"预期影响: {p.get('expected_impact', 'N/A')} | "
                f"风险: {p.get('risk_assessment', 'low')}\n"
            )
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md", "content": proposal_text},
            })

            # Action buttons
            elements.append({
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "通过"},
                        "type": "primary",
                        "value": {"action": "approve", "proposal_id": p["id"]},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "修改"},
                        "type": "default",
                        "value": {"action": "modify", "proposal_id": p["id"]},
                    },
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "拒绝"},
                        "type": "danger",
                        "value": {"action": "reject", "proposal_id": p["id"]},
                    },
                ],
            })

        return {
            "header": {
                "title": {"tag": "plain_text", "content": f"Hermes 每日进化报告 ({date_str})"},
                "template": "blue",
            },
            "elements": elements,
        }

    def _get_client(self):
        """Get or create a cached lark Client instance."""
        if self._client is None:
            import lark_oapi as lark
            self._client = (
                lark.Client.builder()
                .app_id(self.app_id)
                .app_secret(self.app_secret)
                .build()
            )
        return self._client

    def _send_card(self, card: dict):
        """Send an interactive card via Feishu.

        Prefers lark_oapi SDK (same as the gateway), falls back to REST.
        """
        try:
            receive_id, receive_id_type = self._resolve_target()
            if not receive_id:
                logger.warning("No Feishu receive target configured")
                return

            content_str = json.dumps(card, ensure_ascii=False)

            # Try SDK first (using cached client)
            try:
                from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

                client = self._get_client()

                body = CreateMessageRequestBody.builder() \
                    .receive_id(receive_id) \
                    .msg_type("interactive") \
                    .content(content_str) \
                    .build()

                request = CreateMessageRequest.builder() \
                    .receive_id_type(receive_id_type) \
                    .request_body(body) \
                    .build()

                response = client.im.v1.message.create(request)
                if response.success():
                    logger.info("Feishu card sent via SDK")
                    return
                logger.warning("Feishu SDK send failed: code=%s msg=%s", response.code, response.msg)
            except ImportError:
                pass

            # Fallback to REST API
            self._send_card_rest(receive_id, receive_id_type, content_str)

        except Exception as exc:
            logger.warning("Feishu notification failed: %s", exc)

    def _resolve_target(self) -> tuple:
        """Resolve (receive_id, receive_id_type) from env config."""
        deliver_to = os.getenv("SELF_EVOLUTION_FEISHU_DELIVER", "user")
        if deliver_to.startswith("chat:"):
            return deliver_to.replace("chat:", ""), "chat_id"
        user_id = os.getenv("SELF_EVOLUTION_FEISHU_USER_ID", "")
        if not user_id:
            return "", ""
        if user_id.startswith("ou_"):
            return user_id, "open_id"
        if user_id.startswith("oc_"):
            return user_id, "chat_id"
        return user_id, "user_id"

    def _send_card_rest(self, receive_id: str, receive_id_type: str, content: str):
        """Fallback: send card via REST API."""
        import requests

        token = self._get_tenant_token()
        if not token:
            logger.warning("Failed to get Feishu token")
            return

        resp = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            headers={"Authorization": f"Bearer {token}"},
            params={"receive_id_type": receive_id_type},
            json={"receive_id": receive_id, "msg_type": "interactive", "content": content},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("Feishu REST send failed: %s", resp.text)

    def _send_confirmation(self, proposal_id: str, message: str):
        """Send a simple confirmation message."""
        if not self.enabled:
            return
        card = {
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**提案 `{proposal_id}`**: {message}",
                    },
                },
            ],
        }
        self._send_card(card)

    def _get_tenant_token(self) -> Optional[str]:
        """Get Feishu tenant access token with caching (1.5h TTL)."""
        if self._token_cache is not None:
            token, expire_at = self._token_cache
            if time.time() < expire_at:
                return token
        try:
            import requests
            resp = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self.app_id,
                    "app_secret": self.app_secret,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                token = resp.json().get("tenant_access_token")
                if token:
                    # Feishu tokens expire in ~2h; cache for 1.5h
                    self._token_cache = (token, time.time() + 5400)
                return token
        except Exception as exc:
            logger.debug("Failed to get Feishu token: %s", exc)
        return None
