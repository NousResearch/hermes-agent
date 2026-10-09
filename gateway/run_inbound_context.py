"""Inbound context-reference expansion and its request-bound provenance."""
from __future__ import annotations
import asyncio
import logging
import os
from typing import Optional
from agent.i18n import t
from gateway.session import SessionSource
logger = logging.getLogger("gateway.run")


class GatewayInboundContextMixin:
    async def _expand_inbound_context_references(
        self, source: SessionSource, session_key: str, message_text: str, *, source_slices=None
    ) -> Optional[str]:
        """Expand ``@`` context references; returns None when the injection was refused (user notified)."""
        try:
            from agent.context_references import preprocess_context_references_async

            try:
                from tools.terminal_scope import terminal_env as _ts_env
            except ImportError:
                _ts_env = os.environ.get
            _msg_cwd = _ts_env("TERMINAL_CWD") or await asyncio.to_thread(os.path.expanduser, "~")
            _msg_ctx_len = await self._inbound_model_context_length(source, session_key)
            _ctx_result = await preprocess_context_references_async(
                message_text,
                cwd=_msg_cwd,
                context_length=_msg_ctx_len,
                allowed_root=_msg_cwd,
            )
            if _ctx_result.blocked:
                _adapter = self._delivery_adapter_for(source)
                if _adapter:
                    await _adapter.send(
                        source.chat_id,
                        "\n".join(_ctx_result.warnings) or t("gateway.notify.context_injection_refused"),
                    )
                return None
            if source_slices is not None:
                source_slices.extend(_ctx_result.source_slices)
            if _ctx_result.expanded:
                message_text = _ctx_result.message
        except Exception as exc:
            logger.warning("@ context reference expansion failed: %s", exc)
            logger.debug("@ context reference expansion failure detail", exc_info=True)
        return message_text
