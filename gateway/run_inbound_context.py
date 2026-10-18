"""Inbound context-reference expansion and its request-bound provenance."""
from __future__ import annotations
from typing import Optional
from gateway.session import SessionSource


class GatewayInboundContextMixin:
    async def _expand_inbound_context_references(
        self, source: SessionSource, session_key: str, message_text: str, *, source_slices=None
    ) -> Optional[str]:
        """Expand ``@`` context references; returns None when the injection was refused (user notified)."""
        from gateway.session_context_refs import inbound_references
        expanded, refusal = await inbound_references(
            self, source, session_key, message_text, source_slices=source_slices,
        )
        if refusal is None:
            return expanded
        adapter = self._delivery_adapter_for(source)
        if adapter:
            await adapter.send(source.chat_id, refusal)
        return None
