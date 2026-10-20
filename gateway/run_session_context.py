"""Bind the admitted gateway session identity and history delivery context."""
from __future__ import annotations
from gateway.session import SessionContext


class GatewaySessionContextMixin:
    def _set_session_env(self, context: SessionContext) -> list:
        """Set session context variables (contextvars, not os.environ, so concurrent messages can't
        overwrite each other). Returns reset tokens for ``_clear_session_env`` in a ``finally``."""
        from gateway.recall_scope import is_recall_gateway_platform
        from gateway.session_context import set_session_vars
        # Async-delivery capability tells async tools whether this channel can wake a later turn. Default
        # True keeps CLI/unknown paths working; stateless adapters (api_server) declare False.
        _adapter = (getattr(self, "adapters", None) or {}).get(context.source.platform)
        _async_delivery = getattr(_adapter, "supports_async_delivery", True)
        # #98619: an admitted API turn carries its session-id provenance in the admission settings,
        # and its rebound browser-controller identity; every other platform leaves both undeclared.
        from gateway.session_api_turn import api_session_vars
        _api_vars = api_session_vars()
        _history_delivery = _api_vars.pop("session_history_delivery", None)
        return set_session_vars(
            platform=context.source.platform.value,
            chat_id=context.source.chat_id,
            prospective_thread_id=str(
                getattr(context.source, "prospective_thread_id", "") or ""),
            chat_type=str(context.source.chat_type) if context.source.chat_type else "",
            chat_name=context.source.chat_name or "",
            thread_id=str(context.source.thread_id) if context.source.thread_id else "",
            user_id=str(context.source.user_id) if context.source.user_id else "",
            user_id_alt=str(context.source.user_id_alt) if context.source.user_id_alt else "",
            user_name=str(context.source.user_name) if context.source.user_name else "",
            scope_id=str(getattr(context.source, "scope_id", "") or ""),
            parent_chat_id=str(getattr(context.source, "parent_chat_id", "") or ""),
            session_key=context.session_key,
            session_id=context.session_id or "",
            message_id=str(context.source.message_id) if context.source.message_id else "",
            profile=getattr(context.source, "profile", "") or "",
            async_delivery=_async_delivery,
            cron_session="",
            gateway_context=is_recall_gateway_platform(context.source.platform),
            session_history_delivery=_history_delivery, **_api_vars)

