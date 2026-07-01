"""app-chatbot plugin — CRWD support tools + MongoDB-first intent router."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from . import handlers, schemas
from ._utils import default_user_id
from .router import format_router_context

logger = logging.getLogger(__name__)

_TOOLSET = "app-chatbot"


def _mongodb_available() -> bool:
    return bool(os.getenv("MONGODB_URI"))


def _plugin_settings() -> Dict[str, Any]:
    return {"default_user_id": default_user_id()}


def _prefetch_context(user_message: str = "", **_: Any) -> Optional[Dict[str, str]]:
    settings = _plugin_settings()
    context = format_router_context(
        user_message,
        default_user_id=settings["default_user_id"],
    )
    if not context:
        return None
    return {"context": context}


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _prefetch_context)

    tool_pairs = [
        (schemas.GET_ACTIVE_GIGS, handlers.get_active_gigs),
        (schemas.GET_USER_PROFILE_BY_ID, handlers.get_user_profile_by_id),
        (schemas.GET_GIG_DETAILS, handlers.get_gig_details),
        (schemas.GET_USER_GIG_HISTORY, handlers.get_user_gig_history),
        (schemas.GET_USER_JOINED_GIGS, handlers.get_user_joined_gigs),
    ]
    for schema, handler in tool_pairs:
        ctx.register_tool(
            name=schema["name"],
            toolset=_TOOLSET,
            schema=schema,
            handler=handler,
            check_fn=_mongodb_available,
            requires_env=["MONGODB_URI"],
        )
