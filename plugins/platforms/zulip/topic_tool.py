"""Agent tool for starting or continuing an isolated Zulip topic session."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import replace
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SESSION_SEED_PREFIX = "[Hermes to Zulip]"


def _zulip_platform():
    """Return the plugin's dynamically registered platform value."""
    from gateway.config import Platform

    return Platform("zulip")


def _current_zulip_session_user() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return the active Zulip sender's email, name, and profile, if any."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return None, None, None

    if get_session_env("HERMES_SESSION_PLATFORM", "").lower() != "zulip":
        return None, None, None
    email = get_session_env("HERMES_SESSION_USER_ID", "").strip()
    name = get_session_env("HERMES_SESSION_USER_NAME", "").strip()
    profile = get_session_env("HERMES_SESSION_PROFILE", "").strip()
    return email or None, name or email or None, profile or None


def _load_zulip_tool_config() -> tuple[str, str, str, Any, Any]:
    """Load credentials and the config needed to seed a gateway session."""
    from gateway.config import Platform, load_gateway_config
    from .search_tool import _get_zulip_credentials

    config = load_gateway_config()
    platform_config = config.platforms.get(Platform("zulip"))
    if not platform_config or not platform_config.enabled:
        raise ValueError("Zulip platform is not configured or enabled")

    site_url, bot_email, api_key = _get_zulip_credentials(platform_config)
    if not site_url or not bot_email or not api_key:
        raise ValueError(
            "Zulip credentials are incomplete; configure site_url, bot_email, and API key"
        )

    # Credential resolution gives explicit environment credentials precedence.
    # Give the standalone sender that exact resolved account as well, so stream
    # lookup and delivery can never accidentally target different realms.
    send_config = replace(
        platform_config,
        token=api_key,
        api_key=None,
        extra={
            **(platform_config.extra or {}),
            "site_url": site_url,
            "bot_email": bot_email,
        },
    )
    return site_url, bot_email, api_key, config, send_config


def _seed_topic_session(
    *,
    config: Any,
    stream_id: int,
    stream_name: str,
    topic: str,
    user_email: str,
    user_name: str,
    profile: Optional[str],
    message: str,
    message_id: str,
) -> str:
    """Create/reuse the inbound topic session and record the sent seed text."""
    from gateway.session import SessionSource, build_session_key
    from hermes_state import SessionDB

    source = SessionSource(
        platform=_zulip_platform(),
        chat_id=f"{stream_id}:{topic}",
        chat_name=stream_name,
        chat_type="stream",
        user_id=user_email,
        user_name=user_name,
        chat_topic=topic,
        profile=profile,
    )
    session_key = build_session_key(
        source,
        group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
        thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
        profile=profile if getattr(config, "multiplex_profiles", False) else None,
    )
    db = SessionDB()
    existing = db.find_latest_gateway_session_for_peer(
        source="zulip",
        user_id=user_email,
        session_key=session_key,
        chat_id=source.chat_id,
        chat_type=source.chat_type,
    )
    if existing:
        session_id = str(existing["id"])
        db.reopen_session(session_id)
    else:
        session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        db.create_session(
            session_id,
            "zulip",
            user_id=user_email,
            session_key=session_key,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
        )
        db.record_gateway_session_peer(
            session_id,
            source="zulip",
            user_id=user_email,
            session_key=session_key,
            chat_id=source.chat_id,
            chat_type=source.chat_type,
            display_name=stream_name,
            origin_json=json.dumps(source.to_dict()),
        )
    db.append_message(
        session_id,
        role="user",
        content=f"{_SESSION_SEED_PREFIX}\n{message}",
        platform_message_id=message_id,
        timestamp=time.time(),
    )
    return session_id


def zulip_send_topic_message(
    stream: Optional[str] = None,
    topic: Optional[str] = None,
    message: Optional[str] = None,
    session_user_email: Optional[str] = None,
) -> str:
    """Send to a Zulip topic, creating it if needed, and seed its session.

    When invoked from a Zulip conversation the current sender owns the target
    session. Other invocation surfaces must specify ``session_user_email`` so
    a later reply routes to the same per-user stream session.
    """
    stream = (stream or "").strip()
    topic = (topic or "").strip()
    message = message or ""
    if not stream or not topic or not message.strip():
        return json.dumps({"error": "'stream', 'topic', and 'message' are required"})

    current_email, current_name, profile = _current_zulip_session_user()
    owner_email = (session_user_email or current_email or "").strip()
    if not owner_email:
        return json.dumps(
            {
                "error": (
                    "session_user_email is required outside a Zulip conversation "
                    "so Hermes can seed the correct user's topic session"
                )
            }
        )
    owner_name = current_name if owner_email == current_email and current_name else owner_email

    try:
        import zulip
        from model_tools import _run_async
        from .adapter import _standalone_send_zulip

        site_url, bot_email, api_key, config, send_config = _load_zulip_tool_config()
        client = zulip.Client(site=site_url, email=bot_email, api_key=api_key)
        stream_result = client.get_stream_id(stream)
        if stream_result.get("result") != "success" or stream_result.get("stream_id") is None:
            raise ValueError(stream_result.get("msg") or f"Zulip stream {stream!r} was not found")
        stream_id = int(stream_result["stream_id"])

        result = _run_async(
            _standalone_send_zulip(send_config, f"{stream_id}:{topic}", message)
        )
        if not isinstance(result, dict) or not result.get("success"):
            raise ValueError((result or {}).get("error") or "Zulip send failed")
    except Exception as exc:
        logger.warning("zulip_send_topic_message failed: %s", exc)
        return json.dumps({"error": str(exc)})

    response = {
        "success": True,
        "platform": "zulip",
        "stream": stream,
        "topic": topic,
        "chat_id": f"{stream_id}:{topic}",
        "session_user_email": owner_email,
        "message_id": str(result.get("message_id") or ""),
    }
    try:
        response["session_id"] = _seed_topic_session(
            config=config,
            stream_id=stream_id,
            stream_name=stream,
            topic=topic,
            user_email=owner_email,
            user_name=owner_name,
            profile=profile,
            message=message,
            message_id=response["message_id"],
        )
        response["session_seeded"] = True
    except Exception as exc:
        # The external delivery has already succeeded. Surface the durable
        # session problem without claiming the message failed (and inviting a
        # retry that would post a duplicate to Zulip).
        logger.warning("Zulip topic message sent but session seed failed: %s", exc)
        response["session_seeded"] = False
        response["session_error"] = str(exc)
    return json.dumps(response)


_ZULIP_SEND_TOPIC_SCHEMA = {
    "name": "zulip_send_topic_message",
    "description": (
        "Send a message to a Zulip stream topic. Zulip creates a topic when its "
        "first message is sent. Hermes seeds the matching topic session so a "
        "future reply continues with this message in context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "stream": {"type": "string", "description": "Zulip stream name."},
            "topic": {"type": "string", "description": "New or existing Zulip topic."},
            "message": {"type": "string", "description": "Text to post to the topic."},
            "session_user_email": {
                "type": "string",
                "description": (
                    "Zulip email whose session is seeded. Omit in a Zulip "
                    "conversation; required from CLI or another platform."
                ),
            },
        },
        "required": ["stream", "topic", "message"],
    },
}


def _handle_zulip_send_topic_message(args, **_kw):
    return zulip_send_topic_message(
        stream=args.get("stream"),
        topic=args.get("topic"),
        message=args.get("message"),
        session_user_email=args.get("session_user_email"),
    )


def register_zulip_topic_message_tool(ctx) -> None:
    from .search_tool import _check_zulip_search_requirements

    ctx.register_tool(
        name="zulip_send_topic_message",
        toolset="zulip-history",
        schema=_ZULIP_SEND_TOPIC_SCHEMA,
        handler=_handle_zulip_send_topic_message,
        check_fn=_check_zulip_search_requirements,
        emoji="✉️",
    )
