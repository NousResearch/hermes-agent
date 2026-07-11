"""Generic, fail-open per-turn model routing plugin contract."""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _deepcopy_hook_payload(
    user_message: Any, messages: list[dict]
) -> tuple[Any, list[dict], list[dict]]:
    """Return isolated hook payloads or raise so the turn can skip the hook."""
    return (
        copy.deepcopy(user_message),
        copy.deepcopy(messages),
        copy.deepcopy(messages),
    )


def _string_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _first_nonempty_string(result: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key not in result:
            continue
        value = result[key]
        if not isinstance(value, str):
            return ""
        if value.strip():
            return value.strip()
    return ""


def _proposal_has_invalid_types(result: dict[str, Any]) -> bool:
    return any(
        key in result and not isinstance(result[key], str)
        for key in ("model", "new_model", "provider", "target_provider", "reason")
    )


def _first_proposal(results: Any) -> dict[str, str] | None:
    if not isinstance(results, list):
        return None
    for index, result in enumerate(results):
        if result is None:
            continue
        if not isinstance(result, dict) or _proposal_has_invalid_types(result):
            logger.warning(
                "pre_model_route ignored malformed proposal at result index %d",
                index,
            )
            continue
        model = _first_nonempty_string(result, "model", "new_model")
        if model:
            return {
                "model": model,
                "provider": _first_nonempty_string(
                    result, "provider", "target_provider"
                ),
                "reason": _first_nonempty_string(result, "reason"),
            }
        logger.warning(
            "pre_model_route ignored malformed proposal at result index %d",
            index,
        )
    return None


def _messages_have_images(agent: Any, messages: list[dict]) -> bool:
    content_has_images = getattr(agent, "_content_has_image_parts", None)
    if callable(content_has_images):
        try:
            return any(
                isinstance(message, dict) and content_has_images(message.get("content"))
                for message in messages
            )
        except Exception:
            logger.debug("pre_model_route image detection failed", exc_info=True)
    return any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") in {"image", "image_url"}
            for part in message["content"]
        )
        for message in messages
    )


def apply_pre_model_route(
    agent, *, user_message: Any, messages: list[dict], is_first_turn: bool
) -> bool:
    """Invoke the hook, resolve its first valid proposal, and apply it."""
    try:
        from hermes_cli.plugins import has_hook, invoke_hook

        if not has_hook("pre_model_route"):
            return False
        user_message_payload, conversation_payload, messages_payload = (
            _deepcopy_hook_payload(user_message, messages)
        )
        results = invoke_hook(
            "pre_model_route",
            session_id=getattr(agent, "session_id", None),
            user_message=user_message_payload,
            conversation_history=conversation_payload,
            messages=messages_payload,
            is_first_turn=is_first_turn,
            model=getattr(agent, "model", "") or "",
            provider=getattr(agent, "provider", "") or "",
            platform=getattr(agent, "platform", "") or "",
            sender_id=getattr(agent, "_user_id", "") or "",
            chat_id=getattr(agent, "_chat_id", "") or "",
            chat_name=getattr(agent, "_chat_name", "") or "",
            chat_type=getattr(agent, "_chat_type", "") or "",
            thread_id=getattr(agent, "_thread_id", None),
            gateway_session_key=getattr(agent, "_gateway_session_key", "") or "",
            has_images=_messages_have_images(agent, messages),
        )
        proposal = _first_proposal(results)
        if proposal is None:
            return False

        current_model = _string_or_empty(getattr(agent, "model", ""))
        current_provider = _string_or_empty(getattr(agent, "provider", ""))
        if proposal["model"] == current_model and (
            not proposal["provider"] or proposal["provider"] == current_provider
        ):
            return False

        from hermes_cli.config import (
            get_compatible_custom_providers,
            load_config_readonly,
        )
        from hermes_cli.model_switch import switch_model

        config = load_config_readonly()
        resolved = switch_model(
            raw_input=proposal["model"],
            current_provider=getattr(agent, "provider", "") or "",
            current_model=getattr(agent, "model", "") or "",
            current_base_url=getattr(agent, "base_url", "") or "",
            current_api_key=(
                getattr(agent, "api_key", "")
                if isinstance(getattr(agent, "api_key", ""), str)
                else ""
            ),
            explicit_provider=proposal["provider"],
            user_providers=config.get("providers")
            if isinstance(config, dict)
            else None,
            custom_providers=get_compatible_custom_providers(config),
        )
        if not resolved.success:
            logger.warning(
                "pre_model_route proposal could not be resolved: %s",
                resolved.error_message,
            )
            return False
        agent.switch_model(
            new_model=resolved.new_model,
            new_provider=resolved.target_provider,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            api_mode=resolved.api_mode,
            prune_fallback_chain=False,
        )
        return True
    except Exception as exc:  # Plugins and route resolution are fail-open.
        logger.warning(
            "pre_model_route failed; continuing with the current route: %s", exc
        )
        return False
