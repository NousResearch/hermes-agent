from __future__ import annotations

import hashlib
import json
import io
import queue
import threading
from contextlib import redirect_stderr, redirect_stdout
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from hermes_cli.product_config import load_product_config, resolve_runtime_defaults
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_state import SessionDB
if TYPE_CHECKING:
    from run_agent import AIAgent


def _require_username(user: dict[str, Any]) -> str:
    username = str(user.get("preferred_username") or user.get("sub") or "").strip()
    if not username:
        raise ValueError("Signed-in user is missing a usable username")
    return username


def _product_model_route() -> dict[str, str]:
    product_config = load_product_config()
    route = product_config.get("models", {}).get("default_route", {})
    model_name = str(route.get("model") or "").strip()
    if not model_name:
        raise RuntimeError("No default model configured for product chat")
    provider = str(route.get("provider") or "custom").strip().lower() or "custom"
    base_url = str(route.get("base_url") or "").strip()
    api_mode = str(route.get("api_mode") or "chat_completions").strip().lower() or "chat_completions"
    route_config = {
        "model": model_name,
        "provider": provider,
        "api_mode": api_mode,
    }
    if base_url:
        route_config["base_url"] = base_url
    return route_config


def _product_runtime_toolsets() -> list[str]:
    product_config = load_product_config()
    configured_toolsets = product_config.get("tools", {}).get("hermes_toolsets", [])
    if isinstance(configured_toolsets, list):
        normalized = [str(toolset).strip() for toolset in configured_toolsets if str(toolset).strip()]
        if normalized:
            return normalized
    runtime_defaults = resolve_runtime_defaults(product_config)
    toolset = str(runtime_defaults.get("runtime_toolset") or "").strip()
    return [toolset] if toolset else []


def product_chat_session_id(user: dict[str, Any]) -> str:
    username = _require_username(user)
    digest = hashlib.sha1(username.encode("utf-8")).hexdigest()[:12]
    return f"product_{username}_{digest}"


def _conversation_for_agent(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_roles = {"user", "assistant", "system", "tool"}
    conversation: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in allowed_roles:
            continue
        entry: dict[str, Any] = {"role": role, "content": message.get("content")}
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message["tool_call_id"]
        if message.get("tool_name"):
            entry["tool_name"] = message["tool_name"]
        if isinstance(message.get("tool_calls"), list):
            entry["tool_calls"] = message["tool_calls"]
        conversation.append(entry)
    return conversation


def _visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in {"user", "assistant", "system"}:
            continue
        visible.append({"role": role, "content": str(message.get("content") or "")})
    return visible


def _load_session_messages(db: SessionDB, session_id: str) -> list[dict[str, Any]]:
    session = db.get_session(session_id)
    if not session:
        return []
    return db.get_messages_as_conversation(session_id)


def get_product_chat_session(user: dict[str, Any]) -> dict[str, Any]:
    session_id = product_chat_session_id(user)
    db = SessionDB()
    try:
        messages = _load_session_messages(db, session_id)
    finally:
        db.close()
    return {
        "session_id": session_id,
        "messages": _visible_messages(messages),
    }


def _build_agent(
    *,
    session_id: str,
    reasoning_callback: Any = None,
) -> tuple["AIAgent", SessionDB]:
    from run_agent import AIAgent

    route = _product_model_route()
    runtime_kwargs: dict[str, Any]
    if route.get("base_url"):
        runtime_kwargs = {
            "base_url": route["base_url"],
            "api_key": "product-local-route",
            "provider": route["provider"],
            "api_mode": route["api_mode"],
        }
    else:
        resolved = resolve_runtime_provider(requested=route["provider"])
        runtime_kwargs = {
            "base_url": str(resolved.get("base_url") or "").strip() or None,
            "api_key": str(resolved.get("api_key") or "").strip() or None,
            "provider": str(resolved.get("provider") or route["provider"]).strip() or route["provider"],
            "api_mode": str(resolved.get("api_mode") or route["api_mode"]).strip() or route["api_mode"],
            "acp_command": str(resolved.get("command") or "").strip() or None,
            "acp_args": list(resolved.get("args") or []),
        }
    db = SessionDB()
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        agent = AIAgent(
            base_url=runtime_kwargs.get("base_url"),
            api_key=runtime_kwargs.get("api_key"),
            provider=runtime_kwargs.get("provider"),
            api_mode=runtime_kwargs.get("api_mode"),
            acp_command=runtime_kwargs.get("acp_command"),
            acp_args=runtime_kwargs.get("acp_args"),
            model=route["model"],
            quiet_mode=True,
            enabled_toolsets=_product_runtime_toolsets(),
            session_id=session_id,
            session_db=db,
            platform="product",
            reasoning_callback=reasoning_callback,
        )
    return agent, db


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_product_chat_turn(user: dict[str, Any], user_message: str) -> Iterator[str]:
    message = user_message.strip()
    if not message:
        raise ValueError("User message must not be empty")

    session_id = product_chat_session_id(user)
    event_queue: queue.Queue[str | None] = queue.Queue()

    def _emit(event: str, payload: dict[str, Any]) -> None:
        event_queue.put(_sse_event(event, payload))

    def _worker() -> None:
        agent, db = _build_agent(
            session_id=session_id,
            reasoning_callback=lambda delta: _emit("reasoning", {"delta": str(delta or "")}),
        )
        try:
            history = _load_session_messages(db, session_id)
            _emit("start", {"session_id": session_id})
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                result = agent.run_conversation(
                    message,
                    conversation_history=_conversation_for_agent(history),
                    stream_callback=lambda delta: _emit("answer", {"delta": str(delta or "")}),
                    sync_honcho=False,
                )
            updated_messages = _load_session_messages(db, session_id)
            _emit(
                "final",
                {
                    "session_id": session_id,
                    "final_response": str(result.get("final_response") or ""),
                    "messages": _visible_messages(updated_messages),
                },
            )
        except Exception as exc:
            _emit("error", {"detail": str(exc)})
        finally:
            db.close()
            event_queue.put(None)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    while True:
        item = event_queue.get()
        if item is None:
            break
        yield item
