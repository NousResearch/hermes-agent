from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

from hermes_state import SessionDB


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if isinstance(value, str) and value.strip():
        return value
    return default


def _runtime_defaults() -> dict[str, str]:
    return {
        "runtime_mode": "product",
        "runtime_toolsets": "memory,session_search",
        "inference_model": "qwen3.5-9b-local",
    }


def _runtime_toolsets() -> list[str]:
    raw_toolsets = str(os.getenv("HERMES_PRODUCT_TOOLSETS", "")).strip()
    if raw_toolsets:
        normalized = [item.strip() for item in raw_toolsets.split(",") if item.strip()]
        if normalized:
            return normalized
    raw_default_toolsets = _runtime_defaults()["runtime_toolsets"]
    return [item.strip() for item in raw_default_toolsets.split(",") if item.strip()]


class RuntimeTurnRequest(BaseModel):
    user_message: str

    @field_validator("user_message")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_message must not be empty")
        return stripped


class RuntimeMessage(BaseModel):
    role: str
    content: str


class RuntimeSessionResponse(BaseModel):
    session_id: str
    messages: list[RuntimeMessage]
    runtime_mode: str
    runtime_toolsets: list[str]


class RuntimeTurnResponse(RuntimeSessionResponse):
    final_response: str


class RuntimeHealthResponse(BaseModel):
    status: str = "ok"
    runtime_mode: str
    runtime_toolsets: list[str]
    hermes_home: str
    model: str
    session_id: str


def _encode_sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _load_runtime_soul() -> str:
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    soul_path = hermes_home / "SOUL.md"
    if soul_path.exists():
        soul = soul_path.read_text(encoding="utf-8").strip()
        if soul:
            return soul
    raise RuntimeError(f"Product runtime requires a non-empty SOUL.md at {soul_path}")


def _session_id() -> str:
    session_id = str(os.getenv("MYNAH_PRODUCT_SESSION_ID", "")).strip()
    if not session_id:
        raise RuntimeError("MYNAH_PRODUCT_SESSION_ID must be configured for the product runtime")
    return session_id


def _visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    visible: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in {"user", "assistant", "system"}:
            continue
        visible.append({"role": role, "content": str(message.get("content") or "")})
    return visible


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


def _load_session_messages(db: SessionDB, session_id: str) -> list[dict[str, Any]]:
    session = db.get_session(session_id)
    if not session:
        return []
    return db.get_messages_as_conversation(session_id)


def build_runtime_agent(db: SessionDB, session_id: str, *, reasoning_callback: Any = None):
    from run_agent import AIAgent

    provider = _env_or_default("HERMES_PRODUCT_PROVIDER", "custom").strip().lower() or "custom"
    api_mode = _env_or_default("HERMES_PRODUCT_API_MODE", "chat_completions").strip().lower() or "chat_completions"
    model = _env_or_default("HERMES_PRODUCT_MODEL", _runtime_defaults()["inference_model"]).strip()
    base_url = str(os.getenv("OPENAI_BASE_URL", "")).strip() or None
    api_key = str(os.getenv("OPENAI_API_KEY", "")).strip() or None

    return AIAgent(
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        api_mode=api_mode,
        model=model,
        quiet_mode=True,
        enabled_toolsets=_runtime_toolsets(),
        session_id=session_id,
        session_db=db,
        platform="product-runtime",
        reasoning_callback=reasoning_callback,
        skip_context_files=False,
        save_trajectories=False,
        verbose_logging=False,
    )


def create_product_runtime_app() -> FastAPI:
    defaults = _runtime_defaults()
    runtime_mode = _env_or_default("HERMES_PRODUCT_RUNTIME_MODE", defaults["runtime_mode"])
    runtime_toolsets = _runtime_toolsets()
    hermes_home = os.getenv("HERMES_HOME", "")
    model = _env_or_default("HERMES_PRODUCT_MODEL", defaults["inference_model"])
    session_id = _session_id()
    _load_runtime_soul()

    app = FastAPI(title="Hermes Core Product Runtime", version="0.1.0")

    @app.get("/healthz", response_model=RuntimeHealthResponse)
    def healthz() -> RuntimeHealthResponse:
        return RuntimeHealthResponse(
            runtime_mode=runtime_mode,
            runtime_toolsets=runtime_toolsets,
            hermes_home=hermes_home,
            model=model,
            session_id=session_id,
        )

    @app.get("/runtime/session", response_model=RuntimeSessionResponse)
    def runtime_session() -> RuntimeSessionResponse:
        db = SessionDB()
        try:
            messages = _load_session_messages(db, session_id)
        finally:
            db.close()
        return RuntimeSessionResponse(
            session_id=session_id,
            messages=[RuntimeMessage(**message) for message in _visible_messages(messages)],
            runtime_mode=runtime_mode,
            runtime_toolsets=runtime_toolsets,
        )

    @app.post("/runtime/turn", response_model=RuntimeTurnResponse)
    def runtime_turn(request: RuntimeTurnRequest) -> RuntimeTurnResponse:
        db = SessionDB()
        try:
            agent = build_runtime_agent(db, session_id)
            history = _load_session_messages(db, session_id)
            result = agent.run_conversation(
                request.user_message,
                conversation_history=_conversation_for_agent(history),
                sync_honcho=False,
            )
            updated_messages = _load_session_messages(db, session_id)
        finally:
            db.close()
        final_response = str(result.get("final_response") or result.get("response") or "")
        return RuntimeTurnResponse(
            final_response=final_response,
            session_id=session_id,
            messages=[RuntimeMessage(**message) for message in _visible_messages(updated_messages)],
            runtime_mode=runtime_mode,
            runtime_toolsets=runtime_toolsets,
        )

    @app.post("/runtime/turn/stream")
    def runtime_turn_stream(request: RuntimeTurnRequest) -> StreamingResponse:
        event_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()

        def _run() -> None:
            db = SessionDB()
            try:
                agent = build_runtime_agent(
                    db,
                    session_id,
                    reasoning_callback=lambda text: event_queue.put(("reasoning", {"delta": str(text or "")})),
                )
                setattr(
                    agent,
                    "reasoning_callback",
                    lambda text: event_queue.put(("reasoning", {"delta": str(text or "")})),
                )
                history = _load_session_messages(db, session_id)
                event_queue.put(("start", {"session_id": session_id}))
                result = agent.run_conversation(
                    request.user_message,
                    conversation_history=_conversation_for_agent(history),
                    stream_callback=lambda text: event_queue.put(("answer", {"delta": str(text or "")})),
                    sync_honcho=False,
                )
                updated_messages = _load_session_messages(db, session_id)
                event_queue.put(
                    (
                        "final",
                        RuntimeTurnResponse(
                            final_response=str(result.get("final_response") or result.get("response") or ""),
                            session_id=session_id,
                            messages=[RuntimeMessage(**message) for message in _visible_messages(updated_messages)],
                            runtime_mode=runtime_mode,
                            runtime_toolsets=runtime_toolsets,
                        ).model_dump(mode="json"),
                    )
                )
            except Exception as exc:
                event_queue.put(("error", {"detail": str(exc)}))
            finally:
                db.close()
                event_queue.put(("done", {}))

        def _stream() -> Iterator[bytes]:
            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            while True:
                event, payload = event_queue.get()
                if event == "done":
                    break
                yield _encode_sse(event, payload)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app


if os.getenv("MYNAH_PRODUCT_SESSION_ID"):
    app = create_product_runtime_app()
else:  # pragma: no cover
    app = FastAPI(title="Hermes Core Product Runtime", version="0.1.0")


def main() -> int:
    host = _env_or_default("MYNAH_RUNTIME_HOST", "0.0.0.0")
    port = int(_env_or_default("MYNAH_RUNTIME_PORT", "8091"))
    uvicorn.run(create_product_runtime_app(), host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
