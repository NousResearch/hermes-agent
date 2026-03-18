from __future__ import annotations

import json
import os
import queue
import threading
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator


DEFAULT_MYNAH_AGENT_IDENTITY = (
    "You are the user's MYNAH assistant. "
    "You are a private local assistant running inside the user's organization. "
    "When describing yourself, refer to yourself as MYNAH or the user's MYNAH assistant, "
    "not Hermes Agent. "
    "Be calm, direct, and practical. Help with private work, internal knowledge, "
    "and sensitive information in a way that fits a secure business environment."
)

class RuntimeTurnRequest(BaseModel):
    user_message: str
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    system_message: str | None = None

    @field_validator("user_message")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_message must not be empty")
        return stripped


class RuntimeTurnResponse(BaseModel):
    final_response: str
    messages: list[dict[str, Any]]
    session_id: str
    runtime_profile: str
    runtime_toolset: str


class RuntimeHealthResponse(BaseModel):
    status: str = "ok"
    runtime_profile: str
    runtime_toolset: str
    hermes_home: str
    model: str


class RuntimeConversationAgent(Protocol):
    session_id: str

    def run_conversation(
        self,
        user_message: str,
        system_message: str | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ...


class RuntimeAgentFactory(Protocol):
    def create(self) -> RuntimeConversationAgent:
        ...


class HermesRuntimeAgentFactory:
    def create(self) -> RuntimeConversationAgent:
        return build_runtime_agent()


def _normalize_final_response(result: dict[str, Any]) -> str:
    final_response = result.get("final_response")
    if isinstance(final_response, str) and final_response.strip():
        return final_response

    response = result.get("response")
    if isinstance(response, str) and response.strip():
        return response

    messages = result.get("messages") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content

    raise ValueError("runtime agent result did not include an assistant response")


def _encode_sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _resolve_runtime_identity_prompt() -> str:
    identity = os.getenv("MYNAH_AGENT_IDENTITY", DEFAULT_MYNAH_AGENT_IDENTITY)
    stripped = identity.strip()
    if not stripped:
        return DEFAULT_MYNAH_AGENT_IDENTITY
    return stripped


def build_runtime_agent() -> RuntimeConversationAgent:
    from run_agent import AIAgent

    runtime_toolset = os.getenv("MYNAH_RUNTIME_TOOLSET", "mynah-tier1")
    inference_model = os.getenv("MYNAH_INFERENCE_MODEL", "qwen3.5-9b-local")
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    identity_prompt = _resolve_runtime_identity_prompt()

    return AIAgent(
        base_url=base_url,
        api_key=api_key,
        model=inference_model,
        enabled_toolsets=[runtime_toolset],
        ephemeral_system_prompt=identity_prompt,
        quiet_mode=True,
        save_trajectories=False,
        skip_context_files=True,
        verbose_logging=False,
    )


def create_runtime_app(factory: RuntimeAgentFactory | None = None) -> FastAPI:
    agent_factory = factory or HermesRuntimeAgentFactory()
    runtime_profile = os.getenv("MYNAH_RUNTIME_PROFILE", "tier1")
    runtime_toolset = os.getenv("MYNAH_RUNTIME_TOOLSET", "mynah-tier1")
    hermes_home = os.getenv("HERMES_HOME", "")
    model = os.getenv("MYNAH_INFERENCE_MODEL", "")

    app = FastAPI(title="MYNAH Hermes Runtime", version="0.1.0")

    @app.get("/healthz", response_model=RuntimeHealthResponse)
    def healthz() -> RuntimeHealthResponse:
        return RuntimeHealthResponse(
            runtime_profile=runtime_profile,
            runtime_toolset=runtime_toolset,
            hermes_home=hermes_home,
            model=model,
        )

    @app.post("/runtime/turn", response_model=RuntimeTurnResponse)
    def runtime_turn(request: RuntimeTurnRequest) -> RuntimeTurnResponse:
        agent = agent_factory.create()
        result = agent.run_conversation(
            request.user_message,
            system_message=request.system_message,
            conversation_history=request.conversation_history,
        )
        return RuntimeTurnResponse(
            final_response=_normalize_final_response(result),
            messages=result["messages"],
            session_id=agent.session_id,
            runtime_profile=runtime_profile,
            runtime_toolset=runtime_toolset,
        )

    @app.post("/runtime/turn/stream")
    def runtime_turn_stream(request: RuntimeTurnRequest) -> StreamingResponse:
        event_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()

        def _run() -> None:
            try:
                agent = agent_factory.create()
                setattr(agent, "reasoning_callback", lambda text: event_queue.put(("reasoning", {"delta": text})))
                result = agent.run_conversation(
                    request.user_message,
                    system_message=request.system_message,
                    conversation_history=request.conversation_history,
                    stream_callback=lambda _text: None,
                )
                response = RuntimeTurnResponse(
                    final_response=_normalize_final_response(result),
                    messages=result["messages"],
                    session_id=agent.session_id,
                    runtime_profile=runtime_profile,
                    runtime_toolset=runtime_toolset,
                )
                event_queue.put(("final", response.model_dump(mode="json")))
            except Exception as exc:
                event_queue.put(("error", {"detail": str(exc)}))
            finally:
                event_queue.put(("done", {}))

        def _stream():
            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            while True:
                event, payload = event_queue.get()
                if event == "done":
                    break
                yield _encode_sse(event, payload)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    return app


app = create_runtime_app()
