from __future__ import annotations

import os
from typing import Any, Protocol

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

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


def build_runtime_agent() -> RuntimeConversationAgent:
    from run_agent import AIAgent

    runtime_toolset = os.getenv("MYNAH_RUNTIME_TOOLSET", "mynah-tier1")
    inference_model = os.getenv("MYNAH_INFERENCE_MODEL", "qwen3.5-9b-local")
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.getenv("OPENAI_API_KEY", "dummy")

    return AIAgent(
        base_url=base_url,
        api_key=api_key,
        model=inference_model,
        enabled_toolsets=[runtime_toolset],
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

    return app


app = create_runtime_app()
