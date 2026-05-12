"""Small Memori SDK wrapper used by the Hermes provider."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_TIMEOUT_SECS = 30
MEMORI_PLATFORM = "hermes"


class MemoriApiError(RuntimeError):
    """Raised when a Memori SDK request fails."""


def json_dumps_trace(trace: dict[str, Any]) -> str:
    return json.dumps(trace, ensure_ascii=False, default=str)


class MemoriAgentClient:
    """Thin wrapper around the Memori Python SDK for Hermes."""

    def __init__(
        self,
        *,
        api_key: str,
        entity_id: str,
        project_id: str,
        process_id: str | None = None,
        base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECS,
    ) -> None:
        try:
            from memori import Memori
        except ModuleNotFoundError as exc:
            missing = exc.name or "memori"
            raise RuntimeError(
                f"Memori SDK dependency missing: {missing}. "
                "Run: pip install memori"
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "Memori SDK could not be imported. "
                "Run: pip install memori"
            ) from exc

        self.entity_id = entity_id
        self.project_id = project_id
        self.memori = Memori(api_key=api_key, base_url=base_url).attribution(
            entity_id,
            process_id,
        )
        self.memori.config.request_secs_timeout = timeout

    def capture_turn(
        self,
        *,
        user_content: str,
        assistant_content: str,
        session_id: str,
        platform: str,
        trace: dict[str, Any] | None = None,
    ) -> None:
        del platform
        try:
            payload = {
                "user_content": user_content,
                "assistant_content": assistant_content,
                "project_id": self.project_id,
                "session_id": session_id,
                "platform": MEMORI_PLATFORM,
            }
            if trace:
                payload["trace"] = trace
            try:
                self.memori.capture_agent_turn(**payload)
            except TypeError:
                if not trace:
                    raise
                fallback_payload = dict(payload)
                fallback_payload.pop("trace", None)
                fallback_payload["assistant_content"] = (
                    f"{assistant_content}\n\n"
                    "<hermes_tool_trace>\n"
                    f"{json_dumps_trace(trace)}\n"
                    "</hermes_tool_trace>"
                )
                self.memori.capture_agent_turn(**fallback_payload)
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def agent_recall(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.memori.agent_recall(
                query=params.get("query"),
                date_start=params.get("date_start") or params.get("dateStart"),
                date_end=params.get("date_end") or params.get("dateEnd"),
                project_id=params.get("project_id")
                or params.get("projectId")
                or self.project_id,
                session_id=params.get("session_id") or params.get("sessionId"),
                signal=params.get("signal"),
                source=params.get("source"),
            )
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def agent_recall_summary(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.memori.agent_recall_summary(
                date_start=params.get("date_start") or params.get("dateStart"),
                date_end=params.get("date_end") or params.get("dateEnd"),
                project_id=params.get("project_id")
                or params.get("projectId")
                or self.project_id,
                session_id=params.get("session_id") or params.get("sessionId"),
            )
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def quota(self) -> dict[str, Any]:
        try:
            return self.memori.agent.default_api.get("sdk/quota")
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def signup(self, email: str) -> dict[str, Any]:
        try:
            return self.memori.agent.default_api.post("sdk/account", {"email": email})
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc

    def feedback(self, content: str) -> dict[str, Any]:
        try:
            self.memori.agent_feedback(content)
            return {}
        except Exception as exc:  # noqa: BLE001
            raise MemoriApiError(str(exc)) from exc
