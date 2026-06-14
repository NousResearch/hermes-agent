"""Minimal Linear GraphQL client for the AgentSession platform plugin."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is a Hermes dependency
    httpx = None  # type: ignore[assignment]


DEFAULT_LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"


class LinearClient:
    """Small async client for Linear GraphQL mutations used by the MVP."""

    def __init__(self, token: str | None = None, *, endpoint: str = DEFAULT_LINEAR_GRAPHQL_URL):
        self.token = (token or os.getenv("LINEAR_ACCESS_TOKEN") or os.getenv("LINEAR_API_KEY") or "").strip()
        self.endpoint = endpoint

    async def graphql(self, query: str, variables: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx not installed")
        if not self.token:
            raise RuntimeError("Linear token not configured")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.endpoint,
                headers={"Authorization": self.token, "Content-Type": "application/json"},
                json={"query": query, "variables": variables or {}},
            )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            messages = []
            for error in data.get("errors") or []:
                if isinstance(error, dict):
                    message = str(error.get("message") or "").strip()
                    if message:
                        messages.append(message)
            detail = "; ".join(messages[:3]) or json.dumps(data.get("errors"), default=str)[:500]
            raise RuntimeError(f"Linear GraphQL error: {detail}")
        return data

    async def create_agent_activity(
        self,
        *,
        agent_session_id: str,
        content: Dict[str, Any],
        ephemeral: bool | None = None,
        signal: str | None = None,
        signal_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        mutation = """
        mutation HermesAgentActivityCreate($input: AgentActivityCreateInput!) {
          agentActivityCreate(input: $input) {
            success
            agentActivity { id }
          }
        }
        """
        data = await self.graphql(
            mutation,
            {
                "input": {
                    "agentSessionId": agent_session_id,
                    "content": content,
                    **({} if ephemeral is None else {"ephemeral": ephemeral}),
                    **({} if signal is None else {"signal": signal}),
                    **({} if signal_metadata is None else {"signalMetadata": signal_metadata}),
                }
            },
        )
        result = (data.get("data") or {}).get("agentActivityCreate") or {}
        if not result.get("success"):
            raise RuntimeError("Linear agentActivityCreate failed")
        return result.get("agentActivity") or {}

    async def update_agent_session(
        self,
        *,
        agent_session_id: str,
        plan: list[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        mutation = """
        mutation HermesAgentSessionUpdate($id: String!, $input: AgentSessionUpdateInput!) {
          agentSessionUpdate(id: $id, input: $input) {
            success
            agentSession { id status plan }
          }
        }
        """
        input_data: Dict[str, Any] = {}
        if plan is not None:
            input_data["plan"] = plan
        data = await self.graphql(
            mutation,
            {"id": agent_session_id, "input": input_data},
        )
        result = (data.get("data") or {}).get("agentSessionUpdate") or {}
        if not result.get("success"):
            raise RuntimeError("Linear agentSessionUpdate failed")
        return result.get("agentSession") or {}
