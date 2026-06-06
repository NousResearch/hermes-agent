"""Minimal Linear GraphQL client for the AgentSession platform plugin."""

from __future__ import annotations

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
            raise RuntimeError("Linear GraphQL error")
        return data

    async def create_agent_activity(self, *, agent_session_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
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
            {"input": {"agentSessionId": agent_session_id, "content": content}},
        )
        result = (data.get("data") or {}).get("agentActivityCreate") or {}
        if not result.get("success"):
            raise RuntimeError("Linear agentActivityCreate failed")
        return result.get("agentActivity") or {}
