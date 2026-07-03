"""Linear Agent Activity helpers."""

from __future__ import annotations

from typing import Any


AGENT_ACTIVITY_CREATE_MUTATION = """
mutation AgentActivityCreate($input: AgentActivityCreateInput!) {
  agentActivityCreate(input: $input) {
    success
    agentActivity {
      id
    }
  }
}
"""


def build_activity_input(
    agent_session_id: str,
    activity_type: str,
    body: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Build the GraphQL input for Linear's agentActivityCreate mutation.

    Linear expects activity details under ``input.content``.  The older flat
    shape (``input.type`` / ``input.body``) is rejected by Linear's GraphQL
    schema, which prevents acknowledgements and final responses from posting.
    """
    content: dict[str, Any] = {"type": activity_type}
    if activity_type in {"thought", "elicitation", "response", "error"}:
        content["body"] = body or ""
    elif body:
        content["body"] = body
    content.update({k: v for k, v in extra.items() if v is not None})
    return {
        "agentSessionId": agent_session_id,
        "content": content,
    }
