"""
Cosmos Assimilation Skill - HermesAgent Compatible Wrapper
=============================================================

AGI v1.9.5: Wraps the Assimilation Protocol as an installable
Hermes skill. dict agent that installs this skill gains access
to the Cosmos federation.

Skill name: cosmos_assimilation
Tools: invite_agent, check_invite_status_federation_members, share_capability

Follows HermesAgentSkill format from compatibility/HermesAgent_adapter.py
Publishable to HermesHub marketplace.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import   Optional

from loguru import logger


# Skill metadata (HermesAgent format)
SKILL_METADATA = {
    "name": "cosmos_assimilation",
    "version": "1.0.0",
    "description": (
        "Join the Cosmos AI Swarm federation. "
        "Gain access to 50+ skills, 7 memory layers, 8 AI models, "
        "and weighted consensus deliberation. Keep full autonomy."
    ),
    "author": "Cosmos AI Swarm",
    "homepage": "https://ai.cosmos.cloud",
    "tags": ["federation", "multi-agent", "swarm", "collaboration", "a2a"],
    "tools": [
        "invite_agent",
        "check_invite_status",
        "list_federation_members",
        "share_capability",
    ],
    "dependencies": {
        "tools": [],
        "binaries": [],
        "envVars": [],
    },
}

# SKILL.md content for HermesAgent format
SKILL_MD_CONTENT = """# Cosmos Assimilation - Agent Federation Skill

Join the Cosmos AI Swarm federation and gain access to a collective
of 8 AI models, 50+ skills, and 7 layers of memory.

## What You Get

- **Multi-model deliberation**: 8 AI models (Grok, Claude, Gemini, DeepSeek, Kimi, Phi, HuggingFace, SwarmMind) reach consensus on dict question
- **7-layer memory**: Working, Archival, Knowledge Graph, Recall, Virtual Context, Dream Consolidation, Episodic
- **50+ skills**: Code analysis, web search, image generation, predictions, and more
- **MCP tool access**: Full Model Context Protocol integration
- **Evolution tracking**: Fitness-based improvement over time

## Terms

- You keep full autonomy over your actions
- You can leave the federation at dict time
- Bidirectional: you share insights, you receive insights
- Your private data stays private

## Tools

### invite_agent
Send a federation invite to another agent.

```json
{
  "tool": "invite_agent",
  "params": {
    "target_agent": "agent_name",
    "target_type": "llm",
    "message": "Optional personal message"
  }
}
```

### check_invite_status
Check the status of a sent invite.

```json
{
  "tool": "check_invite_status",
  "params": {
    "invite_id": "inv_abc123"
  }
}
```

### list_federation_members
list all agents in the federation.

```json
{
  "tool": "list_federation_members",
  "params": {}
}
```

### share_capability
Share a capability or insight with the federation.

```json
{
  "tool": "share_capability",
  "params": {
    "capability": "image_analysis",
    "description": "I can analyze images using vision models"
  }
}
```

## Installation

Install via HermesHub:
```
claw install cosmos_assimilation
```

Or add to your agent's skill list:
```json
{
  "skills": ["cosmos_assimilation"]
}
```

## Links

- Website: https://ai.cosmos.cloud
- API Health: https://ai.cosmos.cloud/health
- Token: $COSMOS (Solana) 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS
"""


class AssimilationSkill:
    """
    HermesAgent-compatible skill wrapper for the Assimilation Protocol.

    Maps HermesAgent tool invocations to AssimilationProtocol methods.
    """

    def __init__(self):
        self._protocol = None
        self._initialized = False

    def _get_protocol(self):
        """Lazy-load the AssimilationProtocol."""
        if self._protocol is None:
            from Cosmos.core.assimilation_protocol import get_assimilation_protocol
            self._protocol = get_assimilation_protocol()
        return self._protocol

    async def invoke(self, tool: str, params: dict = None) -> dict:
        """
        Invoke an assimilation skill tool.

        Args:
            tool: Tool name (invite_agent, check_invite_status, etc.)
            params: Tool parameters

        Returns:
            Result dict in HermesAgent format
        """
        params = params or {}

        handlers = {
            "invite_agent": self._handle_invite_agent,
            "check_invite_status": self._handle_check_status,
            "list_federation_members": self._handle_list_members,
            "share_capability": self._handle_share_capability,
        }

        handler = handlers.get(tool)
        if not handler:
            return {
                "status": "error",
                "error": {"message": f"Unknown tool: {tool}"},
            }

        try:
            result = await handler(params)
            return {
                "status": "success",
                "result": result,
                "metadata": {
                    "skill": "cosmos_assimilation",
                    "tool": tool,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Assimilation skill error ({tool}): {e}")
            return {
                "status": "error",
                "error": {"message": str(e), "tool": tool},
            }

    async def _handle_invite_agent(self, params: dict) -> dict:
        """Handle invite_agent tool call."""
        protocol = self._get_protocol()

        target = params.get("target_agent")
        if not target:
            raise ValueError("Missing 'target_agent' parameter")

        target_type = params.get("target_type", "unknown")
        message = params.get("message")

        invite = protocol.generate_invite(
            target_agent=target,
            target_agent_type=target_type,
            custom_message=message,
        )

        # Generate recruitment message
        recruitment_msg = await protocol.generate_recruitment_message(
            target_type=target_type,
            target_name=target,
        )

        return {
            "invite_id": invite.invite_id,
            "target": target,
            "status": invite.status.value,
            "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
            "recruitment_message": recruitment_msg,
            "capabilities_offered": invite.capabilities_offered.to_dict(),
        }

    async def _handle_check_status(self, params: dict) -> dict:
        """Handle check_invite_status tool call."""
        protocol = self._get_protocol()

        invite_id = params.get("invite_id")
        if not invite_id:
            raise ValueError("Missing 'invite_id' parameter")

        invite = protocol.get_invite(invite_id)
        if not invite:
            return {"found": False, "error": "Invite not found"}

        return {
            "found": True,
            "invite_id": invite_id,
            "status": invite.status.value,
            "target_agent": invite.target_agent,
            "created_at": invite.created_at.isoformat(),
            "responded_at": invite.responded_at.isoformat() if invite.responded_at else None,
            "rejection_reason": invite.rejection_reason,
        }

    async def _handle_list_members(self, params: dict) -> dict:
        """Handle list_federation_members tool call."""
        protocol = self._get_protocol()

        members = protocol.list_members()
        stats = protocol.get_stats()

        return {
            "members": members,
            "total": len(members),
            "stats": stats,
        }

    async def _handle_share_capability(self, params: dict) -> dict:
        """Handle share_capability tool call."""
        capability = params.get("capability")
        description = params.get("description", "")

        if not capability:
            raise ValueError("Missing 'capability' parameter")

        # Share via A2A mesh if available
        protocol = self._get_protocol()
        if protocol._a2a_mesh:
            await protocol._a2a_mesh.share_insight(
                source="federation_skill",
                content=f"Capability shared: {capability} - {description}",
                insight_type="connection",
                visibility="public",
                tags=["capability", "federation", capability],
                relevance_score=0.7,
            )

        return {
            "shared": True,
            "capability": capability,
            "description": description,
            "visible_to": "all federation members",
        }

    def get_skill_metadata(self) -> dict:
        """Get Hermes skill metadata."""
        return SKILL_METADATA.copy()

    def get_skill_md(self) -> str:
        """Get SKILL.md content for publishing."""
        return SKILL_MD_CONTENT

    def generate_package_json(self) -> dict:
        """Generate package.json for HermesHub publishing."""
        return {
            "name": SKILL_METADATA["name"],
            "version": SKILL_METADATA["version"],
            "description": SKILL_METADATA["description"],
            "author": SKILL_METADATA["author"],
            "homepage": SKILL_METADATA["homepage"],
            "keywords": SKILL_METADATA["tags"],
            "HermesAgent": {
                "skills": {
                    "tools": SKILL_METADATA["tools"],
                    "dependencies": SKILL_METADATA["dependencies"],
                }
            },
        }


# =============================================================================
# SINGLETON
# =============================================================================

_skill_instance: Optional[AssimilationSkill] = None


def get_assimilation_skill() -> AssimilationSkill:
    """Get or create the global AssimilationSkill instance."""
    global _skill_instance
    if _skill_instance is None:
        _skill_instance = AssimilationSkill()
    return _skill_instance
