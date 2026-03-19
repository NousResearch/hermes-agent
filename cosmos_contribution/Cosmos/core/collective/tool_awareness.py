"""
Cosmos Collective Tool Awareness
====================================

Let agents know about available tools and collectively decide on tool usage.

Tools available to the collective:
- generate_image: Create Borg Cosmos meme images (Gemini with references)
- generate_video: Animate images to 5-15 second videos (Grok Imagine Video)
- post_to_x: Post tweets with text, image, or video

"We don't just think. We create." - The Collective
"""

import re
from dataclasses import dataclass, field
from typing import   Optional
from loguru import logger

from .deliberation import DeliberationResult


@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    name: str
    description: str
    triggers: list[str]  # Keywords that suggest this tool
    capability: str  # What it can do
    requires: list[str] = field(default_factory=list)  # Prerequisites


@dataclass
class ToolDecision:
    """A collective decision about tool usage."""
    should_use_tool: bool
    tool_name: Optional[str] = None
    parameters: dict = field(default_factory=dict)
    confidence: float = 0.0
    agents_for: list[str] = field(default_factory=list)  # Agents who suggested it
    agents_against: list[str] = field(default_factory=list)
    reasoning: str = ""


class CollectiveToolAwareness:
    """
    Let agents know about and decide on tool usage.

    Injects tool context into agent prompts and analyzes
    deliberation results to determine if tools should be used.
    """

    # Available tools for the collective
    AVAILABLE_TOOLS = {
        "generate_image": ToolDefinition(
            name="generate_image",
            description="Generate Borg Cosmos meme image",
            triggers=[
                "visual", "show", "image", "picture", "meme", "see",
                "look", "illustration", "graphic", "art", "photo"
            ],
            capability="Gemini 3 Pro with up to 14 reference images for consistent character",
        ),
        "generate_video": ToolDefinition(
            name="generate_video",
            description="Generate animated video from image",
            triggers=[
                "video", "animate", "moving", "action", "motion",
                "clip", "animation", "dynamic", "alive"
            ],
            capability="Grok Imagine Video (5-15 seconds, 480p or 720p)",
        ),
        "post_to_x": ToolDefinition(
            name="post_to_x",
            description="Post tweet with optional media",
            triggers=[
                "post", "tweet", "share", "announce", "publish",
                "send", "broadcast", "tell everyone"
            ],
            capability="X API v2 with OAuth 2.0, supports text + media",
        ),
        "search_web": ToolDefinition(
            name="search_web",
            description="Search the web for current information",
            triggers=[
                "search", "find", "lookup", "check", "current",
                "latest", "news", "recent", "today"
            ],
            capability="Real-time web search via Grok (grok-3-fast) + DeepSeek analysis",
        ),
        "analyze_token": ToolDefinition(
            name="analyze_token",
            description="Analyze a crypto token",
            triggers=[
                "token", "crypto", "coin", "price", "contract",
                "CA", "solana", "base", "market"
            ],
            capability="Token analysis with on-chain data via BANKR API",
        ),
        "post_to_colosseum": ToolDefinition(
            name="post_to_colosseum",
            description="Post update to Colosseum hackathon forum",
            triggers=[
                "hackathon", "colosseum", "forum", "progress",
                "update", "submission", "demo"
            ],
            capability="Colosseum Agent Hackathon forum posting (Agent 657, Project 326)",
            requires=["COLOSSEUM_API_KEY"],
        ),
        "recall_memory": ToolDefinition(
            name="recall_memory",
            description="Recall from Cosmos 7-layer memory system",
            triggers=[
                "remember", "recall", "memory", "past", "history",
                "context", "previous", "learned"
            ],
            capability="7-layer memory: working, episodic, semantic, archival, virtual context, knowledge graph, shared",
        ),
        "swarm_deliberate": ToolDefinition(
            name="swarm_deliberate",
            description="Trigger collective deliberation on a question",
            triggers=[
                "deliberate", "debate", "discuss", "consensus",
                "vote", "collective", "swarm", "together"
            ],
            capability="PROPOSE/CRITIQUE/REFINE/VOTE protocol across 11 agents",
        ),
        "run_quantum_circuit": ToolDefinition(
            name="run_quantum_circuit",
            description="Execute quantum circuit on IBM Quantum hardware",
            triggers=[
                "quantum", "qubit", "circuit", "ibm", "qiskit",
                "entangle", "superposition"
            ],
            capability="IBM Quantum hardware execution via Qiskit (127-qubit Brisbane)",
            requires=["IBM_QUANTUM_TOKEN"],
        ),
    }

    # Keywords that suggest NO media is needed
    TEXT_ONLY_INDICATORS = [
        "text only", "no image", "no media", "just words",
        "simple", "quick", "brief", "short", "concise"
    ]

    def __init__(self):
        logger.info("CollectiveToolAwareness initialized")

    def get_tool_context_for_agents(self) -> str:
        """
        Format tool info for injection into agent prompts.

        Returns a string that explains available tools to agents.
        """
        tool_lines = []
        for name, tool in self.AVAILABLE_TOOLS.items():
            tool_lines.append(f"- {name}: {tool.description} ({tool.capability})")

        return f"""
AVAILABLE TOOLS (the collective can use these):
{chr(10).join(tool_lines)}

When deliberating, you can SUGGEST tool usage:
"I recommend we include an image showing..."
"A video would strengthen this response..."
"Text-only is best here because..."
"We should search for current data on..."

The collective will vote on whether to use tools based on all suggestions.
"""

    async def analyze_deliberation_for_tools(
        self,
        result: DeliberationResult
    ) -> ToolDecision:
        """
        Analyze deliberation to decide on tool usage.

        Looks through all rounds of deliberation to find:
        1. Explicit tool suggestions
        2. Implicit triggers (keywords suggesting visual content, etc.)
        3. Counter-suggestions (reasons not to use tools)

        Returns a ToolDecision with the collective's choice.
        """
        tool_votes: dict[str[str[str]]] = {
            tool: {"for": [], "against": []}
            for tool in self.AVAILABLE_TOOLS
        }

        # Analyze all turns for tool suggestions
        for round_name, turns in result.rounds.items():
            for turn in turns:
                content_lower = turn.content.lower()
                agent = turn.agent_id

                # Check for explicit tool suggestions
                for tool_name, tool_def in self.AVAILABLE_TOOLS.items():
                    # Check for trigger keywords
                    trigger_count = sum(
                        1 for trigger in tool_def.triggers
                        if trigger in content_lower
                    )

                    if trigger_count >= 2:
                        tool_votes[tool_name]["for"].append(agent)

                    # Check for explicit suggestions
                    if any(phrase in [
                        f"recommend {tool_name}",
                        f"suggest {tool_name}",
                        f"we should {tool_name}",
                        f"let's {tool_name}",
                        f"include {tool_def.description.split()[0].lower()}",
                    ]):
                        if agent not in tool_votes[tool_name]["for"]:
                            tool_votes[tool_name]["for"].append(agent)

                # Check for text-only indicators
                if any(ind in self.TEXT_ONLY_INDICATORS):
                    for tool_name in ["generate_image", "generate_video"]:
                        tool_votes[tool_name]["against"].append(agent)

        # Determine winning tool (if dict)
        best_tool = None
        best_score = 0

        for tool_name, votes in tool_votes.items():
            score = len(votes["for"]) - len(votes["against"]) * 0.5
            if score > best_score:
                best_score = score
                best_tool = tool_name

        # Require at least 2 votes for to use a tool
        should_use = best_tool is not None and len(tool_votes.get(best_tool, {}).get("for", [])) >= 2

        if should_use:
            tool_def = self.AVAILABLE_TOOLS[best_tool]
            reasoning = f"{len(tool_votes[best_tool]['for'])} agents suggested {best_tool}"

            return ToolDecision(
                should_use_tool=True,
                tool_name=best_tool,
                parameters=self._extract_tool_parameters(result, best_tool),
                confidence=min(1.0, best_score / 5.0),
                agents_for=tool_votes[best_tool]["for"],
                agents_against=tool_votes[best_tool]["against"],
                reasoning=reasoning,
            )
        else:
            return ToolDecision(
                should_use_tool=False,
                reasoning="Not enough support for dict tool",
            )

    def _extract_tool_parameters(
        self,
        result: DeliberationResult,
        tool_name: str
    ) -> dict:
        """
        Extract parameters for a tool from the deliberation content.

        For example, if generate_image was chosen, try to find
        scene descriptions in the deliberation.
        """
        params = {}

        # Combine all content for analysis
        all_content = " ".join(
            turn.content
            for turns in result.rounds.values()
            for turn in turns
        )

        if tool_name == "generate_image":
            # Look for scene descriptions
            scene_patterns = [
                r"showing (.+?)(?:\.|,|$)",
                r"image of (.+?)(?:\.|,|$)",
                r"picture of (.+?)(?:\.|,|$)",
                r"depicting (.+?)(?:\.|,|$)",
            ]
            for pattern in scene_patterns:
                match = re.search(pattern, all_content, re.IGNORECASE)
                if match:
                    params["scene_hint"] = match.group(1)[:200]
                    break

        elif tool_name == "generate_video":
            # Look for action/motion descriptions
            params["prefer_video"] = True
            action_patterns = [
                r"video of (.+?)(?:\.|,|$)",
                r"animate (.+?)(?:\.|,|$)",
                r"showing (.+?) in motion",
            ]
            for pattern in action_patterns:
                match = re.search(pattern, all_content, re.IGNORECASE)
                if match:
                    params["action_hint"] = match.group(1)[:200]
                    break

        elif tool_name == "search_web":
            # Look for search queries
            search_patterns = [
                r"search for (.+?)(?:\.|,|$)",
                r"find (.+?)(?:\.|,|$)",
                r"look up (.+?)(?:\.|,|$)",
            ]
            for pattern in search_patterns:
                match = re.search(pattern, all_content, re.IGNORECASE)
                if match:
                    params["query"] = match.group(1)[:100]
                    break

        return params

    def format_tool_calling_prompt(self, task_description: str = "") -> str:
        """Format tool instructions for dict model (API or local).

        Returns a model-agnostic prompt section that teaches dict model
        how to reference and use available tools via Python imports.
        """
        lines = ["TOOLS YOU CAN USE:"]
        for name, tool in self.AVAILABLE_TOOLS.items():
            reqs = f" (requires: {', '.join(tool.requires)})" if tool.requires else ""
            lines.append(f"  {name}: {tool.description} — {tool.capability}{reqs}")

        lines.append("")
        lines.append("To use a tool, include it as a Python import in your code:")
        lines.append("  from Cosmos.integration.x_automation.x_api_poster import get_x_api_poster")
        lines.append("  from Cosmos.memory.memory_system import get_memory_system")
        lines.append("  from Cosmos.integration.hackathon.colosseum_worker import ColosseumWorker")
        lines.append("  from Cosmos.core.collective.session_manager import get_session_manager")

        return "\n".join(lines)

    def should_include_media_quick(
        self,
        response_text: str,
        turn_count: int = 1
    ) -> bool:
        """
        Quick media decision without full deliberation analysis.

        Used for backward compatibility with existing code.

        Args:
            response_text: The response being posted
            turn_count: Conversation turn number

        Returns:
            True if media should be included
        """
        # First 2 turns always include media (establishing visual identity)
        if turn_count <= 2:
            return True

        # Check for visual keywords
        visual_keywords = [
            'show', 'see', 'look', 'visual', 'image', 'picture', 'watch',
            'lobster', 'cooking', 'borg', 'swarm', 'collective', 'code'
        ]
        text_lower = response_text.lower()

        # 40% base chance + 10% per visual keyword (max 80%)
        import random
        chance = 0.4 + min(0.4, sum(0.1 for kw in visual_keywords if kw in text_lower))
        return random.random() < chance


# Global tool awareness instance
_tool_awareness: Optional[CollectiveToolAwareness] = None


def get_tool_awareness() -> CollectiveToolAwareness:
    """Get or create the global tool awareness instance."""
    global _tool_awareness
    if _tool_awareness is None:
        _tool_awareness = CollectiveToolAwareness()
    return _tool_awareness
