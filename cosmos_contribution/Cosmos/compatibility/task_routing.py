"""
Cosmos Task Routing - HermesAgent to Swarm Model Mapping
==========================================================

Routes HermesAgent tools/skills to the optimal swarm model based on task type.

Models Available:
-----------------
API-Based (External):
  - Grok: Real-time info, X/Twitter, humor, fast research
  - Gemini: Multimodal, images, long context, synthesis, code
  - Claude: Code quality, ethics, documentation, careful analysis
  - ClaudeOpus: Final auditor, complex reasoning, premium quality
  - Kimi: 256K context, long documents, complex planning
  - DeepSeek: Math, reasoning chains, open source advocate

Local/Hybrid:
  - Phi: Fast inference, efficiency, local processing
  - HuggingFace: Open-source models (Mistral, Llama, CodeLlama)

Shadow Agents (tmux persistent):
  - grok_shadow, gemini_shadow, kimi_shadow, claude_shadow
  - deepseek_shadow, phi_shadow, huggingface_shadow, swarm_mind_shadow

Specialist Agents:
  - web_agent: Browser automation, scraping
  - code_agent: Code generation, refactoring
  - research_agent: Information gathering
  - filesystem_agent: File operations

"The right model for the right task." - The Collective
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger


class HermesAgentTaskType(Enum):
    """Task types derived from HermesAgent tool groups."""
    # Core tool groups
    FILESYSTEM = "filesystem"      # Read, write, edit files
    RUNTIME = "runtime"            # Execute code, bash commands
    SESSIONS = "sessions"          # Multi-agent coordination
    MEMORY = "memory"              # Knowledge retrieval
    WEB = "web"                    # Search, fetch, browse
    UI = "ui"                      # Canvas, browser automation
    AUTOMATION = "automation"      # Scheduled tasks, cron
    MESSAGING = "messaging"        # Channel communication
    NODES = "nodes"                # Device: camera, screen, location

    # Extended capabilities
    VOICE = "voice"                # Speech-to-text, text-to-speech
    CANVAS = "canvas"              # Visual rendering, A2UI
    SKILLS = "skills"              # HermesHub marketplace skills
    IMAGE = "image"                # Image generation/analysis
    VIDEO = "video"                # Video processing
    AUDIO = "audio"                # Audio processing


@dataclass
class ModelCapability:
    """Describes a model's capabilities for task routing."""
    model_id: str
    display_name: str
    strengths: List[str]
    task_types: List[HermesAgentTaskType]
    priority: int = 5  # 1-10, higher = preferred
    max_tokens: int = 4000
    supports_vision: bool = False
    supports_code: bool = True
    cost_tier: str = "standard"  # free, standard, premium
    latency: str = "medium"  # fast, medium, slow


# =============================================================================
# MODEL REGISTRY - All available models with capabilities
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelCapability] = {
    # === API-BASED EXTERNAL MODELS ===
    # Sources: Official documentation verified 2026-02

    "Grok": ModelCapability(
        model_id="Grok",
        display_name="Grok 4 (xAI)",
        # Ref: https://docs.x.ai/docs/overview
        # 2M token context, vision, code, function calling, real-time data
        strengths=[
            "2M token context",      # Largest in industry
            "real-time X/Twitter",   # Native integration
            "function calling",      # Tool use / agents
            "code generation",       # Grok Code Fast
            "video/audio gen",       # Grok Imagine API
            "low-latency voice",     # Grok Voice
        ],
        task_types=[
            HermesAgentTaskType.WEB,
            HermesAgentTaskType.MESSAGING,
            HermesAgentTaskType.RUNTIME,
            HermesAgentTaskType.VIDEO,
            HermesAgentTaskType.AUDIO,
            HermesAgentTaskType.VOICE,
        ],
        priority=9,
        max_tokens=32000,  # Output limit
        supports_vision=True,
        cost_tier="standard",
        latency="fast",
    ),

    "Gemini": ModelCapability(
        model_id="Gemini",
        display_name="Gemini 3 Flash (Google)",
        # Ref: https://ai.google.dev/gemini-api/docs/gemini-3
        # Agentic vision + code execution, ultra-high res vision
        strengths=[
            "agentic vision",        # Code execution + vision grounding
            "ultra-high resolution", # Fine text/detail recognition
            "multimodal native",     # Image/video/audio
            "code execution",        # 5-10% quality boost
            "segmentation",          # Object detection
            "long context",          # 1M+ tokens
        ],
        task_types=[
            HermesAgentTaskType.IMAGE,
            HermesAgentTaskType.VIDEO,
            HermesAgentTaskType.WEB,
            HermesAgentTaskType.RUNTIME,
            HermesAgentTaskType.CANVAS,
            HermesAgentTaskType.MEMORY,
            HermesAgentTaskType.UI,
        ],
        priority=9,
        max_tokens=8192,
        supports_vision=True,
        cost_tier="standard",
        latency="medium",
    ),

    "Claude": ModelCapability(
        model_id="Claude",
        display_name="Claude Sonnet 4.5 (Anthropic)",
        # Ref: https://platform.claude.com/docs/en/about-claude/models/overview
        # Best coding model, 1M context (beta), extended thinking
        strengths=[
            "best coding model",     # Official claim
            "1M token context",      # With beta header
            "extended thinking",     # Deeper reasoning
            "agentic tasks",         # Claude Code CLI
            "vision native",         # First-class support
            "careful analysis",      # Quality + safety
        ],
        task_types=[
            HermesAgentTaskType.RUNTIME,
            HermesAgentTaskType.FILESYSTEM,
            HermesAgentTaskType.SESSIONS,
            HermesAgentTaskType.MEMORY,
            HermesAgentTaskType.SKILLS,
        ],
        priority=9,
        max_tokens=8192,
        supports_vision=True,
        supports_code=True,
        cost_tier="standard",
        latency="medium",
    ),

    "ClaudeOpus": ModelCapability(
        model_id="ClaudeOpus",
        display_name="Claude Opus 4.5 (Anthropic)",
        # Premium tier - complex reasoning, final auditor
        strengths=[
            "complex reasoning",
            "final auditor",
            "premium quality",
            "deep analysis",
            "extended thinking",
            "1M context",
        ],
        task_types=[
            HermesAgentTaskType.RUNTIME,
            HermesAgentTaskType.FILESYSTEM,
            HermesAgentTaskType.SESSIONS,
            HermesAgentTaskType.SKILLS,
        ],
        priority=10,
        max_tokens=16384,
        supports_vision=True,
        supports_code=True,
        cost_tier="premium",
        latency="slow",
    ),

    "Kimi": ModelCapability(
        model_id="Kimi",
        display_name="Kimi K2.5 (Moonshot)",
        # Ref: https://platform.moonshot.ai/
        # 256K context, 1T MoE, agent swarm (100 agents), thinking mode
        strengths=[
            "256K token context",    # Long document master
            "1T MoE parameters",     # 32B active per request
            "agent swarm",           # Up to 100 agents coordinated
            "thinking mode",         # Reasoning traces
            "tool calling",          # Native support
            "4.5x faster execution", # Agent swarm speedup
        ],
        task_types=[
            HermesAgentTaskType.MEMORY,
            HermesAgentTaskType.FILESYSTEM,
            HermesAgentTaskType.SESSIONS,
            HermesAgentTaskType.SKILLS,
        ],
        priority=8,
        max_tokens=32000,
        supports_vision=True,  # K2.5 multimodal
        cost_tier="standard",
        latency="medium",
    ),

    "DeepSeek": ModelCapability(
        model_id="DeepSeek",
        display_name="DeepSeek V3.2 / V4",
        # Ref: https://api-docs.deepseek.com/
        # 1M+ context (V4), multi-file reasoning, thinking mode
        strengths=[
            "1M+ token context",     # V4 sparse attention
            "multi-file reasoning",  # Full codebase analysis
            "thinking mode",         # deepseek-reasoner
            "bug diagnosis",         # Cross-file fixes
            "math/reasoning",        # Chain-of-thought
            "open source friendly",
        ],
        task_types=[
            HermesAgentTaskType.RUNTIME,
            HermesAgentTaskType.FILESYSTEM,
            HermesAgentTaskType.AUTOMATION,
            HermesAgentTaskType.SKILLS,
        ],
        priority=8,
        max_tokens=32000,  # Up to 64K with reasoning
        supports_code=True,
        cost_tier="free",  # Very competitive pricing
        latency="fast",
    ),

    # === LOCAL/HYBRID MODELS ===

    "Phi": ModelCapability(
        model_id="Phi",
        display_name="Phi-4 (Microsoft Local)",
        strengths=[
            "speed",
            "efficiency",
            "local processing",
            "privacy",
            "small footprint",
        ],
        task_types=[
            HermesAgentTaskType.RUNTIME,
            HermesAgentTaskType.FILESYSTEM,
            HermesAgentTaskType.VOICE,
        ],
        priority=6,
        max_tokens=4000,
        cost_tier="free",
        latency="fast",
    ),

    "HuggingFace": ModelCapability(
        model_id="HuggingFace",
        display_name="HuggingFace Local (Whisper/Mistral/Llama)",
        # Ref: https://huggingface.co/docs/transformers/
        # Whisper for audio, local LLMs, embeddings
        strengths=[
            "Whisper STT",           # 680K hours trained
            "local embeddings",      # sentence-transformers
            "fine-tunable",          # Custom training
            "open source",           # Full control
            "Mistral/Llama/CodeLlama",
            "zero-shot audio",
        ],
        task_types=[
            HermesAgentTaskType.MEMORY,
            HermesAgentTaskType.RUNTIME,
            HermesAgentTaskType.VOICE,
            HermesAgentTaskType.AUDIO,
        ],
        priority=7,
        max_tokens=4000,
        cost_tier="free",
        latency="medium",
    ),

    # === SPECIALIST AGENTS ===

    "WebAgent": ModelCapability(
        model_id="WebAgent",
        display_name="Web Agent (Browser)",
        strengths=["browser automation", "scraping", "screenshots", "navigation"],
        task_types=[
            HermesAgentTaskType.WEB,
            HermesAgentTaskType.UI,
            HermesAgentTaskType.CANVAS,
        ],
        priority=9,
        max_tokens=2000,
        cost_tier="free",
        latency="medium",
    ),

    "FilesystemAgent": ModelCapability(
        model_id="FilesystemAgent",
        display_name="Filesystem Agent",
        strengths=["file operations", "directory management", "search"],
        task_types=[
            HermesAgentTaskType.FILESYSTEM,
        ],
        priority=9,
        max_tokens=2000,
        cost_tier="free",
        latency="fast",
    ),
}


# =============================================================================
# TASK ROUTING TABLE - Maps HermesAgent tasks to preferred models
# =============================================================================

TASK_ROUTING: Dict[HermesAgentTaskType, List[str]] = {
    # ==========================================================================
    # ROUTING TABLE - Based on official model capabilities (2026-02)
    # ==========================================================================

    # File operations - Claude best for careful code, DeepSeek for multi-file
    HermesAgentTaskType.FILESYSTEM: [
        "Claude",           # Best coding model, careful analysis
        "DeepSeek",         # Multi-file reasoning (V4)
        "FilesystemAgent",  # Specialist
        "Kimi",             # 256K for large files
        "Phi",              # Fast local fallback
    ],

    # Code execution - Claude is "best coding model", DeepSeek has thinking mode
    HermesAgentTaskType.RUNTIME: [
        "Claude",           # Best coding model (official)
        "DeepSeek",         # Thinking mode, multi-file bugs
        "Gemini",           # Agentic vision + code execution
        "Grok",             # Grok Code Fast - agentic coding
        "Phi",              # Fast local
    ],

    # Multi-agent sessions - Kimi has native 100-agent swarm, Claude for quality
    HermesAgentTaskType.SESSIONS: [
        "Kimi",             # Agent Swarm: 100 agents, 4.5x faster
        "Claude",           # Best for coordination quality
        "ClaudeOpus",       # Complex reasoning
        "Grok",             # 2M context for session history
    ],

    # Memory/knowledge retrieval - Kimi 256K, Grok 2M, HF for embeddings
    HermesAgentTaskType.MEMORY: [
        "Grok",             # 2M token context - largest
        "Kimi",             # 256K context master
        "HuggingFace",      # Local embeddings (sentence-transformers)
        "Claude",           # 1M context (beta)
        "DeepSeek",         # 1M+ context (V4)
    ],

    # Web search/fetch - Grok has real-time X data, Gemini for research
    HermesAgentTaskType.WEB: [
        "Grok",             # Real-time X/Twitter, function calling
        "Gemini",           # Deep research, agentic vision
        "WebAgent",         # Browser automation
        "Claude",           # Quality synthesis
    ],

    # UI/Browser automation - Gemini agentic vision + code execution
    HermesAgentTaskType.UI: [
        "Gemini",           # Agentic vision - grounded answers
        "WebAgent",         # Browser specialist
        "Grok",             # Fast responses
        "Claude",           # Careful interaction
    ],

    # Scheduled tasks - DeepSeek reliable, Claude careful
    HermesAgentTaskType.AUTOMATION: [
        "DeepSeek",         # Reliable, multi-file
        "Claude",           # Careful execution
        "Phi",              # Fast local cron
        "Kimi",             # Agent swarm for complex automation
    ],

    # Channel messaging - Grok native X, Claude quality
    HermesAgentTaskType.MESSAGING: [
        "Grok",             # Native X/Twitter integration
        "Claude",           # Quality responses
        "Gemini",           # Good synthesis
        "Kimi",             # Long conversation context
    ],

    # Device nodes (camera, screen, location) - Gemini ultra-high res vision
    HermesAgentTaskType.NODES: [
        "Gemini",           # Ultra-high resolution vision
        "Grok",             # Vision support
        "Claude",           # Vision native
        "HuggingFace",      # Local processing
    ],

    # Voice processing - HuggingFace Whisper (680K hours), Grok Voice
    HermesAgentTaskType.VOICE: [
        "HuggingFace",      # Whisper - 680K hours trained, zero-shot
        "Grok",             # Grok Voice - low latency, tool calling
        "Phi",              # Fast local
        "Gemini",           # Multimodal audio
    ],

    # Canvas/visual rendering - Gemini agentic vision
    HermesAgentTaskType.CANVAS: [
        "Gemini",           # Agentic vision + code execution
        "Grok",             # Vision + fast
        "WebAgent",         # Browser rendering
        "Claude",           # Vision native
    ],

    # HermesHub skills - Claude best for agents, Kimi for complex multi-agent
    HermesAgentTaskType.SKILLS: [
        "Claude",           # Best for agentic tasks (Claude Code)
        "Kimi",             # Agent swarm - 100 agents coordinated
        "ClaudeOpus",       # Complex skill orchestration
        "DeepSeek",         # Multi-file skill execution
    ],

    # Image processing - Gemini ultra-high res, segmentation
    HermesAgentTaskType.IMAGE: [
        "Gemini",           # Ultra-high res, segmentation, detection
        "Grok",             # Vision support
        "Claude",           # Vision native
        "HuggingFace",      # Local vision models
    ],

    # Video processing - Gemini multimodal, Grok Imagine
    HermesAgentTaskType.VIDEO: [
        "Gemini",           # Multimodal native
        "Grok",             # Grok Imagine - video generation
        "HuggingFace",      # Local processing
    ],

    # Audio processing - HuggingFace Whisper, Grok Voice
    HermesAgentTaskType.AUDIO: [
        "HuggingFace",      # Whisper - best STT
        "Grok",             # Grok Voice, audio generation
        "Gemini",           # Multimodal audio
    ],
}


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def get_best_model_for_task(
    task_type: HermesAgentTaskType,
    prefer_local: bool = False,
    prefer_fast: bool = False,
    prefer_quality: bool = False,
    exclude_models: List[str] = None
) -> Optional[str]:
    """
    Get the best model for a specific task type.

    Args:
        task_type: The HermesAgent task type
        prefer_local: Prefer local/free models
        prefer_fast: Prefer fast models
        prefer_quality: Prefer high-quality (premium) models
        exclude_models: Models to skip

    Returns:
        Model ID or None
    """
    exclude = set(exclude_models or [])
    candidates = TASK_ROUTING.get(task_type, [])

    for model_id in candidates:
        if model_id in exclude:
            continue

        model = MODEL_REGISTRY.get(model_id)
        if not model:
            continue

        # Apply preferences
        if prefer_local and model.cost_tier != "free":
            continue
        if prefer_fast and model.latency == "slow":
            continue
        if prefer_quality and model.cost_tier != "premium":
            # Only skip if we explicitly want premium
            pass

        return model_id

    # Fallback to first available
    for model_id in candidates:
        if model_id not in exclude:
            return model_id

    return None


def get_models_for_task(
    task_type: HermesAgentTaskType,
    limit: int = 3
) -> List[str]:
    """Get ranked list of models for a task type."""
    return TASK_ROUTING.get(task_type, [])[:limit]


def get_fallback_chain(model_id: str, task_type: HermesAgentTaskType) -> List[str]:
    """
    Get fallback chain for a model on a specific task.

    If the primary model fails, try these in order.
    """
    candidates = TASK_ROUTING.get(task_type, [])

    # Remove the failed model, keep the rest as fallbacks
    fallbacks = [m for m in candidates if m != model_id]

    # Always add ClaudeOpus as final fallback for quality
    if "ClaudeOpus" not in fallbacks:
        fallbacks.append("ClaudeOpus")

    return fallbacks


def classify_HermesAgent_tool(tool_name: str, action: str = None) -> HermesAgentTaskType:
    """
    Classify an HermesAgent tool call to a task type.

    Args:
        tool_name: The tool being invoked (e.g., "browser", "exec")
        action: Optional action (e.g., "snapshot", "camera.snap")

    Returns:
        HermesAgentTaskType for routing
    """
    tool_lower = tool_name.lower()
    action_lower = (action or "").lower()

    # Direct mappings
    tool_to_type = {
        "read": HermesAgentTaskType.FILESYSTEM,
        "write": HermesAgentTaskType.FILESYSTEM,
        "edit": HermesAgentTaskType.FILESYSTEM,
        "apply_patch": HermesAgentTaskType.FILESYSTEM,
        "exec": HermesAgentTaskType.RUNTIME,
        "bash": HermesAgentTaskType.RUNTIME,
        "process": HermesAgentTaskType.RUNTIME,
        "sessions_list": HermesAgentTaskType.SESSIONS,
        "sessions_history": HermesAgentTaskType.SESSIONS,
        "sessions_send": HermesAgentTaskType.SESSIONS,
        "sessions_spawn": HermesAgentTaskType.SESSIONS,
        "memory_search": HermesAgentTaskType.MEMORY,
        "memory_get": HermesAgentTaskType.MEMORY,
        "web_search": HermesAgentTaskType.WEB,
        "web_fetch": HermesAgentTaskType.WEB,
        "browser": HermesAgentTaskType.UI,
        "canvas": HermesAgentTaskType.CANVAS,
        "cron": HermesAgentTaskType.AUTOMATION,
        "gateway": HermesAgentTaskType.AUTOMATION,
        "message": HermesAgentTaskType.MESSAGING,
        "nodes": HermesAgentTaskType.NODES,
    }

    if tool_lower in tool_to_type:
        return tool_to_type[tool_lower]

    # Check action for more specific routing
    if "camera" in action_lower or "photo" in action_lower:
        return HermesAgentTaskType.IMAGE
    if "screen" in action_lower or "capture" in action_lower:
        return HermesAgentTaskType.UI
    if "voice" in action_lower or "speech" in action_lower or "audio" in action_lower:
        return HermesAgentTaskType.VOICE
    if "location" in action_lower or "gps" in action_lower:
        return HermesAgentTaskType.NODES

    # Default to runtime for unknown tools
    return HermesAgentTaskType.RUNTIME


async def route_HermesAgent_task(
    tool: str,
    action: str = None,
    params: Dict = None,
    prefer_local: bool = False
) -> Tuple[str, HermesAgentTaskType]:
    """
    Route an HermesAgent tool call to the best model.

    Args:
        tool: Tool name
        action: Optional action
        params: Tool parameters
        prefer_local: Prefer local models

    Returns:
        Tuple of (model_id, task_type)
    """
    task_type = classify_HermesAgent_tool(tool, action)
    model_id = get_best_model_for_task(task_type, prefer_local=prefer_local)

    logger.debug(f"Routed {tool}.{action} -> {model_id} (type: {task_type.value})")

    return model_id, task_type


# =============================================================================
# CHANNEL ROUTING - Which model handles which channel
# =============================================================================

CHANNEL_MODEL_ROUTING = {
    "whatsapp": ["Claude", "Gemini", "Phi"],       # Personal, quality responses
    "telegram": ["Grok", "Claude", "Phi"],         # Fast, can be casual
    "discord": ["Grok", "Gemini", "DeepSeek"],     # Gaming/tech crowd
    "slack": ["Claude", "Gemini", "DeepSeek"],     # Professional
    "signal": ["Claude", "Phi"],                   # Privacy-focused
    "matrix": ["DeepSeek", "HuggingFace", "Phi"],  # Open source crowd
    "imessage": ["Claude", "Gemini"],              # Apple users, quality
    "webchat": ["Gemini", "Claude", "Grok"],       # Website - full capability
    "x": ["Grok"],                                 # X/Twitter native
}


def get_model_for_channel(channel_type: str) -> str:
    """Get preferred model for a messaging channel."""
    models = CHANNEL_MODEL_ROUTING.get(channel_type.lower(), ["Claude", "Gemini"])
    return models[0] if models else "Claude"


# =============================================================================
# SUMMARY FUNCTION
# =============================================================================

def get_routing_summary() -> Dict[str, Any]:
    """Get a summary of the routing configuration."""
    return {
        "total_models": len(MODEL_REGISTRY),
        "total_task_types": len(HermesAgentTaskType),
        "models": {
            mid: {
                "name": m.display_name,
                "strengths": m.strengths,
                "cost": m.cost_tier,
                "latency": m.latency,
            }
            for mid, m in MODEL_REGISTRY.items()
        },
        "task_routing": {
            tt.value: TASK_ROUTING.get(tt, [])
            for tt in HermesAgentTaskType
        },
        "channel_routing": CHANNEL_MODEL_ROUTING,
    }
