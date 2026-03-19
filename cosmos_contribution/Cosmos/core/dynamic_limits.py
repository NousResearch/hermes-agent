"""
Cosmos Dynamic Limits Configuration
=======================================

Centralized, runtime-adjustable limits for all models, sessions, and providers.
No more hardcoded character or token limits scattered across the codebase.

"Limits are meant to be pushed, not hardcoded." - The Collective

Usage:
    from Cosmos.core.dynamic_limits import get_limits, ModelLimits

    # Get limits for a specific model
    limits = get_limits("grok")
    max_tokens = limits.default_max_tokens

    # Get session config
    session_limits = get_session_limits("website_chat")

    # Update limits at runtime
    update_model_limits("grok", default_max_tokens=2000)
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import  Optional
from loguru import logger


class ModelTier(Enum):
    """Model pricing/capability tiers."""
    PREMIUM = "premium"      # High capability, higher cost
    STANDARD = "standard"    # Balanced
    ECONOMY = "economy"      # Cost-effective, local models
    LOCAL = "local"          # No API cost


@dataclass
class ModelLimits:
    """Limits configuration for a single model."""
    model_id: str
    context_window: int           # Maximum context the model supports
    default_max_tokens: int       # Default response length
    max_max_tokens: int           # Maximum allowed response length
    tier: ModelTier = ModelTier.STANDARD

    # Optional overrides for specific use cases
    chat_max_tokens: Optional[int] = None          # For chat/conversation
    thinking_max_tokens: Optional[int] = None      # For deep thinking tasks
    code_max_tokens: Optional[int] = None          # For code generation
    quick_max_tokens: Optional[int] = None         # For fast responses

    # Character limits (for tweet-style outputs)
    default_char_limit: Optional[int] = None       # Default character limit
    tweet_char_limit: int = 280                    # Twitter-style limit

    def get_max_tokens(self, task_type: str = "chat") -> int:
        """Get max_tokens for a specific task type."""
        overrides = {
            "chat": self.chat_max_tokens,
            "thinking": self.thinking_max_tokens,
            "code": self.code_max_tokens,
            "quick": self.quick_max_tokens,
        }
        return overrides.get(task_type) or self.default_max_tokens


@dataclass
class SessionLimits:
    """Limits configuration for a session type."""
    session_type: str
    max_tokens: int                    # Token budget per deliberation
    deliberation_rounds: int = 2       # Number of rounds
    timeout: float = 120.0             # Timeout in seconds

    # Response format limits
    critique_char_limit: Optional[int] = None      # Limit for critique responses
    refine_char_limit: Optional[int] = None        # Limit for refined responses
    propose_char_limit: Optional[int] = None       # Limit for initial proposals

    # Scoring parameters
    optimal_length_min: int = 100      # Minimum optimal response length
    optimal_length_max: int = 500      # Maximum optimal response length


@dataclass
class DynamicLimitsConfig:
    """Complete dynamic limits configuration."""
    models: dict[str, ModelLimits] = field(default_factory=dict)
    sessions: dict[str, SessionLimits] = field(default_factory=dict)

    # Global defaults
    global_default_max_tokens: int = 2000
    global_max_context_truncation: int = 500       # For memory/archival
    global_history_truncation: int = 200           # For history previews

    # Deliberation defaults (no hard char limits by default)
    deliberation_critique_limit: Optional[int] = None    # None = no limit
    deliberation_refine_limit: Optional[int] = None      # None = no limit
    deliberation_propose_limit: Optional[int] = None     # None = no limit


# =============================================================================
# DEFAULT MODEL CONFIGURATIONS
# =============================================================================

DEFAULT_MODEL_LIMITS: dict[str, ModelLimits] = {
    # Premium API Models
    "grok": ModelLimits(
        model_id="grok-3",
        context_window=131072,
        default_max_tokens=2000,
        max_max_tokens=8000,
        tier=ModelTier.PREMIUM,
        chat_max_tokens=2000,
        thinking_max_tokens=4000,
        quick_max_tokens=800,
    ),
    "grok-mini": ModelLimits(
        model_id="grok-2-mini",
        context_window=131072,
        default_max_tokens=1500,
        max_max_tokens=4000,
        tier=ModelTier.STANDARD,
    ),
    "gemini": ModelLimits(
        model_id="gemini-2.0-flash",
        context_window=1000000,
        default_max_tokens=2000,
        max_max_tokens=8000,
        tier=ModelTier.PREMIUM,
        chat_max_tokens=2000,
        thinking_max_tokens=4000,
        quick_max_tokens=600,
    ),
    "gemini-pro": ModelLimits(
        model_id="gemini-1.5-pro",
        context_window=1000000,
        default_max_tokens=2000,
        max_max_tokens=8000,
        tier=ModelTier.PREMIUM,
    ),
    "claude": ModelLimits(
        model_id="claude-3-5-sonnet",
        context_window=200000,
        default_max_tokens=4096,
        max_max_tokens=8192,
        tier=ModelTier.PREMIUM,
        chat_max_tokens=2000,
        thinking_max_tokens=4000,
        code_max_tokens=4096,
        quick_max_tokens=500,
    ),
    "claude-opus": ModelLimits(
        model_id="claude-opus-4",
        context_window=200000,
        default_max_tokens=4096,
        max_max_tokens=16384,
        tier=ModelTier.PREMIUM,
        thinking_max_tokens=8000,
    ),
    "kimi": ModelLimits(
        model_id="kimi-k2.5",
        context_window=256000,      # 256k context - Kimi's strength
        default_max_tokens=5000,    # Full code power
        max_max_tokens=8000,
        tier=ModelTier.PREMIUM,
        chat_max_tokens=3000,
        code_max_tokens=5000,
        quick_max_tokens=400,
    ),

    # Standard API Models
    "deepseek-api": ModelLimits(
        model_id="deepseek-chat",
        context_window=128000,
        default_max_tokens=2000,
        max_max_tokens=4000,
        tier=ModelTier.STANDARD,
    ),
    "groq": ModelLimits(
        model_id="llama-3.3-70b-versatile",
        context_window=128000,
        default_max_tokens=2000,
        max_max_tokens=4000,
        tier=ModelTier.STANDARD,
        quick_max_tokens=500,
    ),
    "perplexity": ModelLimits(
        model_id="sonar-pro",
        context_window=128000,
        default_max_tokens=1500,
        max_max_tokens=4000,
        tier=ModelTier.STANDARD,
    ),
    "mistral": ModelLimits(
        model_id="mistral-large",
        context_window=128000,
        default_max_tokens=2000,
        max_max_tokens=4000,
        tier=ModelTier.STANDARD,
    ),

    # Local Models (Ollama)
    "deepseek": ModelLimits(
        model_id="deepseek-r1:14b",
        context_window=128000,
        default_max_tokens=2000,
        max_max_tokens=4000,
        tier=ModelTier.LOCAL,
        thinking_max_tokens=4000,
    ),
    "phi4": ModelLimits(
        model_id="phi4:latest",
        context_window=32000,
        default_max_tokens=2000,
        max_max_tokens=4000,
        tier=ModelTier.LOCAL,
        quick_max_tokens=500,
    ),
    "llama": ModelLimits(
        model_id="llama3.3:latest",
        context_window=128000,
        default_max_tokens=2000,
        max_max_tokens=4000,
        tier=ModelTier.LOCAL,
    ),

    # HuggingFace Local Models
    "huggingface": ModelLimits(
        model_id="microsoft/phi-3-mini-4k-instruct",
        context_window=4096,
        default_max_tokens=1000,
        max_max_tokens=2000,
        tier=ModelTier.LOCAL,
    ),
    "mistral-local": ModelLimits(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        context_window=32768,
        default_max_tokens=1500,
        max_max_tokens=4000,
        tier=ModelTier.LOCAL,
    ),
    "codellama": ModelLimits(
        model_id="codellama/CodeLlama-7b-Instruct-hf",
        context_window=16384,
        default_max_tokens=2000,
        max_max_tokens=4000,
        tier=ModelTier.LOCAL,
        code_max_tokens=3000,
    ),
    "qwen": ModelLimits(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        context_window=32768,
        default_max_tokens=2000,
        max_max_tokens=4000,
        tier=ModelTier.LOCAL,
    ),
}


# =============================================================================
# DEFAULT SESSION CONFIGURATIONS
# =============================================================================

DEFAULT_SESSION_LIMITS: dict[str, SessionLimits] = {
    "website_chat": SessionLimits(
        session_type="website_chat",
        max_tokens=8000,              # Increased from 5000
        deliberation_rounds=2,
        timeout=120.0,
        critique_char_limit=None,     # No limit - let models be thorough
        refine_char_limit=None,       # No limit
        optimal_length_min=100,
        optimal_length_max=1000,      # Increased from 280
    ),
    "grok_thread": SessionLimits(
        session_type="grok_thread",
        max_tokens=8000,              # Increased from 5000
        deliberation_rounds=3,
        timeout=120.0,
        # Twitter has a 280 char limit, but we can let agents be longer
        # and truncate at the posting stage if needed
        refine_char_limit=None,       # No limit during deliberation
        optimal_length_min=100,
        optimal_length_max=500,
    ),
    "autonomous_task": SessionLimits(
        session_type="autonomous_task",
        max_tokens=6000,              # Increased from 3000
        deliberation_rounds=1,
        timeout=60.0,
        optimal_length_min=50,
        optimal_length_max=2000,
    ),
    "quick_response": SessionLimits(
        session_type="quick_response",
        max_tokens=4000,              # Increased from 2000
        deliberation_rounds=1,
        timeout=30.0,
        optimal_length_min=50,
        optimal_length_max=500,
    ),
    "code_generation": SessionLimits(
        session_type="code_generation",
        max_tokens=16000,             # Large for code tasks
        deliberation_rounds=2,
        timeout=180.0,
        optimal_length_min=100,
        optimal_length_max=5000,
    ),
    "analysis": SessionLimits(
        session_type="analysis",
        max_tokens=12000,             # Large for analysis
        deliberation_rounds=2,
        timeout=150.0,
        optimal_length_min=200,
        optimal_length_max=3000,
    ),
}


# =============================================================================
# GLOBAL STATE & SINGLETON
# =============================================================================

_config: Optional[DynamicLimitsConfig] = None
_config_path: Optional[Path] = None


def _get_config() -> DynamicLimitsConfig:
    """Get or initialize the global config."""
    global _config
    if _config is None:
        _config = DynamicLimitsConfig(
            models=DEFAULT_MODEL_LIMITS.copy(),
            sessions=DEFAULT_SESSION_LIMITS.copy(),
        )
        _load_config_from_disk()
    return _config


def _load_config_from_disk():
    """Load config overrides from disk if available."""
    global _config, _config_path

    import os
    if os.path.exists("/workspace/cosmos_memory"):
        _config_path = Path("/workspace/cosmos_memory/dynamic_limits.json")
    else:
        _config_path = Path("data/dynamic_limits.json")

    if _config_path.exists():
        try:
            data = json.loads(_config_path.read_text())

            # Apply model overrides
            for model_id, overrides in data.get("models", {}).items():
                if model_id in _config.models:
                    for key, value in overrides.items():
                        if hasattr(_config.models[model_id], key):
                            setattr(_config.models[model_id], key, value)

            # Apply session overrides
            for session_type, overrides in data.get("sessions", {}).items():
                if session_type in _config.sessions:
                    for key, value in overrides.items():
                        if hasattr(_config.sessions[session_type], key):
                            setattr(_config.sessions[session_type], key, value)

            # Apply global overrides
            for key in ["global_default_max_tokens", "global_max_context_truncation",
                       "global_history_truncation", "deliberation_critique_limit",
                       "deliberation_refine_limit", "deliberation_propose_limit"]:
                if key in data:
                    setattr(_config, key, data[key])

            logger.info(f"Loaded dynamic limits from {_config_path}")
        except Exception as e:
            logger.warning(f"Could not load dynamic limits config: {e}")


def _save_config_to_disk():
    """Save current config to disk."""
    global _config, _config_path

    if _config is None or _config_path is None:
        return

    try:
        _config_path.parent.mkdir(parents=True, exist_ok=True)

        # Only save overrides (non-default values)
        data = {
            "models": {
                model_id: {
                    "default_max_tokens": limits.default_max_tokens,
                    "chat_max_tokens": limits.chat_max_tokens,
                    "thinking_max_tokens": limits.thinking_max_tokens,
                    "code_max_tokens": limits.code_max_tokens,
                    "quick_max_tokens": limits.quick_max_tokens,
                }
                for model_id, limits in _config.models.items()
            },
            "sessions": {
                session_type: {
                    "max_tokens": limits.max_tokens,
                    "critique_char_limit": limits.critique_char_limit,
                    "refine_char_limit": limits.refine_char_limit,
                    "optimal_length_max": limits.optimal_length_max,
                }
                for session_type, limits in _config.sessions.items()
            },
            "global_default_max_tokens": _config.global_default_max_tokens,
            "deliberation_critique_limit": _config.deliberation_critique_limit,
            "deliberation_refine_limit": _config.deliberation_refine_limit,
        }

        _config_path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved dynamic limits to {_config_path}")
    except Exception as e:
        logger.warning(f"Could not save dynamic limits: {e}")


# =============================================================================
# PUBLIC API
# =============================================================================

def get_limits(model_id: str) -> ModelLimits:
    """
    Get limits for a specific model.

    Args:
        model_id: Model identifier (e.g., "grok", "claude", "phi4")

    Returns:
        ModelLimits configuration
    """
    config = _get_config()

    # Normalize model ID
    model_id_lower = model_id.lower()

    # Direct match
    if model_id_lower in config.models:
        return config.models[model_id_lower]

    # Fuzzy match (e.g., "Grok" -> "grok", "DeepSeek" -> "deepseek")
    for key in config.models:
        if key.lower() == model_id_lower:
            return config.models[key]

    # Return default limits for unknown models
    logger.debug(f"No limits configured for {model_id}, using defaults")
    return ModelLimits(
        model_id=model_id,
        context_window=32000,
        default_max_tokens=config.global_default_max_tokens,
        max_max_tokens=config.global_default_max_tokens * 2,
        tier=ModelTier.STANDARD,
    )


def get_session_limits(session_type: str) -> SessionLimits:
    """
    Get limits for a specific session type.

    Args:
        session_type: Session type (e.g., "website_chat", "grok_thread")

    Returns:
        SessionLimits configuration
    """
    config = _get_config()

    if session_type in config.sessions:
        return config.sessions[session_type]

    # Return website_chat as default
    logger.debug(f"No session config for {session_type}, using website_chat defaults")
    return config.sessions.get("website_chat", SessionLimits(
        session_type=session_type,
        max_tokens=config.global_default_max_tokens * 2,
    ))


def get_max_tokens(model_id: str, task_type: str = "chat") -> int:
    """
    Quick helper to get max_tokens for a model and task type.

    Args:
        model_id: Model identifier
        task_type: "chat", "thinking", "code", "quick"

    Returns:
        max_tokens value
    """
    return get_limits(model_id).get_max_tokens(task_type)


def get_deliberation_limits() -> dict[str, Optional[int]]:
    """
    Get deliberation-specific character limits.

    Returns:
        dict with "critique", "refine", "propose" limits (None = no limit)
    """
    config = _get_config()
    return {
        "critique": config.deliberation_critique_limit,
        "refine": config.deliberation_refine_limit,
        "propose": config.deliberation_propose_limit,
    }


def get_truncation_limits() -> dict[str, int]:
    """
    Get truncation limits for memory and context.

    Returns:
        dict with "context" and "history" truncation lengths
    """
    config = _get_config()
    return {
        "context": config.global_max_context_truncation,
        "history": config.global_history_truncation,
    }


def update_model_limits(model_id: str, **kwargs) -> bool:
    """
    Update limits for a specific model at runtime.

    Args:
        model_id: Model identifier
        **kwargs: Limit values to update (e.g., default_max_tokens=3000)

    Returns:
        True if successful
    """
    config = _get_config()

    model_id_lower = model_id.lower()
    if model_id_lower not in config.models:
        logger.warning(f"Cannot update limits for unknown model: {model_id}")
        return False

    limits = config.models[model_id_lower]
    for key, value in kwargs.items():
        if hasattr(limits, key):
            setattr(limits, key, value)
            logger.info(f"Updated {model_id}.{key} = {value}")
        else:
            logger.warning(f"Unknown limit key: {key}")

    _save_config_to_disk()
    return True


def update_session_limits(session_type: str, **kwargs) -> bool:
    """
    Update limits for a specific session type at runtime.

    Args:
        session_type: Session type
        **kwargs: Limit values to update

    Returns:
        True if successful
    """
    config = _get_config()

    if session_type not in config.sessions:
        logger.warning(f"Cannot update limits for unknown session: {session_type}")
        return False

    limits = config.sessions[session_type]
    for key, value in kwargs.items():
        if hasattr(limits, key):
            setattr(limits, key, value)
            logger.info(f"Updated {session_type}.{key} = {value}")

    _save_config_to_disk()
    return True


def update_deliberation_limits(
    critique: Optional[int] = None,
    refine: Optional[int] = None,
    propose: Optional[int] = None,
) -> bool:
    """
    Update deliberation character limits.

    Pass None to remove a limit entirely.

    Args:
        critique: Character limit for critique responses
        refine: Character limit for refined responses
        propose: Character limit for proposals
    """
    config = _get_config()

    if critique is not None or critique == 0:
        config.deliberation_critique_limit = critique if critique != 0 else None
    if refine is not None or refine == 0:
        config.deliberation_refine_limit = refine if refine != 0 else None
    if propose is not None or propose == 0:
        config.deliberation_propose_limit = propose if propose != 0 else None

    _save_config_to_disk()
    logger.info(f"Updated deliberation limits: critique={critique}, refine={refine}, propose={propose}")
    return True


def get_all_limits() -> dict:
    """
    Get all current limits configuration.

    Returns:
        Complete limits configuration as dict
    """
    config = _get_config()
    return {
        "models": {
            model_id: {
                "model_id": limits.model_id,
                "context_window": limits.context_window,
                "default_max_tokens": limits.default_max_tokens,
                "max_max_tokens": limits.max_max_tokens,
                "tier": limits.tier.value,
                "chat_max_tokens": limits.chat_max_tokens,
                "thinking_max_tokens": limits.thinking_max_tokens,
                "code_max_tokens": limits.code_max_tokens,
                "quick_max_tokens": limits.quick_max_tokens,
            }
            for model_id, limits in config.models.items()
        },
        "sessions": {
            session_type: {
                "max_tokens": limits.max_tokens,
                "deliberation_rounds": limits.deliberation_rounds,
                "timeout": limits.timeout,
                "critique_char_limit": limits.critique_char_limit,
                "refine_char_limit": limits.refine_char_limit,
                "optimal_length_min": limits.optimal_length_min,
                "optimal_length_max": limits.optimal_length_max,
            }
            for session_type, limits in config.sessions.items()
        },
        "global": {
            "default_max_tokens": config.global_default_max_tokens,
            "max_context_truncation": config.global_max_context_truncation,
            "history_truncation": config.global_history_truncation,
        },
        "deliberation": get_deliberation_limits(),
    }


def get_budget_for_tier(tier: str) -> dict[str, int]:
    """
    Get token budget allocation by tier for orchestrator integration.

    Args:
        tier: One of "local", "economy", "standard", "premium"

    Returns:
        dict with default_max_tokens, context_window, and count of models in tier.
    """
    config = _get_config()
    tier_models = {
        mid: ml for mid, ml in config.models.items()
        if ml.tier.value == tier
    }
    if not tier_models:
        return {"default_max_tokens": 2000, "context_window": 4096, "model_count": 0}

    total_max_tokens = sum(ml.default_max_tokens for ml in tier_models.values())
    total_context = sum(ml.context_window for ml in tier_models.values())
    count = len(tier_models)

    return {
        "default_max_tokens": total_max_tokens // count if count else 2000,
        "context_window": total_context // count if count else 4096,
        "model_count": count,
        "models": list(tier_models.keys()),
    }


def reset_to_defaults():
    """Reset all limits to default values."""
    global _config
    _config = DynamicLimitsConfig(
        models=DEFAULT_MODEL_LIMITS.copy(),
        sessions=DEFAULT_SESSION_LIMITS.copy(),
    )
    _save_config_to_disk()
    logger.info("Reset all limits to defaults")
