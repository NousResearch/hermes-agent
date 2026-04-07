"""
Finetune routing hook for Hermes agent.

Registers a pre_llm_call hook that routes prompts to the best fine-tuned
adapter based on cosine similarity with cluster centroids.

This module is imported by the finetune skill when routing is enabled.
It registers itself with the Hermes plugin system.

Integration:
    The hook sets `_finetune_adapter_path` on the agent instance (via the
    session-scoped hook state). The llama.cpp provider checks this attribute
    and passes the adapter path as a LoRA override.

    For providers that don't support runtime LoRA loading, the hook is a no-op.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.finetune.routing")

# Lazy-initialized router
_router = None


def _get_router():
    """Lazy-load the AdapterRouter to avoid import overhead when routing is disabled."""
    global _router
    if _router is None:
        try:
            import sys
            scripts_dir = str(Path(__file__).resolve().parent)
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from route import AdapterRouter
            _router = AdapterRouter()
        except Exception as e:
            logger.debug("Failed to initialize AdapterRouter: %s", e)
            return None
    return _router


def finetune_pre_llm_call_hook(**kwargs) -> Optional[Dict[str, Any]]:
    """
    pre_llm_call hook for adapter routing.

    Checks if the current provider supports adapter routing and, if so,
    embeds the prompt to find the best matching cluster adapter.

    Returns a dict with routing context (injected into user message as
    an ephemeral note) and sets adapter metadata for the provider layer.
    """
    user_message = kwargs.get("user_message", "")
    model = kwargs.get("model", "")
    platform = kwargs.get("platform", "")

    # Only route for local/custom providers
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    is_local = any(indicator in base_url.lower() for indicator in [
        "localhost", "127.0.0.1", "0.0.0.0", "llama", "local",
    ])

    if not is_local:
        return None

    router = _get_router()
    if router is None:
        return None

    if not router.enabled:
        return None

    # Don't route empty or very short messages
    if not user_message or len(user_message.strip()) < 10:
        return None

    try:
        result = router.route(user_message)
    except Exception as e:
        logger.debug("Routing failed: %s", e)
        return None

    if not result.get("adapter_path"):
        return None

    adapter_path = result["adapter_path"]
    cluster_id = result.get("cluster_id", "unknown")
    confidence = result.get("confidence", 0.0)
    label = result.get("label", "")

    logger.info(
        "Finetune routing: %s (cluster=%s, confidence=%.3f, adapter=%s)",
        label, cluster_id, confidence, adapter_path,
    )

    # Store routing decision for the provider layer to pick up.
    # The agent's _build_api_kwargs can check this env var.
    os.environ["_HERMES_FINETUNE_ADAPTER"] = adapter_path

    # Return context for logging — not injected into the prompt
    # (we don't want to pollute the user message with routing info)
    return {
        "finetune_routing": {
            "cluster_id": cluster_id,
            "adapter_path": adapter_path,
            "confidence": confidence,
            "label": label,
        }
    }


def register_routing_hook():
    """Register the routing hook with the Hermes plugin system."""
    try:
        from hermes_cli.plugins import get_plugin_manager
        manager = get_plugin_manager()
        manager._hooks.setdefault("pre_llm_call", []).append(
            finetune_pre_llm_call_hook
        )
        logger.info("Finetune routing hook registered")
    except ImportError:
        logger.debug("Plugin system not available — routing hook not registered")
    except Exception as e:
        logger.warning("Failed to register routing hook: %s", e)


def unregister_routing_hook():
    """Remove the routing hook."""
    try:
        from hermes_cli.plugins import get_plugin_manager
        manager = get_plugin_manager()
        hooks = manager._hooks.get("pre_llm_call", [])
        manager._hooks["pre_llm_call"] = [
            h for h in hooks if h is not finetune_pre_llm_call_hook
        ]
    except Exception:
        pass
