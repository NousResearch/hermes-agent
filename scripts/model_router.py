#!/usr/bin/env python3
"""Model router CLI — score a prompt against routing heuristics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import routing helpers
from agent.smart_model_routing import (
    choose_cheap_model_route,
    has_opus_keyword,
    is_continuation_turn,
)
from agent.routing_telemetry import multiplier_for


def load_config(config_path: Path) -> dict:
    """Load routing config from YAML file."""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback if yaml not available
        return {}
    except FileNotFoundError:
        return {}


def get_default_config() -> dict:
    """Return minimal inline config for routing."""
    return {
        "smart_model_routing": {
            "enabled": True,
            "cheap_model": {
                "model": "gpt-5-mini",
                "provider": "copilot",
            },
            "model": {
                "default": {
                    "model": "claude-opus-4.6",
                    "provider": "copilot",
                }
            }
        }
    }


def decide_route(prompt: str, config: dict) -> tuple:
    """Decide which model to use and why.
    
    Returns (model, provider, reason, multiplier).
    """
    routing_config = config.get("smart_model_routing", {})
    primary = config.get("smart_model_routing", {}).get("model", {}).get("default", {})
    
    if not primary:
        primary = {"model": "claude-opus-4.6", "provider": "copilot"}
    
    # Check for OPUS keyword
    if has_opus_keyword(prompt):
        model = primary.get("model", "claude-opus-4.6")
        provider = primary.get("provider", "copilot")
        reason = "opus_keyword"
        return model, provider, reason, multiplier_for(model)
    
    # Check for continuation
    if is_continuation_turn(prompt):
        model = primary.get("model", "claude-opus-4.6")
        provider = primary.get("provider", "copilot")
        reason = "continuation"
        return model, provider, reason, multiplier_for(model)
    
    # Try cheap model route
    route = choose_cheap_model_route(prompt, routing_config)
    if route:
        model = route.get("model", "gpt-5-mini")
        provider = route.get("provider", "copilot")
        reason = "simple_turn"
        return model, provider, reason, multiplier_for(model)
    
    # Default to primary
    model = primary.get("model", "claude-opus-4.6")
    provider = primary.get("provider", "copilot")
    reason = "primary_default"
    return model, provider, reason, multiplier_for(model)


def main():
    parser = argparse.ArgumentParser(
        description="Route a prompt to the appropriate model",
        prog="model_router",
    )
    parser.add_argument("prompt", help="The prompt text to route")
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and args.config.exists():
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Decide route
    model, provider, reason, multiplier = decide_route(args.prompt, config)
    
    # Print result
    print(f"decision: {model} ({provider})")
    print(f"reason: {reason}")
    print(f"multiplier: {multiplier}x")


if __name__ == "__main__":
    main()
