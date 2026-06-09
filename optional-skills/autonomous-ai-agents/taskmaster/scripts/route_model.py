#!/usr/bin/env python3
"""Model routing table for TaskMaster skill.

Resolves the best model for a given complexity tier based on available
providers. Outputs JSON so the agent can parse it programmatically.

Usage:
    python3 route_model.py --tier LOW
    python3 route_model.py --tier HIGH --provider openrouter
    python3 route_model.py --list-tiers
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Routing table — models grouped by complexity tier and provider
# ---------------------------------------------------------------------------

ROUTING_TABLE: Dict[str, Dict[str, List[str]]] = {
    "LOW": {
        "openrouter": [
            "google/gemini-2.0-flash-001",
            "openai/gpt-4o-mini",
        ],
        "google": ["gemini-2.0-flash"],
        "openai": ["gpt-4o-mini"],
        "zai": ["glm-4-flash"],
    },
    "MEDIUM": {
        "openrouter": [
            "google/gemini-2.5-flash-preview-05-20",
            "openai/gpt-4o",
        ],
        "google": ["gemini-2.5-flash"],
        "openai": ["gpt-4o"],
        "zai": ["glm-5.1"],
    },
    "HIGH": {
        "openrouter": [
            "anthropic/claude-sonnet-4",
            "google/gemini-2.5-pro-preview-06-05",
        ],
        "anthropic": ["claude-sonnet-4"],
        "google": ["gemini-2.5-pro"],
    },
    "VOTE": {
        "openrouter": [
            "google/gemini-2.5-flash-preview-05-20",
            "openai/gpt-4o",
            "meta-llama/llama-4-maverick",
        ],
        "google": ["gemini-2.5-flash"],
    },
}

TIER_DESCRIPTIONS = {
    "LOW": "Formatting, extraction, simple lookups, file I/O",
    "MEDIUM": "Summarization, drafting, categorization, comparison",
    "HIGH": "Complex reasoning, code generation, architectural decisions",
    "VOTE": "Multi-model consensus for quality gating",
}


def resolve_model(tier: str, provider: Optional[str] = None) -> dict:
    """Resolve the best model for a given tier and optional provider.

    Returns a dict with the resolved model, provider, and tier info.
    """
    tier = tier.upper()
    if tier not in ROUTING_TABLE:
        return {
            "error": f"Unknown tier '{tier}'. Valid: {list(ROUTING_TABLE.keys())}",
        }

    tier_models = ROUTING_TABLE[tier]

    if provider:
        provider_models = tier_models.get(provider)
        if not provider_models:
            available_providers = list(tier_models.keys())
            return {
                "error": (
                    f"Provider '{provider}' has no models for tier '{tier}'. "
                    f"Available providers: {available_providers}"
                ),
            }
        return {
            "tier": tier,
            "provider": provider,
            "model": provider_models[0],
            "alternatives": provider_models[1:],
            "description": TIER_DESCRIPTIONS[tier],
        }

    # No provider specified — return first available across all providers
    for prov, models in tier_models.items():
        return {
            "tier": tier,
            "provider": prov,
            "model": models[0],
            "alternatives": models[1:],
            "all_providers": {
                p: ms for p, ms in tier_models.items()
            },
            "description": TIER_DESCRIPTIONS[tier],
        }

    return {"error": f"No models available for tier '{tier}'."}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TaskMaster model routing table",
    )
    parser.add_argument(
        "--tier",
        choices=["LOW", "MEDIUM", "HIGH", "VOTE"],
        help="Complexity tier to resolve",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Specific provider to use (optional)",
    )
    parser.add_argument(
        "--list-tiers",
        action="store_true",
        help="List all tiers and their models",
    )

    args = parser.parse_args()

    if args.list_tiers:
        output = {}
        for tier, desc in TIER_DESCRIPTIONS.items():
            output[tier] = {
                "description": desc,
                "models": ROUTING_TABLE[tier],
            }
        print(json.dumps(output, indent=2))
        return

    if not args.tier:
        parser.error("--tier is required (or use --list-tiers)")

    result = resolve_model(args.tier, args.provider)
    print(json.dumps(result, indent=2))

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
