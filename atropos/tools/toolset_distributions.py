"""
Toolset distributions (Hermes-Agent inspired).

Distributions are optional helpers for data generation runs where you want
probabilistic inclusion of toolsets per trajectory/item.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, TypedDict

from .toolsets import validate_toolset


class DistributionDef(TypedDict):
    description: str
    toolsets: Dict[str, int]


DISTRIBUTIONS: Dict[str, DistributionDef] = {
    "default": {
        "description": "All common sandbox tools.",
        "toolsets": {"sandbox": 100},
    },
    "code_agent_plus_image": {
        "description": "Sandbox tools with optional image generation.",
        "toolsets": {"sandbox": 100, "image_gen": 30},
    },
    "sandbox_only": {
        "description": "Only sandbox tools (terminal + filesystem).",
        "toolsets": {"sandbox": 100},
    },
}


def get_distribution(name: str) -> Optional[DistributionDef]:
    return DISTRIBUTIONS.get(name)


def list_distributions() -> Dict[str, DistributionDef]:
    return DISTRIBUTIONS.copy()


def validate_distribution(name: str) -> bool:
    return name in DISTRIBUTIONS


def sample_toolsets_from_distribution(distribution_name: str) -> List[str]:
    dist = get_distribution(distribution_name)
    if not dist:
        raise ValueError(f"Unknown distribution: {distribution_name}")

    selected: List[str] = []
    for toolset_name, probability in dist["toolsets"].items():
        if not validate_toolset(toolset_name):
            continue
        if random.random() * 100 < probability:
            selected.append(toolset_name)

    # Ensure at least one toolset if the distribution isn't empty.
    if not selected and dist["toolsets"]:
        highest = max(dist["toolsets"].items(), key=lambda x: x[1])[0]
        if validate_toolset(highest):
            selected.append(highest)

    return selected

