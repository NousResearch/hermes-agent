from __future__ import annotations

import copy
from pathlib import Path

from .config import resolve_conductor_config
from .models import CampaignPlan, Step


def plan_from_config(definition: dict, config: dict) -> CampaignPlan:
    """Build a campaign using config.yaml's canonical role routing and limits."""
    resolved = resolve_conductor_config(config)
    cwd = Path(str(definition["cwd"])).expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f"campaign cwd does not exist: {cwd}")
    steps = [Step.from_dict(value) for value in definition.get("steps", [])]
    if not steps:
        raise ValueError("campaign requires at least one classified step")
    return CampaignPlan(
        campaign_id=str(definition["campaign_id"]),
        cwd=str(cwd),
        mutable_manifest=[str(path) for path in definition.get("mutable_manifest", [])],
        steps=steps,
        writer=copy.deepcopy(resolved["writer"]),
        reviewer=copy.deepcopy(resolved["reviewer"]),
        budgets=copy.deepcopy(resolved["budgets"]),
    )
