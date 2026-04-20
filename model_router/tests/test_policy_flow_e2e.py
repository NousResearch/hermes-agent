import copy
from pathlib import Path

import yaml

from model_router import (
    Mode,
    Privacy,
    Priority,
    Quota,
    RouterInput,
    TaskType,
    load_config,
    route_model,
    Model,
)
from propose_config_patch import build_patch_proposal
from validate_router_config import validate_router_config


ROOT = Path(__file__).resolve().parent.parent


def load_base_config_dict():
    return yaml.safe_load((ROOT / "router_config.yaml").read_text(encoding="utf-8"))


def test_generated_patch_produces_valid_config_and_changes_routing(tmp_path: Path):
    config = load_base_config_dict()
    config["policy_overrides"] = []

    suggestions = [
        {
            "type": "segment_low_success",
            "severity": "high",
            "segment": {
                "task_type": "chat",
                "priority": "low",
                "quota": "critical",
                "primary_model": "deepseek",
            },
            "message": "בעיה בסגמנט הזה",
        }
    ]

    result = build_patch_proposal(config, suggestions)
    assert result["generated_count"] == 1

    validation = validate_router_config(result["proposed_config"])
    assert validation["valid"] is True

    patched_path = tmp_path / "router_config.yaml"
    patched_path.write_text(yaml.safe_dump(result["proposed_config"], allow_unicode=True, sort_keys=False), encoding="utf-8")
    patched_config = load_config(patched_path)

    decision = route_model(
        RouterInput(
            task_type=TaskType.CHAT,
            mode=Mode.DRAFT,
            priority=Priority.LOW,
            privacy=Privacy.NORMAL,
            quota=Quota.CRITICAL,
        ),
        patched_config,
    )

    assert decision.primary_model == Model.CLAUDE
    assert any("policy_override" in item for item in decision.trace)


def test_build_patch_proposal_does_not_duplicate_existing_override():
    config = load_base_config_dict()
    existing_count = len(config.get("policy_overrides", []))

    suggestions = [
        {
            "type": "segment_low_success",
            "severity": "high",
            "segment": {
                "task_type": "chat",
                "priority": "medium",
                "quota": "critical",
                "primary_model": "deepseek",
            },
            "message": "כבר יש כלל כזה",
        }
    ]

    result = build_patch_proposal(copy.deepcopy(config), suggestions)
    assert result["generated_count"] == 0
    assert len(result["proposed_config"]["policy_overrides"]) == existing_count


def test_build_patch_proposal_skips_overly_broad_override():
    config = load_base_config_dict()
    config["policy_overrides"] = []

    suggestions = [
        {
            "type": "segment_low_success",
            "severity": "high",
            "segment": {
                "priority": "low",
                "primary_model": "deepseek",
            },
            "message": "רחב מדי",
        }
    ]

    result = build_patch_proposal(copy.deepcopy(config), suggestions)
    assert result["generated_count"] == 0
    assert result["generated_overrides"] == []
