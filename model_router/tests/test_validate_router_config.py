import copy
from pathlib import Path

import yaml

from validate_router_config import validate_router_config


ROOT = Path(__file__).resolve().parent.parent


def load_base_config_dict():
    return yaml.safe_load((ROOT / "router_config.yaml").read_text(encoding="utf-8"))


def test_validate_router_config_accepts_current_config():
    config = load_base_config_dict()
    result = validate_router_config(config)

    assert result["valid"] is True
    assert result["errors"] == []



def test_validate_router_config_rejects_invalid_policy_values():
    config = load_base_config_dict()
    config["policy_overrides"] = [
        {
            "name": "bad-values",
            "when": {
                "task_type": "sales",
                "has_code": "yes",
                "privacy": "private",
            },
            "force": "claude-sonnet-4.6",
        }
    ]

    result = validate_router_config(config)

    assert result["valid"] is False
    assert any("task_type" in item and "ערך לא חוקי" in item for item in result["errors"])
    assert any("has_code" in item and "boolean" in item for item in result["errors"])
    assert any("privacy" in item and "ערך לא חוקי" in item for item in result["errors"])



def test_validate_router_config_warns_on_broad_policy_override():
    config = load_base_config_dict()
    config["policy_overrides"] = [
        {
            "name": "too-broad",
            "when": {"priority": "low"},
            "force": "deepseek",
        }
    ]

    result = validate_router_config(config)

    assert result["valid"] is True
    assert any("רחב מאוד" in item for item in result["warnings"])
    assert any("לא כולל task_type או quota" in item for item in result["warnings"])



def test_validate_router_config_warns_when_local_only_force_is_not_ollama():
    config = load_base_config_dict()
    config["policy_overrides"] = [
        {
            "name": "bad-local-only-force",
            "when": {
                "task_type": "chat",
                "privacy": "local_only",
            },
            "force": "claude-sonnet-4.6",
        }
    ]

    result = validate_router_config(config)

    assert result["valid"] is True
    assert any("privacy=local_only" in item and "ollama" in item for item in result["warnings"])



def test_validate_router_config_warns_on_duplicate_and_shadowed_policy_rules():
    config = load_base_config_dict()
    config["policy_overrides"] = [
        {
            "name": "chat-critical",
            "when": {
                "task_type": "chat",
                "quota": "critical",
            },
            "force": "claude-sonnet-4.6",
        },
        {
            "name": "chat-critical",
            "when": {
                "task_type": "chat",
                "quota": "critical",
            },
            "force": "claude-sonnet-4.6",
        },
        {
            "name": "chat-critical-low",
            "when": {
                "task_type": "chat",
                "quota": "critical",
                "priority": "low",
            },
            "force": "gpt-5.4",
        },
    ]

    result = validate_router_config(config)

    assert result["valid"] is True
    assert any("name כפול" in item for item in result["warnings"])
    assert any("when זהה" in item for item in result["warnings"])
    assert any("לא יופעל" in item or "כלל מאוחר מיותר" in item for item in result["warnings"])



def test_validate_router_config_warns_on_duplicate_fallback_entries():
    config = load_base_config_dict()
    config["fallbacks"] = copy.deepcopy(config["fallbacks"])
    config["fallbacks"]["claude-sonnet-4.6"] = ["gpt-5.4", "gpt-5.4", "deepseek"]

    result = validate_router_config(config)

    assert result["valid"] is True
    assert any("כפילות" in item and "gpt-5.4" in item for item in result["warnings"])
