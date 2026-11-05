# server/config.py
"""Configuration loader — supports coding/reviewers/supervisor structure."""

import yaml
from pathlib import Path
from typing import Optional, List


def load_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_server_config(config: Optional[dict] = None) -> dict:
    if config is None:
        config = load_config()
    return config.get("server", {"host": "0.0.0.0", "port": 8765})


def get_channels(config: Optional[dict] = None) -> dict:
    if config is None:
        config = load_config()
    return config.get("channels", {})


# ── New: coding / reviewers / supervisor ──

def get_coding_config(config: Optional[dict] = None) -> dict:
    """Get the coding (execution) agent configuration."""
    if config is None:
        config = load_config()
    return config.get("coding", {
        "name": "coder",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "temperature": 0.3,
        "max_tokens": 16384,
    })


def get_reviewers_config(config: Optional[dict] = None) -> list[dict]:
    """Get the list of reviewer agent configurations."""
    if config is None:
        config = load_config()
    return config.get("reviewers", [])


def get_supervisor_config(config: Optional[dict] = None) -> dict:
    """Get the supervisor configuration."""
    if config is None:
        config = load_config()
    return config.get("supervisor", {
        "name": "supervisor",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
    })


def get_workflow_config(config: Optional[dict] = None) -> dict:
    if config is None:
        config = load_config()
    return config.get("workflow", {
        "max_iterations": 50,
        "review_timeout_seconds": 180,
        "max_task_retries": 2,
    })


def get_project_config(config: Optional[dict] = None) -> dict:
    if config is None:
        config = load_config()
    return config.get("project", {"workdir": "."})


# ── Backward-compatible wrappers ──

def get_agent_config(config: dict, agent_key: str) -> dict:
    """Legacy: get a specific agent config by its old key.
    Prefer get_coding_config() / get_reviewers_config() for new code."""
    # Try new structure first
    if agent_key == "deepseek" or agent_key == "coding":
        return get_coding_config(config)
    if agent_key == "supervisor":
        return get_supervisor_config(config)
    # Fall back to old agents.{key} structure
    return config.get("agents", {}).get(agent_key, {})
