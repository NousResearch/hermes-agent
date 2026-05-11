"""CLI helpers for the Cognee memory provider."""

from __future__ import annotations

from getpass import getpass
from typing import Any

from hermes_cli.config import get_env_path, save_env_value


def _prompt(default: str = "", *, label: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def cmd_setup(args: Any = None) -> None:
    """Interactive Cognee setup.

    Stores secrets in Hermes' profile-scoped .env file. The generic
    ``hermes memory setup`` path can use CogneeMemoryProvider.get_config_schema();
    this helper exists for plugin-specific routing if needed later.
    """

    print("\nCognee memory setup\n")
    api_key = getpass("LLM API key for Cognee (LLM_API_KEY): ").strip()
    provider = _prompt("openai", label="LLM provider (LLM_PROVIDER)")
    base_url = _prompt("", label="Optional LLM base URL (LLM_BASE_URL)")
    dataset_name = _prompt("hermes_memory", label="Default Cognee dataset (COGNEE_DATASET_NAME)")

    if api_key:
        save_env_value("LLM_API_KEY", api_key)
    if provider:
        save_env_value("LLM_PROVIDER", provider)
    if base_url:
        save_env_value("LLM_BASE_URL", base_url)
    if dataset_name:
        save_env_value("COGNEE_DATASET_NAME", dataset_name)

    print(f"\nSaved Cognee environment config to {get_env_path()}")
    print("Enable it with: hermes config set memory.provider cognee")
    print("Restart Hermes after enabling the provider.\n")
