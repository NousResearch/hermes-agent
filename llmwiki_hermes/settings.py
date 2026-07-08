"""Configuration loading for LLM-Wiki on Hermes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llmwiki_hermes.constants import (
    CONFIG_ENV_VAR,
    DEFAULT_AUTO_WRITEBACK,
    DEFAULT_CONFIG_PATH,
    DEFAULT_TOP_K_EPISODIC,
    DEFAULT_TOP_K_SEMANTIC,
)
from llmwiki_hermes.errors import ConfigurationError


class WikiSettings(BaseModel):
    """Persisted runtime configuration."""

    vault_path: Path
    top_k_semantic: int = Field(default=DEFAULT_TOP_K_SEMANTIC, ge=1, le=50)
    top_k_episodic: int = Field(default=DEFAULT_TOP_K_EPISODIC, ge=1, le=50)
    auto_writeback: bool = DEFAULT_AUTO_WRITEBACK

    @classmethod
    def load(
        cls,
        vault_path: Path | None = None,
        config_path: Path | None = None,
    ) -> "WikiSettings":
        """Load settings from explicit args, config file, env, and defaults."""

        resolved_path = config_path or DEFAULT_CONFIG_PATH
        data: dict[str, Any] = {}
        if resolved_path.exists():
            raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                data = raw
        effective_vault = vault_path
        if effective_vault is None and data.get("vault_path"):
            effective_vault = Path(str(data["vault_path"]))
        if effective_vault is None:
            raw_env = os.environ.get(CONFIG_ENV_VAR)
            if raw_env:
                effective_vault = Path(raw_env)
        if effective_vault is None:
            raise ConfigurationError(
                "Vault path is required. Pass --vault or set LLMWIKI_VAULT_PATH."
            )
        merged = {
            "vault_path": effective_vault,
            "top_k_semantic": data.get("top_k_semantic", DEFAULT_TOP_K_SEMANTIC),
            "top_k_episodic": data.get("top_k_episodic", DEFAULT_TOP_K_EPISODIC),
            "auto_writeback": data.get("auto_writeback", DEFAULT_AUTO_WRITEBACK),
        }
        return cls.model_validate(merged)

    def save(self, config_path: Path | None = None) -> Path:
        """Persist settings to YAML."""

        path = config_path or DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vault_path": str(self.vault_path),
            "top_k_semantic": self.top_k_semantic,
            "top_k_episodic": self.top_k_episodic,
            "auto_writeback": self.auto_writeback,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path
