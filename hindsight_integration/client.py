"""Hindsight client initialization and configuration.

Reads the global ~/.hindsight/config.json when available, falling back
to environment variables.

Config file format:
  {
    "apiKey": "...",
    "banks": {
      "hermes": {
        "bankId": "hermes",
        "budget": "mid",
        "enabled": true
      }
    }
  }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hindsight_client import Hindsight

logger = logging.getLogger(__name__)

GLOBAL_CONFIG_PATH = Path.home() / ".hindsight" / "config.json"
HOST = "hermes"

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_VALID_BUDGETS = {"low", "mid", "high"}


@dataclass
class HindsightClientConfig:
    """Configuration for Hindsight client, resolved for a specific bank."""

    api_key: str | None = None
    base_url: str = _DEFAULT_API_URL
    bank_id: str = HOST
    budget: str = "mid"
    enabled: bool = False

    @classmethod
    def from_env(cls) -> HindsightClientConfig:
        """Create config from environment variables (fallback)."""
        api_key = os.environ.get("HINDSIGHT_API_KEY")
        bank_id = os.environ.get("HINDSIGHT_BANK_ID", HOST)
        budget = os.environ.get("HINDSIGHT_BUDGET", "mid")
        base_url = os.environ.get("HINDSIGHT_API_URL", _DEFAULT_API_URL)
        return cls(
            api_key=api_key,
            base_url=base_url,
            bank_id=bank_id,
            budget=budget if budget in _VALID_BUDGETS else "mid",
            enabled=bool(api_key and bank_id),
        )

    @classmethod
    def from_global_config(
        cls,
        host: str = HOST,
        config_path: Path | None = None,
    ) -> HindsightClientConfig:
        """Create config from ~/.hindsight/config.json.

        Falls back to environment variables if the file doesn't exist.
        """
        path = config_path or GLOBAL_CONFIG_PATH
        if not path.exists():
            logger.debug("No global Hindsight config at %s, falling back to env", path)
            return cls.from_env()

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s, falling back to env", path, e)
            return cls.from_env()

        bank_block = (raw.get("banks") or {}).get(host, {})

        api_key = raw.get("apiKey") or os.environ.get("HINDSIGHT_API_KEY")
        base_url = raw.get("baseUrl") or os.environ.get("HINDSIGHT_API_URL", _DEFAULT_API_URL)
        bank_id = bank_block.get("bankId") or os.environ.get("HINDSIGHT_BANK_ID", host)

        raw_budget = bank_block.get("budget") or os.environ.get("HINDSIGHT_BUDGET", "mid")
        budget = raw_budget if raw_budget in _VALID_BUDGETS else "mid"

        # enabled: bank-level wins, then auto-enable if key + bank_id present
        bank_enabled = bank_block.get("enabled")
        if bank_enabled is not None:
            enabled = bool(bank_enabled)
        else:
            enabled = bool(api_key and bank_id)

        return cls(
            api_key=api_key,
            base_url=base_url,
            bank_id=bank_id,
            budget=budget,
            enabled=enabled,
        )


_hindsight_client: Hindsight | None = None


def get_hindsight_client(config: HindsightClientConfig | None = None) -> Hindsight:
    """Get or create the Hindsight client singleton.

    When no config is provided, attempts to load ~/.hindsight/config.json
    first, falling back to environment variables.
    """
    global _hindsight_client

    if _hindsight_client is not None:
        return _hindsight_client

    if config is None:
        config = HindsightClientConfig.from_global_config()

    if not config.api_key:
        raise ValueError(
            "Hindsight API key not found. "
            "Get your API key at https://app.hindsight.vectorize.io, "
            "then run 'hermes hindsight setup' or set HINDSIGHT_API_KEY."
        )

    try:
        from hindsight_client import Hindsight
    except ImportError:
        raise ImportError(
            "hindsight-client is required for Hindsight integration. "
            "Install it with: pip install 'hindsight-client>=0.4.0'"
        )

    logger.info(
        "Initializing Hindsight client (url: %s, bank: %s)",
        config.base_url,
        config.bank_id,
    )

    _hindsight_client = Hindsight(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=30.0,
    )
    return _hindsight_client


def reset_hindsight_client() -> None:
    """Reset the Hindsight client singleton (useful for testing)."""
    global _hindsight_client
    _hindsight_client = None
