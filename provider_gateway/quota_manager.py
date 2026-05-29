"""Quota manager and guard for the opt-in provider gateway.

Checks accumulated USD spends against configured daily and monthly limits
to prevent out-of-control LLM budget exhaustion.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Any

from provider_gateway.usage_tracker import ProviderUsageTracker

logger = logging.getLogger(__name__)


class QuotaExceededError(RuntimeError):
    """Exception raised when API usage quota is exceeded."""
    pass


class QuotaManager:
    """Manages LLM budget consumption tracking and quota enforcement."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = ProviderUsageTracker().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Establish a WAL connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def get_daily_spend(self) -> float:
        """Get total cost in USD for today since 00:00:00 local time."""
        now = datetime.datetime.now()
        start_of_today = datetime.datetime(now.year, now.month, now.day)
        start_epoch = start_of_today.timestamp()

        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT SUM(estimated_cost_usd) 
                FROM provider_usage 
                WHERE created_at >= ? AND status = 'success'
                """,
                (start_epoch,),
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as exc:
            logger.debug("Failed to calculate daily spend: %s", exc)
            return 0.0
        finally:
            conn.close()

    def get_monthly_spend(self) -> float:
        """Get total cost in USD for this month since 1st day 00:00:00 local time."""
        now = datetime.datetime.now()
        start_of_month = datetime.datetime(now.year, now.month, 1)
        start_epoch = start_of_month.timestamp()

        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT SUM(estimated_cost_usd) 
                FROM provider_usage 
                WHERE created_at >= ? AND status = 'success'
                """,
                (start_epoch,),
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as exc:
            logger.debug("Failed to calculate monthly spend: %s", exc)
            return 0.0
        finally:
            conn.close()

    def check_quota(self, agent: Any) -> bool:
        """Check budget spends. If limits exceeded, apply block (raise error) or fallback.

        Returns True if quota is within limits or successfully falls back.
        Raises QuotaExceededError when block action triggers.
        """
        config = getattr(agent, "_provider_gateway_config", None)
        if config is None or not config.enabled:
            return True

        daily_limit = config.daily_limit_usd
        monthly_limit = config.monthly_limit_usd

        if daily_limit is None and monthly_limit is None:
            return True

        daily_spend = self.get_daily_spend()
        monthly_spend = self.get_monthly_spend()

        exceeded = False
        reason = ""

        if daily_limit is not None and daily_spend >= daily_limit:
            exceeded = True
            reason = f"Daily budget limit exceeded! Spend: {daily_spend:.4f} USD, Limit: {daily_limit:.4f} USD"

        if monthly_limit is not None and monthly_spend >= monthly_limit:
            exceeded = True
            reason = f"Monthly budget limit exceeded! Spend: {monthly_spend:.4f} USD, Limit: {monthly_limit:.4f} USD"

        if exceeded:
            logger.warning("Quota Guard Triggered: %s. Action: %s", reason, config.quota_action)
            if config.quota_action == "fallback":
                logger.info("Quota Guard: falling back to local Ollama (free of charge).")
                agent.provider = "ollama"
                # If we have fallback models, use the first one, otherwise use llama3
                if config.fallback_models:
                    agent.model = config.fallback_models[0]
                else:
                    agent.model = "llama3"
                # Route to local free endpoint
                agent.base_url = "http://localhost:11434/v1"
                agent.api_key = "ollama"
                agent.api_mode = "chat_completions"
                return True
            else:
                raise QuotaExceededError(reason)

        return True
