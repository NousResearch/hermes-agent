"""Kilo AI balance provider — fetches from the Kilo profile API."""

from __future__ import annotations

import httpx

from agent.balance_provider import BalanceConfig, BalanceProvider, ProviderBalance


class KiloBalanceProvider(BalanceProvider):
    provider_slug = "kilocode"

    def fetch(self, api_key: str, config: BalanceConfig) -> ProviderBalance:
        endpoint = config.endpoint or "https://api.kilo.ai/api/profile/balance"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(endpoint, headers=headers)
                resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return ProviderBalance(
                provider_name=self.provider_slug,
                label="Kilo AI",
                value=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        balance = float(data.get("balance", 0.0))
        depleted = bool(data.get("isDepleted", False))

        return ProviderBalance(
            provider_name=self.provider_slug,
            label="Kilo AI",
            value=balance,
            is_depleted=depleted,
        )

    @classmethod
    def default_config(cls) -> BalanceConfig:
        return BalanceConfig(
            endpoint="https://api.kilo.ai/api/profile/balance",
            api_key_env="KILOCODE_API_KEY",
            enabled=True,
            cache_ttl_seconds=60.0,
        )