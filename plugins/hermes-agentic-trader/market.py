"""Network-safe market normalization for Hermes Agentic Trader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PoolSnapshot:
    network: str
    pool_address: str
    base_token_address: str
    quote_token_address: str
    base_symbol: str
    quote_symbol: str
    liquidity_usd: Optional[float]
    volume_24h_usd: Optional[float]
    price_change_24h_pct: Optional[float]

    @property
    def trade_target_address(self) -> str:
        """The candidate asset address; never the liquidity-pool contract."""
        return self.base_token_address


def build_trending_request(network: str = "base") -> tuple[str, dict[str, Any]]:
    """Return the exact defi-trading-mcp@2.1.3 network-specific request."""
    network_id = str(network).strip().lower()
    if not network_id:
        raise ValueError("network is required")
    return (
        "get_trending_pools_by_network",
        {
            "network": network_id,
            "include": "base_token,quote_token,dex",
            "duration": "24h",
        },
    )


def _decode_json_layers(value: Any, *, max_depth: int = 4) -> Any:
    current = value
    for _ in range(max_depth):
        if isinstance(current, str):
            text = current.strip()
            if not text:
                return current
            try:
                current = json.loads(text)
            except json.JSONDecodeError:
                return current
            continue
        if isinstance(current, dict) and "result" in current:
            candidate = current.get("result")
            if isinstance(candidate, (str, dict)):
                current = candidate
                continue
        break
    return current


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resource_network_and_address(resource_id: Any) -> tuple[str, str]:
    text = str(resource_id or "").strip().lower()
    marker = "_0x"
    index = text.rfind(marker)
    if index <= 0:
        return "", ""
    return text[:index], text[index + 1 :]


def _relationship_id(item: dict[str, Any], name: str) -> str:
    relationships = item.get("relationships")
    if not isinstance(relationships, dict):
        return ""
    relation = relationships.get(name)
    if not isinstance(relation, dict):
        return ""
    data = relation.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("id") or "")


def _included_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    included = payload.get("included")
    if not isinstance(included, list):
        return {}
    return {
        str(item.get("id")): item
        for item in included
        if isinstance(item, dict) and item.get("id")
    }


def _token_details(
    token_id: str,
    included: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    item = included.get(token_id) or {}
    attrs = item.get("attributes") if isinstance(item, dict) else {}
    if not isinstance(attrs, dict):
        attrs = {}

    _network, fallback_address = _resource_network_and_address(token_id)
    address = str(attrs.get("address") or fallback_address or "").lower()
    symbol = str(attrs.get("symbol") or "")
    return address, symbol


def _unwrap_market_payload(result: Any) -> Optional[dict[str, Any]]:
    payload = _decode_json_layers(result)
    if not isinstance(payload, dict):
        return None

    provider = payload.get("data")
    if isinstance(provider, dict) and isinstance(provider.get("data"), list):
        return provider

    if isinstance(payload.get("data"), list):
        return payload
    return None


def parse_trending_result(
    result: Any,
    *,
    expected_network: str,
) -> list[PoolSnapshot]:
    """Normalize only pools that provably belong to the requested network."""
    network = str(expected_network).strip().lower()
    if not network:
        raise ValueError("expected_network is required")

    payload = _unwrap_market_payload(result)
    if payload is None:
        return []

    included = _included_index(payload)
    pools: list[PoolSnapshot] = []

    for item in payload.get("data", []):
        if not isinstance(item, dict) or item.get("type") != "pool":
            continue

        resource_network, _resource_address = _resource_network_and_address(item.get("id"))
        if resource_network and resource_network != network:
            continue

        network_relation = _relationship_id(item, "network")
        if network_relation and network_relation.lower() != network:
            continue

        attrs = item.get("attributes")
        if not isinstance(attrs, dict):
            continue

        pool_address = str(attrs.get("address") or "").lower()
        if not pool_address:
            continue

        base_id = _relationship_id(item, "base_token")
        quote_id = _relationship_id(item, "quote_token")
        base_network, _ = _resource_network_and_address(base_id)
        quote_network, _ = _resource_network_and_address(quote_id)
        if base_network and base_network != network:
            continue
        if quote_network and quote_network != network:
            continue

        base_address, base_symbol = _token_details(base_id, included)
        quote_address, quote_symbol = _token_details(quote_id, included)
        if not base_address or not quote_address:
            continue

        volume = attrs.get("volume_usd")
        volume_24h = volume.get("h24") if isinstance(volume, dict) else None
        changes = attrs.get("price_change_percentage")
        change_24h = changes.get("h24") if isinstance(changes, dict) else None

        pools.append(
            PoolSnapshot(
                network=network,
                pool_address=pool_address,
                base_token_address=base_address,
                quote_token_address=quote_address,
                base_symbol=base_symbol,
                quote_symbol=quote_symbol,
                liquidity_usd=_coerce_float(attrs.get("reserve_in_usd")),
                volume_24h_usd=_coerce_float(volume_24h),
                price_change_24h_pct=_coerce_float(change_24h),
            )
        )

    return pools
