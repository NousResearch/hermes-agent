from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from hermes_olin.profile import RuntimeProfile
from hermes_olin.runtime import run_runtime_cycle
from hermes_olin.store import TradingStateStore
from hermes_t.tech_data import TechDataProvider


def _missing_required_fields(item: dict, required_fields: tuple[str, ...]) -> list[str]:
    missing_fields: list[str] = []
    for field in required_fields:
        value = item.get(field)
        if isinstance(value, str):
            if not value.strip():
                missing_fields.append(field)
            continue
        if not value:
            missing_fields.append(field)
    return missing_fields


def _validated_positive_int(*, value: object, field_name: str, idx: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"profile item {idx} field {field_name} must be a positive integer")
    return value


def _build_runtime_profile_from_item(*, item: dict, idx: int) -> RuntimeProfile:
    missing_fields = _missing_required_fields(item, ("profile_id", "symbol"))
    if missing_fields:
        raise ValueError(f"profile item {idx} missing required fields: {', '.join(missing_fields)}")

    for field_name in ("profile_id", "symbol"):
        value = item[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"profile item {idx} field {field_name} must be a non-empty string")

    trade_unit = _validated_positive_int(value=item.get("trade_unit", 10000), field_name="trade_unit", idx=idx)
    max_trades = _validated_positive_int(value=item.get("max_trades", 4), field_name="max_trades", idx=idx)
    return RuntimeProfile(
        profile_id=item["profile_id"],
        symbol=item["symbol"],
        trade_unit=trade_unit,
        max_trades=max_trades,
    )


def load_runtime_profiles_from_json(config_path: str | Path) -> list[RuntimeProfile]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("runtime profiles config must be a JSON list")

    profiles: list[RuntimeProfile] = []
    seen_profile_ids: set[str] = set()
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"profile item {idx} must be a JSON object")

        profile = _build_runtime_profile_from_item(item=item, idx=idx)
        if profile.profile_id in seen_profile_ids:
            raise ValueError(f"duplicate profile_id '{profile.profile_id}' at item {idx}")
        seen_profile_ids.add(profile.profile_id)
        profiles.append(profile)
    return profiles


def run_profiles_from_config(
    *,
    config_path: str | Path,
    base_dir: str | Path,
    tech_data_provider: TechDataProvider,
    effective_trade_date: str,
    dispatch: bool = False,
    channel: str = "feishu",
    chat_id: str | None = None,
    thread_id: str | None = None,
    runner: Callable = run_runtime_cycle,
) -> dict:
    results = []
    for profile in load_runtime_profiles_from_json(config_path):
        store = TradingStateStore(base_dir, profile=profile)
        payload = runner(
            store,
            tech_data=tech_data_provider.get(profile.symbol),
            effective_trade_date=effective_trade_date,
            dispatch=dispatch,
            channel=channel,
            chat_id=chat_id,
            thread_id=thread_id,
        )
        results.append({"profile_id": profile.profile_id, "symbol": profile.symbol, "payload": payload})
    return {"config_path": str(config_path), "total_profiles": len(results), "results": results}
