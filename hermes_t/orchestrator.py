"""Sequential multi-profile orchestration for hermes_t runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hermes_t.cli_shared import RuntimeProfile
from hermes_t.runtime import dispatch_pending_signal, run_runtime_cycle
from hermes_t.store import TradingStateStore, validate_profile_id
from hermes_t.tech_data import TechDataProvider


def _validated_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _validated_non_blank_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _load_profiles_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("profiles config must be a JSON object")
    return payload


def load_runtime_profiles_from_json(path: str | Path) -> list[RuntimeProfile]:
    payload = _load_profiles_payload(path)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("profiles config must contain a non-empty profiles list")

    profiles: list[RuntimeProfile] = []
    for idx, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"profiles[{idx}] must be an object")
        profiles.append(
            RuntimeProfile(
                profile_id=validate_profile_id(_validated_non_blank_string(raw_profile.get("profile_id"), "profile_id")),
                symbol=_validated_non_blank_string(raw_profile.get("symbol"), "symbol"),
                trade_unit=_validated_positive_int(raw_profile.get("trade_unit"), "trade_unit"),
                max_trades=_validated_positive_int(raw_profile.get("max_trades", 4), "max_trades"),
            )
        )
    return profiles


def run_profiles(
    *,
    base_dir: str | Path | None,
    profiles: list[RuntimeProfile],
    tech_data_provider: TechDataProvider,
    dispatch: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for profile in profiles:
        store = TradingStateStore(base_dir=base_dir, profile_id=profile.profile_id)
        payload = run_runtime_cycle(
            store=store,
            tech_data=tech_data_provider.get(profile.symbol),
            profile_id=profile.profile_id,
            symbol=profile.symbol,
            trade_unit=profile.trade_unit,
            max_trades=profile.max_trades,
        )
        if dispatch:
            payload["dispatch"] = dispatch_pending_signal(
                store=store,
                profile_id=profile.profile_id,
            )
        results.append(
            {
                "profile_id": profile.profile_id,
                "symbol": profile.symbol,
                "payload": payload,
            }
        )
    return {
        "profile_count": len(results),
        "results": results,
    }


def run_profiles_from_config(
    *,
    base_dir: str | Path | None,
    profiles_config_path: str | Path,
    tech_data_provider: TechDataProvider,
    dispatch: bool = False,
) -> dict[str, Any]:
    payload = _load_profiles_payload(profiles_config_path)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("profiles config must contain a profiles list")
    if not raw_profiles:
        return {"profile_count": 0, "results": []}
    return run_profiles(
        base_dir=base_dir,
        profiles=load_runtime_profiles_from_json(profiles_config_path),
        tech_data_provider=tech_data_provider,
        dispatch=dispatch,
    )
