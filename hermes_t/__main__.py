"""Single-profile CLI entrypoint for hermes_t runtime cycle."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from typing import Sequence

from hermes_t.cli_shared import (
    RuntimeProfile,
    build_runtime_parser,
    build_runtime_profile_from_args,
    build_runtime_store,
)
from hermes_t.orchestrator import run_profiles_from_config
from hermes_t.runtime import dispatch_pending_signal, run_runtime_cycle
from hermes_t.tech_data import build_tech_data_provider


def _resolve_tech_data(args: Namespace, profile: RuntimeProfile) -> dict[str, object]:
    if args.tech_data_config or args.quote_data_config or args.quote_snapshot_config:
        provider = build_tech_data_provider(
            tech_data_config_path=args.tech_data_config,
            quote_data_config_path=args.quote_data_config,
            quote_snapshot_config_path=args.quote_snapshot_config,
            default_tech_data={"signal": str(args.signal), "score": int(args.score)},
        )
        return provider.get(profile.symbol)
    return {
        "signal": str(args.signal),
        "score": int(args.score),
    }


def _build_payload(args: Namespace, profile: RuntimeProfile) -> dict[str, object]:
    store = build_runtime_store(args.base_dir, profile, prefer_legacy_olin_store=False)
    tech_data = _resolve_tech_data(args, profile)
    payload = run_runtime_cycle(
        store=store,
        tech_data=tech_data,
        profile_id=profile.profile_id,
        symbol=profile.symbol,
        trade_unit=profile.trade_unit,
        max_trades=profile.max_trades,
    )
    if args.dispatch:
        payload["dispatch"] = dispatch_pending_signal(
            store=store,
            profile_id=profile.profile_id,
        )
    return payload


def _build_profiles_payload(args: Namespace) -> dict[str, object]:
    provider = build_tech_data_provider(
        tech_data_config_path=args.tech_data_config,
        quote_data_config_path=args.quote_data_config,
        quote_snapshot_config_path=args.quote_snapshot_config,
        default_tech_data={"signal": str(args.signal), "score": int(args.score)},
    )
    return run_profiles_from_config(
        base_dir=args.base_dir,
        profiles_config_path=args.profiles_config,
        tech_data_provider=provider,
        dispatch=args.dispatch,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_runtime_parser()
    args = parser.parse_args(argv)
    if args.profiles_config is not None:
        payload = _build_profiles_payload(args)
    else:
        payload = _build_payload(args, build_runtime_profile_from_args(args))
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
