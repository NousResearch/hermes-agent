from __future__ import annotations

import json
from pathlib import Path

from hermes_t.cli_shared import build_runtime_parser, build_runtime_profile_from_args, build_runtime_store
from hermes_t.orchestrator import run_profiles_from_config
from hermes_t.tech_data import build_tech_data_provider
from hermes_olin.runtime import run_runtime_cycle


def _build_parser():
    parser = build_runtime_parser(
        description="Run one Hermes generalized T-runtime cycle",
        default_base_dir_name=".hermes_t_runtime",
        channel_env_vars=("HERMES_T_CHANNEL", "HERMES_OLIN_CHANNEL"),
        chat_id_env_vars=("HERMES_T_CHAT_ID", "HERMES_OLIN_CHAT_ID"),
        thread_id_env_vars=("HERMES_T_THREAD_ID", "HERMES_OLIN_THREAD_ID"),
    )
    parser.add_argument("--profiles-config", help="JSON file defining multiple runtime profiles for orchestrated execution.")
    parser.add_argument("--tech-data-config", help="Optional JSON file mapping symbol -> tech_data payload for multi-profile runs.")
    parser.add_argument("--quote-data-config", help="Optional JSON file mapping symbol -> quote payload; when provided, nested quote_payload.tech_data is used as runtime tech_data.")
    parser.add_argument("--quote-snapshot-config", help="Optional JSON file for offline quote snapshots: either a raw symbol -> snapshot map, or a factory config like {source, snapshot_path|snapshots_by_symbol}; nested snapshot.tech_data is used as runtime tech_data.")
    return parser


def _runtime_delivery_kwargs(args) -> dict[str, object]:
    return {
        "effective_trade_date": args.trade_date,
        "dispatch": args.dispatch,
        "channel": args.channel,
        "chat_id": args.chat_id,
        "thread_id": args.thread_id,
    }


def main() -> None:
    args = _build_parser().parse_args()
    default_tech_data = {"summary_signal": args.signal, "score": {"total": args.score}}
    tech_data_provider = build_tech_data_provider(
        tech_data_config_path=args.tech_data_config,
        quote_data_config_path=args.quote_data_config,
        quote_snapshot_config_path=args.quote_snapshot_config,
        default_tech_data=default_tech_data,
    )
    runtime_kwargs = _runtime_delivery_kwargs(args)
    if args.profiles_config:
        payload = run_profiles_from_config(
            config_path=args.profiles_config,
            base_dir=args.base_dir,
            tech_data_provider=tech_data_provider,
            **runtime_kwargs,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    profile = build_runtime_profile_from_args(args)
    store = build_runtime_store(
        base_dir=args.base_dir,
        profile=profile,
        prefer_legacy_olin_store=False,
    )
    payload = run_runtime_cycle(
        store,
        tech_data=tech_data_provider.get(profile.symbol),
        **runtime_kwargs,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
