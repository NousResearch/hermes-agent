from __future__ import annotations

import json

from hermes_t.cli_shared import build_runtime_parser, build_runtime_profile_from_args, build_runtime_store

from .runtime import run_runtime_cycle


def _build_parser():
    return build_runtime_parser(
        description="Run one Hermes Olin minimal runtime cycle",
        default_base_dir_name=".hermes_olin_runtime",
        channel_env_vars=("HERMES_OLIN_CHANNEL",),
        chat_id_env_vars=("HERMES_OLIN_CHAT_ID",),
        thread_id_env_vars=("HERMES_OLIN_THREAD_ID",),
    )


def main() -> None:
    args = _build_parser().parse_args()
    profile = build_runtime_profile_from_args(args)
    store = build_runtime_store(
        base_dir=args.base_dir,
        profile=profile,
        prefer_legacy_olin_store=True,
    )
    payload = run_runtime_cycle(
        store,
        tech_data={"summary_signal": args.signal, "score": {"total": args.score}},
        effective_trade_date=args.trade_date,
        dispatch=args.dispatch,
        channel=args.channel,
        chat_id=args.chat_id,
        thread_id=args.thread_id,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
