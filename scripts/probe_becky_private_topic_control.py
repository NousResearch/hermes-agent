#!/usr/bin/env python3
"""Guarded, manual proof for private Telegram topic close/reopen."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path


_ID_RE = re.compile(r"^-?\d+$")
_ACK = "DISPOSABLE"
_GUARD = "--i-understand-this-mutates-a-disposable-topic"


def _status_for_code(code: str) -> str:
    if code in {"topic_control_unsupported", "method_not_found"}:
        return "unsupported"
    if code in {"topic_not_found", "chat_not_found", "channel_invalid"}:
        return "not_found"
    if code in {
        "forbidden",
        "topic_forbidden",
        "topic_control_forbidden",
        "chat_admin_required",
    }:
        return "forbidden"
    return "unavailable"


def _valid_target(chat_id: str, thread_id: str) -> bool:
    return (
        bool(_ID_RE.fullmatch(chat_id))
        and bool(_ID_RE.fullmatch(thread_id))
        and thread_id not in {"0", "1"}
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove private-topic close and same-topic reopen on a disposable topic."
    )
    parser.add_argument(_GUARD, action="store_true", dest="mutating_ack")
    return parser.parse_args()


def _read_disposable_ack() -> bool:
    return sys.stdin.readline().rstrip("\r\n") == _ACK


def _read_visual_confirmation() -> bool:
    return sys.stdin.readline() in {"\n", "\r\n"}


async def _run(chat_id: str, thread_id: str) -> int:
    # Keep the repository root importable when this file is launched directly.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv(hermes_home=Path.home() / ".hermes")
    from gateway.telegram_mtproto import (
        MTProtoPrivateTopicController,
        MtprotoTopicControlError,
    )

    controller = MTProtoPrivateTopicController.from_environment(chat_id=chat_id)
    if controller is None:
        print("close=unavailable")
        return 1
    try:
        try:
            await controller.start()
        except MtprotoTopicControlError as error:
            print(f"close={_status_for_code(error.code)}")
            return 1
        try:
            await controller.set_topic_closed(
                chat_id=chat_id, thread_id=thread_id, closed=True
            )
        except MtprotoTopicControlError as error:
            print(f"close={_status_for_code(error.code)}")
            return 1
        print("close=success")
        # The operator must inspect Telegram while the topic is visibly closed.
        if not _read_visual_confirmation():
            print("reopen=unavailable")
            return 1
        try:
            await controller.set_topic_closed(
                chat_id=chat_id, thread_id=thread_id, closed=False
            )
        except MtprotoTopicControlError as error:
            print(f"reopen={_status_for_code(error.code)}")
            return 1
        print("reopen=success")
        return 0
    finally:
        await controller.stop()


def main() -> int:
    args = _parse_args()
    if not args.mutating_ack:
        print("error=guard_required", file=sys.stderr)
        return 2
    chat_id = os.getenv("BECKY_PROOF_CHAT_ID", "").strip()
    thread_id = os.getenv("BECKY_PROOF_THREAD_ID", "").strip()
    if not _valid_target(chat_id, thread_id):
        print("error=disposable_target_required", file=sys.stderr)
        return 2
    if not _read_disposable_ack():
        print("error=acknowledgement_required", file=sys.stderr)
        return 2
    return asyncio.run(_run(chat_id, thread_id))


if __name__ == "__main__":
    raise SystemExit(main())
