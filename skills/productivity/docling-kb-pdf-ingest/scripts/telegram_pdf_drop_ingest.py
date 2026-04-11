from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdf_ingest_config import INGRESS_TELEGRAM
from pdf_ingest_lib import copy_to_telegram_staging
from pdf_ingest_pipeline import ingest_pdf


def ingest_telegram_pdf(*, cached_path: Path, chat_id: str, message_id: str, original_filename: str | None = None, caption: str | None = None) -> dict:
    staged_path = copy_to_telegram_staging(cached_path, original_filename=original_filename)
    source_context = {
        "telegram_chat_id": chat_id,
        "telegram_message_id": message_id,
        "telegram_file_name": original_filename or cached_path.name,
        "telegram_caption": caption or "",
    }
    result = ingest_pdf(
        pdf_path=staged_path,
        ingress_channel=INGRESS_TELEGRAM,
        source_context=source_context,
        dry_run_promotion=True,
    )
    result["staged_path"] = str(staged_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a Telegram PDF drop into the Hermes evidence layer")
    parser.add_argument("cached_path")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--message-id", required=True)
    parser.add_argument("--original-filename")
    parser.add_argument("--caption", default="")
    args = parser.parse_args()

    payload = ingest_telegram_pdf(
        cached_path=Path(args.cached_path),
        chat_id=args.chat_id,
        message_id=args.message_id,
        original_filename=args.original_filename,
        caption=args.caption,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
