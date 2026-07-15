#!/usr/bin/env python3
"""Encoding-safe search adapter for the packaged SkyVision mail helper.

Python's stdlib IMAP client ASCII-encodes string SEARCH criteria.  A Cyrillic
criterion therefore raises ``UnicodeEncodeError`` before the server can answer.
This adapter makes one deterministic transport choice: criteria containing a
non-ASCII codepoint use the helper's bounded POP3 local-filter implementation.
ASCII criteria retain the helper's IMAP path.  An unexpected UnicodeError on
that path also falls back locally.  No word, sender, subject, or intent is
classified here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


def _load_helper(path: Path) -> ModuleType:
    if not path.is_absolute() or path.name != "skyvision_email_ops.py":
        raise ValueError("skyvision_email_helper_path_invalid")
    spec = importlib.util.spec_from_file_location(
        "_muncho_packaged_skyvision_email_ops", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("skyvision_email_helper_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _non_ascii_criterion(args: argparse.Namespace) -> bool:
    return any(
        isinstance(value, str) and any(ord(character) > 127 for character in value)
        for value in (
            args.from_addr,
            args.to_addr,
            args.subject,
            args.text,
        )
    )


def _emit(value: dict[str, Any]) -> int:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if value.get("status") == "PASS" else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skyvision-email-utf8-search")
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--since")
    parser.add_argument("--from", dest="from_addr")
    parser.add_argument("--to", dest="to_addr")
    parser.add_argument("--subject")
    parser.add_argument("--text")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    helper = _load_helper(args.asset)
    address = helper.account_email(args.account)
    limit = max(1, min(args.limit, helper.MAX_LIMIT))
    if _non_ascii_criterion(args):
        return _emit(
            helper.pop3_search(
                address,
                args,
                limit,
                "imap_ascii_criteria_incompatible",
            )
        )
    try:
        helper.cmd_search(args)
    except UnicodeError:
        return _emit(
            helper.pop3_search(
                address,
                args,
                limit,
                "imap_unicode_encoding_error",
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
