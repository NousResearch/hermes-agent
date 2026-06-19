"""Slash handler for /ai-employees."""

from __future__ import annotations

import json

from . import core


def handle_slash(raw_args: str) -> str:
    args = (raw_args or "").strip().lower()
    if args in ("", "status"):
        return json.dumps(core.status(), ensure_ascii=False, indent=2)
    if args == "install":
        result = core.install_all(dry_run=False)
        return json.dumps(result, ensure_ascii=False, indent=2)
    return (
        "Usage: /ai-employees [status|install]\n"
        "CLI: hermes ai-employees install"
    )
