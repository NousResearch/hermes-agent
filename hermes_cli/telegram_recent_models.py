"""Recently used models for the Telegram /model picker (🕘 button).

Mirrors the OpenCode bot's "Recent" list: the last few models the user picked,
most recent first, de-duplicated and capped.

Storage: ``$HERMES_HOME/model_recent.json`` (atomic write). Read/parse failures
degrade to an empty list and never break a chat turn.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

#: how many entries we keep on disk (display is capped separately)
KEEP = 20
#: how many the picker shows by default
DEFAULT_LIMIT = 10

_FILENAME = "model_recent.json"


def _resolve(path=None) -> Path:
    if path is not None:
        return Path(path)
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / _FILENAME


def _load(path: Path) -> list:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "").strip()
        model = str(item.get("model") or "").strip()
        if provider and model:
            out.append({"provider": provider, "model": model})
    return out


def recent_models(limit: int = DEFAULT_LIMIT, path=None) -> list:
    """Most-recently-used models first: ``[{"provider":…, "model":…}, …]``."""
    items = _load(_resolve(path))
    try:
        limit = int(limit)
    except Exception:
        limit = DEFAULT_LIMIT
    return items[:limit] if limit and limit > 0 else items


def record_recent(provider: str, model: str, path=None, keep: int = KEEP) -> bool:
    """Push ``(provider, model)`` to the front of the history.

    Re-picking an existing entry moves it to the front instead of duplicating.
    Returns True when the history file was written.
    """
    provider = str(provider or "").strip()
    model = str(model or "").strip()
    if not provider or not model:
        return False

    target = _resolve(path)
    items = [
        item
        for item in _load(target)
        if not (item["provider"] == provider and item["model"] == model)
    ]
    items.insert(0, {"provider": provider, "model": model})
    items = items[: max(1, int(keep))]

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        return True
    except Exception:
        return False