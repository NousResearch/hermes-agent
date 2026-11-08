"""Readable, distinguishable button labels for the Telegram /model picker.

Display-only: callback data still carries the catalog index, so the full model id is what gets
selected. Bot API button text has no tight length cap; the budget below is what a phone shows
on a full-width button before the client ellipsizes it.
"""

from __future__ import annotations

import os
import re

MODEL_LABEL_BUDGET = 44
_NAMESPACE_DOT = re.compile(r"\.")


def shared_label_prefix(model_ids: list[str]) -> str:
    """The namespace every id shares (``global.anthropic.`` / ``ri.language-model-service..``), cut
    after a dot so no label starts mid-word. Only when some id is over budget (short catalogs such as
    ``gpt-5.1`` / ``gpt-5.2`` keep their names) and computed over the WHOLE list so labels stay stable
    across pages; a single id keeps its full name."""
    if len(model_ids) < 2 or max(len(m) for m in model_ids) <= MODEL_LABEL_BUDGET:
        return ""
    common = os.path.commonprefix(model_ids)
    cut = max((m.end() for m in _NAMESPACE_DOT.finditer(common)), default=0)
    # Never strip an id down to nothing (one id may BE the shared prefix plus a separator).
    return common[:cut] if all(len(m) > cut for m in model_ids) else ""


def model_button_label(model_id: str, prefix: str, suffix: str = "") -> str:
    """``model_id`` without the shared namespace; over budget, the middle goes so the
    distinguishing tail (family, version, date) stays visible."""
    short = model_id[len(prefix):] if prefix and model_id.startswith(prefix) else model_id
    if "/" in short:
        short = short.split("/")[-1]
    room = MODEL_LABEL_BUDGET - (len(suffix) + 3 if suffix else 0)
    if len(short) > room:
        head = max(room // 3, 4)
        short = f"{short[:head]}…{short[-(room - head - 1):]}"
    return f"{short} · {suffix}" if suffix else short
