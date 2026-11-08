"""Readable, distinguishable button labels for the Telegram /model picker.

Display-only: callback data still carries the catalog index, so the full model id is what gets
selected. Bot API button text has no tight length cap; the budget below is what a phone shows
on a full-width button before the client ellipsizes it.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Callable

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


def _strip(model_id: str, prefix: str) -> str:
    short = model_id[len(prefix):] if prefix and model_id.startswith(prefix) else model_id
    return short.split("/")[-1]


def _with_suffix(short: str, suffix: str) -> str:
    return f"{short} · {suffix}" if suffix else short


def _room(suffix: str) -> int:
    return MODEL_LABEL_BUDGET - (len(suffix) + 3 if suffix else 0)


def model_button_label(model_id: str, prefix: str, suffix: str = "") -> str:
    """``model_id`` without the shared namespace; over budget, the middle goes so the
    distinguishing tail (family, version, date) stays visible."""
    short = _strip(model_id, prefix)
    room = _room(suffix)
    if len(short) > room:
        head = max(room // 3, 4)
        short = f"{short[:head]}…{short[-(room - head - 1):]}"
    return _with_suffix(short, suffix)


def _around_difference(name: str, others: list[str], room: int) -> str:
    """``name`` shortened to ``room`` keeping the span where it differs from its look-alikes:
    a short head, then a window starting just before the first differing character."""
    if len(name) <= room:
        return name
    diff = max(len(os.path.commonprefix([name, o])) for o in others)
    if diff < room - 4:
        return name[:room - 1] + "…"
    head = max(room // 4, 4)
    start = max(diff - 4, head + 1)
    body = name[start:start + room - head - 1]
    if start + len(body) < len(name):
        body = body[:-1] + "…"
    return f"{name[:head]}…{body}"


def _clashes(labels: list[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    return [g for g in groups.values() if len(g) > 1]


def model_button_labels(model_ids: list[str], suffix_for: Callable[[str], str]) -> list[str]:
    """Labels for the WHOLE catalog (stable across pages), unique within it: ids whose head/tail
    names collide are re-shortened around the span where they differ, and an ordinal is the last
    resort so two buttons never read the same while selecting different models."""
    prefix = shared_label_prefix(model_ids)
    names = [_strip(m, prefix) for m in model_ids]
    suffixes = [suffix_for(m) for m in model_ids]
    labels = [model_button_label(m, prefix, s) for m, s in zip(model_ids, suffixes)]
    # Compare names without the plan word: ``a…tail · Large`` beside ``a…tail`` is still ambiguous.
    for group in _clashes([model_button_label(m, prefix) for m in model_ids]):
        for i in group:
            others = [names[j] for j in group if names[j] != names[i]] or [""]
            labels[i] = _with_suffix(_around_difference(names[i], others, _room(suffixes[i])), suffixes[i])
    for group in _clashes(labels):
        for n, i in enumerate(group, 1):
            tag = f" #{n}"
            labels[i] = _with_suffix(_around_difference(names[i], [""], _room(suffixes[i]) - len(tag)) + tag, suffixes[i])
    return labels
