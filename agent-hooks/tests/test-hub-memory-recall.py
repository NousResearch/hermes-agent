#!/usr/bin/env python3
"""PERS-117: gbrain scores overlap, so score-only recall is noise.

Live hub measurement (recall_fast, 2026-08-28):

    telegram flood control ban   real     0.152, 0.152, 0.152, 0.152
    zzqqxx-style junk            nonsense scores in the same band

A floor at 0.10 keeps both. A floor above 0.152 drops the real query too.
The filter must require lexical overlap so signal stays and nonsense is silent.

Run:
  python3 agent-hooks/tests/test-hub-memory-recall.py
  pytest tests/agent/test_hub_memory_recall.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FILTER_PATH = Path(__file__).resolve().parents[1] / "gbrain_semantic_filter.py"

failures: list[str] = []


def _load_filter():
    spec = importlib.util.spec_from_file_location("gbrain_semantic_filter", FILTER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {FILTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIGNAL_PROMPT = "telegram flood control ban"
NONSENSE_PROMPT = "zzqqxx blorptrix vvvv wwww quandle fizzbap"

# Same numeric score on a relevant hit and an unrelated hit — the PERS-117 trap.
SIGNAL_HIT = {
    "title": "Telegram flood-limit prevention",
    "score": "0.152",
    "text": "Telegram flood control ban after too many outbound messages.",
}
NOISE_HIT = {
    "title": "WhatsApp both-accounts rule",
    "score": "0.152",
    "text": "Always check personal_nl AND personal_us, sent and received.",
}
LOW_HIT = {
    "title": "Telegram flood control ban",
    "score": "0.001",
    "text": "Telegram flood control ban, but scored far below the floor.",
}


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def test_overlapping_0_152_score_keeps_signal_query() -> None:
    filt = _load_filter()
    kept = filt.filter_gbrain_hits(SIGNAL_PROMPT, [SIGNAL_HIT, NOISE_HIT])
    texts = " ".join(str(h.get("text") or "") for h in kept)
    assert kept, "signal query at 0.152 must still recall"
    assert "flood control" in texts
    assert "personal_us" not in texts


def test_overlapping_0_152_score_ignores_nonsense_query() -> None:
    filt = _load_filter()
    kept = filt.filter_gbrain_hits(NONSENSE_PROMPT, [SIGNAL_HIT, NOISE_HIT])
    assert kept == [], f"nonsense at 0.152 must be silent, got {kept!r}"


def test_below_floor_score_is_dropped_even_with_word_overlap() -> None:
    filt = _load_filter()
    kept = filt.filter_gbrain_hits(SIGNAL_PROMPT, [LOW_HIT])
    assert kept == []


def test_empty_prompt_does_not_wipe_the_tier() -> None:
    filt = _load_filter()
    kept = filt.filter_gbrain_hits("", [SIGNAL_HIT, NOISE_HIT])
    assert len(kept) == 2


def main() -> int:
    print("hub_memory_recall PERS-117 regression")
    tests = [
        ("overlapping 0.152 keeps signal", test_overlapping_0_152_score_keeps_signal_query),
        ("overlapping 0.152 ignores nonsense", test_overlapping_0_152_score_ignores_nonsense_query),
        ("below-floor score is dropped", test_below_floor_score_is_dropped_even_with_word_overlap),
        ("empty prompt does not wipe the tier", test_empty_prompt_does_not_wipe_the_tier),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            check(name, False, f"{type(exc).__name__}: {exc}")
        else:
            check(name, True)
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
