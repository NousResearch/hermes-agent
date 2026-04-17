"""Invariant TDD for agent.task_state (parallel suite)."""
from __future__ import annotations

import random

import pytest

from agent import task_state as ts


CONTINUATION_WORDS = ["sigue", "continua", "continúa", "dale", "ok", "hazlo", "ya"]
CONTINUATION_PHRASES = ["haz lo tuyo", "sigue con eso", "mismo tema", "go on", "keep going"]


@pytest.mark.parametrize("w", CONTINUATION_WORDS)
def test_continuation_word_always_detected(w: str):
    assert ts.is_continuation(w)
    assert ts.is_continuation(w.upper())
    assert ts.is_continuation(f" {w} ")


@pytest.mark.parametrize("p", CONTINUATION_PHRASES)
def test_continuation_phrase_always_detected(p: str):
    assert ts.is_continuation(p)
    assert ts.is_continuation(p.capitalize())


@pytest.mark.parametrize("seed", list(range(8)))
def test_random_new_task_prompts_are_not_continuations(seed: int):
    rng = random.Random(seed)
    new_tasks = [
        "refactor sistema de pagos",
        "analiza la latencia de la API",
        "investiga proveedores de GPUs",
        "escribe un poema de despedida",
        "debug pipeline de datos",
        "implementa cache LRU",
    ]
    for _ in range(5):
        assert not ts.is_continuation(rng.choice(new_tasks))


def test_silence_is_idempotent_on_active_task():
    s = ts.start_task(ts.default_state(), tier=4, model="kimi-k2.5", category="code")
    for _ in range(5):
        s = ts.record_turn(s, "", was_easy=False)
        assert s["active_task"] is True
        assert s["last_tier"] == 4


def test_continuation_never_flips_active_task_off():
    s = ts.start_task(ts.default_state(), tier=3, model="kimi-k2.5", category="analysis")
    for w in CONTINUATION_WORDS + CONTINUATION_PHRASES:
        s = ts.record_turn(s, w, was_easy=False)
        assert s["active_task"] is True
