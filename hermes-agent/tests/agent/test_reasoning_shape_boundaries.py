"""Structured reasoning keeps text and answer channels intact on every chat consumer."""
from evals.providers.reasoning_shapes import run_matrix


def test_stream_reasoning_shapes_preserve_text_and_answer():
    result = run_matrix(("main", "relay", "aux-sync", "aux-async"))
    failures = [row for row in result["results"] if not row["ok"]]
    assert not failures, failures


def test_completed_reasoning_shapes_preserve_text_and_answer():
    result = run_matrix(("nonstream",))
    failures = [row for row in result["results"] if not row["ok"]]
    assert not failures, failures
