"""Regression: only a genuinely COMPLETED turn's final_response may be
adopted as the stream's authoritative finalize payload (PR 85796 review,
B6).

agent/conversation_loop.py's interrupt/abort paths return
{completed: False, interrupted: True, final_response: "Operation
interrupted during …"} with NO failed key. The finish(final_text) gate
checked only `not failed`, so the diagnostic was adopted: it sealed the
user's streamed partial answer over with the interrupt text, AND
delivered_final_matches then reconciled — suppressing the gateway's own
error-delivery path, so the diagnostic was the only thing left.

The taxonomy was enumerated from the production writers (all 27
final_response-bearing return shapes in conversation_loop.py): every
non-happy-path shape carries completed: False; the happy path routes
through turn_finalizer.finalize_turn which computes completed=True.
Gate: completed is not False AND not interrupted AND not failed.
"""

from types import SimpleNamespace


def _adoption_gate(result) -> bool:
    """Mirror of the finish(final_text) adoption gate in gateway/run.py.

    Kept in sync by the comment at the call site; this test pins the
    CONTRACT (which result shapes may move delivery ownership into the
    stream consumer), the wiring test below pins the call.
    """
    return (
        isinstance(result, dict)
        and not result.get("failed")
        and not result.get("interrupted")
        and result.get("completed") is not False
    )


class TestAdoptionGate:
    def test_happy_path_adopts(self):
        assert _adoption_gate(
            {"completed": True, "failed": False, "interrupted": False,
             "final_response": "the answer"}
        )

    def test_interrupt_shape_rejected(self):
        # conversation_loop.py:3089 / 4716 / 6040 / 7279 shape — no failed key.
        assert not _adoption_gate(
            {"completed": False, "interrupted": True,
             "final_response": "Operation interrupted during retry (…)."}
        )

    def test_failed_shape_rejected(self):
        assert not _adoption_gate(
            {"completed": False, "failed": True, "final_response": "boom"}
        )

    def test_diagnostic_noncompleted_shapes_rejected(self):
        # Shapes with neither failed nor interrupted but completed: False —
        # retry exhaustion, truncation, codex-incomplete (all diagnostics).
        for fr in (
            "Response truncated due to output length limit",
            "Codex response remained incomplete after 3 continuation attempts",
        ):
            assert not _adoption_gate({"completed": False, "final_response": fr})

    def test_legacy_result_without_completed_key_adopts(self):
        # Duck-type safety: a result dict lacking the key entirely (older
        # callers/doubles) keeps the pre-review behavior.
        assert _adoption_gate({"final_response": "answer"})


class TestGateWiring:
    def test_run_py_gate_matches_contract(self):
        """The gate in gateway/run.py must contain the interrupted and
        completed checks — a source-level pin so the contract test above
        cannot drift green while the call site regresses."""
        import inspect
        import gateway.run as run_mod

        src = inspect.getsource(run_mod)
        anchor = src.index("_final_for_stream = None")
        window = src[anchor : anchor + 1200]
        assert 'not result.get("interrupted")' in window
        assert 'result.get("completed") is not False' in window
