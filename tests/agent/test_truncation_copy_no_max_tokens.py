"""Regression test for #118333.

Hermes no longer reads ``model.max_tokens``, ``HERMES_MAX_TOKENS``, provider
output-cap settings, or ``model_overrides.*.*.max_output_tokens`` (see
``website/docs/integrations/providers.md``, "Context Length Detection"). The
truncation-recovery copy must not tell the user to raise/increase max_tokens
for a knob that does not exist and cannot change the outcome.
"""

from __future__ import annotations

from agent.turn_failure_copy import site_copy
from agent.turn_truncation import _CEILING_NO_TEXT, _THINKING_EXHAUSTED


def test_truncated_site_copy_does_not_mention_max_tokens():
    copy = site_copy("truncated")
    assert "max_tokens" not in copy
    assert "/reasoning low" in copy


def test_thinking_exhausted_copy_does_not_mention_max_tokens():
    for variant in _THINKING_EXHAUSTED:
        assert "max_tokens" not in variant
    assert "lowering reasoning effort" in _THINKING_EXHAUSTED[2]


def test_ceiling_no_text_copy_does_not_mention_max_tokens():
    assert "max_tokens" not in _CEILING_NO_TEXT
    assert "/model" in _CEILING_NO_TEXT
