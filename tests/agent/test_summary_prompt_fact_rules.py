"""Summarizer preamble fact rules: CORRECTIONS WIN and ONE FACT, ONE SECTION.

Behavior contracts, not snapshots:

- The preamble instructs the summarizer that a corrected fact appears ONLY in
  its corrected form, including a fact carried in the previous summary that
  the new turns correct (without the previous-summary clause, a corrected
  value keeps traveling forward through compaction generations inside the
  previous summary the iterative-update prompt feeds back).
- The section list the ONE FACT, ONE SECTION rule enumerates equals the
  model-owned headings the template renders, so a template edit without a
  rule update (or vice versa) fails here instead of drifting silently.
"""

import re
from unittest.mock import patch

from agent.context_compressor import (
    _PRUNED_SKILLS_SECTION_HEADING,
    _SECTION_INSTRUCTIONS,
    _SUMMARY_MODEL_SECTION_NAMES,
    ContextCompressor,
)


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="test", quiet_mode=True)


def test_corrections_win_covers_fact_carried_in_previous_summary():
    prompt = _compressor()._build_summary_prompt("turns", 500, None, "", True)
    assert "CORRECTIONS WIN" in prompt
    assert "carried in the previous summary that the new turns correct" in prompt
    assert "superseded values are dropped" in prompt


def test_one_fact_rule_section_list_matches_rendered_template():
    compressor = _compressor()
    prompt = compressor._build_summary_prompt("turns", 500, None, "", True)
    m = re.search(
        r"ONE FACT, ONE SECTION: record each fact in exactly ONE section of the template \(([^)]*)\)",
        prompt,
    )
    assert m, "ONE FACT, ONE SECTION rule missing from the summarizer prompt"
    rule_sections = {s.strip() for s in m.group(1).replace(", or ", ", ").split(",")}
    assert rule_sections == set(_SUMMARY_MODEL_SECTION_NAMES)

    template = compressor._summary_template_sections(_SECTION_INSTRUCTIONS[True], 500, "")
    headings = {
        line.lstrip("# ").strip()
        for line in template.splitlines()
        if line.startswith("## ")
    }
    # Augmenter-owned sections (Pruned Skills) are not model-owned.
    model_headings = headings - {_PRUNED_SKILLS_SECTION_HEADING.lstrip("# ").strip()}
    assert model_headings == set(_SUMMARY_MODEL_SECTION_NAMES)
