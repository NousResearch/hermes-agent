"""Interrupt placeholder wording substituted into empty non-final wire rows.

Split out of ``agent/agent_runtime_helpers.py`` (code-health size ratchet); a leaf module with no
imports so every reader (runtime helpers, turn prep, the conversation loop) can import it directly.
"""

# Placeholder for an empty non-final message the provider would reject. This is the single source of
# the wording: every healing path imports it from here, so healed transcripts read consistently.
#
# Wording matters: this text is substituted into an assistant row's ``content`` on the wire copy
# (``fill_empty_non_final_wire_payload``, ``repair_empty_non_final_messages``, and the projection in
# ``run_conversation``), so the model reads it as something IT said. A short natural-language phrase
# in that position gets echoed verbatim — a clean ``finish_reason=stop`` turn answering an ordinary
# instruction with just the placeholder (#132949, and #81841 for the same hazard on the scaffold).
# Keep it a structural label the model has no conversational reason to reproduce, never prose.
_INTERRUPTED_PLACEHOLDER = "[interrupt: no assistant output for this turn]"

# The spelling shipped before #132949. Models reproduce it verbatim, so rows already carrying it
# stay poisoned after the wording change; the replay filters reference this name to retire them.
_LEGACY_INTERRUPTED_PLACEHOLDER = "[response interrupted]"


def hidden_interrupt_placeholder_row(reasoning: str = "") -> dict:
    """A fresh assistant row that is invisible in transcripts but carries the placeholder as its
    non-empty ``api_content``, so the pre-call sanitizer does not re-heal it every call (#88955).

    ``reasoning`` attaches reasoning already streamed before the interrupt so it survives the
    persist (#134946); it lives in ``reasoning_content`` only — never ``content``/``api_content``,
    which the wire replays byte-for-byte. An empty ``reasoning_content`` key is omitted so the
    row keeps its exact pre-#134946 shape for every caller with no reasoning."""
    row = {"role": "assistant", "content": "", "display_kind": "hidden", "api_content": _INTERRUPTED_PLACEHOLDER}
    if isinstance(reasoning, str) and reasoning.strip():
        row["reasoning_content"] = reasoning
    return row
