"""Interrupt placeholder wording substituted into empty non-final wire rows.

Split out of ``agent/agent_runtime_helpers.py`` (code-health size ratchet); a leaf module with no
imports so every reader (runtime helpers, turn prep, the conversation loop) can import it directly.
"""

# Placeholder for an empty non-final message the provider would reject. Kept identical to the stub
# placeholder in chat_completion_helpers so healed transcripts read consistently.
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
