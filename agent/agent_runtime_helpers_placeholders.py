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
_INTERRUPT_SCAFFOLD_MARKER = "[This response was interrupted by a user correction.]"
_INTERRUPT_ECHO_HAZARDS = frozenset({
    _INTERRUPTED_PLACEHOLDER,
    _LEGACY_INTERRUPTED_PLACEHOLDER,
    _INTERRUPT_SCAFFOLD_MARKER,
})
_ECHOED_PLACEHOLDER_NUDGE = (
    "Your previous response only repeated an internal interrupt placeholder instead of answering. "
    "Please respond directly to the user's message above."
)


def is_interrupt_placeholder_echo(text: str | None) -> bool:
    """True when ``text`` (stripped) is one of Hermes's interrupt placeholder/scaffold markers (#150052)."""
    return isinstance(text, str) and text.strip() in _INTERRUPT_ECHO_HAZARDS


def has_content_after_think_block(agent, content: str | None) -> bool:
    """True when non-placeholder text remains after stripping reasoning blocks (#150052)."""
    return bool(content) and bool(s := agent._strip_think_blocks(content).strip()) and not is_interrupt_placeholder_echo(s)


def hidden_interrupt_placeholder_row() -> dict:
    """A fresh assistant row that is invisible in transcripts but carries the placeholder as its
    non-empty ``api_content``, so the pre-call sanitizer does not re-heal it every call (#88955)."""
    return {"role": "assistant", "content": "", "display_kind": "hidden", "api_content": _INTERRUPTED_PLACEHOLDER}
