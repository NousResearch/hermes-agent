"""Input-area placeholder rendering for the classic CLI composer."""

from prompt_toolkit.layout.processors import Processor, Transformation


class _PlaceholderProcessor(Processor):
    """Render grayed-out placeholder text inside the input when empty."""
    def __init__(self, get_text):
        self._get_text = get_text

    def apply_transformation(self, ti):
        if not ti.document.text and ti.lineno == 0:
            # A parked next-prompt suggestion is already appended as ghost text; the example
            # placeholder would render glued onto it.
            suggestion = ti.buffer_control.buffer.suggestion
            if suggestion is not None and suggestion.text:
                return Transformation(fragments=ti.fragments)
            text = self._get_text()
            if text:
                # Append after existing fragments (preserves the ❯ prompt).
                return Transformation(fragments=ti.fragments + [('class:placeholder', text)])
        return Transformation(fragments=ti.fragments)
