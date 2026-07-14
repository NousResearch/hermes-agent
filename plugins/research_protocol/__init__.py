
"""Opt-in Hermes Research Protocol plugin.

PR 0 intentionally registers no runtime behavior.  Keeping the loader entry
point valid lets discovery distinguish an explicitly enabled empty baseline
from a malformed plugin.
"""


def register(_ctx) -> None:
    """Register no tools, hooks, or commands until the gated runtime PRs."""
