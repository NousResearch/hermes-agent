"""Canonical entry boundary for the Hermes CLI.

Command ownership moves here incrementally. Commands not yet migrated cross exactly
one explicit legacy seam in :mod:`nous_cli.legacy`.
"""

from __future__ import annotations

from nous_cli.legacy import dispatch_legacy


def main():
    """Dispatch through the strangler boundary."""
    return dispatch_legacy()
