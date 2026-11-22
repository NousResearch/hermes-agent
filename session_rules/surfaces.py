"""The creation-surface rules the store applies when it stamps ``created_source``: the creation labels, the surface each maps to and the labels stored as ``local``.

Standard library only.
"""

from __future__ import annotations


# Creation label -> agent surface. ``tool`` (third-party integrations, ``hermes chat --source tool``) and
# ``oneshot`` (finite ``chat -q`` / ``-z`` runs) run as the CLI but keep their own stored label, so human
# pickers hide them (INTERNAL_LISTING_SOURCES).
SURFACES = {'cli': 'cli', 'tui': 'tui', 'gui': 'desktop', 'acp': 'acp', 'tool': 'cli', 'oneshot': 'cli'}
# ``tool`` and ``oneshot`` stay stored under their own labels so pickers can hide them.
# Every other creation label is rewritten to ``source='local'`` on peer refresh.
_KEEP_STORED_LABEL = frozenset({'tool', 'oneshot'})
STORED_AS_LOCAL = frozenset(SURFACES) - _KEEP_STORED_LABEL
# Routing labels and the accounting placeholder: never a creating surface.
_NOT_A_SURFACE = frozenset({None, '', 'local', 'unknown'})


def created_surface(label):
    """``created_source`` for a policy or live label.

    ``local`` and ``unknown`` are not surfaces. A label stored as ``local`` maps
    through ``SURFACES`` (``gui`` is ``desktop``). Any other label, including
    ``tool`` and ``oneshot``, is kept as itself: created_source is provenance.
    """
    if label in _NOT_A_SURFACE:
        return None
    return SURFACES[label] if label in STORED_AS_LOCAL else label
