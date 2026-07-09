"""Brand alias for :mod:`hermes_constants` (Phase 6).

``import ht_constants`` returns the exact same module object as
``import hermes_constants`` — the two names are interchangeable. ``hermes_constants``
remains the canonical module (all internal code and any pickled references keep
using it); this alias just makes the HT-branded name importable too.
"""
import sys

import hermes_constants as _module

# Replace this shim in sys.modules with the real module so ``ht_constants`` and
# ``hermes_constants`` resolve to the same object (attributes, singletons, and
# isinstance checks all agree; no duplicate module state).
sys.modules[__name__] = _module
