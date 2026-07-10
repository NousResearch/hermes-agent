"""Brand alias for :mod:`hermes_time` (Phase 6).

``import ht_time`` returns the exact same module object as
``import hermes_time`` — the two names are interchangeable. ``hermes_time``
remains the canonical module (all internal code and any pickled references keep
using it); this alias just makes the HT-branded name importable too.
"""
import sys

import hermes_time as _module

# Replace this shim in sys.modules with the real module so ``ht_time`` and
# ``hermes_time`` resolve to the same object (attributes, singletons, and
# isinstance checks all agree; no duplicate module state).
sys.modules[__name__] = _module
