"""Brand alias for :mod:`hermes_state` (Phase 6).

``import ht_state`` returns the exact same module object as
``import hermes_state`` — the two names are interchangeable. ``hermes_state``
remains the canonical module (all internal code and any pickled references keep
using it); this alias just makes the HT-branded name importable too.
"""
import sys

import hermes_state as _module

# Replace this shim in sys.modules with the real module so ``ht_state`` and
# ``hermes_state`` resolve to the same object (attributes, singletons, and
# isinstance checks all agree; no duplicate module state).
sys.modules[__name__] = _module
