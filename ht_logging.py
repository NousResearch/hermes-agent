"""Brand alias for :mod:`hermes_logging` (Phase 6).

``import ht_logging`` returns the exact same module object as
``import hermes_logging`` — the two names are interchangeable. ``hermes_logging``
remains the canonical module (all internal code and any pickled references keep
using it); this alias just makes the HT-branded name importable too.
"""
import sys

import hermes_logging as _module

# Replace this shim in sys.modules with the real module so ``ht_logging`` and
# ``hermes_logging`` resolve to the same object (attributes, singletons, and
# isinstance checks all agree; no duplicate module state).
sys.modules[__name__] = _module
