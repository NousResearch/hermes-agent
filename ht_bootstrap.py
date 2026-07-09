"""Brand alias for :mod:`hermes_bootstrap` (Phase 6).

``import ht_bootstrap`` returns the exact same module object as
``import hermes_bootstrap`` — the two names are interchangeable. ``hermes_bootstrap``
remains the canonical module (all internal code and any pickled references keep
using it); this alias just makes the HT-branded name importable too.
"""
import sys

import hermes_bootstrap as _module

# Replace this shim in sys.modules with the real module so ``ht_bootstrap`` and
# ``hermes_bootstrap`` resolve to the same object (attributes, singletons, and
# isinstance checks all agree; no duplicate module state).
sys.modules[__name__] = _module
