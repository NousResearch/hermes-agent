"""Oryn Dev control-plane boundary inside Hermes.

This package owns the stable internal contracts for Dev execution plans,
runtime routing, normalized events, supervisor state, and Oryn-facing read
models. Existing gateway modules may keep compatibility facades, but new Dev
control-plane code should prefer importing contracts from here.
"""

