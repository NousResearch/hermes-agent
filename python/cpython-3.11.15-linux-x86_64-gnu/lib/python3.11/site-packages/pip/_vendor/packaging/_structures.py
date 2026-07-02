# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

"""Backward-compatibility shim for unpickling Version objects serialized before
packaging 26.1.

Old pickles reference ``packaging._structures.InfinityType`` and
``packaging._structures.NegativeInfinityType``.  This module provides minimal
stand-in classes so that ``pickle.loads()`` can resolve those references.
The deserialized objects are not used for comparisons — ``Version.__setstate__``
discards the stale ``_key`` cache and recomputes it from the core version fields.
"""

from __future__ import annotations


class InfinityType:
    """Stand-in for the removed ``InfinityType`` used in old comparison keys."""

    def __repr__(self) -> str:
        return "Infinity"


class NegativeInfinityType:
    """Stand-in for the removed ``NegativeInfinityType`` used in old comparison keys."""

    def __repr__(self) -> str:
        return "-Infinity"


Infinity = InfinityType()
NegativeInfinity = NegativeInfinityType()
