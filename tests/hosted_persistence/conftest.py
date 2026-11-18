"""Fixtures shared across this directory's files, registered here so a test names them as parameters without
importing them (an imported fixture that a parameter shadows reads as a redefinition)."""

from tests.hosted_persistence.test_provider_discovery import bundled_root, entry_points
from tests.hosted_persistence.test_turn_fence_lifetime import claimed
from tests.hosted_persistence.test_turn_store_resolved_once import provider
