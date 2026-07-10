"""Shared fixtures for the AMDP test suite.

The plan cache and config cache are process-global; reset them around every
test so a cached plan/refusal from one test can't leak into another (the
once-per-turn cache is keyed on prompt+prefix, and several tests share prompts).
"""

import pytest

from agent.amdp import loop


@pytest.fixture(autouse=True)
def _reset_amdp_caches():
    loop.reset_cache_for_tests()
    yield
    loop.reset_cache_for_tests()
