"""The hand-off suite, release N-1 to HEAD.

A git install of the previous release tag, set up as that release set it up, updated to HEAD:
the N-1 updater fetches, pulls and hands off to HEAD's updater, which restarts everything.

Every property is a test over one recorded scenario (see ``_scenario.py``).
"""

from __future__ import annotations

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.handoff._scenario import HandoffProperties

pytestmark = [
    pytest.mark.skipif(not H.BWRAP_OK, reason="bubblewrap sandbox unavailable"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
    pytest.mark.live_system_guard_bypass,  # one bwrap sandbox per column; killing it reaps everything
]


class TestFromReleaseN1(HandoffProperties):
    column = "n1"
