"""The gateway fixture yields only once the gateway dispatches the room's messages directly."""

from __future__ import annotations

import pytest

from tests.integration.matrix_live.conftest import _gateway_ready


ROOM = "!room:matrix.test"
JOINED = f"INFO hermes_plugins.platforms__matrix.adapter: Matrix: joined {ROOM}"
RUNNING = "INFO gateway.run: Gateway running with 1 platform(s)"
STARTED = "INFO gateway.run: Press Ctrl+C to stop"


@pytest.mark.parametrize(
    ("lines", "ready"),
    [
        pytest.param([RUNNING, JOINED], False, id="joined-during-startup-restore"),
        pytest.param([RUNNING, STARTED], False, id="started-before-join"),
        pytest.param([JOINED, RUNNING, STARTED], True, id="joined-then-started"),
        pytest.param([RUNNING, STARTED, JOINED], True, id="started-then-joined"),
    ],
)
def test_gateway_is_ready_after_joining_and_finishing_startup(lines: list[str], ready: bool) -> None:
    assert _gateway_ready("\n".join(lines), ROOM) is ready
