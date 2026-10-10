"""A separate encrypted client verifies original withdrawal."""

import time
from pathlib import Path
from collections.abc import Callable

import pytest

from tests.integration.matrix_live.conftest import (
    LiveRoom,
    LinuxNioObserver,
)
from tests.integration.matrix_live.live_gateway import LiveGateway
from tests.integration.matrix_live.context_client import group_gateway as group_gateway
from tests.integration.matrix_live.context_client import group_member as group_member
from tests.integration.matrix_live.resolution_client import check_resolution


@pytest.mark.parametrize("gateway", ["pause-resolution"], indirect=True)
@pytest.mark.parametrize(
    "scope,barrier",
    [
        ("event", "fetch"),
        ("thread-root", "fetch"),
        ("event", "original"),
        ("room", "original"),
        ("thread-child", "original"),
    ],
)
def test_resolution_original(
    tmp_path: Path,
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
    scope: str,
    barrier,
) -> None:
    started = time.monotonic()
    try:
        check_resolution(
            tmp_path,
            group_gateway,
            live_room,
            linux_nio_observer,
            scope=scope,
            barrier=barrier,
            replacement=False,
        )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
