"""A separate encrypted client withdraws a later reaction during the first decryption."""

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveRoom,
)
from tests.integration.matrix_live.live_gateway import LiveGateway
from tests.integration.matrix_live.context_client import group_gateway as group_gateway
from tests.integration.matrix_live.context_client import group_member as group_member
from tests.integration.matrix_live.integrity_client import check_reaction_page


@pytest.mark.parametrize("gateway", ["pause-resolution"], indirect=True)
@pytest.mark.parametrize("eviction", [False, True])
def test_reaction_page_decrypt(
    tmp_path: Path,
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
    eviction: bool,
) -> None:
    started = time.monotonic()
    try:
        check_reaction_page(
            tmp_path, group_gateway, live_room, linux_nio_observer, eviction=eviction
        )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
