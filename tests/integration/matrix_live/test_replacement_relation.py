"""A separate encrypted client checks the authenticated replacement target."""

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.integration.matrix_live.conftest import (
    LinuxNioObserver,
    LiveGateway,
    LiveRoom,
)
from tests.integration.matrix_live.context_client import group_gateway as group_gateway
from tests.integration.matrix_live.context_client import group_member as group_member
from tests.integration.matrix_live.integrity_client import check_replacement_relation


@pytest.mark.parametrize("gateway", ["pause-resolution"], indirect=True)
@pytest.mark.parametrize("relation_kind", ["matching", "outer-only", "wrong-target"])
def test_replacement_relation(
    tmp_path: Path,
    group_gateway: LiveGateway,
    live_room: LiveRoom,
    linux_nio_observer: LinuxNioObserver,
    record_property: Callable[[str, object], None],
    relation_kind: str,
) -> None:
    started = time.monotonic()
    try:
        check_replacement_relation(
            tmp_path,
            group_gateway,
            live_room,
            linux_nio_observer,
            relation_kind=relation_kind,
        )
    finally:
        record_property("body_seconds", round(time.monotonic() - started, 3))
