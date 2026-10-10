from pathlib import Path

import pytest

from pm.filesystem import durable_write_bytes


@pytest.mark.platforms("windows")
def test_publish_lands_while_another_handle_reads_the_target(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_bytes(b"plugins: {enabled: []}\n")
    with target.open("rb"):
        durable_write_bytes(target, b"plugins: {enabled: [example]}\n")

    assert target.read_bytes() == b"plugins: {enabled: [example]}\n"
    assert not list(tmp_path.glob(".publish-*"))
