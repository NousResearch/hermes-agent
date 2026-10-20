"""Native leased completion children keep reclaim closed after owner exit."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.scripts.desktop_update.windows_handoff_support import (
    MARKER, _dead_pid, _op, _CHECKOUT_HOLDER,
)


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('lease', [1, 16])
def test_marker_op_reclaim_reports_held_while_a_leased_child_outlives_its_owner(tmp_path: Path, lease: int) -> None:
    """R5b: the owner died (its byte at 1 MiB is free) but a completion child the job refused
    still holds its lease byte. The checkout is busy until it exits: reclaim answers `held`."""
    install = tmp_path / 'hermes-agent'
    install.mkdir()
    dead = f'{_dead_pid()}\n{int(time.time())}\nct:5.000\n'.encode()
    (tmp_path / MARKER).write_bytes(dead)
    release = tmp_path / 'release-holder'
    holder = subprocess.Popen([sys.executable, '-c', _CHECKOUT_HOLDER, str(install / '.hermes-update.lock'),
                               str(release), str(lease)])
    try:
        deadline = time.monotonic() + 30
        while not Path(str(release) + '.ready').exists():
            assert time.monotonic() < deadline and holder.poll() is None
            time.sleep(0.05)
        assert _op(tmp_path, '-MarkerOp', 'reclaim')[:2] == (0, 'held\n')
        assert (tmp_path / MARKER).read_bytes() == dead
    finally:
        release.touch()
        holder.wait(timeout=30)
