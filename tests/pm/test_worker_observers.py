"""Progress observers cannot stall other range writers in the isolated worker."""
from __future__ import annotations

import json
import threading
import time

import pytest

from pm import paths
from pm.lock import Facts
from tests.pm._fixtures import client as client, isolated_python as isolated_python
from tests.pm._range_server import dl_server as dl_server
from tests.pm.test_worker import _node_archive


@pytest.mark.platforms("posix")
def test_slow_worker_observer_does_not_hold_range_writers(client, dl_server):
    # Every range contains several chunks: one write per range could otherwise
    # appear to make progress even while all writers wait on the bitmap lock.
    _node_archive(dl_server, b"#!/bin/sh\nexit 0\n#" + b"x" * (64 << 20))
    snapshots = []
    advanced = []

    def observe(done, total, ranges):
        snapshots.append((done, total, ranges))
        if not done or advanced:
            return
        part, = paths.partials_root().glob("*.part")
        # The eighth range's second chunk is untouched at the first callback
        # on the old implementation. All writes are actual local HTTP bytes.
        position = 7 * total // 8 + (1 << 20)
        started = time.monotonic()
        deadline = started + 3
        while time.monotonic() < deadline:
            with part.open("rb") as stream:
                stream.seek(position)
                if stream.read(1) == b"x":
                    advanced.append(True)
                    break
            time.sleep(0.01)
        else:
            advanced.append(False)

    client.ensure("node", explicit=True, download_progress=observe)
    assert advanced == [True], "range writers stalled until the progress callback returned"
    assert snapshots[-1][0] == snapshots[-1][1]
    assert [done for done, _, _ in snapshots] == sorted(done for done, _, _ in snapshots)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("action", ["pause", "error"])
def test_worker_observer_failure_preserves_resumable_bytes(client, dl_server, action):
    from pm.downloader import DownloadPaused

    _node_archive(dl_server, b"#!/bin/sh\nexit 0\n#" + b"x" * (32 << 20))
    pause = threading.Event()
    failure = ValueError("observer failed")
    seen = []

    def observe(done, total, ranges):
        seen.append((done, total, ranges))
        if done:
            if action == "pause":
                pause.set()
            else:
                raise failure

    with pytest.raises(DownloadPaused if action == "pause" else ValueError) as caught:
        client.ensure("node", explicit=True, pause_event=pause, download_progress=observe)
    if action == "error":
        assert caught.value is failure
    assert Facts(paths.facts_path()).get("node") is None
    sidecar, = paths.partials_root().glob("*.ranges")
    durable = sum(end - start for start, end in json.loads(sidecar.read_text())["ranges"])
    assert durable >= seen[-1][0] and durable > 0
    pause.clear()
    resumed = []
    client.ensure("node", explicit=True, pause_event=pause,
                  download_progress=lambda *args: resumed.append(args))
    assert resumed[0][0] == durable
    assert resumed[-1][0] == resumed[-1][1]
    assert not list(paths.partials_root().glob("*.part"))
    assert Facts(paths.facts_path()).get("node") is not None
