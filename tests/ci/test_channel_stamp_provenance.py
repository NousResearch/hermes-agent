"""Channel package recording retains the producer's update-owner boundaries."""
import json

import pytest

from scripts.termux import stage_apt_repo
from tests.ci.channel_termux_helpers import COMMIT, read_stamp, stage_from_workflow, stamp_from_shell
from tests.ci.test_darwin_feed_handoff import run_stage
from tests.scripts.test_release_r2 import r2_server
from tests.termux_fixtures import build_deb

TAG = "channel.20261028T120000Z-v1.2.3"


@pytest.mark.parametrize("changed", [
    {"payload": "bundled"}, {"distribution": "desktop-app"}, {"updateMechanism": "electron-updater"},
    {"baseVersion": "9.9.9"}, {"source": "commit-build"},
])
def test_termux_recorder_rejects_substituted_channel_stamp(tmp_path, request, changed):
    server = request.getfixturevalue("r2_server")
    result = stamp_from_shell(tmp_path, TAG)
    assert result.returncode == 0, result.stdout + result.stderr
    stamp = {**read_stamp(tmp_path), **changed}
    root = tmp_path / "termux-build"
    (root / "deb").mkdir(parents=True)
    build_deb(root / "deb/package.deb", {
        "Package": "hermes-agent", "Version": "1.2.3~channel.20261028T120000Z-1", "Architecture": "aarch64",
    }, {"data/data/com.termux/files/usr/lib/hermes-agent/app/install-stamp.json": json.dumps(stamp).encode()})
    assert stage_apt_repo.main(["--pool", str(root / "deb"), "--out", str(root / "apt"),
                               "--suite", "hermes-channel", "--pool-subdir", TAG]) == 3
    result = stage_from_workflow(tmp_path, server, TAG, "https://archive.example")
    assert result.returncode != 0
    assert "provenance" in result.stderr or "ownership" in result.stderr
    assert not (root / "metadata-termux-aarch64.json").exists()
    assert not server.store


def test_desktop_recorder_cannot_accept_channel_claim_without_request(tmp_path, request):
    server = request.getfixturevalue("r2_server")
    result = run_stage(tmp_path, server, TAG, "arm64", feed=True)
    assert result.returncode != 0
    assert "channel request" in result.stderr
    assert not server.store
