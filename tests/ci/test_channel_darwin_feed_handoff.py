"""Channel claims keep a mandatory macOS feed when commit claims omit theirs."""
import pytest

from tests.ci.test_darwin_feed_handoff import COMMIT, assert_receipt, run_stage
from tests.scripts.test_channel_build_versions import CLAIM, channel_request
from tests.scripts.test_release_r2 import r2_server


@pytest.mark.parametrize("arch", ["arm64", "x64"])
@pytest.mark.parametrize("feed", [True, False])
def test_channel_claim_staging_requires_its_packagers_feed(tmp_path, request, arch, feed):
    server = request.getfixturevalue("r2_server")
    result = run_stage(tmp_path, server, CLAIM, arch, feed=feed, channel_request=channel_request(COMMIT))
    if feed:
        assert result.returncode == 0, result.stdout + result.stderr
        assert_receipt(server, tmp_path, CLAIM, arch, feed=True)
    else:
        assert result.returncode != 0
        assert "No files match release handoff pattern: *-mac.yml" in result.stderr
        assert not server.store
