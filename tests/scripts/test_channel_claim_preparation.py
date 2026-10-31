"""Tagged channel requests survive every preparation boundary, not just creation."""
import json
import subprocess
from pathlib import Path

import pytest

from scripts.bundles import desktop_prepare, desktop_toolchain
from scripts.bundles.desktop_inputs import identity_environment
from tests.scripts.test_channel_build_versions import CLAIM, channel_request
from tests.scripts.test_desktop_preparation import _project


def claimed_request(tmp_path):
    source, commit = _project(tmp_path)
    subprocess.run(['git', 'tag', CLAIM, commit], cwd=source, check=True)
    return desktop_prepare.BuildRequest.create(
        source, tag=CLAIM, commit=None, release_commit=commit, variant='bundled',
        work=tmp_path / 'work', cache=tmp_path / 'cache', bundle_env={},
        channel_request=channel_request(commit),
    )


def test_claim_roundtrip_and_build_environment_preserve_the_admitted_request(tmp_path):
    original = claimed_request(tmp_path)
    restored = desktop_prepare.BuildRequest.from_data(json.loads(json.dumps(original.data())))
    assert restored.data() == original.data()
    env = identity_environment(restored, 'bundled', {})
    assert env['HERMES_PAYLOAD_TAG'] == CLAIM
    assert json.loads(env['_HERMES_CHANNEL_REQUEST_JSON']) == original.channel_request
    for tag in ('v1.2.3', 'channel.20261028T120001Z-v1.2.3', ''):
        changed = {**original.data(), 'tag': tag}
        with pytest.raises(ValueError, match='prepared build differs'):
            desktop_prepare.BuildRequest.from_data(changed)


@pytest.mark.platforms('windows', 'macos')
def test_claim_preparation_reaches_the_provider_with_its_serialized_identity(tmp_path, monkeypatch):
    request = claimed_request(tmp_path)

    class ReachedProvider(Exception):
        pass

    def run_preparation(source, work, cache, request_file):
        restored = desktop_prepare.BuildRequest.from_data(json.loads(Path(request_file).read_text()))
        assert (source, work, cache) == (request.source, request.work, request.cache)
        assert restored.data() == request.data()
        raise ReachedProvider

    # The external tool-provisioning process is the only substituted boundary.
    monkeypatch.setattr(desktop_toolchain, 'run_preparation', run_preparation)
    with pytest.raises(ReachedProvider):
        desktop_prepare.prepare(request)
