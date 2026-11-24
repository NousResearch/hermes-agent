"""Commit claim build environments must reach the standalone icon renderer."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli.update_channel import release_version
from scripts.bundles.desktop_inputs import identity_environment

ROOT = Path(__file__).resolve().parents[2]
TAG = "commit.20261026T180000Z-v1.2.3"
COMMIT = "0123456" + "a" * 33


def icon_environment():
    request = SimpleNamespace(
        validate_channel=lambda: None, channel_request=None, commit=COMMIT,
        tag=TAG, version=release_version(TAG), bundle_env={}, release_epoch=1793037600,
    )
    return identity_environment(request, "bundled", os.environ)


def render(out, env):
    return subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/generate_icons.py"),
         "--source", str(ROOT), "--out", str(out)],
        env=env, text=True, capture_output=True, timeout=180, check=False,
    )


def test_commit_claim_renders_the_same_icons_as_its_tagless_commit(tmp_path):
    env = icon_environment()
    claimed = tmp_path / "claimed"
    result = render(claimed, env)
    assert result.returncode == 0, result.stdout + result.stderr
    env.pop("HERMES_PAYLOAD_TAG")
    tagless = tmp_path / "tagless"
    result = render(tagless, env)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = [p.relative_to(claimed) for p in claimed.rglob("*") if p.is_file()]
    assert Path("apps/desktop/assets/icon.png") in outputs
    for relative in outputs:
        assert (claimed / relative).read_bytes() == (tagless / relative).read_bytes(), relative


@pytest.mark.parametrize("tag", [
    "v1.2.3", "canary.20261026T180000Z-v1.2.3", "channel.20261026T180000Z-v1.2.3",
    "commit.not-a-claim", "commit.20261026T180000Z-v2026.1.1", TAG + "\n",
])
def test_commit_icons_reject_incompatible_payload_tags_before_writing(tmp_path, tag):
    env = icon_environment()
    env["HERMES_PAYLOAD_TAG"] = tag
    result = render(tmp_path / "icons", env)
    assert result.returncode != 0
    assert "Commit builds cannot also select" in result.stderr
    assert not (tmp_path / "icons").exists()
