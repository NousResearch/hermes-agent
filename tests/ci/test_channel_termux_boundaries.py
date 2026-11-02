"""Termux channel claims reach real metadata, APT indices, handoffs and candidate assembly."""
import hashlib
import json

import pytest

from scripts.bundles import release_artifacts
from scripts.releases import handoff, stable
from scripts.termux import deb_version, stage_apt_repo
from tests.ci.channel_termux_helpers import COMMIT, read_stamp, stage_from_workflow, stamp_from_shell
from tests.ci.test_darwin_feed_handoff import run_stage
from tests.scripts.test_channel_build_versions import channel_request
from tests.scripts.test_release_r2 import r2_server
from tests.scripts.test_stable_release import https_origin
from tests.termux_fixtures import build_deb


@pytest.mark.parametrize("kind", ["channel", "canary"])
def test_termux_claim_shell_stamp_to_candidate(tmp_path, request, kind):
    server = request.getfixturevalue("r2_server")
    origin = request.getfixturevalue("https_origin")
    origin.store = server.store
    tag = f"{kind}.20261028T120000Z-v1.2.3"
    result = stamp_from_shell(tmp_path, tag)
    assert result.returncode == 0, result.stdout + result.stderr
    stamp = read_stamp(tmp_path)
    assert (stamp["payload"], stamp["distribution"], stamp["updateMechanism"]) == ("runtime", "apt-termux", "external")
    assert (stamp["tag"], stamp["commit"], stamp["baseVersion"]) == (tag, COMMIT, f"1.2.3+{kind}.20261028T120000Z")
    assert "channelBuild" not in stamp and stamp["source"] == "bundle"
    root = tmp_path / "termux-build"
    (root / "deb").mkdir(parents=True)
    version = deb_version.deb_version_for_tag(tag)
    assert version == f"1.2.3~{kind}.20261028T120000Z-1"
    package = root / "deb" / f"hermes-agent_{version}_aarch64.deb"
    build_deb(package, {"Package": "hermes-agent", "Version": version, "Architecture": "aarch64"},
              {"data/data/com.termux/files/usr/lib/hermes-agent/app/install-stamp.json":
               (tmp_path / "app/install-stamp.json").read_bytes()}, compression="xz")
    suite = "hermes-" + deb_version.channel_for_tag(tag)
    # Signing is the external boundary: the real stage emits all unsigned index and pool bytes.
    assert stage_apt_repo.main(["--pool", str(root / "deb"), "--out", str(root / "apt"),
                               "--suite", suite, "--pool-subdir", tag]) == 3
    index = (root / "apt/dists" / suite / "main/binary-aarch64/Packages").read_text(encoding="utf-8")
    assert f"Version: {version}\n" in index and f"Filename: pool/{tag}/" in index
    result = stage_from_workflow(tmp_path, server, tag, origin.base)
    assert result.returncode == 0, result.stdout + result.stderr
    # Exercise a mixed desktop/Termux candidate with the real macOS record/handoff fixture.
    desktop = run_stage(tmp_path / "desktop", server, tag, "arm64", feed=True,
                        channel_request=channel_request(COMMIT) if kind == "channel" else None)
    assert desktop.returncode == 0, desktop.stdout + desktop.stderr
    fetched = tmp_path / "fetched"
    handoff.fetch(tag, COMMIT, ["termux", "darwin-arm64"], fetched, ["metadata-*.json"], public_base=origin.base)
    manifest = release_artifacts.assemble(
        fetched, tag, COMMIT, origin.base, tmp_path / "candidate.json",
        smoke_results={name: {"result": "success" if name == "smoke-darwin-arm64" else "skipped"}
                       for name in stable.SMOKE_TARGETS},
        release_epoch=stable.stamped_epoch(tag), archive=tag, lanes=["termux", "macos-arm64"])
    row, = [row for row in manifest["packages"] if row["platform"] == "termux"]
    assert row["version"] == version and row["identity"] == "hermes-agent"
    assert row["artifact"]["sha256"] == hashlib.sha256(package.read_bytes()).hexdigest()
    prefix = f"releases/tag/{tag}/"
    assert all(key.startswith(prefix) for key in server.store)
    receipt = json.loads(server.store[prefix + "handoff-termux.json"][0])
    for item in receipt["files"]:
        body = (root / item["path"]).read_bytes()
        assert server.store[prefix + item["path"]][0] == body
        assert item["sha256"] == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize("substitution", [
    ("HERMES_DESKTOP_VARIANT=runtime", "HERMES_DESKTOP_VARIANT=bundled"),
    ("HERMES_DESKTOP_VARIANT=runtime", "HERMES_DESKTOP_VARIANT=light"),
    ("HERMES_DESKTOP_VARIANT=runtime", "HERMES_DESKTOP_VARIANT=store"),
    ("--distribution apt-termux", "--distribution desktop-app"),
    ("--distribution apt-termux", "--distribution docker"),
    ("--distribution apt-termux", ""),
    ("--update-mechanism external", "--update-mechanism self"),
    ("--update-mechanism external", "--update-mechanism electron-updater"),
    ("--update-mechanism external", "--update-mechanism app-installer"),
])
def test_channel_tag_without_request_is_not_a_desktop_or_other_owner(tmp_path, substitution):
    result = stamp_from_shell(tmp_path, "channel.20261028T120000Z-v1.2.3", substitutions=[substitution])
    assert result.returncode != 0
    assert "requires" in result.stderr
    assert not (tmp_path / "app/install-stamp.json").exists()


@pytest.mark.parametrize("suite", ["hermes-stable", "hermes-canary"])
def test_channel_apt_pool_cannot_stage_into_protected_suite(tmp_path, suite):
    with pytest.raises(SystemExit, match="2"):
        stage_apt_repo.main(["--pool", str(tmp_path), "--out", str(tmp_path / "apt"),
                             "--suite", suite, "--pool-subdir", "channel.20261028T120000Z-v1.2.3"])
    assert not list((tmp_path / "apt").rglob("*"))
