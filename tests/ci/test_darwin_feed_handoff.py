"""Execute macOS staging with real recorders, receipts and loopback object storage.

Only codesign (unavailable off macOS) and the storage endpoint are substituted.
Commit packages intentionally have no updater feed, as electron-builder emits none.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import plistlib
import shlex
import subprocess
import sys
import zipfile

import pytest

from scripts.bundles.release_artifacts import release_version
from tests.ci.desktop_release_roles import load, native_builds, stage_step
from tests.ci.test_desktop_release_tag_admission import _child_env, _native_tool
from tests.scripts.test_release_r2 import r2_server

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
TAGS = ["v1.2.3", "canary.20261028T120000Z-v1.2.3", "commit.20261028T120000Z-v1.2.3"]


def packages(root, tag, arch, channel_request=None):
    version = release_version(tag)
    identity = "com.nousresearch.hermes-bundled"
    stamp = {"tag": tag, "commit": COMMIT, "baseVersion": version,
             "source": "commit-build" if tag.startswith("commit.") else "build"}
    if channel_request:
        identity = channel_request["identity"]["appId"]
        stamp.update(source="channel-build", channelBuild=channel_request)
    app = "HermesBundled"
    (root / f"mac-{arch}" / f"{app}.app").mkdir(parents=True)
    stem = f"{app}-{version}-mac-{arch}"
    with zipfile.ZipFile(root / f"{stem}.zip", "w") as archive:
        archive.writestr(f"{app}.app/Contents/Info.plist", plistlib.dumps({
            "CFBundleIdentifier": identity, "CFBundleShortVersionString": version,
        }))
        archive.writestr(f"{app}.app/Contents/Resources/install-stamp.json", json.dumps(stamp))
    for suffix in ("dmg", "dmg.blockmap", "zip.blockmap"):
        (root / f"{stem}.{suffix}").write_bytes(b"unsigned transport fixture")


def run_stage(tmp_path, server, tag, arch, *, feed=False, missing=None, channel_request=None):
    root = tmp_path / "apps/desktop/release"
    packages(root, tag, arch, channel_request)
    if feed:
        channel = "stable" if tag.startswith("v") else "canary" if tag.startswith("canary.") else "latest"
        (root / f"{arch}-{channel}-mac.yml").write_text("version: 1.2.3\n")
    if missing:
        for file in root.glob(missing):
            file.unlink()
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import runpy, subprocess, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from scripts.releases import r2\n"
        f"r2.s3_endpoint = lambda _: 'http://127.0.0.1:{server.server_port}'\n"
        "real_run = subprocess.run\n"
        "def run(argv, *args, **kwargs):\n"
        "    if argv[0] == 'codesign':\n"
        "        return subprocess.CompletedProcess(argv, 0, '', 'TeamIdentifier=ABCDEFGHIJ\\n')\n"
        "    return real_run(argv, *args, **kwargs)\n"
        "subprocess.run = run\n"
        "assert sys.argv[1] == '-m', sys.argv\n"
        "module = sys.argv[2]\nsys.argv = sys.argv[2:]\n"
        "runpy.run_module(module, run_name='__main__')\n",
        encoding="utf-8",
    )
    # An explicit interpreter avoids platform-specific executable script/shebang handling.
    launcher = f"python() {{ {shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(driver.as_posix())} \"$@\"; }}\n"
    jobs = load()["jobs"]
    script = stage_step(jobs[native_builds(jobs)[f"darwin-{arch}"]])["run"]
    env = _child_env(
        TARGET=f"darwin-{arch}", RELEASE_PHASE="candidate", RELEASE_COMMIT=COMMIT,
        HERMES_DESKTOP_VARIANT="bundled", HERMES_PAYLOAD_TAG=tag,
        HERMES_ARCHIVE_TAG=f"rc.1-{tag}" if tag.startswith("v") else tag,
        HERMES_BUILD_COMMIT=COMMIT if tag.startswith("commit.") else "",
        CHANNEL_BUILD=json.dumps(channel_request) if channel_request else "",
        RUNNER_TEMP=tmp_path.as_posix(), R2_DISPOSABLE_RUN="",
        CLOUDFLARE_R2_PUBLIC_URL=f"http://127.0.0.1:{server.server_port}/hermes-releases",
    )
    if channel_request:
        (tmp_path / "channel-request.json").write_text(json.dumps(channel_request))
    return subprocess.run([_native_tool("bash"), "-e", "-o", "pipefail", "-c", launcher + script],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30, check=False)


def assert_receipt(server, tmp_path, tag, arch, *, feed):
    archive = f"rc.1-{tag}" if tag.startswith("v") else tag
    prefix = f"releases/tag/{archive}/"
    receipt = json.loads(server.store[prefix + f"handoff-darwin-{arch}.json"][0])
    assert (receipt["tag"], receipt["commit"]) == (archive, COMMIT)
    files = {row["path"]: row for row in receipt["files"]}
    assert any(name.endswith(".dmg") for name in files)
    assert any(name.endswith(".zip") for name in files)
    assert any(name.endswith(".blockmap") for name in files)
    assert any(name.endswith("-mac.yml") for name in files) == feed
    metadata = f"metadata-macos-{arch}.json"
    assert metadata in files
    record = json.loads(server.store[prefix + metadata][0])
    assert (record["tag"], record["commit"]) == (tag, COMMIT)
    for name, row in files.items():
        body = (tmp_path / "apps/desktop/release" / name).read_bytes()
        assert server.store[prefix + name][0] == body
        assert row["sha256"] == hashlib.sha256(body).hexdigest()
    assert all(key.startswith(prefix) for key in server.store)


@pytest.mark.parametrize("arch", ["arm64", "x64"])
@pytest.mark.parametrize("tag", TAGS)
def test_candidate_stages_only_the_inputs_its_packager_produces(tmp_path, request, tag, arch):
    server = request.getfixturevalue("r2_server")
    feed = not tag.startswith("commit.")
    result = run_stage(tmp_path, server, tag, arch, feed=feed)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_receipt(server, tmp_path, tag, arch, feed=feed)


@pytest.mark.parametrize("arch", ["arm64", "x64"])
@pytest.mark.parametrize("tag", TAGS[:2])
def test_updating_candidates_still_require_the_feed(tmp_path, request, tag, arch):
    server = request.getfixturevalue("r2_server")
    result = run_stage(tmp_path, server, tag, arch)
    assert result.returncode != 0
    assert "No files match release handoff pattern: *-mac.yml" in result.stderr
    assert not server.store


@pytest.mark.parametrize("missing", ["*.dmg", "*.blockmap"])
def test_commit_claim_still_requires_its_package_inputs(tmp_path, request, missing):
    server = request.getfixturevalue("r2_server")
    result = run_stage(tmp_path, server, TAGS[-1], "arm64", missing=missing)
    assert result.returncode != 0
    assert f"No files match release handoff pattern: {missing}" in result.stderr
    assert not server.store
