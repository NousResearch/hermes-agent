"""Native artifact metadata stays bound to tagged or tagless package inputs."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import plistlib
import stat
import subprocess
import zipfile

import hermes_yaml
import pytest

from scripts.bundles import release_artifacts
from tests.scripts.test_channel_build_versions import channel_request


@pytest.mark.parametrize("arch", ["arm64", "x64"])
@pytest.mark.parametrize("channel", [False, True])
def test_macos_record_binds_filename_version_and_provenance(tmp_path, monkeypatch, arch, channel):
    commit = "a" * 40
    request = channel_request(commit, sequence=6) if channel else None
    tag = None if channel else "v1.2.3"
    version = request["version"] if request else "1.2.3"
    identity = request["identity"]["appId"] if request else "ai.hermes.test"
    stamp = ({"source": "channel-build", "channelBuild": request, "commit": commit, "tag": None}
             if request else {"commit": commit, "tag": tag, "baseVersion": version})
    root = tmp_path / "release"
    app = root / "mac" / "Product.app"
    app.mkdir(parents=True)
    package = root / f"Product-{version}-mac-{arch}.zip"
    out = root / "metadata.json"

    def write_package(app_version=version, package_stamp=stamp):
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("Product.app/Contents/Info.plist", plistlib.dumps({
                "CFBundleIdentifier": identity, "CFBundleShortVersionString": app_version,
            }))
            archive.writestr("Product.app/Contents/Resources/install-stamp.json",
                             json.dumps(package_stamp))
        _add_outer_dmg(root, package, feed=not channel)

    def codesign(argv, **kwargs):
        assert argv[0] == "codesign"
        return subprocess.CompletedProcess(argv, 0, stderr="TeamIdentifier=ABCDEFGHIJ\n")

    monkeypatch.setattr(release_artifacts.subprocess, "run", codesign)
    write_package()
    original = package.read_bytes()
    release_artifacts.record("macos", arch, root, tag, commit, out, channel_request=request)
    metadata = json.loads(out.read_text(encoding="utf-8-sig"))
    assert metadata["identity"] == identity
    assert metadata["version"] == version
    assert metadata["filename"] == package.name
    assert metadata["teamId"] == "ABCDEFGHIJ"
    assert metadata.get("request") == request
    assert package.read_bytes() == original

    wrong_name = package.with_name(f"Product-9.9.9-mac-{arch}.zip")
    package.rename(wrong_name)
    with pytest.raises(ValueError, match="filename"):
        release_artifacts.record("macos", arch, root, tag, commit, out, channel_request=request)
    wrong_name.rename(package)
    write_package(app_version="9.9.9")
    with pytest.raises(ValueError, match="version"):
        release_artifacts.record("macos", arch, root, tag, commit, out, channel_request=request)
    write_package(package_stamp={**stamp, "commit": "b" * 40})
    with pytest.raises(ValueError, match="provenance"):
        release_artifacts.record("macos", arch, root, tag, commit, out, channel_request=request)


# Squirrel.Mac's ShipIt aborts every macOS update when one entry of the downloaded
# app is read-only (#138659); record() is the release step that opens the ZIP.
PAYLOAD = "Product.app/Contents/Resources/payload/"
UNIX = 16  # a ZIP entry's Unix mode is the high word of external_attr
OWNER_WRITABLE = [
    (PAYLOAD, (stat.S_IFDIR | 0o755) << UNIX, b""),
    (PAYLOAD + "hermes/__pycache__/cli.cpython-313.pyc", (stat.S_IFREG | 0o644) << UNIX, b"pyc"),
    (PAYLOAD + "hermes/cli.py", (stat.S_IFREG | 0o644) << UNIX, b"py"),
    (PAYLOAD + "bin/python3", (stat.S_IFLNK | 0o755) << UNIX, b"python3.13"),
    # Only an MS-DOS attribute (archive), as a Windows archiver writes it: no Unix mode.
    (PAYLOAD + "no-unix-mode.txt", 0x20, b"dos"),
]


def record_mac_update(tmp_path, monkeypatch, extra=(), size=None):
    """record() a tagged arm64 update ZIP: a valid app plus OWNER_WRITABLE plus `extra`, grown to `size`."""
    commit, root = "a" * 40, tmp_path / "release"
    (root / "mac" / "Product.app").mkdir(parents=True)
    package = root / "Product-1.2.3-mac-arm64.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("Product.app/Contents/Info.plist", plistlib.dumps({
            "CFBundleIdentifier": "ai.hermes.test", "CFBundleShortVersionString": "1.2.3",
        }))
        archive.writestr("Product.app/Contents/Resources/install-stamp.json",
                         json.dumps({"commit": commit, "tag": "v1.2.3", "baseVersion": "1.2.3"}))
        for name, external_attr, data in [*OWNER_WRITABLE, *extra]:
            entry = zipfile.ZipInfo(name)
            entry.external_attr = external_attr
            archive.writestr(entry, data)
    if size is not None:
        # Sparse padding ahead of the archive: ZIP offsets count from the end, so it still opens.
        data = package.read_bytes()
        with package.open("wb") as grown:
            grown.truncate(size - len(data))
            grown.seek(size - len(data))
            grown.write(data)
    _add_outer_dmg(root, package, feed=True)
    monkeypatch.setattr(release_artifacts.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(
        argv, 0, stderr="TeamIdentifier=ABCDEFGHIJ\n"))
    out = root / "metadata.json"
    release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)
    return out


def test_macos_record_accepts_an_owner_writable_update_zip(tmp_path, monkeypatch):
    out = record_mac_update(tmp_path, monkeypatch)
    assert json.loads(out.read_text(encoding="utf-8"))["version"] == "1.2.3"


@pytest.mark.parametrize("locked", [
    pytest.param([(PAYLOAD + f"pkg/__pycache__/m{n}.cpython-313.pyc", (stat.S_IFREG | 0o444) << UNIX, b"pyc")
                  for n in range(5)], id="pycs-0444"),
    pytest.param([(PAYLOAD + "pkg/vendored.py", (stat.S_IFREG | 0o444) << UNIX, b"py"),
                  (PAYLOAD + "pkg/_speedups.cpython-313-darwin.so", (stat.S_IFREG | 0o555) << UNIX, b"so")],
                 id="source-0444-and-extension-0555"),
    pytest.param([(PAYLOAD + "tools/", (stat.S_IFDIR | 0o555) << UNIX, b"")], id="directory-0555"),
])
def test_macos_record_refuses_an_update_zip_entry_without_owner_write(tmp_path, monkeypatch, locked):
    with pytest.raises(ValueError, match=f"^{len(locked)} update ZIP entries lack owner write") as refused:
        record_mac_update(tmp_path, monkeypatch, locked)
    named = [name for name, _, _ in locked]
    assert all(name in str(refused.value) for name in named[:3])
    assert not any(name in str(refused.value) for name in named[3:])
    assert not (tmp_path / "release" / "metadata.json").exists()


def test_macos_record_refuses_an_update_zip_squirrel_cannot_download(tmp_path, monkeypatch):
    """v0.22.0's Squirrel.Mac (Electron 40) dies downloading an update ZIP of 1 GiB or more; a release
    that published one would crash every v0.22.0 Mac that clicks Update now."""
    with pytest.raises(ValueError, match="1,073,741,824 bytes"):
        record_mac_update(tmp_path, monkeypatch, size=release_artifacts.SQUIRREL_ZIP_LIMIT)
    assert not (tmp_path / "release" / "metadata.json").exists()


def test_macos_record_accepts_an_update_zip_just_under_one_gib(tmp_path, monkeypatch):
    out = record_mac_update(tmp_path, monkeypatch, size=release_artifacts.SQUIRREL_ZIP_LIMIT - 1)
    assert json.loads(out.read_text(encoding="utf-8"))["version"] == "1.2.3"


def _write_blockmap(path, total_size: int) -> None:
    payload = {"version": "2", "files": [{"name": "file", "offset": 0, "checksums": ["abc"], "sizes": [total_size]}]}
    path.write_bytes(gzip.compress(json.dumps(payload).encode("utf-8")))


def _sha512_base64(path) -> str:
    with path.open("rb") as handle:  # streamed: the Squirrel-limit fixtures are a sparse 1 GiB ZIP
        return base64.b64encode(hashlib.file_digest(handle, "sha512").digest()).decode("ascii")


def _write_mac_feed(path, version: str, zip_path, dmg_path, *, dmg_size: int | None = None, dmg_sha512: str | None = None) -> None:
    zip_sha = _sha512_base64(zip_path)
    feed = {
        "version": version,
        "files": [
            {"url": zip_path.name, "sha512": zip_sha, "size": zip_path.stat().st_size},
            {"url": dmg_path.name, "sha512": dmg_sha512 or _sha512_base64(dmg_path),
             "size": dmg_path.stat().st_size if dmg_size is None else dmg_size},
        ],
        "path": zip_path.name,
        "sha512": zip_sha,
    }
    path.write_text(hermes_yaml.safe_dump(feed, sort_keys=False), encoding="utf-8")


def _add_outer_dmg(root, package, *, feed: bool) -> None:
    """The signed DMG (and, for tagged releases, the feed) a record needs beside its update ZIP."""
    dmg = package.with_suffix(".dmg")
    dmg.write_bytes(b"signed-stapled-dmg")
    _write_blockmap(dmg.with_name(dmg.name + ".blockmap"), dmg.stat().st_size)
    if feed:
        _write_mac_feed(root / "stable-mac.yml", "1.2.3", package, dmg)


def test_macos_record_verifies_outer_dmg_signature_blockmap_and_feed_consistency(tmp_path, monkeypatch):
    commit, root = "a" * 40, tmp_path / "release"
    (root / "mac" / "Product.app").mkdir(parents=True)
    package = root / "Product-1.2.3-mac-arm64.zip"
    dmg = root / "Product-1.2.3-mac-arm64.dmg"
    blockmap = root / "Product-1.2.3-mac-arm64.dmg.blockmap"
    feed = root / "arm64-stable-mac.yml"
    out = root / "metadata-macos-arm64.json"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("Product.app/Contents/Info.plist", plistlib.dumps({
            "CFBundleIdentifier": "ai.hermes.test", "CFBundleShortVersionString": "1.2.3",
        }))
        archive.writestr("Product.app/Contents/Resources/install-stamp.json",
                         json.dumps({"commit": commit, "tag": "v1.2.3", "baseVersion": "1.2.3"}))

    pre_staple = b"udif-dmg-unsigned-bytes"
    post_staple = pre_staple + b"-codesign-cms-and-stapled-ticket"
    dmg.write_bytes(post_staple)
    _write_blockmap(blockmap, len(post_staple))
    _write_mac_feed(feed, "1.2.3", package, dmg)

    calls: list[list[str]] = []
    teams = {"app": "ABCDEFGHIJ", "dmg": "ABCDEFGHIJ"}

    def fake_codesign(argv, **kwargs):
        calls.append(list(argv))
        assert argv[0] == "codesign"
        target = argv[-1]
        team_id = teams["dmg"] if target.endswith(".dmg") else teams["app"]
        return subprocess.CompletedProcess(argv, 0, stderr=f"TeamIdentifier={team_id}\n")

    monkeypatch.setattr(release_artifacts.subprocess, "run", fake_codesign)
    release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)
    assert ["codesign", "--verify", "--verbose=2", str(dmg)] in calls
    assert ["codesign", "-dv", "--verbose=4", str(dmg)] in calls

    _write_blockmap(blockmap, len(pre_staple))
    with pytest.raises(ValueError, match="blockmap"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)
    _write_blockmap(blockmap, len(post_staple))

    _write_mac_feed(feed, "1.2.3", package, dmg, dmg_size=len(pre_staple))
    with pytest.raises(ValueError, match="feed"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)

    pre_sha = base64.b64encode(hashlib.sha512(pre_staple).digest()).decode("ascii")
    _write_mac_feed(feed, "1.2.3", package, dmg, dmg_size=len(post_staple), dmg_sha512=pre_sha)
    with pytest.raises(ValueError, match="feed"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)

    zip_bytes = package.read_bytes()
    zip_sha = base64.b64encode(hashlib.sha512(zip_bytes).digest()).decode("ascii")
    feed.write_text(hermes_yaml.safe_dump({
        "version": "1.2.3",
        "files": [{"url": package.name, "sha512": zip_sha, "size": len(zip_bytes)}],
        "path": package.name,
        "sha512": zip_sha,
    }, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must list exactly one entry"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)
    _write_mac_feed(feed, "1.2.3", package, dmg)

    blockmap.unlink()
    with pytest.raises(ValueError, match="Missing DMG blockmap"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)
    _write_blockmap(blockmap, len(post_staple))

    feed.unlink()
    with pytest.raises(ValueError, match="Missing macOS update feed"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)
    _write_mac_feed(feed, "1.2.3", package, dmg)

    teams["dmg"] = "ZZZZZZZZZZ"
    with pytest.raises(ValueError, match="team"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)
    teams["dmg"] = "ABCDEFGHIJ"

    wrong_dmg = root / "Product-9.9.9-mac-arm64.dmg"
    dmg.rename(wrong_dmg)
    with pytest.raises(ValueError, match="filename"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)

    wrong_dmg.unlink()
    with pytest.raises(ValueError, match="Expected exactly one artifact"):
        release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)


def test_macos_record_matches_commit_claim_dmg_contract_for_new_pre_pr_and_legacy_commits(tmp_path, monkeypatch):
    commit = "a" * 40
    claim_tag = "commit.20261125T000000Z-v1.2.3"

    def make_root(name: str, *, stamp: dict, version: str, with_blockmap: bool):
        root = tmp_path / name
        (root / "mac" / "Product.app").mkdir(parents=True)
        package = root / f"Product-{version}-mac-arm64.zip"
        dmg = root / f"Product-{version}-mac-arm64.dmg"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("Product.app/Contents/Info.plist", plistlib.dumps({
                "CFBundleIdentifier": "ai.hermes.test", "CFBundleShortVersionString": version,
            }))
            archive.writestr("Product.app/Contents/Resources/install-stamp.json", json.dumps(stamp))
        dmg.write_bytes(b"commit-dmg-bytes")
        if with_blockmap:
            _write_blockmap(root / f"{dmg.name}.blockmap", dmg.stat().st_size)
        return root, dmg

    calls: list[str] = []

    def codesign(argv, **kwargs):
        calls.append(argv[-1])
        return subprocess.CompletedProcess(argv, 0, stderr="TeamIdentifier=ABCDEFGHIJ\n")

    monkeypatch.setattr(release_artifacts.subprocess, "run", codesign)

    # New commit claim (signedDmg: true) enforces outer DMG + blockmap without requiring a feed yml.
    claim_version = "1.2.3+commit.20261125T000000Z"
    new_root, new_dmg = make_root(
        "new-claim",
        stamp={"source": "commit-build", "commit": commit, "tag": claim_tag, "baseVersion": claim_version, "signedDmg": True},
        version=claim_version,
        with_blockmap=True,
    )
    release_artifacts.record("macos", "arm64", new_root, claim_tag, commit, new_root / "meta.json")
    assert str(new_dmg) in calls

    (new_root / f"{new_dmg.name}.blockmap").unlink()
    with pytest.raises(ValueError, match="Missing DMG blockmap"):
        release_artifacts.record("macos", "arm64", new_root, claim_tag, commit, new_root / "meta.json")

    # Pre-PR claim-aware commit (signedDmg absent) and legacy commit both skip outer DMG while checking .app.
    calls.clear()
    pre_pr_root, pre_pr_dmg = make_root(
        "pre-pr-claim",
        stamp={"source": "commit-build", "commit": commit, "tag": claim_tag, "baseVersion": claim_version},
        version=claim_version,
        with_blockmap=False,
    )
    release_artifacts.record("macos", "arm64", pre_pr_root, claim_tag, commit, pre_pr_root / "meta.json")
    assert str(pre_pr_dmg) not in calls

    legacy_root, legacy_dmg = make_root(
        "legacy-claim",
        stamp={"source": "commit-build", "commit": commit, "tag": None, "baseVersion": "1.2.3"},
        version="1.2.3",
        with_blockmap=False,
    )
    release_artifacts.record(
        "macos", "arm64", legacy_root, claim_tag, commit, legacy_root / "meta.json", legacy_commit_build=True,
    )
    assert str(legacy_dmg) not in calls
    assert json.loads((legacy_root / "meta.json").read_text(encoding="utf-8"))["legacyCommitBuild"] is True


@pytest.mark.parametrize("kind", ["tagged", "channel", "signed commit claim"])
def test_macos_record_refuses_a_root_without_its_outer_dmg(tmp_path, monkeypatch, kind):
    """A record that must check the outer DMG fails closed when the root holds only the update ZIP."""
    commit = "a" * 40
    request = channel_request(commit, sequence=6) if kind == "channel" else None
    tag, version, stamp = {
        "tagged": ("v1.2.3", "1.2.3", {"commit": commit, "tag": "v1.2.3", "baseVersion": "1.2.3"}),
        "channel": (None, request and request["version"],
                    {"source": "channel-build", "channelBuild": request, "commit": commit, "tag": None}),
        "signed commit claim": ("commit.20261125T000000Z-v1.2.3", "1.2.3+commit.20261125T000000Z", {
            "source": "commit-build", "commit": commit, "tag": "commit.20261125T000000Z-v1.2.3",
            "baseVersion": "1.2.3+commit.20261125T000000Z", "signedDmg": True,
        }),
    }[kind]
    root = tmp_path / "release"
    (root / "mac" / "Product.app").mkdir(parents=True)
    with zipfile.ZipFile(root / f"Product-{version}-mac-arm64.zip", "w") as archive:
        archive.writestr("Product.app/Contents/Info.plist", plistlib.dumps({
            "CFBundleIdentifier": request["identity"]["appId"] if request else "ai.hermes.test",
            "CFBundleShortVersionString": version,
        }))
        archive.writestr("Product.app/Contents/Resources/install-stamp.json", json.dumps(stamp))
    monkeypatch.setattr(release_artifacts.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(
        argv, 0, stderr="TeamIdentifier=ABCDEFGHIJ\n"))
    out = root / "metadata.json"
    with pytest.raises(ValueError, match="Expected exactly one artifact"):
        release_artifacts.record("macos", "arm64", root, tag, commit, out, channel_request=request)
    assert not out.exists()
