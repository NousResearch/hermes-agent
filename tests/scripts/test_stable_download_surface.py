"""Behavior contracts for the stable macOS DMG URL and releases/stable/index.html."""
from __future__ import annotations

import contextlib
import hashlib
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.release_channels import ChannelError, canonical_json
from scripts.bundles import release_artifacts as artifacts
from scripts.releases import channel_releases, docker, handoff, lanes as lane_set, r2, sequencer, store
from tests.scripts.test_release_channels import object_server, publisher
from tests.scripts.test_release_darwin import _inputs, base64_sha512
from tests.scripts.test_release_r2 import r2_server
from tests.scripts.test_release_sequencer import _sequencer_fixture
from tests.scripts.test_stable_release import https_origin

STABLE_DMG_KEY = "releases/darwin/stable/HermesBundled-mac-arm64.dmg"
STABLE_PAGE_KEY = "releases/stable/index.html"
STABLE_APPINSTALLER_KEY = "releases/win32/stable/stable.appinstaller"
STABLE_MAC_FEED_KEY = "releases/darwin/stable/stable-mac.yml"
SMOKE_RESULTS = {
    name: {"result": "success"}
    for name in ("smoke-darwin-arm64", "smoke-win32-arm64", "smoke-win32-x64")
}
BASE_EPOCH = 1_787_965_323


def _stage_stable_release(
    tmp_path: Path,
    server,
    origin,
    *,
    version: str,
    attempt_num: int = 1,
    epoch_offset: int = 0,
    dmg_bytes: bytes | None = None,
    include_dmg: bool = True,
) -> tuple[dict, bytes]:
    import hermes_yaml

    tag = f"v{version}"
    attempt = f"rc.{attempt_num}-{tag}"
    commit = "a" * 40
    epoch = BASE_EPOCH + epoch_offset
    windows_version = f"2026.5761.{123 + epoch_offset}.0"
    base = origin.base
    origin.store = server.store

    built = tmp_path / f"built-{version}-{attempt_num}"
    built.mkdir(parents=True, exist_ok=True)

    zip_name = f"HermesBundled-{version}-mac-arm64.zip"
    dmg_name = f"HermesBundled-{version}-mac-arm64.dmg"
    zip_data = f"zip payload for {version}".encode()
    dmg_data = dmg_bytes if dmg_bytes is not None else f"dmg payload for {version}".encode()

    (built / zip_name).write_bytes(zip_data)
    mac_files = [
        {"url": zip_name, "size": len(zip_data), "sha512": base64_sha512(zip_data)},
    ]
    mac_includes = [zip_name, "metadata-macos-arm64.json", "arm64-stable-mac.yml"]
    if include_dmg:
        (built / dmg_name).write_bytes(dmg_data)
        mac_files.append({"url": dmg_name, "size": len(dmg_data), "sha512": base64_sha512(dmg_data)})
        mac_includes.append(dmg_name)

    (built / "arm64-stable-mac.yml").write_text(
        hermes_yaml.safe_dump(
            {
                "version": version,
                "files": mac_files,
                "path": zip_name,
                "sha512": base64_sha512(zip_data),
                "releaseDate": "2026-10-14T00:00:00Z",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (built / "metadata-macos-arm64.json").write_text(
        json.dumps(
            {
                "platform": "macos",
                "arch": "arm64",
                "tag": tag,
                "commit": commit,
                "identity": "com.nousresearch.hermes-bundled",
                "version": version,
                "teamId": "ABCDEFGHIJ",
                "filename": zip_name,
            }
        ),
        encoding="utf-8",
    )
    handoff.stage(attempt, commit, "darwin-arm64", built, mac_includes)

    for arch in ("x64", "arm64"):
        msix_name = f"HermesBundled-{version}-win-{arch}.msix"
        (built / msix_name).write_bytes(f"msix {version} {arch}".encode())
        (built / f"metadata-windows-{arch}.json").write_text(
            json.dumps(
                {
                    "platform": "windows",
                    "arch": arch,
                    "tag": tag,
                    "commit": commit,
                    "identity": "NousResearch.HermesBundled",
                    "publisher": "CN=Test",
                    "applicationId": "HermesBundled",
                    "version": windows_version,
                    "executableVersion": windows_version,
                }
            ),
            encoding="utf-8",
        )
        handoff.stage(attempt, commit, f"win32-{arch}", built, [msix_name, f"metadata-windows-{arch}.json"])

    bundle_name = f"HermesBundled-{windows_version}-win.msixbundle"
    with zipfile.ZipFile(built / bundle_name, "w") as archive:
        archive.writestr(
            "AppxMetadata/AppxBundleManifest.xml",
            f'<Bundle><Identity Name="NousResearch.HermesBundled" Publisher="CN=Test" '
            f'Version="{windows_version}"/><Packages>'
            '<Package Type="application" Architecture="arm64"/>'
            '<Package Type="application" Architecture="x64"/></Packages></Bundle>',
        )
    (built / f"Store-{bundle_name}").write_bytes(b"store bundle fixture")
    handoff.stage(attempt, commit, "windows-universal", built, ["*.msixbundle"])

    fetched = tmp_path / f"fetched-{version}-{attempt_num}"
    handoff.fetch(
        attempt,
        commit,
        ["win32-x64", "win32-arm64", "darwin-arm64", "windows-universal"],
        fetched,
        ["metadata-*.json", "*.msixbundle"],
    )
    manifest = artifacts.assemble(
        fetched,
        tag,
        commit,
        base,
        tmp_path / f"candidates-{version}-{attempt_num}.json",
        smoke_results=SMOKE_RESULTS,
        release_epoch=epoch,
        archive=attempt,
        lanes=lane_set.DEFAULT,
    )
    return manifest, dmg_data


def test_cache_control_marks_stable_dmg_alias_and_page_no_store():
    assert r2.cache_control_for(STABLE_DMG_KEY) == "no-store"
    assert r2.cache_control_for(STABLE_PAGE_KEY) == "no-store"
    assert r2.cache_control_for("releases/tag/rc.3-v0.22.0/HermesBundled-0.22.0-mac-arm64.dmg") is None


def test_stable_promotion_publishes_unversioned_dmg_and_downloads_page(
    tmp_path,
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    https_origin,  # health: allow F811 -- pytest injects the imported https_origin fixture by parameter name
):
    monkeypatch.setattr(urllib.request, "urlopen", https_origin.opener)
    manifest, dmg_bytes = _stage_stable_release(tmp_path, r2_server, https_origin, version="1.2.3")
    # Same version and table-shaped, so a page built from a bucket listing would show it.
    orphan = f"releases/tag/{manifest['archive']}/HermesLight-1.2.3-win-x64.msix"
    r2_server.store[orphan] = (b"unreceipted light msix", '"e"')
    r2_server.requests.clear()

    artifacts.promote(manifest, tmp_path / "promote", https_origin.base)

    assert STABLE_DMG_KEY in r2_server.store
    assert r2_server.store[STABLE_DMG_KEY][0] == dmg_bytes
    dmg_puts = [
        headers
        for method, path, headers in r2_server.requests
        if method == "PUT" and path == f"/hermes-releases/{STABLE_DMG_KEY}"
    ]
    assert len(dmg_puts) == 1
    assert dmg_puts[0]["x-amz-copy-source"] == (
        f"/hermes-releases/releases/tag/{manifest['archive']}/HermesBundled-1.2.3-mac-arm64.dmg"
    )
    assert dmg_puts[0]["x-amz-metadata-directive"] == "REPLACE"
    assert dmg_puts[0]["Content-Length"] == "0"
    assert dmg_puts[0]["Cache-Control"] == "no-store"
    assert dmg_puts[0]["Content-Disposition"] == 'attachment; filename="HermesBundled-mac-arm64.dmg"'
    assert dmg_puts[0]["Content-Type"] == "application/x-apple-diskimage"
    assert not (tmp_path / "promote" / "HermesBundled-1.2.3-mac-arm64.dmg").exists()

    assert STABLE_PAGE_KEY in r2_server.store
    page_puts = [
        headers
        for method, path, headers in r2_server.requests
        if method == "PUT" and path == f"/hermes-releases/{STABLE_PAGE_KEY}"
    ]
    assert len(page_puts) == 1
    assert page_puts[0]["Cache-Control"] == "no-store"
    assert page_puts[0]["Content-Type"] == "text/html; charset=utf-8"

    page = r2_server.store[STABLE_PAGE_KEY][0].decode("utf-8")
    assert '<meta name="hermes-build" content="v1.2.3">' in page
    assert f"{https_origin.base}/releases/tag/{manifest['archive']}/HermesBundled-1.2.3-mac-arm64.dmg" in page
    assert f"{https_origin.base}/releases/tag/{manifest['archive']}/HermesBundled-1.2.3-win-x64.msix" in page
    assert f"{https_origin.base}/releases/tag/{manifest['archive']}/HermesBundled-1.2.3-win-arm64.msix" in page
    assert "Attempt builds are not upgrade-safe" not in page
    assert orphan not in page
    assert "Hermes Light" not in page
    assert page.count("Passed") == len(SMOKE_RESULTS)

    r2_server.requests.clear()
    artifacts.promote(manifest, tmp_path / "promote-retry", https_origin.base)
    assert not [
        path
        for method, path, _ in r2_server.requests
        if method == "PUT" and path in (f"/hermes-releases/{STABLE_DMG_KEY}", f"/hermes-releases/{STABLE_PAGE_KEY}")
    ]


class _PublicOriginServingStalePage(dict):
    """The public origin's view of the bucket, still serving an old stable downloads page."""

    def __init__(self, bucket: dict):
        super().__init__()
        self.bucket = bucket

    def get(self, key, default=None):
        if key == STABLE_PAGE_KEY:
            return (b"stale cached page", '"cdn"')
        return self.bucket.get(key, default)

def test_stable_promotion_rejects_stale_order_for_dmg_and_page(
    tmp_path,
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    https_origin,  # health: allow F811 -- pytest injects the imported https_origin fixture by parameter name
):
    monkeypatch.setattr(urllib.request, "urlopen", https_origin.opener)
    older, _older_dmg = _stage_stable_release(
        tmp_path, r2_server, https_origin, version="1.2.3", attempt_num=1, epoch_offset=0
    )
    newer, newer_dmg = _stage_stable_release(
        tmp_path, r2_server, https_origin, version="1.2.4", attempt_num=1, epoch_offset=10
    )
    artifacts.promote(newer, tmp_path / "promote-newer", https_origin.base)
    newer_page = r2_server.store[STABLE_PAGE_KEY][0]
    assert r2_server.store[STABLE_DMG_KEY][0] == newer_dmg

    r2_server.requests.clear()
    with pytest.raises(ValueError, match="backward"):
        artifacts.promote(older, tmp_path / "promote-older", https_origin.base)
    assert not any(method == "PUT" for method, _, _ in r2_server.requests)
    assert r2_server.store[STABLE_DMG_KEY][0] == newer_dmg
    assert r2_server.store[STABLE_PAGE_KEY][0] == newer_page

    del r2_server.store[STABLE_MAC_FEED_KEY]
    r2_server.requests.clear()
    with pytest.raises(ValueError, match="backward"):
        artifacts.promote(older, tmp_path / "promote-older-dmg-guard", https_origin.base)
    assert not any(method == "PUT" for method, _, _ in r2_server.requests)
    assert r2_server.store[STABLE_DMG_KEY][0] == newer_dmg
    assert r2_server.store[STABLE_PAGE_KEY][0] == newer_page

    # Only the page ahead is the state render-builds-table's `complete` step leaves when it
    # renders a newer release while an older advance is still pending. The sequencer advances
    # oldest first, so the older promotion must land and leave that page alone, not wedge.
    # The page it leaves is not its own, so it does not read that page back either: a public copy
    # still catching up must not fail the advance after the feeds have moved.
    del r2_server.store[STABLE_DMG_KEY]
    r2_server.metadata.pop(STABLE_DMG_KEY, None)
    r2_server.requests.clear()
    https_origin.store = _PublicOriginServingStalePage(r2_server.store)
    artifacts.promote(older, tmp_path / "promote-older-page-ahead", https_origin.base)
    https_origin.store = r2_server.store
    assert r2_server.store[STABLE_DMG_KEY][0] == _older_dmg
    assert r2_server.store[STABLE_PAGE_KEY][0] == newer_page
    assert not any(method == "PUT" and path == f"/hermes-releases/{STABLE_PAGE_KEY}"
                   for method, path, _ in r2_server.requests)

    artifacts.promote(newer, tmp_path / "restore-newer", https_origin.base)
    assert r2_server.store[STABLE_DMG_KEY][0] == newer_dmg
    assert r2_server.store[STABLE_PAGE_KEY][0] == newer_page
    different_same_ver, _ = _stage_stable_release(
        tmp_path,
        r2_server,
        https_origin,
        version="1.2.4",
        attempt_num=2,
        epoch_offset=10,
        dmg_bytes=b"forged different dmg bytes for 1.2.4",
    )
    r2_server.requests.clear()
    with pytest.raises(ValueError, match="different artifact"):
        artifacts.promote(different_same_ver, tmp_path / "promote-same-ver-diff", https_origin.base)
    assert not any(method == "PUT" for method, _, _ in r2_server.requests)
    assert r2_server.store[STABLE_DMG_KEY][0] == newer_dmg


def test_stable_promotion_refuses_a_canary_page_at_the_stable_key(
    tmp_path,
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    https_origin,  # health: allow F811 -- pytest injects the imported https_origin fixture by parameter name
):
    """Only a newer stable page is left in place; a page from another channel fails closed."""
    monkeypatch.setattr(urllib.request, "urlopen", https_origin.opener)
    manifest, _ = _stage_stable_release(tmp_path, r2_server, https_origin, version="1.2.3")
    wrong_channel = b'<meta name="hermes-build" content="canary.20261120T064609Z-v9.9.9">\n'
    r2_server.store[STABLE_PAGE_KEY] = (wrong_channel, '"canary-page"')
    r2_server.requests.clear()
    with pytest.raises(ValueError, match="canary"):
        artifacts.promote(manifest, tmp_path / "wrong-channel-page", https_origin.base)
    assert not any(method == "PUT" for method, _, _ in r2_server.requests)
    assert r2_server.store[STABLE_PAGE_KEY][0] == wrong_channel

def test_stable_promotion_fails_closed_when_dmg_missing_from_manifest_or_r2(
    tmp_path,
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    https_origin,  # health: allow F811 -- pytest injects the imported https_origin fixture by parameter name
):
    monkeypatch.setattr(urllib.request, "urlopen", https_origin.opener)
    no_dmg_manifest, _ = _stage_stable_release(
        tmp_path, r2_server, https_origin, version="1.2.3", attempt_num=1, include_dmg=False
    )
    r2_server.requests.clear()
    with pytest.raises(ValueError, match="macOS DMG"):
        artifacts.promote(no_dmg_manifest, tmp_path / "no-dmg", https_origin.base)
    assert not any(method == "PUT" for method, _, _ in r2_server.requests)

    manifest, _ = _stage_stable_release(
        tmp_path, r2_server, https_origin, version="1.2.4", attempt_num=1, epoch_offset=10
    )
    source_dmg_key = f"releases/tag/{manifest['archive']}/HermesBundled-1.2.4-mac-arm64.dmg"
    saved = r2_server.store.pop(source_dmg_key)
    r2_server.requests.clear()
    with pytest.raises(r2.R2RequestError):
        artifacts.promote(manifest, tmp_path / "missing-src-dmg", https_origin.base)
    assert not any(method == "PUT" for method, _, _ in r2_server.requests)
    r2_server.store[source_dmg_key] = saved


def test_stable_promotion_checks_the_dmg_copy_limit_before_any_feed_moves(
    tmp_path,
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    https_origin,  # health: allow F811 -- pytest injects the imported https_origin fixture by parameter name
):
    """A DMG over the single-copy limit stops promotion before the first feed write, not after
    the macOS feed has moved."""
    monkeypatch.setattr(urllib.request, "urlopen", https_origin.opener)
    manifest, dmg_bytes = _stage_stable_release(tmp_path, r2_server, https_origin, version="1.2.3")
    monkeypatch.setattr(r2, "MAX_SINGLE_COPY_SIZE", len(dmg_bytes) - 1)
    r2_server.requests.clear()
    with pytest.raises(ValueError, match="CopyObject size"):
        artifacts.promote(manifest, tmp_path / "oversized", https_origin.base)
    assert not any(method == "PUT" for method, _, _ in r2_server.requests)

@contextlib.contextmanager
def _stable_channel_controller(tmp_path, monkeypatch):
    """Run publish_release against a real ChannelPublisher over an object server;
    promote_stable_feeds only records the feed promotion, or fails it on request."""
    from scripts.releases.channels import preview_identity
    from tests.scripts.test_release_channels import _stage_accepted_native

    identity = preview_identity("stable", "9" * 16)
    commit = "c" * 40
    with object_server() as (url, objects, _headers, _requests, _faults):
        pub = publisher(url)
        base = pub.public_base
        objects["releases/channels/stable.json"] = canonical_json(
            {
                "schema": 1,
                "name": "stable",
                "repository": pub.repository,
                "policy": "stable-release",
                "state": "active",
                "revision": 1,
                "nextSequence": 1,
                "head": None,
                "identity": identity,
            }
        )
        monkeypatch.setattr(r2, "credentials", lambda: (pub.store.creds, url, "bucket"))
        monkeypatch.setattr(r2, "public_base_url", lambda: base)
        monkeypatch.setattr(r2, "put", lambda **kw: pub.store.put(kw["key"], Path(kw["file"]).read_bytes()))

        def setup_release(version: str, attempt: str, win_ver: str):
            _stage_accepted_native(tmp_path, objects, identity, attempt, commit, version, win_ver, "stable")
            # Override metadata tags to plain payload tag vX.Y.Z as stable releases stamp them.
            payload_tag = f"v{version}"
            for name in ("metadata-macos-arm64.json", "metadata-windows-x64.json", "metadata-windows-arm64.json"):
                key = f"releases/tag/{attempt}/{name}"
                row = json.loads(objects[key])
                row["tag"] = payload_tag
                body = canonical_json(row)
                objects[key] = body
            # Refresh receipts to match updated metadata bytes.
            for leg, meta_names in (
                ("darwin-arm64", ["metadata-macos-arm64.json"]),
                ("win32-x64", ["metadata-windows-x64.json"]),
                ("win32-arm64", ["metadata-windows-arm64.json"]),
            ):
                rkey = f"releases/tag/{attempt}/{handoff.receipt_name(leg)}"
                receipt = json.loads(objects[rkey])
                for entry in receipt["files"]:
                    if entry["path"] in meta_names:
                        raw = objects[f"releases/tag/{attempt}/{entry['path']}"]
                        entry["size"] = len(raw)
                        entry["sha256"] = hashlib.sha256(raw).hexdigest()
                objects[rkey] = canonical_json(receipt)

            prefix = f"releases/tag/{attempt}/"
            zip_name = f"{identity['artifactNamePascal']}-{version}-mac-arm64.zip"
            bundle_name = f"{identity['artifactNamePascal']}-{win_ver}-win.msixbundle"
            return {
                "tag": payload_tag,
                "archive": attempt,
                "packages": [
                    {
                        "platform": "macos",
                        "arch": "arm64",
                        "identity": identity["appId"],
                        "version": version,
                        "teamId": "ABCDEFGHIJ",
                        "artifact": {
                            "url": f"{base}/{prefix}{zip_name}",
                            "sha256": hashlib.sha256(objects[prefix + zip_name]).hexdigest(),
                        },
                    },
                    *[
                        {
                            "platform": "windows",
                            "arch": arch,
                            "identity": identity["msixAppIdWithOrg"],
                            "version": win_ver,
                            "publisher": "CN=Fixture",
                            "applicationId": identity["appNamePascal"],
                            "artifact": {
                                "url": f"{base}/{prefix}{bundle_name}",
                                "sha256": hashlib.sha256(objects[prefix + bundle_name]).hexdigest(),
                            },
                        }
                        for arch in ("x64", "arm64")
                    ],
                ],
            }

        staged: dict[str, dict] = {}
        current: dict = {}
        promoted: list[str] = []

        monkeypatch.setattr(
            channel_releases, "admit_transaction", lambda policy, env, **_kw: (current["tag"], commit)
        )
        monkeypatch.setattr(
            channel_releases.stable,
            "final_context",
            lambda _env: (current["tag"], commit, {"claim_epoch": BASE_EPOCH, "lanes": lane_set.DEFAULT}),
        )
        monkeypatch.setattr(channel_releases, "accepted_stable", lambda *args, **kwargs: current["accepted"])

        def promote_feeds(candidate, *_args):
            if current["fail_feeds"]:
                raise RuntimeError("feed promotion failed")
            promoted.append(candidate["tag"])

        monkeypatch.setattr(channel_releases, "promote_stable_feeds", promote_feeds)
        monkeypatch.setattr(channel_releases, "product_identity", lambda _tag: dict(identity))
        env = {"GITHUB_REPOSITORY": pub.repository}

        def publish(attempt: str, version: str, win_ver: str, *, fail_feeds: bool = False) -> None:
            if attempt not in staged:
                staged[attempt] = setup_release(version, attempt, win_ver)
            current.update(tag=attempt, accepted=staged[attempt], fail_feeds=fail_feeds)
            root = tmp_path / f"pub-{len(promoted)}-{attempt}-{fail_feeds}"
            channel_releases.publish_release("stable-release", env, root)

        yield SimpleNamespace(publish=publish, promoted=promoted)


def test_stale_stable_channel_publish_stops_before_feed_promotion(tmp_path, monkeypatch):
    with _stable_channel_controller(tmp_path, monkeypatch) as channel:
        channel.publish("rc.1-v2.1.0", "2.1.0", "2026.5761.200.0")
        assert channel.promoted == ["v2.1.0"]

        with pytest.raises(ChannelError, match="stale"):
            channel.publish("rc.1-v2.0.0", "2.0.0", "2026.5761.100.0")
        assert channel.promoted == ["v2.1.0"]


def test_out_of_order_stable_retry_stops_before_feed_promotion(tmp_path, monkeypatch):
    """A release allocated before an older one published keeps its lower sequence on retry.
    The head update refuses it on that sequence, so its feeds must not move either."""
    with _stable_channel_controller(tmp_path, monkeypatch) as channel:
        with pytest.raises(RuntimeError, match="feed promotion failed"):
            channel.publish("rc.1-v2.2.0", "2.2.0", "2026.5761.300.0", fail_feeds=True)
        channel.publish("rc.1-v2.1.0", "2.1.0", "2026.5761.200.0")
        assert channel.promoted == ["v2.1.0"]

        with pytest.raises(ChannelError, match="stale"):
            channel.publish("rc.1-v2.2.0", "2.2.0", "2026.5761.300.0")
        assert channel.promoted == ["v2.1.0"]


@pytest.mark.parametrize("lanes", [["windows-store"], ["windows"], ["macos-arm64"]])
def test_store_only_and_partial_desktop_release_leaves_stable_dmg_and_page_untouched(
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    lanes,
):
    archived = json.dumps({"files": [{"path": "Store-Product-win.msixbundle"}]}).encode()
    manifest_digest = hashlib.sha256(archived).hexdigest()
    _tags, releases, run = _sequencer_fixture(
        "0.23.5", manifest_digest=manifest_digest, docker_digest=None, lanes=lanes
    )
    r2_server.store[STABLE_DMG_KEY] = (b"previous v0.22.0 dmg bytes", '"prev-dmg"')
    r2_server.store[STABLE_PAGE_KEY] = (b'<meta name="hermes-build" content="v0.22.0">', '"prev-page"')
    r2_server.store[STABLE_APPINSTALLER_KEY] = (b"<AppInstaller/>", '"prev-app"')
    r2_server.store[STABLE_MAC_FEED_KEY] = (b"version: 0.22.0\n", '"prev-feed"')
    before = dict(r2_server.store)
    r2_server.requests.clear()

    monkeypatch.setattr(channel_releases, "stable_head_version", lambda _env: "0.22.0")
    monkeypatch.setattr(docker, "stable_alias_digest", lambda: None)
    monkeypatch.setattr(store, "release_from_env", lambda *_a: "released")

    steps = sequencer.reconcile(
        {"GITHUB_REPOSITORY": "example/project", "REQUESTED_VERSION": "0.23.5"},
        run=run,
        read_archive=lambda _key: archived,
    )

    assert steps == [{"flip": "0.23.5"}, {"advance": "0.23.5"}]
    assert releases[0]["draft"] is False
    assert r2_server.store == before
    assert not any(method == "PUT" for method, _, _ in r2_server.requests)


def test_r2_pruner_protects_stable_dmg_alias_and_downloads_page(
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
):
    from scripts.releases.darwin import merge_mac_feeds

    legs, _ = _inputs("0.22.0")
    stable_feed = merge_mac_feeds(legs, "v0.22.0", archive="rc.3-v0.22.0")
    live_archive_dmg = "releases/tag/rc.3-v0.22.0/HermesBundled-0.22.0-mac-arm64.dmg"
    live_archive_zip = "releases/tag/rc.3-v0.22.0/HermesBundled-0.22.0-mac-arm64.zip"
    expired_canary = "releases/tag/canary.20200101T000000Z-v0.20.0/HermesBundled-0.20.0-mac-arm64.dmg"
    expired_commit = "releases/tag/commit.20200101T000000Z-v0.20.0/HermesBundled-0.20.0-mac-arm64.dmg"

    protected_keys = {
        STABLE_DMG_KEY,
        STABLE_PAGE_KEY,
        "releases/canary/index.html",
        STABLE_MAC_FEED_KEY,
        live_archive_dmg,
        live_archive_zip,
        live_archive_zip + ".blockmap",
    }
    for key in protected_keys | {expired_canary, expired_commit}:
        r2_server.store[key] = (b"fixture bytes", '"e"')
        r2_server.last_modified[key] = "2020-01-01T00:00:00Z"
    r2_server.store[STABLE_MAC_FEED_KEY] = (stable_feed["text"].encode(), '"feed"')

    r2.prune(keep_days=14, now_epoch=datetime(2026, 11, 24, tzinfo=UTC).timestamp())

    assert expired_canary not in r2_server.store
    assert expired_commit not in r2_server.store
    for key in protected_keys:
        assert key in r2_server.store, key


def test_stable_promotion_fails_on_public_readback_mismatch_for_dmg_and_page(
    tmp_path,
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    https_origin,  # health: allow F811 -- pytest injects the imported https_origin fixture by parameter name
):
    monkeypatch.setattr(urllib.request, "urlopen", https_origin.opener)
    manifest, _ = _stage_stable_release(tmp_path, r2_server, https_origin, version="1.2.3")

    r2_server.corrupt_put = STABLE_PAGE_KEY
    with pytest.raises(ValueError, match="Channel read-back differs: releases/stable/index.html"):
        artifacts.promote(manifest, tmp_path / "bad-page-readback", https_origin.base)
    r2_server.corrupt_put = None
    del r2_server.store[STABLE_PAGE_KEY]

    r2_server.store[STABLE_DMG_KEY] = (b"wrong size on public origin", '"stale"')
    r2_server.metadata[STABLE_DMG_KEY] = {}
    del r2_server.store[STABLE_MAC_FEED_KEY]
    del r2_server.store[STABLE_APPINSTALLER_KEY]
    r2_server.corrupt_put = STABLE_DMG_KEY
    with pytest.raises(r2.R2RequestError, match="size mismatch"):
        artifacts.promote(manifest, tmp_path / "bad-dmg-copy", https_origin.base)
    assert STABLE_APPINSTALLER_KEY not in r2_server.store
    assert STABLE_PAGE_KEY not in r2_server.store
    r2_server.corrupt_put = None


class _PublicOriginServingStaleAlias(dict):
    """The public origin's view of the bucket, serving an old size for the DMG alias only."""

    def __init__(self, bucket: dict):
        super().__init__()
        self.bucket = bucket

    def get(self, key, default=None):
        if key == STABLE_DMG_KEY:
            return (b"stale cached alias", '"cdn"')
        return self.bucket.get(key, default)


def test_stable_promotion_fails_when_public_origin_serves_another_dmg_size(
    tmp_path,
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    https_origin,  # health: allow F811 -- pytest injects the imported https_origin fixture by parameter name
):
    monkeypatch.setattr(urllib.request, "urlopen", https_origin.opener)
    manifest, dmg_bytes = _stage_stable_release(tmp_path, r2_server, https_origin, version="1.2.3")
    https_origin.store = _PublicOriginServingStaleAlias(r2_server.store)

    with pytest.raises(ValueError, match=f"Channel read-back differs: {STABLE_DMG_KEY}"):
        artifacts.promote(manifest, tmp_path / "stale-public-alias", https_origin.base)
    # R2 took the copy; the public read-back stopped promotion before the later pointers moved.
    assert r2_server.store[STABLE_DMG_KEY][0] == dmg_bytes
    assert STABLE_APPINSTALLER_KEY not in r2_server.store
    assert STABLE_PAGE_KEY not in r2_server.store

def test_stable_promotion_upgrades_existing_dmg_and_page_and_detects_etag_race(
    tmp_path,
    monkeypatch,
    r2_server,  # health: allow F811 -- pytest injects the imported r2_server fixture by parameter name
    https_origin,  # health: allow F811 -- pytest injects the imported https_origin fixture by parameter name
):
    monkeypatch.setattr(urllib.request, "urlopen", https_origin.opener)
    v1, v1_dmg = _stage_stable_release(tmp_path, r2_server, https_origin, version="1.2.3", epoch_offset=0)
    v2, v2_dmg = _stage_stable_release(tmp_path, r2_server, https_origin, version="1.2.4", epoch_offset=10)

    artifacts.promote(v1, tmp_path / "promote-v1", https_origin.base)
    assert r2_server.store[STABLE_DMG_KEY][0] == v1_dmg
    assert '<meta name="hermes-build" content="v1.2.3">' in r2_server.store[STABLE_PAGE_KEY][0].decode()

    r2_server.race_key = STABLE_PAGE_KEY
    with pytest.raises(r2.R2RequestError, match="412"):
        artifacts.promote(v2, tmp_path / "promote-v2-race", https_origin.base)
    r2_server.race_key = None

    artifacts.promote(v2, tmp_path / "promote-v2-retry", https_origin.base)
    assert r2_server.store[STABLE_DMG_KEY][0] == v2_dmg
    assert '<meta name="hermes-build" content="v1.2.4">' in r2_server.store[STABLE_PAGE_KEY][0].decode()
    assert f"{https_origin.base}/releases/tag/{v2['archive']}/HermesBundled-1.2.4-mac-arm64.dmg" in (
        r2_server.store[STABLE_PAGE_KEY][0].decode()
    )
