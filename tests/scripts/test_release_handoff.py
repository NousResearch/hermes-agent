"""Release handoffs use exact R2 bytes without advancing a channel."""
import copy
import json

import pytest

from scripts.releases import r2
from tests.scripts.test_release_r2 import r2_server  # noqa: F401


@pytest.mark.parametrize('tag', ['v1.2.3', 'v1.2.3-canary.20260908232538'])
def test_stage_and_fetch_bind_tag_commit_and_files_without_feed_writes(tmp_path, r2_server, tag):
    from scripts.releases import handoff

    commit = "a" * 40
    root = tmp_path / "built"
    root.mkdir()
    data = b"multi-chunk package transport fixture\n" * 90000
    (root / "app.msix").write_bytes(data)
    (root / "Store-app.msix").write_bytes(b"store transport fixture")
    (root / "metadata-windows-x64.json").write_text("{}", encoding="utf-8")
    nested = root / "apt" / "dists" / "hermes-stable" / "InRelease"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"APT index transport fixture")
    handoff.main(["stage", "--tag", tag, "--commit", commit, "--name", "win32-x64",
                  "--root", str(root), "--include", "*.msix", "--include", "metadata-*.json", "--include", "apt/**/*"])
    receipt_key = f"releases/tag/{tag}/handoff-win32-x64.json"
    receipt = json.loads(r2_server.store[receipt_key][0])
    assert receipt["tag"] == tag and receipt["commit"] == commit
    assert {row["path"] for row in receipt["files"]} == {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert all(key.startswith(f"releases/tag/{tag}/") for key in r2_server.store)
    puts = [path for method, path, _ in r2_server.requests if method == "PUT"]
    assert puts[-1].endswith(receipt_key)
    assert all(headers.get("If-None-Match") == "*" for method, _, headers in r2_server.requests if method == "PUT")

    downloaded = tmp_path / "downloaded"
    handoff.main(["fetch", "--tag", tag, "--commit", commit, "--name", "win32-x64",
                  "--root", str(downloaded), "--include", "*.msix"])
    assert (downloaded / "app.msix").read_bytes() == data
    assert (downloaded / "Store-app.msix").read_bytes() == b"store transport fixture"
    assert not (downloaded / "metadata-windows-x64.json").exists()
    handoff.stage(tag, commit, "win32-x64", root, ["*.msix", "metadata-*.json", "apt/**/*"])
    handoff.fetch(tag, commit, ["win32-x64"], downloaded, ["apt/*"])
    assert (downloaded / nested.relative_to(root)).read_bytes() == nested.read_bytes()

    with pytest.raises(ValueError, match="identity"):
        handoff.fetch(tag, "b" * 40, ["win32-x64"], tmp_path / "wrong")

    for bad_path in ("../outside", "C:/outside", "dir\\outside", "a//b", "./a", "a%2fb", "a:b", "a?b", "a#b"):
        bad = copy.deepcopy(receipt)
        bad["files"][0]["path"] = bad_path
        r2_server.store[receipt_key] = (json.dumps(bad).encode(), '"e"')
        r2_server.requests.clear()
        with pytest.raises(ValueError, match="path"):
            handoff.fetch(tag, commit, ["win32-x64"], tmp_path / "unsafe")
        assert len(r2_server.requests) == 1
    r2_server.store[receipt_key] = (json.dumps(receipt).encode(), '"e"')
    r2_server.store[f"releases/tag/{tag}/app.msix"] = (b"different bytes", '"e"')
    with pytest.raises(ValueError, match="checksum mismatch"):
        handoff.fetch(tag, commit, ["win32-x64"], downloaded)
    assert (downloaded / "app.msix").read_bytes() == data


def test_failed_stage_never_publishes_a_receipt_or_channel(tmp_path, monkeypatch, r2_server):
    from scripts.releases import handoff

    tag, commit = "v1.2.3", "a" * 40
    (tmp_path / "one.msix").write_bytes(b"first")
    (tmp_path / "two.msix").write_bytes(b"second")
    real_put = r2.put

    def fail_second(**kwargs):
        # The transfer API receives the full key, not the basename.
        if kwargs["key"] == f"releases/tag/{tag}/two.msix":
            raise RuntimeError("upload interrupted")
        real_put(**kwargs)

    monkeypatch.setattr(r2, "put", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        handoff.stage(tag, commit, "win32-x64", tmp_path, ["*.msix"])
    assert set(r2_server.store) == {f"releases/tag/{tag}/one.msix"}
    r2_server.requests.clear()
    with pytest.raises(ValueError, match="No files"):
        handoff.stage(tag, commit, "win32-x64", tmp_path, ["*.zip"])
    assert r2_server.requests == []
    with pytest.raises(ValueError, match="identity"):
        handoff.stage("../bad", commit, "win32-x64", tmp_path, ["*.msix"])
    assert r2_server.requests == []


@pytest.mark.parametrize("commit_only", [False, True])
def test_shared_receipt_files_download_once_and_conflicts_leave_targets_intact(tmp_path, r2_server, commit_only):
    from scripts.releases import handoff

    commit, tag = "a" * 40, "v1.2.3"
    artifact = tmp_path / "shared.bin"
    artifact.write_bytes(b"same shared bytes")
    if commit_only:
        stage = lambda name: handoff.stage_commit_build(commit, name, tmp_path, ["*.bin"])
        fetch = lambda target: handoff.fetch_commit_build(commit, ["one", "two"], target)
        prefix = r2.commit_prefix_for(commit)
    else:
        stage = lambda name: handoff.stage(tag, commit, name, tmp_path, ["*.bin"])
        fetch = lambda target: handoff.fetch(tag, commit, ["one", "two"], target)
        prefix = r2.staging_key_for(tag, "")
    stage("one")
    stage("two")
    r2_server.requests.clear()
    target = tmp_path / "download"
    fetch(target)
    assert (target / "shared.bin").read_bytes() == artifact.read_bytes()
    gets = [url for method, url, _ in r2_server.requests if method == "GET"]
    assert sum(url.endswith(prefix + "shared.bin") for url in gets) == 1

    receipt_key = prefix + "handoff-two.json"
    receipt = json.loads(r2_server.store[receipt_key][0])
    receipt["files"][0]["sha256"] = "0" * 64
    r2_server.store[receipt_key] = (json.dumps(receipt).encode(), '"changed"')
    before = {file.name: file.read_bytes() for file in target.iterdir()}
    with pytest.raises(ValueError, match="Conflicting"):
        fetch(target)
    assert {file.name: file.read_bytes() for file in target.iterdir()} == before


def test_conflicting_bytes_at_the_transport_never_replace_the_object(tmp_path, r2_server):
    """A conflicting immutable upload must preserve the existing object."""
    from scripts.releases import handoff

    tag, commit = "v1.2.3", "a" * 40
    (tmp_path / "one.msix").write_bytes(b"first")
    handoff.stage(tag, commit, "win32-x64", tmp_path, ["*.msix"])
    key = f"releases/tag/{tag}/one.msix"
    assert r2_server.store[key][0] == b"first"
    handoff.stage(tag, commit, "win32-x64", tmp_path, ["*.msix"])
    (tmp_path / "one.msix").write_bytes(b"conflicting")
    with pytest.raises(Exception):
        handoff.stage(tag, commit, "win32-x64", tmp_path, ["*.msix"])
    assert r2_server.store[key][0] == b"first"
