"""Unit tests for the plugin upgrade-preservation verifier.

tests/install/e2e-assets/verify-plugin-preservation.py is the standalone
hook the release E2E drivers call before and after a real upgrade. These
tests exercise it against a real temp HERMES_HOME (real files, real
symlinks) — no source-reading, no mocks of the filesystem.

The verifier must be read-only against the scanned home and must catch
deletion and modification of every recorded entry kind: regular files,
wrapper markers, directory trees, symlinks (identity + target), and the
externally-owned sidecar witness file a symlinked plugin runtime points at.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
VERIFIER = os.path.join(
    _HERE, "..", "install", "e2e-assets", "verify-plugin-preservation.py"
)

_spec = importlib.util.spec_from_file_location("verify_plugin_preservation", VERIFIER)
vpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vpp)


def _make_link(target, link):
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Windows without symlink privilege: same reparse-point shape.
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def _remove_link(link):
    if os.path.islink(str(link)):
        os.remove(str(link))
    else:  # NTFS junction
        os.rmdir(str(link))


@pytest.fixture()
def home(tmp_path):
    """A controlled temp HERMES_HOME with a non-dependency directory wrapper
    plugin: marker + payload + a symlink to an external runtime whose witness
    file lives OUTSIDE the home (externally-owned), plus a second plugin in a
    profile tree. No pyproject anywhere in the scanned root — the fixture is
    directory-only, so the scanner cannot recurse into a dependency graph and
    the test needs no network/Torch."""
    h = tmp_path / "hermes-home"
    # active-home plugin: directory wrapper with marker + payload
    plugin = h / "plugins" / "mnemosyne-wrapper"
    plugin.mkdir(parents=True)
    (plugin / "mnemosyne-wrapper.json").write_text('{"wrapper": true}\n', encoding="utf-8")
    (plugin / "plugin.py").write_bytes(b"PAYLOAD-BYTES-0\n")
    # external runtime, owned outside the home, reached through a symlink
    external = tmp_path / "external-mnemosyne-runtime"
    external.mkdir()
    (external / "sidecar-witness.txt").write_text("external-witness-v1\n", encoding="utf-8")
    (external / "engine.bin").write_bytes(b"\x00\x01\x02")
    _make_link(external, plugin / "runtime")
    # profile plugin tree
    pplugin = h / "profiles" / "work" / "plugins" / "second-plugin"
    pplugin.mkdir(parents=True)
    (pplugin / "marker.json").write_text('{"p": 1}\n', encoding="utf-8")
    (pplugin / "data.bin").write_bytes(b"profile-bytes\n")
    return h


def _snapshot(home, out):
    snap = vpp.snapshot_home(str(home))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(snap, fh)
    return snap


def _verify(home, snap):
    return vpp.verify_home(str(home), snap)


def test_snapshot_records_all_trees(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    keys = set(snap["entries"])
    assert "plugins/mnemosyne-wrapper/mnemosyne-wrapper.json" in keys
    assert "plugins/mnemosyne-wrapper/plugin.py" in keys
    assert "profiles/work/plugins/second-plugin/data.bin" in keys
    assert snap["roots"] == [
        "<home>/plugins",
        "<home>/profiles/work/plugins",
    ]


def test_snapshot_captures_bytes_and_symlinks(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    e = snap["entries"]
    assert e["plugins/mnemosyne-wrapper/plugin.py"]["sha256"] != ""
    assert e["plugins/mnemosyne-wrapper/plugin.py"]["size"] == 16
    link = e["plugins/mnemosyne-wrapper/runtime"]
    assert link["kind"] == "symlink"
    assert link["target_resolves"] is True
    assert link["target_kind"] == "dir"
    tree = link["target_tree"]
    assert tree["sidecar-witness.txt"]["sha256"] != ""
    assert tree["engine.bin"]["kind"] == "file"


def test_untouched_home_verifies_clean(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    report = _verify(home, snap)
    assert report["ok"] is True
    assert report["counts"]["deleted"] == 0
    assert report["counts"]["modified"] == 0


def test_catches_plugin_file_deletion(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    (home / "plugins" / "mnemosyne-wrapper" / "plugin.py").unlink()
    report = _verify(home, snap)
    assert report["ok"] is False
    assert "plugins/mnemosyne-wrapper/plugin.py" in report["deleted"]


def test_catches_whole_plugin_root_deletion(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    shutil.rmtree(home / "plugins")
    report = _verify(home, snap)
    assert report["ok"] is False
    assert report["counts"]["deleted"] > 0


def test_catches_wrapper_marker_modification(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    (home / "plugins" / "mnemosyne-wrapper" / "mnemosyne-wrapper.json").write_text(
        '{"wrapper": false}\n', encoding="utf-8"
    )
    report = _verify(home, snap)
    assert report["ok"] is False
    assert "plugins/mnemosyne-wrapper/mnemosyne-wrapper.json" in report["modified"]


def test_catches_symlink_target_repoint(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    link = home / "plugins" / "mnemosyne-wrapper" / "runtime"
    other = tmp_path / "other-runtime"
    other.mkdir()
    (other / "sidecar-witness.txt").write_text("different\n", encoding="utf-8")
    _remove_link(link)
    _make_link(other, link)
    report = _verify(home, snap)
    assert report["ok"] is False
    assert "plugins/mnemosyne-wrapper/runtime" in report["modified"]


def test_catches_external_witness_modification(home, tmp_path):
    # The externally-owned sidecar witness lives OUTSIDE the home; an upgrade
    # that tramples it must still be caught through the symlink fingerprint.
    snap = _snapshot(home, tmp_path / "snap.json")
    witness = tmp_path / "external-mnemosyne-runtime" / "sidecar-witness.txt"
    witness.write_text("external-witness-TAMPERED\n", encoding="utf-8")
    report = _verify(home, snap)
    assert report["ok"] is False
    # Depending on the platform's walk, the tamper surfaces either as the
    # link entry (target fingerprint) or as the linked file itself.
    assert any(
        "runtime" in key for key in list(report["modified"]) + report["deleted"]
    )


def test_catches_external_witness_deletion(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    (tmp_path / "external-mnemosyne-runtime" / "engine.bin").unlink()
    report = _verify(home, snap)
    assert report["ok"] is False


def test_catches_profile_plugin_deletion(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    (home / "profiles" / "work" / "plugins" / "second-plugin" / "data.bin").unlink()
    report = _verify(home, snap)
    assert report["ok"] is False
    assert "profiles/work/plugins/second-plugin/data.bin" in report["deleted"]


def test_added_entries_do_not_fail(home, tmp_path):
    # An upgrade may ADD files (new bundled plugin, caches); only taking
    # away or changing existing entries is a violation.
    snap = _snapshot(home, tmp_path / "snap.json")
    newp = home / "plugins" / "fresh-from-upgrade"
    newp.mkdir()
    (newp / "b.txt").write_text("new\n", encoding="utf-8")
    report = _verify(home, snap)
    assert report["ok"] is True
    assert "plugins/fresh-from-upgrade/b.txt" in report["added"]


def test_verifier_is_read_only_against_home(home, tmp_path):
    snap = _snapshot(home, tmp_path / "snap.json")
    before = sorted(
        (p, p.stat().st_size if p.is_file() else "dir")
        for p in home.rglob("*")
    )
    _verify(home, snap)
    _verify(home, snap)
    after = sorted(
        (p, p.stat().st_size if p.is_file() else "dir")
        for p in home.rglob("*")
    )
    assert before == after


def test_catches_empty_dir_deletion(home, tmp_path):
    # A plugin directory emptied (or an empty dir removed) must be caught:
    # directories themselves are recorded, not skipped.
    empty = home / "plugins" / "wrapper-b" / "empty-cache"
    empty.mkdir(parents=True)
    snap2 = _snapshot(home, tmp_path / "snap2.json")
    assert snap2["entries"]["plugins/wrapper-b/empty-cache"] == {"kind": "dir"}
    empty.rmdir()
    (home / "plugins" / "wrapper-b").rmdir()
    report = _verify(home, snap2)
    assert report["ok"] is False
    assert "plugins/wrapper-b/empty-cache" in report["deleted"]


def test_empty_snapshot_is_inconclusive(home, tmp_path):
    # Zero recorded entries cannot prove anything: the CLI refuses.
    empty_home = tmp_path / "bare-home"
    empty_home.mkdir()
    snap_file = tmp_path / "empty-snap.json"
    r1 = subprocess.run(
        [sys.executable, VERIFIER, "snapshot", "--home", str(empty_home),
         "--out", str(snap_file)],
        capture_output=True, text=True,
    )
    assert r1.returncode == 3
    assert "ZERO entries" in r1.stderr
    snap_file.write_text(json.dumps(vpp.snapshot_home(str(empty_home))), encoding="utf-8")
    r2 = subprocess.run(
        [sys.executable, VERIFIER, "verify", "--home", str(empty_home),
         "--snapshot", str(snap_file)],
        capture_output=True, text=True,
    )
    assert r2.returncode == 3
    assert "INCONCLUSIVE" in r2.stderr


@pytest.mark.platforms("posix")
def test_unreadable_path_is_hard_error(home, tmp_path):
    # A scanner that cannot see a path must fail loudly, not skip silently.
    # Skip where chmod-based unreadability is not enforceable (Windows).
    if os.geteuid() == 0:
        pytest.skip("root can read chmod-000 directories")
    secret = home / "plugins" / "mnemosyne-wrapper" / "locked"
    secret.mkdir()
    (secret / "x.txt").write_text("data", encoding="utf-8")
    os.chmod(secret, 0o000)
    try:
        with pytest.raises((OSError, vpp.ScanError)):
            _snapshot(home, tmp_path / "snap2.json")
    finally:
        os.chmod(secret, 0o755)


def test_missing_home_fails_snapshot(tmp_path):
    proc = subprocess.run(
        [sys.executable, VERIFIER, "snapshot", "--home", str(tmp_path / "nope"),
         "--out", str(tmp_path / "x.json")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2


def test_cli_roundtrip_end_to_end(home, tmp_path):
    """The exact command shape the E2E drivers use."""
    snap_file = tmp_path / "snap.json"
    r1 = subprocess.run(
        [sys.executable, VERIFIER, "snapshot", "--home", str(home), "--out", str(snap_file)],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(
        [sys.executable, VERIFIER, "verify", "--home", str(home), "--snapshot", str(snap_file)],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    (home / "plugins" / "mnemosyne-wrapper" / "plugin.py").unlink()
    r3 = subprocess.run(
        [sys.executable, VERIFIER, "verify", "--home", str(home), "--snapshot", str(snap_file)],
        capture_output=True, text=True,
    )
    assert r3.returncode == 1
    assert "PLUGIN PRESERVATION FAILED" in r3.stderr


def test_release_fixture_seed_is_shared_and_never_repairs_damage(tmp_path):
    home, external = tmp_path / "home", tmp_path / "external"
    args = [sys.executable, VERIFIER, "seed", "--home", str(home), "--external", str(external)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    snap = vpp.snapshot_home(str(home))
    runtime = snap["entries"]["plugins/mnemosyne-wrapper/runtime"]
    assert runtime["target_tree"]["engine.bin"]["kind"] == "file"
    witness = external / "sidecar-witness.txt"
    witness.unlink()
    retry = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert retry.returncode != 0
    assert not witness.exists()
    assert not vpp.verify_home(str(home), snap)["ok"]
