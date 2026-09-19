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
    if os.name != "nt":
        vpp.seed_fixtures(h, tmp_path / "external-mnemosyne-runtime")
        (h / "profiles/e2e-preserve").rename(h / "profiles/work")
        return h
    # Retained native NTFS fixture until the shared-seed successor runs on Windows.
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


@pytest.mark.parametrize("relative,action,category", [
    ("plugins/mnemosyne-wrapper/plugin.py", "delete", "deleted"),
    ("plugins", "tree", "deleted"),
    ("plugins/mnemosyne-wrapper/mnemosyne-wrapper.json", "change", "modified"),
    ("plugins/mnemosyne-wrapper/runtime", "repoint", "modified"),
    ("plugins/mnemosyne-wrapper/runtime/sidecar-witness.txt", "change", "modified"),
    ("plugins/mnemosyne-wrapper/runtime/engine.bin", "delete", "modified"),
    ("profiles/work/plugins/second-plugin/data.bin", "delete", "deleted"),
    ("plugins/wrapper-b/empty-cache", "tree", "deleted"),
    ("plugins/fresh-from-upgrade/b.txt", "add", "added"),
])
def test_preservation_cli_fault_matrix(home, tmp_path, relative, action, category):
    empty = home / "plugins/wrapper-b/empty-cache"
    empty.mkdir(parents=True)
    snapshot = tmp_path / "snap.json"
    report = tmp_path / "report.json"
    command = [sys.executable, VERIFIER, "verify", "--home", str(home),
               "--snapshot", str(snapshot), "--report", str(report)]
    def fingerprint():
        return {str(p): (p.read_bytes() if p.is_file() else None, p.lstat().st_mtime_ns,
                         os.readlink(p) if p.is_symlink() or p.is_junction() else None)
                for root in (home, tmp_path / "external-mnemosyne-runtime") for p in root.rglob("*")}
    before = fingerprint()
    result = subprocess.run([sys.executable, VERIFIER, "snapshot", "--home", str(home), "--out", str(snapshot)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    entries = json.loads(snapshot.read_text(encoding="utf-8-sig"))["entries"]
    assert {"plugins/mnemosyne-wrapper/plugin.py", "profiles/work/plugins/second-plugin/data.bin",
            "plugins/mnemosyne-wrapper/mnemosyne-wrapper.json", "plugins/wrapper-b/empty-cache"} <= entries.keys()
    assert entries["plugins/wrapper-b/empty-cache"] == {"kind": "dir"}
    link = entries["plugins/mnemosyne-wrapper/runtime"]
    assert link["kind"] == "symlink" and link["target_resolves"] and link["target_kind"] == "dir"
    assert {"engine.bin", "sidecar-witness.txt"} <= link["target_tree"].keys()
    assert subprocess.run(command, capture_output=True, text=True, timeout=30).returncode == 0
    assert fingerprint() == before
    target = home / relative
    if action == "repoint":
        other = tmp_path / "other-runtime"
        other.mkdir()
        _remove_link(target)
        _make_link(other, target)
    elif action == "tree":
        shutil.rmtree(target)
    elif action == "delete":
        target.unlink()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"tampered or added")
    before = fingerprint()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if action == "add" else 1), result.stderr
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    affected = "plugins/mnemosyne-wrapper/runtime" if "/runtime/" in relative else relative
    assert affected in data[category]
    assert data["ok"] == (action == "add")
    assert fingerprint() == before


def test_empty_snapshot_is_inconclusive(tmp_path):
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
            vpp.snapshot_home(str(home))
    finally:
        os.chmod(secret, 0o755)


def test_missing_home_fails_snapshot(tmp_path):
    proc = subprocess.run(
        [sys.executable, VERIFIER, "snapshot", "--home", str(tmp_path / "nope"),
         "--out", str(tmp_path / "x.json")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2


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
