"""POSIX Node install/upgrade must stage-then-swap, never delete the live tree before its
replacement is ready (mirrors the Windows _stage_windows_node_zip/_swap_node_tree treatment in
hermes_constants.py). See issue #106456.
"""

from __future__ import annotations

import stat
import subprocess
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_BOOTSTRAP = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"

# node-bootstrap.sh maps `uname -m`'s raw value to the nodejs.org release-arch label used in
# tarball names (e.g. raw "x86_64" -> release label "x64"). Fixtures below build tarballs named
# with the release label, so the fake `uname -m` stub must return the matching raw value.
_UNAME_M_FOR_NODE_ARCH = {"x64": "x86_64", "arm64": "aarch64", "armv7l": "armv7l"}


def _make_fake_node_tarball(dest: Path, *, major: int, os_name: str, arch: str, version: str) -> Path:
    """A tarball shaped like a real nodejs.org release: node-vX.Y.Z-<os>-<arch>/bin/{node,npm,npx}."""
    root = dest / f"node-v{version}-{os_name}-{arch}"
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("node", "npm", "npx"):
        script = bin_dir / name
        script.write_text(f"#!/bin/sh\necho v{version}\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    tarball = dest / f"node-v{version}-{os_name}-{arch}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(root, arcname=root.name)
    return tarball


def _run_install(hermes_home: Path, *, fixture_dir: Path, target_major: int, os_name: str, arch: str) -> subprocess.CompletedProcess:
    """Source node-bootstrap.sh with a fake `curl` that serves the fixture tarball, then call
    _nb_install_bundled_node directly (skip the version-manager/package-manager tiers)."""
    fake_curl = f"""
curl() {{
    local url="${{@: -1}}"
    if [[ "$url" == *"latest-v{target_major}.x/"* ]]; then
        cat "{fixture_dir}/index.html"
    else
        cp "{fixture_dir}"/*.tar.gz "$(echo "$@" | grep -oE '\\-o [^ ]+' | cut -d' ' -f2)" 2>/dev/null || \\
        cat "{fixture_dir}"/*.tar.gz
    fi
}}
export -f curl
export HERMES_HOME="{hermes_home}"
export HERMES_NODE_TARGET_MAJOR="{target_major}"
export HERMES_NODE_SKIP_LINKS="1"
source "{NODE_BOOTSTRAP}"
uname() {{ [ "$1" = "-m" ] && echo "{_UNAME_M_FOR_NODE_ARCH[arch]}" || echo "{os_name.capitalize()}"; }}
export -f uname
_nb_install_bundled_node
"""
    return subprocess.run(["bash", "-c", fake_curl], capture_output=True, text=True)


def test_upgrade_stages_before_replacing_live_tree(tmp_path):
    """A second install (simulating an upgrade) must not remove the live tree until the new one
    is fully staged, and must leave no node.new-*/node.old-* litter behind on success."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    os_name, arch = "linux", "x64"

    fixture_v22 = tmp_path / "fixture-22"
    fixture_v22.mkdir()
    (fixture_v22 / "index.html").write_text("node-v22.9.0-linux-x64.tar.gz", encoding="utf-8")
    _make_fake_node_tarball(fixture_v22, major=22, os_name=os_name, arch=arch, version="22.9.0")

    result = _run_install(home, fixture_dir=fixture_v22, target_major=22, os_name=os_name, arch=arch)
    assert result.returncode == 0, result.stderr
    assert (home / "node" / "bin" / "node").is_file()

    fixture_v24 = tmp_path / "fixture-24"
    fixture_v24.mkdir()
    (fixture_v24 / "index.html").write_text("node-v24.9.1-linux-x64.tar.gz", encoding="utf-8")
    _make_fake_node_tarball(fixture_v24, major=24, os_name=os_name, arch=arch, version="24.9.1")

    result = _run_install(home, fixture_dir=fixture_v24, target_major=24, os_name=os_name, arch=arch)
    assert result.returncode == 0, result.stderr

    version = subprocess.run(
        [str(home / "node" / "bin" / "node"), "--version"], capture_output=True, text=True)
    assert version.stdout.strip() == "v24.9.1"
    # No leftover staged/backup directories after a clean run.
    assert not list(home.glob("node.new-*"))
    assert not list(home.glob("node.old-*"))


def test_failed_download_never_touches_a_working_live_tree(tmp_path):
    """If the new tarball can't be fetched, the existing working install must be left completely
    untouched — not deleted, not partially replaced."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    os_name, arch = "linux", "x64"

    fixture_v22 = tmp_path / "fixture-22"
    fixture_v22.mkdir()
    (fixture_v22 / "index.html").write_text("node-v22.9.0-linux-x64.tar.gz", encoding="utf-8")
    _make_fake_node_tarball(fixture_v22, major=22, os_name=os_name, arch=arch, version="22.9.0")
    result = _run_install(home, fixture_dir=fixture_v22, target_major=22, os_name=os_name, arch=arch)
    assert result.returncode == 0, result.stderr

    empty_fixture = tmp_path / "fixture-empty"
    empty_fixture.mkdir()
    (empty_fixture / "index.html").write_text("", encoding="utf-8")  # no matching tarball name
    result = _run_install(home, fixture_dir=empty_fixture, target_major=24, os_name=os_name, arch=arch)
    assert result.returncode != 0

    version = subprocess.run(
        [str(home / "node" / "bin" / "node"), "--version"], capture_output=True, text=True)
    assert version.stdout.strip() == "v22.9.0"  # untouched


def test_uses_rename_based_swap_not_rm_then_extract(tmp_path):
    """Static check: the extract-then-replace step must be a stage-and-rename, not `rm -rf` +
    `mv` straight into the live path (the shape that offers no rollback on a bad extraction)."""
    script = NODE_BOOTSTRAP.read_text(encoding="utf-8")
    assert 'rm -rf "$HERMES_HOME/node"\n    mv "$extracted" "$HERMES_HOME/node"' not in script
    assert "node.new-" in script
    assert "node.old-" in script


def _run_install_with_failing_mv_call(
    hermes_home: Path, *, fixture_dir: Path, target_major: int, os_name: str, arch: str,
    fail_on_call: int,
) -> subprocess.CompletedProcess:
    """Like `_run_install`, but the Nth `mv` invocation inside the sourced script fails (returns
    1) instead of actually moving anything — used to force the final stage->live rename itself to
    fail, the one branch none of the tests above exercise."""
    fake_curl = f"""
curl() {{
    local url="${{@: -1}}"
    if [[ "$url" == *"latest-v{target_major}.x/"* ]]; then
        cat "{fixture_dir}/index.html"
    else
        cp "{fixture_dir}"/*.tar.gz "$(echo "$@" | grep -oE '\\-o [^ ]+' | cut -d' ' -f2)" 2>/dev/null || \\
        cat "{fixture_dir}"/*.tar.gz
    fi
}}
export -f curl
declare -i __mv_calls=0
mv() {{
    __mv_calls+=1
    if [ "$__mv_calls" -eq {fail_on_call} ]; then
        return 1
    fi
    command mv "$@"
}}
export -f mv
export HERMES_HOME="{hermes_home}"
export HERMES_NODE_TARGET_MAJOR="{target_major}"
export HERMES_NODE_SKIP_LINKS="1"
source "{NODE_BOOTSTRAP}"
uname() {{ [ "$1" = "-m" ] && echo "{_UNAME_M_FOR_NODE_ARCH[arch]}" || echo "{os_name.capitalize()}"; }}
export -f uname
_nb_install_bundled_node
"""
    return subprocess.run(["bash", "-c", fake_curl], capture_output=True, text=True)


def test_final_swap_failure_rolls_back_to_the_working_live_tree(tmp_path):
    """The tests above never force the actual `mv "$staged" "$HERMES_HOME/node"` rename itself
    to fail (only a failed *download*, which never reaches the swap). If that final rename fails
    (e.g. a race, or the target volume rejects the rename), the pre-upgrade tree must be restored
    exactly, not left half-swapped or gone — mirroring the Windows
    test_second_rename_failure_rolls_back coverage for _swap_node_tree."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    os_name, arch = "linux", "x64"

    fixture_v22 = tmp_path / "fixture-22"
    fixture_v22.mkdir()
    (fixture_v22 / "index.html").write_text("node-v22.9.0-linux-x64.tar.gz", encoding="utf-8")
    _make_fake_node_tarball(fixture_v22, major=22, os_name=os_name, arch=arch, version="22.9.0")
    result = _run_install(home, fixture_dir=fixture_v22, target_major=22, os_name=os_name, arch=arch)
    assert result.returncode == 0, result.stderr

    fixture_v24 = tmp_path / "fixture-24"
    fixture_v24.mkdir()
    (fixture_v24 / "index.html").write_text("node-v24.9.1-linux-x64.tar.gz", encoding="utf-8")
    _make_fake_node_tarball(fixture_v24, major=24, os_name=os_name, arch=arch, version="24.9.1")

    # Call order for an upgrade (live tree present): 1) extracted->staged, 2) live->backup,
    # 3) staged->live (the one we fail here).
    result = _run_install_with_failing_mv_call(
        home, fixture_dir=fixture_v24, target_major=24, os_name=os_name, arch=arch, fail_on_call=3)

    assert result.returncode != 0

    version = subprocess.run(
        [str(home / "node" / "bin" / "node"), "--version"], capture_output=True, text=True)
    assert version.stdout.strip() == "v22.9.0"  # rolled back to the pre-upgrade tree
    assert not list(home.glob("node.new-*"))
    assert not list(home.glob("node.old-*"))


def test_corrupted_download_leaves_a_working_live_tree_untouched(tmp_path):
    """A tarball that downloads successfully (curl exits 0) but isn't a valid archive — a
    truncated transfer, or an HTML error page served with a `.tar.gz`-looking name — must fail
    extraction cleanly rather than partially replace the live tree or leave litter behind."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    os_name, arch = "linux", "x64"

    fixture_v22 = tmp_path / "fixture-22"
    fixture_v22.mkdir()
    (fixture_v22 / "index.html").write_text("node-v22.9.0-linux-x64.tar.gz", encoding="utf-8")
    _make_fake_node_tarball(fixture_v22, major=22, os_name=os_name, arch=arch, version="22.9.0")
    result = _run_install(home, fixture_dir=fixture_v22, target_major=22, os_name=os_name, arch=arch)
    assert result.returncode == 0, result.stderr

    corrupt_fixture = tmp_path / "fixture-corrupt"
    corrupt_fixture.mkdir()
    (corrupt_fixture / "index.html").write_text("node-v24.9.1-linux-x64.tar.gz", encoding="utf-8")
    (corrupt_fixture / "node-v24.9.1-linux-x64.tar.gz").write_bytes(b"not actually a gzip tarball")

    result = _run_install(home, fixture_dir=corrupt_fixture, target_major=24, os_name=os_name, arch=arch)
    assert result.returncode != 0

    version = subprocess.run(
        [str(home / "node" / "bin" / "node"), "--version"], capture_output=True, text=True)
    assert version.stdout.strip() == "v22.9.0"  # untouched
    assert not list(home.glob("node.new-*"))
    assert not list(home.glob("node.old-*"))


def test_stale_staging_litter_is_swept_before_install(tmp_path):
    """Mirrors the Windows heal's stale-litter sweep coverage: a node.new-*/node.old-* directory
    left by an interrupted previous run, older than the 10-minute cutoff, must be cleaned up by
    the next install rather than accumulating forever."""
    import os as _os
    import time as _time

    home = tmp_path / "hermes-home"
    home.mkdir()
    os_name, arch = "linux", "x64"
    stale_new = home / "node.new-deadbeef-111"
    stale_new.mkdir()
    stale_old = home / "node.old-deadbeef-111"
    stale_old.mkdir()
    old_ts = _time.time() - 3600
    _os.utime(stale_new, (old_ts, old_ts))
    _os.utime(stale_old, (old_ts, old_ts))

    fixture = tmp_path / "fixture-22"
    fixture.mkdir()
    (fixture / "index.html").write_text("node-v22.9.0-linux-x64.tar.gz", encoding="utf-8")
    _make_fake_node_tarball(fixture, major=22, os_name=os_name, arch=arch, version="22.9.0")

    result = _run_install(home, fixture_dir=fixture, target_major=22, os_name=os_name, arch=arch)

    assert result.returncode == 0, result.stderr
    assert not stale_new.exists()
    assert not stale_old.exists()


def test_fresh_concurrent_staging_litter_survives_the_sweep(tmp_path):
    """A node.new-*/node.old-* directory younger than the 10-minute cutoff must NOT be swept —
    it may belong to another `hermes update`/`--upgrade-node` process mid-swap right now. Sweeping
    it would gut a concurrent install rather than merely tidying up an abandoned one."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    os_name, arch = "linux", "x64"
    fresh_new = home / "node.new-cafef00d-222"
    fresh_new.mkdir()
    (fresh_new / "sentinel").write_text("in-flight", encoding="utf-8")

    fixture = tmp_path / "fixture-22"
    fixture.mkdir()
    (fixture / "index.html").write_text("node-v22.9.0-linux-x64.tar.gz", encoding="utf-8")
    _make_fake_node_tarball(fixture, major=22, os_name=os_name, arch=arch, version="22.9.0")

    result = _run_install(home, fixture_dir=fixture, target_major=22, os_name=os_name, arch=arch)

    assert result.returncode == 0, result.stderr
    assert fresh_new.exists()
    assert (fresh_new / "sentinel").read_text(encoding="utf-8") == "in-flight"
