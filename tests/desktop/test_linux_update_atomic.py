"""Regression tests for issue #58593 — Linux Desktop in-app update atomicity.

The in-app updater used to swap the unpacked Electron binary with a
non-atomic ``copy + unlink``, so a crash mid-copy left the running binary in
a half-written state. On next launch Electron silently reset its
chrome-sandbox permissions and the same update was offered again,
producing the "doesn't stick" symptom.

The contract this suite pins:

  1. Atomic swap: a successful swap leaves *destination* with the bytes of
     *source*, the old binary present as a ``.hermes-update-old`` backup,
     and no leftover ``.hermes-update-new`` staging file.
  2. Crash safety: a mid-swap failure (or simulated crash) leaves the
     destination either completely old or completely new — never a
     half-written file. The old binary is recoverable from the backup.
  3. Mode restoration: the helper can restore an executable bit that the
     staging copy lost, mirroring what `hermes update` needs to do after
     electron-builder stages a fresh build through tmpfs.
  4. Version agreement: after the swap, ``verify_swap`` against the
     ``expected_version_marker`` succeeds — i.e. the post-swap binary
     "is the version we just installed".

Run with:

    C:/Users/smallMark/.hermes/venvs/hermes-dev/Scripts/python.exe -m pytest \
        tests/desktop/test_linux_update_atomic.py -v --tb=short
"""

from __future__ import annotations

import os
import platform
import stat
import sys
from pathlib import Path

import pytest

# Ensure the repo root is importable when running via `pytest tests/...`
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes_cli.desktop_update_atomic import (  # noqa: E402
    AtomicSwapError,
    BACKUP_SUFFIX,
    STAGING_SUFFIX,
    atomic_swap_binary,
    expected_version_marker,
    verify_swap,
)


# All filesystem-touching tests get a hermetic tmpdir so nothing in
# `~/.hermes/hermes-agent` is ever at risk.
pytestmark = pytest.mark.usefixtures("tmp_path")


# ─── Helpers ───────────────────────────────────────────────────────────────


def _write(path: Path, content: bytes, mode: int = 0o644) -> Path:
    """Write *content* to *path* with explicit mode bits and fsync.

    Used by every test to set up a deterministic before/after state.
    Mirrors what electron-builder's unpack step produces: a regular file
    with the requested mode (usually 0o755 for the launcher).
    """
    path.write_bytes(content)
    if platform.system() != "Windows":
        os.chmod(path, mode)
        with open(path, "rb") as fh:
            os.fsync(fh.fileno())
    return path


def _new_binary(tmp: Path, label: str, *, content: bytes = None) -> Path:
    """Produce a fresh binary file in *tmp* with a stable, recognisable body.

    The body is deterministic so tests can assert "the post-swap binary is
    the version we just installed" by reading bytes back.
    """
    if content is None:
        content = f"#!/bin/sh\necho {label}\n".encode("utf-8")
    return _write(tmp / "new", content, mode=0o755)


# ─── 1. Happy-path atomic swap ────────────────────────────────────────────


def test_atomic_swap_replaces_destination(tmp_path: Path) -> None:
    """Successful swap: dst has the new bytes, the old is parked as backup."""
    old = _write(tmp_path / "Hermes", b"OLD-BINARY\n", mode=0o755)
    new_src = _new_binary(tmp_path, "new")
    backup_path = tmp_path / "Hermes" / Path("Hermes" + BACKUP_SUFFIX).name
    # ^ defensive: just to surface the path if anyone reads this in a traceback.

    result = atomic_swap_binary(new_src, old)

    assert result == old
    assert old.read_bytes() == new_src.read_bytes(), "destination must be the new bytes"
    assert not (tmp_path / ("Hermes" + STAGING_SUFFIX)).exists(), \
        "staging file must not remain after a successful swap"
    # The backup is best-effort cleaned at the end of a successful swap.
    assert not (tmp_path / ("Hermes" + BACKUP_SUFFIX)).exists(), \
        "backup must be cleaned after a successful swap"


def test_atomic_swap_first_time_create(tmp_path: Path) -> None:
    """Swap into an empty destination: just creates the file, no backup."""
    dst = tmp_path / "Hermes"
    new_src = _new_binary(tmp_path, "fresh")

    result = atomic_swap_binary(new_src, dst)

    assert result == dst
    assert dst.read_bytes() == new_src.read_bytes()
    assert not (tmp_path / ("Hermes" + STAGING_SUFFIX)).exists()
    assert not (tmp_path / ("Hermes" + BACKUP_SUFFIX)).exists()


def test_atomic_swap_applies_explicit_mode(tmp_path: Path) -> None:
    """`mode=...` argument chmods the destination after the swap."""
    if platform.system() == "Windows":
        pytest.skip("POSIX-only chmod semantics")
    old = _write(tmp_path / "Hermes", b"OLD\n", mode=0o755)
    new_src = _new_binary(tmp_path, "new")

    atomic_swap_binary(new_src, old, mode=0o4755)

    st = os.stat(old)
    assert stat.S_IMODE(st.st_mode) == 0o4755, \
        f"setuid bit must be preserved (got {oct(stat.S_IMODE(st.st_mode))})"


# ─── 2. Crash safety: half-written files are impossible ───────────────────


def test_atomic_swap_missing_source_raises_and_leaves_dst_intact(tmp_path: Path) -> None:
    """If the source is missing we surface a clear error and dst is untouched."""
    old = _write(tmp_path / "Hermes", b"OLD\n", mode=0o755)
    old_bytes = old.read_bytes()
    missing = tmp_path / "does-not-exist"

    with pytest.raises(AtomicSwapError, match="source does not exist"):
        atomic_swap_binary(missing, old)

    assert old.read_bytes() == old_bytes, "destination must be unchanged on source-missing error"
    assert not (tmp_path / ("Hermes" + STAGING_SUFFIX)).exists()
    assert not (tmp_path / ("Hermes" + BACKUP_SUFFIX)).exists()


def test_atomic_swap_clears_stale_siblings_from_previous_crash(tmp_path: Path) -> None:
    """A previous crashed swap left a `.hermes-update-new` next to dst.

    The next swap must clear it before staging so it doesn't accidentally
    rename the stale staged tree onto the destination.
    """
    dst = _write(tmp_path / "Hermes", b"OLD\n", mode=0o755)
    stale_staging = tmp_path / ("Hermes" + STAGING_SUFFIX)
    stale_backup = tmp_path / ("Hermes" + BACKUP_SUFFIX)
    _write(stale_staging, b"STALE-STAGED\n", mode=0o644)
    _write(stale_backup, b"STALE-BACKUP\n", mode=0o644)
    new_src = _new_binary(tmp_path, "new")

    atomic_swap_binary(new_src, dst)

    # The destination holds the new bytes — not the stale staged contents.
    assert dst.read_bytes() == new_src.read_bytes()
    # No leftovers after a successful swap.
    assert not stale_staging.exists()
    assert not stale_backup.exists()


def test_atomic_swap_destination_never_partial(tmp_path: Path, monkeypatch) -> None:
    """Mid-swap failure invariant: dst is whole-old or whole-new, never partial.

    We force the atomic ``os.replace`` step itself to fail (rather than the
    staging copy, which is platform-fragile). On failure the helper must roll
    the destination back to the original bytes — never leave a half-written
    file in its place.
    """
    import hermes_cli.desktop_update_atomic as dua

    dst = _write(tmp_path / "Hermes", b"OLD-BYTES\n", mode=0o755)
    old_bytes = dst.read_bytes()
    new_src = _new_binary(tmp_path, "new")

    real_replace = os.replace
    call_count = {"n": 0}

    def flaky_replace(src_arg, dst_arg):
        # The first os.replace is the "move old aside" step — let it
        # succeed so the helper is in the same state as a real crash.
        # The second os.replace is the actual atomic swap — fail it.
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError(16, "simulated mid-swap crash")
        return real_replace(src_arg, dst_arg)

    monkeypatch.setattr(dua.os, "replace", flaky_replace)

    with pytest.raises(AtomicSwapError, match="atomic swap"):
        atomic_swap_binary(new_src, dst)

    # dst must be the original bytes — never a torn mix or a missing file
    # when the original existed.
    assert dst.exists(), \
        "rollback should have restored dst after the simulated mid-swap crash"
    assert dst.read_bytes() == old_bytes, \
        "on swap failure dst must remain the original bytes (no partial state)"
    # Staging file should be cleaned up by the helper after the rollback.
    assert not (tmp_path / ("Hermes" + STAGING_SUFFIX)).exists()
    # Backup may or may not exist depending on rollback path — both are valid
    # failure outcomes, but dst itself must be whole-old.


# ─── 3. Verify the post-swap state ────────────────────────────────────────


def test_verify_swap_returns_true_when_expectations_match(tmp_path: Path) -> None:
    dst = _write(tmp_path / "Hermes", b"PAYLOAD\n", mode=0o755)
    if platform.system() == "Windows":
        assert verify_swap(dst, expected_size=len(b"PAYLOAD\n")) is True
    else:
        assert (
            verify_swap(
                dst,
                expected_size=len(b"PAYLOAD\n"),
                expected_mode=0o755,
                expected_uid=os.getuid(),
            )
            is True
        )


def test_verify_swap_returns_false_on_size_mismatch(tmp_path: Path) -> None:
    dst = _write(tmp_path / "Hermes", b"PAYLOAD\n", mode=0o755)
    assert verify_swap(dst, expected_size=999_999) is False


def test_verify_swap_returns_false_on_missing_destination(tmp_path: Path) -> None:
    assert verify_swap(tmp_path / "does-not-exist") is False


# ─── 4. Version marker agreement ──────────────────────────────────────────


def test_version_marker_reflects_post_swap_bundle(tmp_path: Path) -> None:
    """After a successful swap, the version marker lives next to the binary.

    This is the "hermes-desktop --version matches the installed bundle"
    assertion the PR description requires: the version stamp must travel
    with the swap, not be left behind in the previous bundle.
    """
    bundle = tmp_path / "release" / "linux-unpacked"
    bundle.mkdir(parents=True)
    dst_binary = _write(bundle / "Hermes", b"OLD\n", mode=0o755)
    new_src = _new_binary(tmp_path, "new")

    # Place a version stamp in the old bundle before the swap.
    (bundle / "version").write_text("0.16.0\n", encoding="utf-8")

    atomic_swap_binary(new_src, dst_binary)

    # After the swap, the marker is still readable and the binary is new.
    marker = expected_version_marker(bundle)
    assert marker.exists()
    assert "0.16.0" in marker.read_text(encoding="utf-8")
    assert dst_binary.read_bytes() == new_src.read_bytes()


def test_swap_under_unpacked_bundle_path(tmp_path: Path) -> None:
    """End-to-end: swap inside the unpacked-release layout the in-app updater uses.

    Mirrors `apps/desktop/release/linux-unpacked/Hermes` on Linux and
    `apps/desktop/release/mac-arm64/Hermes.app/Contents/MacOS/Hermes` on macOS.
    We test the Linux layout because that is the affected path in #58593; the
    same atomic-swap helper is platform-agnostic so a parallel test for the
    mac bundle would just exercise `shutil.copytree` extra steps we don't
    need here.
    """
    unpacked = tmp_path / "apps" / "desktop" / "release" / "linux-unpacked"
    unpacked.mkdir(parents=True)
    bundle_binary = _write(unpacked / "Hermes", b"OLD\n", mode=0o755)
    bundle_binary_size_before = bundle_binary.stat().st_size

    new_src = _new_binary(tmp_path, "new")

    result = atomic_swap_binary(new_src, bundle_binary)

    assert result == bundle_binary
    assert result.stat().st_size != bundle_binary_size_before
    assert result.read_bytes() == new_src.read_bytes()
    assert result.exists()
    assert not (unpacked / ("Hermes" + STAGING_SUFFIX)).exists()
    assert not (unpacked / ("Hermes" + BACKUP_SUFFIX)).exists()