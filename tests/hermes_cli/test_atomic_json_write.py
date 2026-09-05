"""Tests for utils.atomic_json_write — crash-safe JSON file writes."""

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_cli.config as config
import utils
from utils import atomic_json_write


posix_only = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX permission bits are advisory on Windows",
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class TestAtomicJsonWrite:
    """Core atomic write behavior."""







    def test_cleans_up_temp_file_on_baseexception(self, tmp_path):
        class SimulatedAbort(BaseException):
            pass

        target = tmp_path / "data.json"
        original = {"preserved": True}
        target.write_text(json.dumps(original), encoding="utf-8")

        with patch("utils.json.dump", side_effect=SimulatedAbort):
            with pytest.raises(SimulatedAbort):
                atomic_json_write(target, {"new": True})

        tmp_files = [f for f in tmp_path.iterdir() if ".tmp" in f.name]
        assert len(tmp_files) == 0
        assert json.loads(target.read_text(encoding="utf-8")) == original




    def test_mode_does_not_crash_without_fchmod(self, tmp_path):
        """Regression: os.fchmod is Unix-only and absent on Windows. Passing a
        mode must not raise AttributeError when fchmod is unavailable.

        Simulates the Windows os module by removing fchmod from the namespace.
        Previously this crashed in `hermes memory setup` while saving the
        Hindsight config with mode=0o600 (GitHub: Windows setup traceback).
        """
        import utils

        target = tmp_path / "secret.json"
        no_fchmod = {k: getattr(os, k) for k in dir(os) if k != "fchmod"}
        fake_os = type("FakeOs", (), no_fchmod)
        assert not hasattr(fake_os, "fchmod")

        with patch.object(utils, "os", fake_os):
            atomic_json_write(target, {"api_key": "secret"}, mode=0o600)

        assert json.loads(target.read_text(encoding="utf-8")) == {"api_key": "secret"}


    def test_concurrent_writes_dont_corrupt(self, tmp_path):
        """Multiple rapid writes should each produce valid JSON."""
        import threading

        target = tmp_path / "concurrent.json"
        errors = []

        def writer(n):
            try:
                atomic_json_write(target, {"writer": n, "data": list(range(100))})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # File should contain valid JSON from one of the writers
        result = json.loads(target.read_text(encoding="utf-8"))
        assert "writer" in result
        assert len(result["data"]) == 100


@posix_only
def test_open_private_append_creates_fresh_file_with_requested_mode(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.jsonl"

    old_umask = os.umask(0o022)
    try:
        with utils.open_private_append(target, mode=0o600) as handle:
            handle.write("first\n")
    finally:
        os.umask(old_umask)

    assert _mode(target) == 0o600
    assert target.read_text(encoding="utf-8") == "first\n"


@posix_only
def test_open_private_append_preserves_existing_mode_and_appends(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.jsonl"
    target.write_text("first\n", encoding="utf-8")
    os.chmod(target, 0o640)

    old_umask = os.umask(0o022)
    try:
        with utils.open_private_append(target, mode=0o600) as handle:
            handle.write("second\n")
    finally:
        os.umask(old_umask)

    assert _mode(target) == 0o640
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


@posix_only
def test_internal_append_tightens_before_writing_without_touching_explicit_outputs(tmp_path):
    for internal in (False, True):
        target = tmp_path / str(internal)
        target.write_text("old\n", encoding="utf-8")
        target.chmod(0o644)
        with utils.open_private_append(target, tighten_existing=internal) as handle:
            assert _mode(target) == (0o600 if internal else 0o644)
            handle.write("new\n")
        assert target.read_text(encoding="utf-8") == "old\nnew\n"


@posix_only
def test_internal_append_reports_links_and_fails_before_bytes_on_chmod_error(tmp_path, monkeypatch, caplog):
    target = tmp_path / "target"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o644)
    link = tmp_path / "link"
    link.symlink_to(target)
    with utils.open_private_append(link, tighten_existing=True) as handle:
        handle.write("linked\n")
    assert _mode(target) == 0o644
    assert link.is_symlink()
    assert "verify its target permissions" in caplog.text
    before = target.read_bytes()

    def denied(*args):
        raise PermissionError("chmod denied")

    monkeypatch.setattr(utils.os, "fchmod", denied)
    with pytest.raises(PermissionError):
        with utils.open_private_append(target, tighten_existing=True) as handle:
            handle.write("must not be written")
    assert target.read_bytes() == before


@posix_only
def test_open_private_append_uses_managed_group_writable_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    target = tmp_path / "artifact.jsonl"

    old_umask = os.umask(0o022)
    try:
        with utils.open_private_append(
            target, mode=config.artifact_file_mode()
        ) as handle:
            handle.write("managed\n")
    finally:
        os.umask(old_umask)

    assert _mode(target) == 0o660
    assert target.read_text(encoding="utf-8") == "managed\n"


@posix_only
@pytest.mark.require_symlinks
def test_open_private_append_creates_through_dangling_symlink_privately(
    tmp_path: Path,
) -> None:
    """A dangling symlinked artifact path must still be appendable.

    Managed deployments symlink state files into a git-tracked profile
    package, so an artifact path can be a symlink whose target does not exist
    yet. Plain open(path, "a") creates through it at umask default
    (0o644 -- world-readable); this helper must create through it too, but at
    the requested private mode rather than failing the write.
    """
    real = tmp_path / "real.jsonl"
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)

    old_umask = os.umask(0o022)
    try:
        with utils.open_private_append(link, mode=0o600) as handle:
            handle.write("through\n")
    finally:
        os.umask(old_umask)

    assert link.is_symlink(), "symlinked artifact path must survive"
    assert _mode(real) == 0o600
    assert real.read_text(encoding="utf-8") == "through\n"


@posix_only
@pytest.mark.require_symlinks
def test_open_private_append_preserves_live_symlink_target_mode(
    tmp_path: Path,
) -> None:
    """An existing symlink target keeps its own mode across an append."""
    real = tmp_path / "real.jsonl"
    real.write_text("first\n", encoding="utf-8")
    os.chmod(real, 0o644)
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)

    with utils.open_private_append(link, mode=0o600) as handle:
        handle.write("second\n")

    assert link.is_symlink()
    assert _mode(real) == 0o644
    assert real.read_text(encoding="utf-8") == "first\nsecond\n"
