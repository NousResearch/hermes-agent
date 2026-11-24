"""Bot-loop reset epoch file, a sibling of ``<root>/config.yaml``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gateway.bot_loop_reset import EPOCH_NAME, epoch_path, read_epoch, write_epoch


def test_epoch_name_and_path(tmp_path: Path) -> None:
    assert EPOCH_NAME == "bot-loop-reset-epoch"
    assert epoch_path(tmp_path) == tmp_path / EPOCH_NAME


def test_missing_file_reads_as_zero(tmp_path: Path) -> None:
    assert not epoch_path(tmp_path).exists()
    assert read_epoch(tmp_path) == 0.0


def test_write_then_read_returns_the_same_float(tmp_path: Path) -> None:
    assert write_epoch(tmp_path, 0.1) is None
    assert epoch_path(tmp_path).read_text(encoding="utf-8") == "0.100000\n"
    assert read_epoch(tmp_path) == 0.1


def test_later_write_replaces_the_file_with_one_line(tmp_path: Path) -> None:
    write_epoch(tmp_path, 1.5)
    write_epoch(tmp_path, 2.25)
    text = epoch_path(tmp_path).read_text(encoding="utf-8")
    assert text == "2.250000\n"
    assert text.splitlines() == ["2.250000"]
    assert len(text.splitlines()) == 1
    assert read_epoch(tmp_path) == 2.25


def test_garbage_non_finite_and_non_positive_read_as_zero(tmp_path: Path) -> None:
    path = epoch_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for body in ("not-a-float\n", "", "1e99999\n", "nan\n", "inf\n", "-inf\n", "0\n", "0.0\n", "-1.5\n"):
        path.write_text(body, encoding="utf-8")
        assert read_epoch(tmp_path) == 0.0

    path.write_bytes(b"\xff\xfe not utf-8 epoch")
    assert read_epoch(tmp_path) == 0.0


def test_write_creates_root_not_a_gateway_state_subdirectory(tmp_path: Path) -> None:
    root = tmp_path / "fresh-home"
    assert not root.exists()
    write_epoch(root, 1.25)
    assert root.is_dir()
    assert epoch_path(root) == root / EPOCH_NAME
    assert epoch_path(root).is_file()
    assert not (root / "gateway-state").exists()
    assert epoch_path(root).read_text(encoding="utf-8") == "1.250000\n"


def test_write_replaces_a_temp_file_in_the_same_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def _replace(src, dst):
        replaced.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr("gateway.bot_loop_reset.os.replace", _replace)
    root = tmp_path / "install"
    assert write_epoch(root, 2.5) is None
    assert read_epoch(root) == 2.5
    assert len(replaced) == 1
    src, dst = replaced[0]
    assert dst == epoch_path(root)
    assert src.parent == dst.parent
    assert src.name != dst.name
    assert not src.exists()
    assert [p.name for p in dst.parent.iterdir()] == [EPOCH_NAME]


def test_read_uses_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = epoch_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("4.5\n".encode("utf-8"))
    encodings: list[object] = []
    real_read_text = Path.read_text

    def _read_text(self: Path, encoding=None, errors=None, newline=None):
        if self == path:
            encodings.append(encoding)
        return real_read_text(self, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "read_text", _read_text)
    assert read_epoch(tmp_path) == 4.5
    assert encodings == ["utf-8"]


def test_module_does_not_import_config() -> None:
    """Storage helpers stay config-free. ``note_human_reset`` is the policy caller."""
    import inspect

    import gateway.bot_loop_reset as mod

    source_path = mod.__file__
    assert source_path
    storage = "\n".join(
        inspect.getsource(getattr(mod, name))
        for name in ("epoch_path", "read_epoch", "write_epoch")
    )
    assert "hermes_cli" not in storage
    assert "import config" not in storage
    assert "bot_loop_reset_on_human" not in storage
