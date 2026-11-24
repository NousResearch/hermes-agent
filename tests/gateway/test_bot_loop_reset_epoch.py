"""Bot-loop reset epoch file under ``<root>/gateway-state``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gateway.bot_loop_reset import EPOCH_NAME, epoch_path, read_epoch, write_epoch


def test_epoch_name_and_path(tmp_path: Path) -> None:
    assert EPOCH_NAME == "bot-loop-reset-epoch"
    assert epoch_path(tmp_path) == tmp_path / "gateway-state" / EPOCH_NAME


def test_missing_file_reads_as_none(tmp_path: Path) -> None:
    assert not epoch_path(tmp_path).exists()
    assert read_epoch(tmp_path) is None


def test_write_then_read_returns_the_same_float(tmp_path: Path) -> None:
    stored = write_epoch(tmp_path, when=0.1)
    assert stored == 0.1
    assert read_epoch(tmp_path) == stored


def test_second_write_is_greater_or_equal_and_file_has_one_line(tmp_path: Path) -> None:
    first = write_epoch(tmp_path)
    second = write_epoch(tmp_path)
    assert isinstance(first, float)
    assert isinstance(second, float)
    assert second >= first
    text = epoch_path(tmp_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines == [text.strip()]
    assert len(lines) == 1
    assert read_epoch(tmp_path) == second


def test_corrupt_file_reads_as_none(tmp_path: Path) -> None:
    path = epoch_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not-a-float\n", encoding="utf-8")
    assert read_epoch(tmp_path) is None

    path.write_text("", encoding="utf-8")
    assert read_epoch(tmp_path) is None

    path.write_bytes(b"\xff\xfe not utf-8 epoch")
    assert read_epoch(tmp_path) is None

    path.write_text("1e99999\n", encoding="utf-8")
    assert read_epoch(tmp_path) is None


def test_write_creates_gateway_state_directory(tmp_path: Path) -> None:
    root = tmp_path / "fresh-home"
    assert not root.exists()
    write_epoch(root, when=1.25)
    assert epoch_path(root).parent.is_dir()
    assert epoch_path(root).is_file()


def test_write_replaces_a_temp_file_in_the_same_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def _replace(src, dst):
        replaced.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr("gateway.bot_loop_reset.os.replace", _replace)
    stored = write_epoch(tmp_path, when=2.5)
    assert read_epoch(tmp_path) == stored
    assert len(replaced) == 1
    src, dst = replaced[0]
    assert dst == epoch_path(tmp_path)
    assert src.parent == dst.parent
    assert src.name != dst.name
    assert not src.exists()
    assert [p.name for p in dst.parent.iterdir()] == [EPOCH_NAME]


def test_read_uses_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = epoch_path(tmp_path)
    path.parent.mkdir(parents=True)
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
    import gateway.bot_loop_reset as mod

    source_path = mod.__file__
    assert source_path
    source = Path(source_path).read_text(encoding="utf-8")
    assert "hermes_cli" not in source
    assert "import config" not in source
    assert "bot_loop_reset_on_human" not in source
