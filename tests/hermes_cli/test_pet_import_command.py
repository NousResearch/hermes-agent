"""CLI coverage for ``hermes pets import``."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from agent.pet import store
from hermes_cli import pets
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def isolated_home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    pets_parser = parser.add_subparsers(dest="command", required=True).add_parser("pets")
    pets.register_cli(pets_parser)
    return parser


def _atlas(path) -> None:
    image = Image.new("RGBA", (1536, 1872), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((30, 20, 160, 190), fill=(80, 120, 240, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    path.write_bytes(output.getvalue())


def test_pet_import_parser_installs_and_selects(monkeypatch, tmp_path, capsys):
    path = tmp_path / "fox.png"
    _atlas(path)
    selected = []
    monkeypatch.setattr(pets, "_set_active", selected.append)
    monkeypatch.setattr(pets, "_has_active_pet", lambda: True)
    args = _parser().parse_args(["pets", "import", str(path), "--name", "Blue Fox", "--select"])

    assert args.func(args) == 0
    assert selected == ["blue-fox"]
    assert "imported Blue Fox" in capsys.readouterr().out


def test_pet_import_parser_reports_missing_and_invalid_files(tmp_path, capsys):
    missing = _parser().parse_args(["pets", "import", str(tmp_path / "missing.zip")])
    assert missing.func(missing) == 1
    assert "not found" in capsys.readouterr().err

    invalid_path = tmp_path / "bad.png"
    invalid_path.write_bytes(b"not an image")
    invalid = _parser().parse_args(["pets", "import", str(invalid_path)])
    assert invalid.func(invalid) == 1
    assert "import failed" in capsys.readouterr().err


def test_pet_import_parser_rejects_oversized_file_before_reading(monkeypatch, tmp_path, capsys):
    path = tmp_path / "huge.zip"
    with path.open("wb") as output:
        output.truncate(store.PET_IMPORT_MAX_BYTES + 1)

    read_paths = []

    def track_read_bytes(file_path):
        read_paths.append(file_path)
        raise AssertionError("oversized import must not be read")

    monkeypatch.setattr(Path, "read_bytes", track_read_bytes)
    args = _parser().parse_args(["pets", "import", str(path)])

    assert args.func(args) == 1
    assert read_paths == []
    assert "pet import exceeds 32 MB" in capsys.readouterr().err
