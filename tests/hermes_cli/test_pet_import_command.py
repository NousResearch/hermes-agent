"""CLI coverage for ``hermes pets import``."""

from __future__ import annotations

import argparse
import io

import pytest

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

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
