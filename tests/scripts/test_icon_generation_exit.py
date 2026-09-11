"""Icon generation reports per-target failures without hiding later targets."""
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image


@pytest.mark.parametrize("failure", [None, "render", "directory", "verify"])
def test_write_status_includes_every_target(tmp_path, monkeypatch, capsys, failure):
    # The renderer is build-only. This test injects failures at its byte boundary.
    monkeypatch.setitem(sys.modules, "resvg_py", ModuleType("resvg_py"))
    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_icons.py"
    spec = importlib.util.spec_from_file_location("icon_generator_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "immutable source"
    source.mkdir()
    monkeypatch.setattr(module, "IconArt", lambda root: root)
    monkeypatch.setattr(sys, "argv", [str(script), "--source", str(source), "--out", str(tmp_path)])

    image = io.BytesIO()
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(image, "PNG")
    good_bytes = image.getvalue()
    first = "blocked/icon.png" if failure == "directory" else "first.png"
    if failure == "directory":
        (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(module, "TARGETS", [(first, "png", "first"), ("last.png", "png", "last")])

    def target_bytes(art, kind, target):
        assert art == source
        assert not list(source.iterdir())
        if target == "first":
            if failure == "render":
                raise RuntimeError("injected render failure")
            if failure == "verify":
                return b"not an image"
        return good_bytes

    monkeypatch.setattr(module, "target_bytes", target_bytes)
    code = 0
    try:
        module.main()
    except SystemExit as stopped:
        code = stopped.code
    assert bool(code) is (failure is not None)
    assert (tmp_path / "last.png").read_bytes() == good_bytes
    output = capsys.readouterr().out
    assert "last.png: PNG (2, 2)" in output
    assert ("FAILED" in output) is (failure is not None)
