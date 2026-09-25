"""Icon generation reports per-target failures without hiding later targets."""
import importlib.util
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
@pytest.mark.parametrize("editor_export", [False, True])
def test_svg_readers_accept_bom_without_rewriting_assets(tmp_path, monkeypatch, bom, editor_export):
    monkeypatch.setitem(sys.modules, "resvg_py", ModuleType("resvg_py"))
    script = Path(__file__).resolve().parents[2] / "scripts/generate_icons.py"
    spec = importlib.util.spec_from_file_location("icon_readers_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "art.svg"
    element = '<path d="M0 0 L1 1" aria-label="café 東京"/>'
    declaration = '<?xml version="1.0" encoding="UTF-8"?>\n' if editor_export else ""
    namespaces = ' xmlns="http://www.w3.org/2000/svg" xmlns:editor="urn:editor"'
    metadata = '<editor:namedview editor:zoom="1"/>' if editor_export else ""
    document = f'<svg{namespaces} viewBox="0 0 20 30">{metadata}{element}</svg>'
    raw = bom + (declaration + document).encode("utf-8")
    path.write_bytes(raw)
    art = SimpleNamespace(girls={"black": path}, paths={}, backgrounds=tmp_path, colors=None)
    assert module.girl_path(art, "black") == element
    inner, width, height = module.background_inner(art, path.name)
    composed = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{inner}</svg>')
    assert (width, height) == (20, 30)
    assert [ET.tostring(child) for child in composed] == [
        ET.tostring(child) for child in ET.fromstring(document)
    ]
    assert path.read_bytes() == raw
    path.write_bytes(bom + b"<svg/>")
    art.paths.clear()
    with pytest.raises(AssertionError, match="no <path>"):
        module.girl_path(art, "black")
    with pytest.raises(AssertionError, match="viewBox"):
        module.background_inner(art, path.name)


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


@pytest.mark.parametrize("platform", ["", "linux-", "mac-"])
@pytest.mark.parametrize("appearance,girl", [("light", "black"), ("dark", "white")])
@pytest.mark.parametrize("colors", [None, ("#f5cc32", "#443808"), ("#e34850", "#4a1117")])
def test_icon_portrait_overlays_border_inside_outer_silhouette(monkeypatch, platform, appearance, girl, colors):
    monkeypatch.setitem(sys.modules, "resvg_py", ModuleType("resvg_py"))
    source = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("icon_geometry_under_test", source / "scripts/generate_icons.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    art = SimpleNamespace(
        backgrounds=source / "assets/backgrounds", colors=colors, commit="0123456",
        bboxes={girl: (0, 0, 100, 90)}, paths={girl: '<path d="M 0 0 L 100 0 L 100 90 L 0 90 z"/>'},
    )
    name = f"squircle-{platform}{appearance}.svg"
    ns = {"svg": "http://www.w3.org/2000/svg"}
    original = ET.parse(art.backgrounds / name).find("svg:rect", ns)
    result = ET.fromstring(module.compose_svg(art, girl, name))
    tile = result.find("svg:rect", ns)
    assert original is not None and tile is not None
    x, y, width, height, radius = (float(original.attrib[key]) for key in ("x", "y", "width", "height", "rx"))
    thickness = width * module.BORDER_FRACTION
    assert float(tile.attrib["stroke-width"]) == pytest.approx(thickness)
    assert tile.get("stroke") == ("#000000" if appearance == "light" else "#ffffff")
    assert float(tile.attrib["x"]) - thickness / 2 == pytest.approx(x)
    assert float(tile.attrib["y"]) - thickness / 2 == pytest.approx(y)
    assert float(tile.attrib["width"]) + thickness == pytest.approx(width)
    assert float(tile.attrib["height"]) + thickness == pytest.approx(height)
    assert float(tile.attrib["rx"]) + thickness / 2 == pytest.approx(radius)
    expected_fill = (colors or ("#ffffff", module.DARK_HEX))[appearance == "dark"]
    assert tile.get("fill") == expected_fill
    clip = result.find("svg:defs/svg:clipPath/svg:rect", ns)
    assert clip is not None
    assert tuple(float(clip.attrib[key]) for key in ("x", "y", "width", "height", "rx")) == (
        x, y, width, height, radius,
    )
    group = result[-1]
    portrait = group[-1]
    assert len(group) == 1, "extend the existing contour, not a duplicate strip"
    assert portrait is not None and group is not None
    assert portrait.get("preserveAspectRatio") == "xMidYMax meet"
    assert tuple(float(portrait.attrib[key]) for key in ("x", "y", "width", "height")) == module.GIRL_BOXES[name]
    clip_path = result.find("svg:defs/svg:clipPath", ns)
    assert clip_path is not None
    assert group.get("clip-path") == f"url(#{clip_path.attrib['id']})"
    assert portrait.get("overflow") == "visible"
    assert len(result.findall(".//svg:path", ns)) == 2  # one badge and one portrait


def test_bottom_node_drag_preserves_upper_geometry_and_path_transform(monkeypatch):
    monkeypatch.setitem(sys.modules, "resvg_py", ModuleType("resvg_py"))
    script = Path(__file__).resolve().parents[2] / "scripts/generate_icons.py"
    spec = importlib.util.spec_from_file_location("icon_nodes_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = ET.fromstring('<path transform="matrix(2,0,0,2,10,-100)" d="M 0 0 C 0 90 10 98 20 100 L 0 100 z"/>')
    module.drag_bottom_nodes(path, cutoff=80, band=20, distance=10)
    assert path.get("transform") == "matrix(2,0,0,2,10,-100)"
    assert path.get("d") == "M 0 0 C 0 90 10 102.48 20 105 L 0 105 z"
    with pytest.raises(ValueError, match="absolute M/L/C"):
        module.drag_bottom_nodes(ET.fromstring('<path d="m 0 0 l 1 1"/>'), cutoff=0, band=1, distance=1)
