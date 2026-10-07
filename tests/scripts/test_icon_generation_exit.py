"""Icon generation reports per-target failures without hiding later targets."""
import importlib.util
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image


NS = {"svg": "http://www.w3.org/2000/svg"}


def load_module(name="icon_generator_under_test"):
    sys.modules.setdefault("resvg_py", ModuleType("resvg_py"))
    script = Path(__file__).resolve().parents[2] / "scripts/generate_icons.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_png():
    image = io.BytesIO()
    Image.new("RGBA", (2, 2), (0, 0, 0, 255)).save(image, "PNG")
    return image.getvalue()


def test_svg_readers_accept_bom_and_editor_exports_without_rewriting_assets(tmp_path):
    """background_inner parses the plate through ElementTree, so a UTF-8 BOM,
    an XML declaration and editor-scoped namespaces must not trip it, and the
    asset file itself is never rewritten."""
    module = load_module("icon_readers_under_test")
    path = tmp_path / "squircle-light.svg"
    element = '<rect x="0" y="0" width="20" height="30" rx="6" fill="#ffffff"/>'
    declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    namespaces = ' xmlns="http://www.w3.org/2000/svg" xmlns:editor="urn:editor"'
    metadata = '<editor:namedview editor:zoom="1"/>'
    document = f'<svg{namespaces} viewBox="0 0 20 30">{metadata}{element}</svg>'
    raw = b"\xef\xbb\xbf" + (declaration + document).encode("utf-8")
    path.write_bytes(raw)
    art = SimpleNamespace(backgrounds=tmp_path, colors=None)
    inner, width, height = module.background_inner(art, path.name)
    composed = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{inner}</svg>')
    assert (width, height) == (20, 30)
    rect = composed.find("svg:rect", NS)
    assert rect is not None and rect.get("fill") == "#ffffff"
    assert path.read_bytes() == raw
    with pytest.raises(AssertionError, match="viewBox"):
        path.write_bytes(b"\xef\xbb\xbf<svg/>")
        module.background_inner(art, path.name)


def test_masters_register_the_same_logo_rect():
    """The committed master SVGs are the brand source of truth: both embed a
    logo PNG at the same rect, light plate for the light master, dark for dark."""
    module = load_module("icon_masters_under_test")
    assets = Path(__file__).resolve().parents[2] / "assets"
    light = module.load_master(assets / "icon-master.svg")
    dark = module.load_master(assets / "icon-master-dark.svg")
    assert light["rect"] == dark["rect"]
    assert light["bg"] != dark["bg"]
    assert light["png"] != dark["png"] and len(light["png"]) > 100 and len(dark["png"]) > 100


@pytest.mark.parametrize("platform", ["", "mac-"])
@pytest.mark.parametrize("appearance,ink", [("light", "black"), ("dark", "white")])
@pytest.mark.parametrize("colors", [None, ("#f5cc32", "#443808"), ("#e34850", "#4a1117")])
def test_logo_sits_at_the_registered_rect_on_the_plain_tile(platform, appearance, ink, colors):
    """The tile keeps the background's own geometry and fill with no stroke
    (the ring is disabled), the commit badge is scaled into the free band
    above the logo rect, and the logo PNG is embedded at the masters'
    registered rect — mac plates re-map the canvas onto Apple's 824 grid."""
    module = load_module("icon_geometry_under_test")
    source = Path(__file__).resolve().parents[2]
    logo = {"black": tiny_png(), "white": tiny_png()}
    art = SimpleNamespace(
        backgrounds=source / "assets/backgrounds", colors=colors, commit="0123456",
        rect=(102.4, 102.4, 819.2, 819.2), logo=logo,
    )
    name = f"squircle-{platform}{appearance}.svg"
    original = ET.parse(art.backgrounds / name).find("svg:rect", NS)
    result = ET.fromstring(module.compose_svg(art, ink, name))
    tile = result.find("svg:rect", NS)
    assert original is not None and tile is not None
    geometry = tuple(float(original.attrib[key]) for key in ("x", "y", "width", "height", "rx"))
    assert tuple(float(tile.attrib[key]) for key in ("x", "y", "width", "height", "rx")) == geometry
    assert tile.get("stroke") is None and tile.get("stroke-width") is None
    expected_fill = (colors or ("#ffffff", module.DARK_HEX))[appearance == "dark"]
    assert tile.get("fill") == expected_fill
    clip = result.find("svg:defs/svg:clipPath/svg:rect", NS)
    assert clip is not None
    assert tuple(float(clip.attrib[key]) for key in ("x", "y", "width", "height", "rx")) == geometry
    group = result.find("svg:g", NS)
    assert group is not None and group.get("clip-path") == "url(#icon-silhouette)"
    # The badge rides inside the same clip, scaled into the band above the logo
    # rect so the two never overlap, and it follows the mac HIG inset.
    badge = group.find("svg:g", NS)
    assert badge is not None, "commit builds carry the badge"
    tx, ty, scale = module.BADGE_TRANSFORM
    expected_transform = f"translate({tx} {ty}) scale({scale})"
    if platform:
        expected_transform = f"translate(100 100) scale({module.MAC_PLATE}) {expected_transform}"
    assert badge.get("transform") == expected_transform
    bx, by, bw, bh = module.BADGE_RECT
    badge_bottom = ty + (by + bh) * scale
    logo_x, logo_y, logo_w, logo_h = art.rect
    assert badge_bottom <= logo_y, "the badge never overlaps the logo"
    # The logo is the embedded master PNG at the registered rect, mac-transformed
    # only on the mac plates.
    children = [child for child in group if child.tag == f"{{{NS['svg']}}}g"]
    assert children[0] is badge and len(children) == 2
    logo_group = children[1]
    expected_logo_transform = f"translate(100 100) scale({module.MAC_PLATE})" if platform else None
    assert logo_group.get("transform") == expected_logo_transform
    image = logo_group.find("svg:image", NS)
    assert image is not None
    assert tuple(float(image.attrib[key]) for key in ("x", "y", "width", "height")) == art.rect
    assert image.get("preserveAspectRatio") == "xMidYMid meet"
    import base64
    href = image.get("{http://www.w3.org/1999/xlink}href")
    assert href is not None and base64.b64decode(href.removeprefix("data:image/png;base64,")) == logo[ink]


def test_stable_tiles_carry_no_badge():
    module = load_module("icon_stable_under_test")
    source = Path(__file__).resolve().parents[2]
    art = SimpleNamespace(
        backgrounds=source / "assets/backgrounds", colors=None, commit="",
        rect=(102.4, 102.4, 819.2, 819.2), logo={"black": tiny_png(), "white": tiny_png()},
    )
    result = ET.fromstring(module.compose_svg(art, "black", "squircle-light.svg"))
    inner = [child for child in result.find("svg:g", NS) if child.tag == f"{{{NS['svg']}}}g"]
    assert len(inner) == 1 and inner[0].find("svg:image", NS) is not None, "only the logo group" 


@pytest.mark.parametrize("failure", [None, "render", "directory", "verify"])
def test_write_status_includes_every_target(tmp_path, monkeypatch, capsys, failure):
    # The renderer is build-only. This test injects failures at its byte boundary.
    monkeypatch.setitem(sys.modules, "resvg_py", ModuleType("resvg_py"))
    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_icons.py"
    spec = importlib.util.spec_from_file_location("icon_write_status_under_test", script)
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
