"""Build native icons on the runtime interpreter. Measure pixels, not SVG text."""
import colorsys
import io
import itertools
import math
import os
import shutil
from pathlib import Path
import struct
import subprocess
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter
import pytest


ROOT = Path(__file__).resolve().parents[2]
LAYERED_ICON = "icon.icon"  # apps/desktop/assets/icon.icon: the macOS 26 Icon Composer package
DMG_VOLUME = Path("apps/desktop/packaging/dmg-volume.icns")  # hand-made drive artwork, not a tile


@pytest.fixture(scope="module")
def generate(tmp_path_factory):
    root = tmp_path_factory.mktemp("icon-flavors")
    source = root / "source with spaces"
    source.mkdir()
    foreign = root / "foreign-site"
    foreign.mkdir()
    (foreign / "sitecustomize.py").write_text("raise SystemExit('foreign interpreter path leaked')\n", encoding="utf-8")
    shutil.copytree(ROOT / "assets", source / "assets")
    from scripts.build.icon_environment import prepare_icon_environment
    python = prepare_icon_environment(ROOT, root / "runtime", root / "cache")
    node = shutil.which("node")
    assert node, "icon acceptance requires prepared Node"
    outputs = {}
    sequence = itertools.count()

    def build(tag="", commit="", *, rejected=False):
        key = (tag, commit)
        if key not in outputs:
            out = root / str(next(sequence))
            # The runtime interpreter renders with its own packages: foreign
            # interpreter paths must not leak in, and nothing may be installed.
            env = {**os.environ, "HERMES_HOME": str(root / "home"),
                   "HERMES_RUNTIME_DIR": str(root / "tools"),
                   "HERMES_PAYLOAD_TAG": tag, "HERMES_BUILD_COMMIT": commit,
                   "HERMES_PYTHON": str(python), "PYTHONPATH": str(root / "foreign-site"),
                   "PYTHONHOME": str(root / "foreign-python"), "HERMES_DISABLE_LAZY_INSTALLS": "1"}
            command = [node, str(ROOT / "scripts/generate-icons.mjs"),
                       "--source", str(source), "--out", str(out)]
            result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
            if rejected:
                assert result.returncode != 0, "invalid build identity generated icons"
                assert not out.exists()
                return
            assert result.returncode == 0, result.stdout + result.stderr
            if not outputs:
                checked = subprocess.run([*command, "--check"], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
                assert checked.returncode == 0, checked.stdout + checked.stderr
            outputs[key] = out
        return outputs[key]

    return build


def frames(path):
    """Read every native frame, including ICO entries Pillow's n_frames misses."""
    data = path.read_bytes()
    if path.suffix == ".ico":
        count = struct.unpack_from("<H", data, 4)[0]
        assert {data[6 + i * 16] or 256 for i in range(count)} == {16, 24, 32, 48, 64, 128, 256}
        for index in range(count):
            width, height, _, _, _, _, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + index * 16)
            image = Image.open(io.BytesIO(data[offset:offset + length])).convert("RGBA")
            assert image.size == (width or 256, height or 256)
            yield image
    elif path.suffix == ".icns":
        assert data[:4] == b"icns"
        assert struct.unpack_from(">I", data, 4)[0] == len(data)
        image = Image.open(path)
        assert {w * scale for w, h, scale in image.info["sizes"]} == {32, 64, 128, 256, 512, 1024}
        for size in image.info["sizes"]:
            frame = image.icns.getimage(size).convert("RGBA")
            alpha = frame.getchannel("A").point(lambda a: 255 if a >= 128 else 0)
            expected = [frame.width * x / 1024 for x in (100, 100, 924, 924)]
            assert all(abs(a - b) <= 1 for a, b in zip(alpha.getbbox(), expected, strict=True))
            if frame.width == 1024:
                template = Image.new("L", frame.size)
                ImageDraw.Draw(template).rounded_rectangle((100, 100, 923, 923), radius=185.4, fill=255)
                # A three-pixel AA band around Apple's rounded-square template.
                assert ImageChops.subtract(alpha, template.filter(ImageFilter.MaxFilter(7))).getbbox() is None
            yield frame
    else:
        yield Image.open(path).convert("RGBA")


def tile_color(image):
    # The flavor background is the tile fill. Tiles carry an inward contrasting
    # border (black or white) and the artwork sits above the centre, both of
    # which LANCZOS smears across tiny frames — so read the most chromatic pixel
    # of the tile's lower half instead of one fixed coordinate: on a flavored
    # tile that is the fill colour, on a stable tile every candidate is grey.
    x0, y0, x1, y1 = image.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
    pixels = [rgba[:3] for rgba in image.crop((x0, (y0 + y1) // 2, x1, y1)).getdata() if rgba[3] >= 128]
    # Saturation weighted by chroma: a near-black anti-aliased edge pixel has high HSV
    # saturation but almost no colour, the fill has both.
    return max(pixels, key=lambda rgb: max(rgb) - min(rgb))


def is_dark_tile(path):
    # `-dark` outputs and the MSIX dark-theme form (`_altform-unplated`; the
    # light theme's is `_altform-lightunplated`) carry the dark tile.
    return "dark" in path.name or path.name.endswith("_altform-unplated.png")


def assert_same_geometry(original, flavored):
    assert original.size == flavored.size
    assert original.getchannel("A").tobytes() == flavored.getchannel("A").tobytes()
    # LANCZOS container downscales can leave sub-visible alpha at corners.
    assert flavored.getpixel((0, 0))[3] <= 3
    assert flavored.getpixel((flavored.width - 1, flavored.height - 1))[3] <= 3


def assert_unbranded_outputs(stable, flavored):
    for directory in ("assets", "website", "web", "apps/bootstrap-installer"):
        for path in (stable / directory).rglob("*"):
            if path.is_file():
                assert path.read_bytes() == (flavored / path.relative_to(stable)).read_bytes(), path


def test_renderer_canary_rule_is_the_canonical_one(monkeypatch):
    """The renderer runs without the application package installed, so it carries
    its own copy of the canary rule; both rules must agree on every tag shape."""
    import importlib.util
    import types
    from hermes_cli.update_channel import is_canary_tag

    monkeypatch.setitem(sys.modules, "resvg_py", types.ModuleType("resvg_py"))
    spec = importlib.util.spec_from_file_location("generate_icons", ROOT / "scripts/generate_icons.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for tag in ("v1.2.3+canary.20260911T010203Z", "v2026.9.15+canary.20260916T120000Z",
                "v1.2.3", "v1.2.3-canary.20260911010203", "v1.2.3+canary.20260911010203", ""):
        assert bool(module._CANARY_TAG_RE.match(tag)) == is_canary_tag(tag), tag


def test_canary_changes_only_desktop_background_preserving_art_and_native_geometry(generate):
    stable = generate("v1.2.3")
    canary = generate("v1.2.3+canary.20260911T010203Z")
    for path in (stable / "apps/desktop").rglob("*"):
        if not path.is_file() or LAYERED_ICON in path.parts or path.relative_to(stable) == DMG_VOLUME:
            continue  # the layered macOS icon carries its flavor in icon.json; the DMG volume art is unflavoured (both tested below)
        original_frames = list(frames(path))
        canary_frames = list(frames(canary / path.relative_to(stable)))
        assert len(original_frames) == len(canary_frames)
        for original, yellow in zip(original_frames, canary_frames, strict=True):
            assert_same_geometry(original, yellow)
            hue, saturation, value = colorsys.rgb_to_hsv(*(v / 255 for v in tile_color(yellow)))
            assert 0.10 < hue < 0.18 and saturation > 0.65, (path, yellow.size, tile_color(yellow))
            assert (value < 0.4) if is_dark_tile(path) else (value > 0.8), (path, yellow.size, tile_color(yellow))
            # Compare art in direct renders. Tiny container frames use LANCZOS,
            # whose ringing legitimately depends on adjacent background colors.
            if path.suffix == ".png" and original.width >= 256:
                ink = (255, 255, 255, 255) if is_dark_tile(path) else (0, 0, 0, 255)
                assert [p == ink for p in original.get_flattened_data()] == [p == ink for p in yellow.get_flattened_data()]
    assert_unbranded_outputs(stable, canary)


def test_commit_icons_are_red_and_print_only_the_actual_seven_digit_prefix(generate, monkeypatch):
    module = load_generator(monkeypatch)
    (gx, gy), badge_min_size = module.BADGE_GLYPH_ORIGIN, module.BADGE_MIN_SIZE
    stable = generate("v1.2.3")
    first = generate(commit="0123456" + "a" * 33)
    changed = generate(commit="abcdef9" + "a" * 33)
    same_prefix = generate(commit="0123456" + "b" * 33)
    for path in (first / "apps/desktop").rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(first)
        assert path.read_bytes() == (same_prefix / rel).read_bytes(), rel
        if LAYERED_ICON in path.parts or rel == DMG_VOLUME:
            continue  # the layered macOS icon carries its flavor in icon.json; the DMG volume art is unflavoured (both tested below)
        first_frames = list(frames(path))
        other_frames = list(frames(changed / rel))
        stable_frames = list(frames(stable / rel))
        for original, red, other in zip(stable_frames, first_frames, other_frames, strict=True):
            assert_same_geometry(original, red)
            hue, saturation, value = colorsys.rgb_to_hsv(*(v / 255 for v in tile_color(red)))
            assert (hue < 0.05 or hue > 0.95) and saturation > 0.6, (rel, tile_color(red))
            assert (value < 0.4) if is_dark_tile(path) else (value > 0.8), (rel, red.size, tile_color(red))
            # No SHA change may move the tile/art or alter the region below its top quarter.
            bbox = red.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
            diff = ImageChops.difference(red.convert("RGB"), other.convert("RGB")).convert("L")
            opaque = red.getchannel("A").point(lambda a: 255 if a >= 128 else 0)
            changed_box = ImageChops.multiply(diff, opaque).getbbox()
            if path.suffix == ".png" and red.width < badge_min_size:
                # Direct renders this small carry no badge at all, so the SHA is invisible.
                assert changed_box is None, (rel, red.size)
                continue
            assert changed_box is not None, (rel, red.size)
            assert changed_box[1] >= bbox[1]
            assert changed_box[3] <= bbox[1] + (bbox[3] - bbox[1]) * 0.25 + 3
    # At full resolution, read back each glyph's bitmap from pixels. These
    # digit forms spell 0123456, not a generic badge or a hash of the SHA.
    expected = (
        (14, 17, 19, 21, 25, 17, 14), (4, 12, 4, 4, 4, 4, 14),
        (14, 17, 1, 2, 4, 8, 31), (30, 1, 1, 14, 1, 1, 30),
        (2, 6, 10, 18, 31, 2, 2), (31, 16, 16, 30, 1, 1, 30),
        (14, 16, 16, 30, 17, 17, 14),
    )
    # The portrait renders in front of the badge (her hair crosses its lower rows), so a
    # cell is only judged where the stable icon shows no art at that spot; every glyph
    # must still be identified by a majority of its uncovered cells.
    for name in ("icon.png", "icon-dark.png"):
        image = Image.open(first / "apps/desktop/assets" / name).convert("RGB")
        unbadged = Image.open(stable / "apps/desktop/assets" / name).convert("RGB")
        art = (0, 0, 0) if name == "icon.png" else (255, 255, 255)
        for digit, rows in enumerate(expected):
            judged = 0
            for y, row in enumerate(rows):
                for x in range(5):
                    point = (gx + (digit * 6 + x) * 16 + 8, gy + y * 16 + 8)
                    if unbadged.getpixel(point) == art:
                        continue
                    judged += 1
                    pixel = image.getpixel(point)
                    assert (min(pixel) > 240) == bool(row & (1 << (4 - x))), (name, digit, x, y)
            assert judged >= 18, (name, digit, judged)
    assert_unbranded_outputs(stable, first)


def load_generator(monkeypatch):
    import importlib.util
    import types

    monkeypatch.setitem(sys.modules, "resvg_py", types.ModuleType("resvg_py"))
    spec = importlib.util.spec_from_file_location("generate_icons", ROOT / "scripts/generate_icons.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_fills(package):
    """(light, dark) fill colours of an Icon Composer package as 0..1 RGB tuples."""
    import json

    manifest = json.loads((package / "icon.json").read_text(encoding="utf-8"))
    fills = {}
    for spec in manifest["fill-specializations"]:
        prefix, _, channels = spec["value"]["solid"].partition(":")
        assert prefix == "srgb"
        fills[spec.get("appearance", "light")] = tuple(float(v) for v in channels.split(","))[:3]
    layers = {layer["name"]: layer for group in manifest["groups"] for layer in group["layers"]}
    assert "border" not in layers, "the ring is disabled everywhere"
    # A fixed "image-name" makes actool ignore the per-appearance images, so
    # the art layer picks its image through specializations only, with a dark
    # one (else dark mode shows the black girl on the dark fill). Clear and
    # Tinted come from the single mono layer, shown only under "tinted" while
    # the art layer hides there.
    art, mono = layers["art"], layers["mono"]
    assert "image-name" not in art
    art_specs = {spec.get("appearance"): spec["value"] for spec in art["image-name-specializations"]}
    assert set(art_specs) == {None, "dark"}
    assert art["hidden-specializations"] == [{"value": False}, {"appearance": "tinted", "value": True}]
    assert mono["hidden-specializations"] == [{"value": True}, {"appearance": "tinted", "value": False}]
    assert mono["glass"] is True
    referenced = set(art_specs.values()) | {mono["image-name"]}
    assert referenced == {p.name for p in (package / "Assets").iterdir()}, "every layer image is referenced, none dangle"
    return fills["light"], fills["dark"]


def test_layered_macos_icon_mono_layer_and_flavor_stays_in_the_fill(generate, monkeypatch):
    """macOS 26 masks the layers itself. The girl is dragged past the plate edge
    so the mask crops her (no gap below), the mono layer is one image of two
    materials — near-black frosted ink and white — so nothing stacks, and build
    flavors recolour the fill in icon.json only; the layers never change."""
    module = load_generator(monkeypatch)
    stable = generate("v1.2.3") / "apps/desktop/assets" / LAYERED_ICON
    canary = generate("v1.2.3+canary.20260911T010203Z") / "apps/desktop/assets" / LAYERED_ICON
    commit = generate(commit="0123456" + "a" * 33) / "apps/desktop/assets" / LAYERED_ICON

    canvas = module.ICON_CANVAS
    for name in ("art-light.png", "art-dark.png"):
        layer = Image.open(stable / "Assets" / name).convert("RGBA")
        assert layer.size == (canvas, canvas)
        bottom_row = layer.crop((0, canvas - 1, canvas, canvas)).getchannel("A").getbbox()
        assert bottom_row is not None, f"{name}: the girl must reach the plate edge"
    mono = Image.open(stable / "Assets" / "mono.png").convert("RGBA")
    ink_tone = round(module.MONO_INK[0] * 255)
    tones = {px[0] for px in mono.getdata() if px[3] > 127}
    assert tones == {ink_tone, 255}, tones
    light_art = Image.open(stable / "Assets" / "art-light.png").convert("RGBA")
    dark_art = Image.open(stable / "Assets" / "art-dark.png").convert("RGBA")
    for x, y in ((canvas // 2, canvas // 2), (canvas // 3, canvas // 3), (2 * canvas // 3, canvas // 2)):
        px = mono.getpixel((x, y))
        if dark_art.getpixel((x, y))[3] > 200:      # the girl's white parts stay white and opaque
            assert px[:3] == (255, 255, 255) and px[3] == 255, (x, y, px)
        elif light_art.getpixel((x, y))[3] > 200:   # her black parts become the frosted ink
            assert px[0] == ink_tone and abs(px[3] - round(module.MONO_INK[1] * 255)) <= 1, (x, y, px)

    for name in ("art-light.png", "art-dark.png", "mono.png"):
        assert (stable / "Assets" / name).read_bytes() == (canary / "Assets" / name).read_bytes(), name

    # The DMG volume icon is the hand-made drive artwork, shipped as a full ICNS
    # and never flavoured: an installer's disk looks the same for every channel.
    volume = stable.parents[3] / DMG_VOLUME
    for other in (canary, commit):
        assert volume.read_bytes() == (other.parents[3] / DMG_VOLUME).read_bytes()
    icns = Image.open(volume)
    assert {w * scale for w, h, scale in icns.info["sizes"]} == {32, 64, 128, 256, 512, 1024}  # same reps as the app icns
    assert ImageChops.difference(icns.icns.getimage((512, 512, 2)).convert("RGBA"),
                                 Image.open(ROOT / "assets/dmg-volume.png").convert("RGBA")).getbbox() is None
    for name in ("art-light.png", "art-dark.png"):
        # The commit badge is the only difference, and it lives in the top quarter.
        plain = Image.open(stable / "Assets" / name).convert("RGBA")
        badged = Image.open(commit / "Assets" / name).convert("RGBA")
        changed = ImageChops.difference(plain, badged).convert("L").getbbox()
        assert changed is not None and changed[3] <= canvas * 0.25 + 3, (name, changed)

    light, dark = manifest_fills(stable)
    assert light == (1.0, 1.0, 1.0) and max(dark) < 0.1
    for package, low, high in ((canary, 0.10, 0.18), (commit, -0.05, 0.05)):
        light, dark = manifest_fills(package)
        for fill, dark_fill in ((light, False), (dark, True)):
            hue, saturation, value = colorsys.rgb_to_hsv(*fill)
            hue = hue - 1 if hue > 0.5 else hue  # red straddles the hue wrap
            assert low < hue < high and saturation > 0.6, (package, fill)
            assert (value < 0.4) if dark_fill else (value > 0.8), (package, fill)


@pytest.mark.parametrize("tag,commit", [
    ("", "abcdef0"), ("", "a" * 41), ("", "A" * 40),
    ("", "a" * 39 + "g"), ("", "a" * 40 + "\n"),
    ("v1.2.3", "a" * 40),
])
def test_invalid_or_conflicting_build_identity_cannot_emit_icons(generate, tag, commit):
    generate(tag=tag, commit=commit, rejected=True)


def test_mac_mask_offset_keeps_the_ring_geometry_honest(monkeypatch):
    """The ring is disabled, but its geometry stays available: the inward
    offset of Apple's fitted mask must sit exactly one thickness inside the
    outline at every sample, or re-enabling BORDER_ENABLED ships a wobbly band."""
    module = load_generator(monkeypatch)
    canvas = float(module.ICON_CANVAS)
    thickness = canvas * module.BORDER_FRACTION
    outline = module.mac_mask_outline(canvas, 160)
    inner = module.offset_inward(outline, thickness)
    assert len(inner) == len(outline) >= 160
    for (ox, oy), (ix, iy) in zip(outline, inner, strict=True):
        assert math.hypot(ox - ix, oy - iy) == pytest.approx(thickness, abs=1e-6)
        assert 0 <= ox <= canvas and 0 <= oy <= canvas


def test_msix_logos_resolve_every_slot_windows_draws(generate, monkeypatch):
    """Windows picks `<Logo>.scale-N` / `.targetsize-N` by qualifier; the manifest
    only names the bases. Every base the manifest references must therefore have
    its scaled siblings at Microsoft's pixel sizes (never an upscaled 44px), and
    the theme forms must be real variants: dark theme (unplated) gets the dark
    tile, light theme the light one — the unqualified file stays light."""
    module = load_generator(monkeypatch)
    appx = generate("v1.2.3") / module.APPX_DIR
    for name, base in module.APPX_LOGOS.items():
        for scale in module.APPX_SCALES:
            qualifier = "" if scale == 100 else f".scale-{scale}"
            image = Image.open(appx / f"{name}{qualifier}.png")
            expected = tuple(module.appx_scaled(side, scale) for side in (base if isinstance(base, tuple) else (base, base)))
            assert image.size == expected, (name, scale, image.size)
    # Microsoft's own table: 150 @ 125% is 188, not 187.
    assert module.appx_scaled(150, 125) == 188 and module.appx_scaled(44, 400) == 176
    assert {48, 256} <= set(module.APPX_TARGET_SIZES)  # taskbar @200% and the largest Start pin
    light_plate = Image.open(appx / "Square44x44Logo.png").convert("RGBA").getpixel((2, 22))[:3]
    for size in module.APPX_TARGET_SIZES:
        plain = Image.open(appx / f"Square44x44Logo.targetsize-{size}.png").convert("RGBA")
        dark = Image.open(appx / f"Square44x44Logo.targetsize-{size}_altform-unplated.png").convert("RGBA")
        light = Image.open(appx / f"Square44x44Logo.targetsize-{size}_altform-lightunplated.png").convert("RGBA")
        assert plain.size == dark.size == light.size == (size, size)
        edge = (max(1, size // 24), size // 2)  # on the plate, left of the girl
        assert light.getpixel(edge)[:3] == plain.getpixel(edge)[:3] == light_plate == (255, 255, 255), size
        dark_plate = dark.getpixel(edge)
        assert dark_plate[3] == 255 and max(dark_plate[:3]) < 60, (size, dark_plate)
