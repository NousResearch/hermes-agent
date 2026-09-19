"""Build native icons with PM's locked renderer. Measure pixels, not SVG text."""
import colorsys
import io
import itertools
import os
import json
import shutil
import tomllib
import zipfile
from pathlib import Path
import struct
import subprocess
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generate(tmp_path_factory):
    root = tmp_path_factory.mktemp("icon-flavors")
    source = root / "source with spaces"
    source.mkdir()
    foreign = root / "foreign-site"
    foreign.mkdir()
    (foreign / "sitecustomize.py").write_text("raise SystemExit('foreign interpreter path leaked')\n", encoding="utf-8")
    shutil.copytree(ROOT / "assets", source / "assets")
    # A real app-only wheel makes --only-group load-bearing: startup dies if
    # the driver accidentally includes application dependencies in the renderer.
    from tests.pm._fixtures import _wheel
    wheel = _wheel(source, "application_only")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("application_only.pth", "import sys; sys.exit('application dependency leaked into icon renderer')\n")
    group = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8-sig"))["dependency-groups"]["icon-build"]
    (source / "pyproject.toml").write_text(
        '[project]\nname="icon-fixture"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["application-only==1.0"]\n[dependency-groups]\nicon-build=' + json.dumps(group) + '\n'
        '[tool.uv]\npackage=false\n[tool.uv.sources]\napplication-only={path=' + json.dumps(wheel.as_posix()) + '}\n', encoding="utf-8")
    uv, node = shutil.which("uv"), shutil.which("node")
    assert uv and node, "icon acceptance requires prepared uv and Node"
    subprocess.run([uv, "lock", "--python", sys.executable], cwd=source, check=True, capture_output=True, timeout=60)
    outputs = {}
    sequence = itertools.count()

    def build(tag="", commit="", *, rejected=False, on_demand=False):
        key = (tag, commit, on_demand)
        if key not in outputs:
            out = root / str(next(sequence))
            env = {**os.environ, "HERMES_HOME": str(root / "home"),
                   "HERMES_RUNTIME_DIR": str(root / "tools"),
                   "HERMES_PAYLOAD_TAG": tag, "HERMES_BUILD_COMMIT": commit,
                   "HERMES_PYTHON": sys.executable, "PYTHONPATH": str(root / "foreign-site"),
                   "PYTHONHOME": str(root / "foreign-python"), "HERMES_DISABLE_LAZY_INSTALLS": "1"}
            command = [node, str(ROOT / "scripts/generate-icons.mjs"),
                       "--source", str(source), "--out", str(out), *(["--on-demand"] if on_demand else [])]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
            if rejected:
                assert result.returncode != 0, "invalid build identity generated icons"
                assert not out.exists()
                if on_demand:
                    assert "disabled" in (result.stdout + result.stderr).lower()
                return
            assert result.returncode == 0, result.stdout + result.stderr
            if not outputs:
                checked = subprocess.run([*command, "--check"], env=env, capture_output=True, text=True, timeout=180)
                assert checked.returncode == 0, checked.stdout + checked.stderr
            outputs[key] = out
        return outputs[key]

    return build


def test_on_demand_build_obeys_disabled_lazy_install_admission(generate):
    generate(on_demand=True, rejected=True)


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
    # Sample below the artwork, inside the tile (also works for mac margins
    # and the centered wide Appx tile). Ignore sub-visible resampling alpha.
    x0, y0, x1, y1 = image.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
    return image.getpixel(((x0 + x1) // 2, y1 - 1))[:3]


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
    for tag in ("v1.2.3-canary.20260911", "v1.2.3-canary.20260911010203", "v2026.9.15-canary.20260916120000",
                "v1.2.3", "v1.2.3-canary.2026091101", "v1.2.3-canary.20260911010203123", "01.2.3-canary.20260911", ""):
        assert bool(module._CANARY_TAG_RE.match(tag)) == is_canary_tag(tag), tag


def test_canary_changes_only_desktop_background_preserving_art_and_native_geometry(generate):
    stable = generate("v1.2.3")
    canary = generate("v1.2.3-canary.20260911010203")
    legacy = generate("v1.2.3-canary.20260911")
    assert (canary / "apps/desktop/assets/icon.png").read_bytes() == (legacy / "apps/desktop/assets/icon.png").read_bytes()
    for path in (stable / "apps/desktop").rglob("*"):
        if not path.is_file():
            continue
        original_frames = list(frames(path))
        canary_frames = list(frames(canary / path.relative_to(stable)))
        assert len(original_frames) == len(canary_frames)
        for original, yellow in zip(original_frames, canary_frames, strict=True):
            assert_same_geometry(original, yellow)
            hue, saturation, value = colorsys.rgb_to_hsv(*(v / 255 for v in tile_color(yellow)))
            assert 0.10 < hue < 0.18 and saturation > 0.65, (path, yellow.size, tile_color(yellow))
            assert (value < 0.4) if "dark" in path.name else (value > 0.8)
            # Compare art in direct renders. Tiny container frames use LANCZOS,
            # whose ringing legitimately depends on adjacent background colors.
            if path.suffix == ".png" and original.width >= 256:
                ink = (255, 255, 255, 255) if "dark" in path.name else (0, 0, 0, 255)
                assert [p == ink for p in original.get_flattened_data()] == [p == ink for p in yellow.get_flattened_data()]
    assert_unbranded_outputs(stable, canary)


def test_commit_icons_are_red_and_print_only_the_actual_seven_digit_prefix(generate):
    stable = generate("v1.2.3")
    first = generate(commit="0123456" + "a" * 33)
    changed = generate(commit="abcdef9" + "a" * 33)
    same_prefix = generate(commit="0123456" + "b" * 33)
    for path in (first / "apps/desktop").rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(first)
        assert path.read_bytes() == (same_prefix / rel).read_bytes(), rel
        first_frames = list(frames(path))
        other_frames = list(frames(changed / rel))
        stable_frames = list(frames(stable / rel))
        for original, red, other in zip(stable_frames, first_frames, other_frames, strict=True):
            assert_same_geometry(original, red)
            hue, saturation, value = colorsys.rgb_to_hsv(*(v / 255 for v in tile_color(red)))
            assert (hue < 0.05 or hue > 0.95) and saturation > 0.6, (rel, tile_color(red))
            assert (value < 0.4) if "dark" in path.name else (value > 0.8)
            # No SHA change may move the tile/art or alter the region below its top quarter.
            bbox = red.getchannel("A").point(lambda a: 255 if a >= 128 else 0).getbbox()
            diff = ImageChops.difference(red.convert("RGB"), other.convert("RGB")).convert("L")
            opaque = red.getchannel("A").point(lambda a: 255 if a >= 128 else 0)
            changed_box = ImageChops.multiply(diff, opaque).getbbox()
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
    for name in ("icon.png", "icon-dark.png"):
        image = Image.open(first / "apps/desktop/assets" / name).convert("RGB")
        for digit, rows in enumerate(expected):
            for y, row in enumerate(rows):
                for x in range(5):
                    pixel = image.getpixel((184 + (digit * 6 + x) * 16 + 8, 48 + y * 16 + 8))
                    assert (min(pixel) > 240) == bool(row & (1 << (4 - x))), (digit, x, y)
    assert_unbranded_outputs(stable, first)


@pytest.mark.parametrize("tag,commit", [
    ("", "abcdef0"), ("", "a" * 41), ("", "A" * 40),
    ("", "a" * 39 + "g"), ("", "a" * 40 + "\n"),
    ("v1.2.3", "a" * 40),
])
def test_invalid_or_conflicting_build_identity_cannot_emit_icons(generate, tag, commit):
    generate(tag=tag, commit=commit, rejected=True)
