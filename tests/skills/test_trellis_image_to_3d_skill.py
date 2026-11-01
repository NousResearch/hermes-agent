"""Offline tests for the trellis-image-to-3d skill.

The Space is never called: a fake ``gradio_client`` module plus a fake client
exercise the call chain, the GLB extraction retry, and the manifest writer.
"""
import importlib.util
import json
import struct
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "optional-skills" / "gaming" / "trellis-image-to-3d" / "scripts"


def _load(module_name: str, filename: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def trellis(monkeypatch):
    fake = types.ModuleType("gradio_client")
    fake.handle_file = lambda path: path
    fake.Client = object
    monkeypatch.setitem(sys.modules, "gradio_client", fake)
    return _load("trellis_gen_under_test", "trellis_gen.py")


@pytest.fixture()
def gates():
    return _load("pipeline_qgates_under_test", "pipeline_qgates.py")


def _glb(path: Path, *, verts: int = 5000, textured: bool = True) -> Path:
    doc = {
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"count": verts}],
    }
    if textured:
        doc["images"] = [{"uri": "tex.png"}]
    payload = json.dumps(doc).encode()
    payload += b" " * ((4 - len(payload) % 4) % 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(payload))
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )
    return path


def _args(**overrides) -> Namespace:
    base = dict(
        seed=1234, preprocess=False, resolution="512", decimation=300000,
        texture_size=2048, stage1=(7.5, 0.7, 12, 5), stage2=(7.5, 0.7, 12, 5),
        stage3=(1.0, 0.0, 12, 3),
    )
    base.update(overrides)
    return Namespace(**base)


class FakeClient:
    """Returns queued results per api_name; a queued exception is raised."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def predict(self, *args, api_name=None):
        self.calls.append((api_name, args))
        queue = self.script.get(api_name) or []
        if not queue:
            raise AssertionError(f"unexpected call to {api_name}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_gen_one_copies_the_glb_and_returns_the_seed(trellis, tmp_path):
    produced = _glb(tmp_path / "server" / "out.glb")
    client = FakeClient({
        "/image_to_3d": ["preview"],
        "/extract_glb": [(str(produced),)],
    })

    seed, error = trellis.gen_one(client, "in.png", tmp_path / "goblin.glb", "goblin", _args())

    assert (seed, error) == (1234, None)
    assert (tmp_path / "goblin.glb").read_bytes() == produced.read_bytes()


def test_extract_glb_retries_and_succeeds_on_the_second_attempt(trellis, tmp_path, monkeypatch):
    monkeypatch.setattr(trellis.time, "sleep", lambda _s: None)
    produced = _glb(tmp_path / "server" / "out.glb")
    client = FakeClient({
        "/image_to_3d": ["preview"],
        "/extract_glb": [RuntimeError("queue timeout"), (str(produced),)],
    })

    seed, error = trellis.gen_one(client, "in.png", tmp_path / "goblin.glb", "goblin", _args())

    assert error is None
    assert seed == 1234
    assert [c[0] for c in client.calls] == ["/image_to_3d", "/extract_glb", "/extract_glb"]


def test_extract_glb_gives_up_after_four_attempts(trellis, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(trellis.time, "sleep", sleeps.append)
    client = FakeClient({
        "/image_to_3d": ["preview"],
        "/extract_glb": [(object(),)] * 4,
    })

    seed, error = trellis.gen_one(client, "in.png", tmp_path / "goblin.glb", "goblin", _args())

    assert seed == 1234
    assert "no .glb in extract result" in error
    assert sleeps == [15, 30, 45, 60]
    assert not (tmp_path / "goblin.glb").exists()


def test_preprocess_is_requested_before_generation(trellis, tmp_path):
    produced = _glb(tmp_path / "server" / "out.glb")
    client = FakeClient({
        "/preprocess_image": [{"path": str(_glb(tmp_path / "masked.png"))}],
        "/image_to_3d": ["preview"],
        "/extract_glb": [(str(produced),)],
    })

    trellis.gen_one(client, "in.png", tmp_path / "goblin.glb", "goblin", _args(preprocess=True))

    assert next(c[0] for c in client.calls) == "/preprocess_image"


def test_manifest_header_is_written_once(trellis, tmp_path):
    row = dict(slug="01_fire", prompt="ant", image_path="i.png", glb_path="g.glb",
               seed=1, resolution="512", status="ok", error="")

    trellis.append_manifest(tmp_path, row)
    trellis.append_manifest(tmp_path, {**row, "slug": "02_ice"})

    lines = (tmp_path / "manifest.csv").read_text().splitlines()
    assert lines[0].startswith("slug,prompt,image_path,glb_path")
    assert len(lines) == 3


def test_engine_preset_sets_decimation_and_texture(gates):
    args = _args(target="godot-mobile", decimation=300000, texture_size=2048)
    gates.apply_engine_preset(args)
    assert (args.decimation, args.texture_size) == (100000, 1024)


def test_glb_gate_accepts_a_textured_mesh(gates, tmp_path):
    ok, message = gates.gate_glb_geometry(_glb(tmp_path / "a.glb", verts=5000))
    assert ok, message


def test_glb_gate_rejects_an_untextured_mesh(gates, tmp_path):
    ok, message = gates.gate_glb_geometry(_glb(tmp_path / "a.glb", verts=5000, textured=False))
    assert not ok
    assert "NO TEXTURES" in message


def test_glb_gate_rejects_a_stub_mesh(gates, tmp_path):
    ok, message = gates.gate_glb_geometry(_glb(tmp_path / "a.glb", verts=12))
    assert not ok
    assert "only 12 verts" in message


def test_alpha_gate_skips_cleanly_without_pillow(gates, monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    ok, message = gates.gate_alpha_mask(Path("whatever.png"))
    assert ok
    assert "alpha gate skipped" in message


def test_alpha_gate_flags_an_image_that_is_not_masked(gates, tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGBA", (10, 10), (255, 0, 0, 255))
    target = tmp_path / "opaque.png"
    image.save(target)

    ok, message = gates.gate_alpha_mask(target)

    assert not ok
    assert "NOT alpha-masked" in message


def test_alpha_gate_flags_an_unreadable_file(gates, tmp_path):
    pytest.importorskip("PIL.Image")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")

    ok, message = gates.gate_alpha_mask(broken)

    assert not ok
    assert "alpha gate error" in message


def test_scripts_never_fall_back_to_a_home_directory_import():
    """Review blocker: cross-skill imports through ~/.hermes break hub installs."""
    for filename in ("trellis_gen.py", "pipeline_qgates.py"):
        source = (SCRIPTS / filename).read_text()
        assert "~/.hermes" not in source
        assert "Path.home()" not in source
