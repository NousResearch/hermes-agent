"""Offline tests for the skintokens-rigging skill.

No Space is called: a fake ``gradio_client`` and a fake client exercise the
rig chain, the voxel-postprocess auto-escalation, and the local rig gate.
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
SCRIPTS = REPO / "optional-skills" / "gaming" / "skintokens-rigging" / "scripts"


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
def rig(monkeypatch):
    fake = types.ModuleType("gradio_client")
    fake.handle_file = lambda path: path
    fake.Client = object
    monkeypatch.setitem(sys.modules, "gradio_client", fake)
    return _load("skintokens_rig_under_test", "skintokens_rig.py")


@pytest.fixture()
def rig_gates():
    return _load("rig_gates_under_test", "rig_gates.py")


def _glb(path: Path, *, joints: int = 22, weights: bool = True) -> Path:
    attributes = {"POSITION": 0}
    if weights:
        attributes.update({"JOINTS_0": 1, "WEIGHTS_0": 2})
    doc = {
        "meshes": [{"primitives": [{"attributes": attributes}]}],
        "accessors": [{"count": 5000}],
        "skins": [{"joints": list(range(joints))}] if joints else [],
    }
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
        top_k=5, top_p=0.95, temperature=1.0, rep_penalty=2.0, beams=10,
        use_existing_skeleton=False, preserve_texture_scale=True, voxel_postprocess=False,
    )
    base.update(overrides)
    return Namespace(**base)


class FakeClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def predict(self, *args, api_name=None):
        self.calls.append((api_name, args))
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_rig_one_downloads_the_listed_glb(rig, tmp_path):
    produced = _glb(tmp_path / "server" / "rigged.glb")
    client = FakeClient([("ok", [{"path": str(produced)}])])

    error = rig.rig_one(client, "model.glb", tmp_path / "model_rigged.glb", _args())

    assert error is None
    assert (tmp_path / "model_rigged.glb").read_bytes() == produced.read_bytes()
    assert client.calls[0][0] == "/run_gradio"


def test_no_output_files_escalates_to_voxel_postprocess(rig, tmp_path, monkeypatch):
    monkeypatch.setattr(rig.time, "sleep", lambda _s: None)
    produced = _glb(tmp_path / "server" / "rigged.glb")
    client = FakeClient([
        RuntimeError("No output files were produced"),
        ("ok", [{"path": str(produced)}]),
    ])

    error = rig.rig_one(client, "model.glb", tmp_path / "model_rigged.glb", _args())

    assert error is None
    first, second = client.calls
    assert first[1][8] is False
    assert second[1][8] is True


def test_other_failures_do_not_flip_voxel_postprocess(rig, tmp_path, monkeypatch):
    monkeypatch.setattr(rig.time, "sleep", lambda _s: None)
    client = FakeClient([RuntimeError("queue timeout")] * 3)

    error = rig.rig_one(client, "model.glb", tmp_path / "model_rigged.glb", _args())

    assert "queue timeout" in error
    assert all(call[1][8] is False for call in client.calls)
    assert len(client.calls) == 3


def test_missing_glb_in_the_result_is_an_error(rig, tmp_path, monkeypatch):
    monkeypatch.setattr(rig.time, "sleep", lambda _s: None)
    client = FakeClient([("failed", "not-a-glb")] * 3)

    error = rig.rig_one(client, "model.glb", tmp_path / "out.glb", _args())

    assert "no rigged .glb" in error


def test_rig_gate_accepts_a_skinned_glb(rig_gates, tmp_path):
    ok, message = rig_gates.gate_glb_rig(_glb(tmp_path / "a.glb"))
    assert ok, message
    assert "22 joints" in message


def test_rig_gate_rejects_a_static_glb(rig_gates, tmp_path):
    ok, message = rig_gates.gate_glb_rig(_glb(tmp_path / "a.glb", joints=0))
    assert not ok
    assert "no skins" in message


def test_rig_gate_rejects_a_skin_without_weights(rig_gates, tmp_path):
    ok, message = rig_gates.gate_glb_rig(_glb(tmp_path / "a.glb", weights=False))
    assert not ok
    assert "JOINTS_0/WEIGHTS_0" in message


def test_script_resolves_the_gate_locally_not_through_a_home_directory():
    """Review blocker: the old fallback imported a sibling skill via ~/.hermes."""
    source = (SCRIPTS / "skintokens_rig.py").read_text()
    assert "from rig_gates import gate_glb_rig" in source
    assert "~/.hermes" not in source
    assert "Path.home()" not in source
    assert (SCRIPTS / "rig_gates.py").exists()
