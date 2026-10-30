"""Offline tests for the puppeteer-animation skill.

No Blender, no GPU, no Puppeteer checkout: the driver script is exercised with a
fake install tree and a recording ``run`` function, and the Blender-side helper
is loaded against stubbed ``bpy``/``numpy``/``mathutils``.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "optional-skills" / "gaming" / "puppeteer-animation" / "scripts"


def _load(module_name: str, filename: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def driver():
    return _load("puppeteer_animate_under_test", "puppeteer_animate.py")


def _fake_puppeteer(root: Path) -> Path:
    pup = root / "Puppeteer"
    (pup / ".venv" / "bin").mkdir(parents=True)
    (pup / ".venv" / "bin" / "python").write_text("")
    (pup / "animation").mkdir()
    return pup


def _run_main(driver, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["puppeteer_animate.py", *argv])
    driver.main()


def test_missing_puppeteer_path_fails_with_instructions(driver, tmp_path, monkeypatch):
    monkeypatch.delenv("PUPPETEER_DIR", raising=False)
    (tmp_path / "model.glb").write_bytes(b"glb")
    with pytest.raises(SystemExit, match="PUPPETEER_DIR"):
        _run_main(driver, monkeypatch, [str(tmp_path / "model.glb"), "-o", str(tmp_path / "out")])


def test_missing_venv_fails_before_any_subprocess(driver, tmp_path, monkeypatch):
    empty = tmp_path / "Puppeteer"
    empty.mkdir()
    (tmp_path / "model.glb").write_bytes(b"glb")
    calls = []
    monkeypatch.setattr(driver, "run", calls.append)

    with pytest.raises(SystemExit, match="venv"):
        _run_main(driver, monkeypatch, [
            str(tmp_path / "model.glb"), "-o", str(tmp_path / "out"), "--puppeteer", str(empty),
        ])
    assert calls == []


def test_pipeline_runs_conversion_extraction_precompute_and_optimization(driver, tmp_path, monkeypatch):
    pup = _fake_puppeteer(tmp_path)
    glb = tmp_path / "model.glb"
    glb.write_bytes(b"glb")
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"mp4")
    calls = []
    monkeypatch.setattr(driver, "run", lambda cmd, **kw: calls.append((cmd, kw)))

    _run_main(driver, monkeypatch, [
        str(glb), "--video", str(video), "--puppeteer", str(pup),
        "-o", str(tmp_path / "out"), "--img-size", "512", "--fps", "10",
        "--decimate", "20000", "--iter", "200",
    ])

    flat = [" ".join(str(part) for part in cmd) for cmd, _ in calls]
    assert any("glb_to_rignet.py" in line and "--decimate 20000" in line for line in flat)
    assert any(line.startswith("ffmpeg") and "fps=10,scale=512:512" in line for line in flat)
    assert any("utils/save_flow.py" in line for line in flat)
    assert any("precompute_depth.py" in line for line in flat)
    assert any("optimization.py" in line and "--iter 200" in line for line in flat)
    assert (tmp_path / "out" / "input.mp4").exists()


def test_sequence_name_defaults_to_the_glb_stem(driver, tmp_path, monkeypatch):
    pup = _fake_puppeteer(tmp_path)
    glb = tmp_path / "goblin_rigged.glb"
    glb.write_bytes(b"glb")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "input.mp4").write_bytes(b"mp4")
    calls = []
    monkeypatch.setattr(driver, "run", lambda cmd, **kw: calls.append((cmd, kw)))

    _run_main(driver, monkeypatch, [
        str(glb), "--puppeteer", str(pup), "-o", str(tmp_path / "out"), "--skip-convert",
    ])

    optimization = next(cmd for cmd, _ in calls if any("optimization.py" in str(part) for part in cmd))
    assert "goblin_rigged" in optimization


def test_missing_video_fails_before_precompute(driver, tmp_path, monkeypatch):
    pup = _fake_puppeteer(tmp_path)
    glb = tmp_path / "model.glb"
    glb.write_bytes(b"glb")
    calls = []
    monkeypatch.setattr(driver, "run", lambda cmd, **kw: calls.append((cmd, kw)))

    with pytest.raises(SystemExit, match="input.mp4"):
        _run_main(driver, monkeypatch, [
            str(glb), "--puppeteer", str(pup), "-o", str(tmp_path / "out"), "--skip-convert",
        ])
    assert calls == []


def _load_bake():
    """bake_animation.py runs inside Blender; stub the modules it imports."""
    stubs = {
        "bpy": {},
        "numpy": {},
        "mathutils": {"Matrix": object, "Quaternion": object, "Vector": object},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        for attribute, value in attrs.items():
            setattr(module, attribute, value)
        sys.modules.setdefault(name, module)
    return _load("bake_animation_under_test", "bake_animation.py")


def test_parse_rig_txt_returns_joint_order_and_root(tmp_path):
    bake = _load_bake()
    rig = tmp_path / "rig.txt"
    rig.write_text(
        "joints root\n"
        "joints spine_01\n"
        "joints head\n"
        "root root\n"
        "hier root spine_01\n"
        "skin root 0.0 0.0 0.0\n"
    )

    joints, root = bake.parse_rig_txt(rig)

    assert joints == ["root", "spine_01", "head"]
    assert root == "root"


def test_parse_rig_txt_ignores_blank_lines(tmp_path):
    bake = _load_bake()
    rig = tmp_path / "rig.txt"
    rig.write_text("\n\njoints a\n\nroot a\n\n")

    assert bake.parse_rig_txt(rig) == (["a"], "a")


def test_driver_has_no_hardcoded_home_checkout(driver):
    """A baked-in ~/dev/Puppeteer default breaks for every other user."""
    source = (SCRIPTS / "puppeteer_animate.py").read_text()
    assert 'expanduser("~/' not in source
    assert "/home/" not in source
    assert "PUPPETEER_DIR" in source
