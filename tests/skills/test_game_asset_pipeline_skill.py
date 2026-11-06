"""Offline tests for the game-asset-pipeline skill's stage ledger.

The ledger never touches the network: it plans and runs phase commands, so all
of its behaviour is testable with a temp directory. Nothing here imports
gradio_client or reaches a Hugging Face Space.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "optional-skills" / "gaming" / "game-asset-pipeline" / "scripts"


def _load_stage():
    name = "asset_stage_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "asset_stage.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def stage(tmp_path):
    mod = _load_stage()
    mod.main(["init", str(tmp_path), "--name", "ants", "--target", "godot"])
    return mod


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _through_rig(stage, root: Path, slug: str = "01_fire") -> None:
    stage.main(["add", str(root), slug, "fire ant, six legs"])
    stage.main(["import", str(root), slug, "--stage", "image",
                "--file", str(_write(root / "images" / f"{slug}.png", b"png"))])
    stage.main(["import", str(root), slug, "--stage", "mask",
                "--file", str(_write(root / "images_masked" / f"{slug}.png", b"mask"))])
    stage.main(["import", str(root), slug, "--stage", "mesh",
                "--file", str(_write(root / "assets" / "glb" / f"{slug}.glb", b"glb"))])
    stage.main(["import", str(root), slug, "--stage", "rig",
                "--file", str(_write(root / "assets" / "rigged" / f"{slug}_rigged.glb", b"rig"))])


def test_add_records_prompt_and_leaves_image_missing(stage, tmp_path):
    stage.main(["add", str(tmp_path), "01_fire", "fire ant, six legs"])
    project = stage.load(tmp_path)
    stages = project["assets"]["01_fire"]["stages"]
    assert stages["prompt"]["status"] == "ok"
    assert stages["image"]["status"] == "missing"


def test_editing_the_prompt_stales_only_downstream_stages(stage, tmp_path):
    _through_rig(stage, tmp_path)

    stage.main(["set-prompt", str(tmp_path), "01_fire", "ice ant, six legs"])
    stages = stage.load(tmp_path)["assets"]["01_fire"]["stages"]

    assert stages["prompt"]["status"] == "ok"
    assert stages["image"]["status"] == "stale"
    assert stages["mesh"]["status"] == "stale"
    assert stage.load(tmp_path)["assets"]["01_fire"]["prompt"] == "ice ant, six legs"


def test_a_mesh_overwritten_on_disk_does_not_stale_the_image(stage, tmp_path):
    _through_rig(stage, tmp_path)
    (tmp_path / "assets" / "glb" / "01_fire.glb").write_bytes(b"glb-blender-edit")

    stage.main(["refresh", str(tmp_path)])
    stages = stage.load(tmp_path)["assets"]["01_fire"]["stages"]

    assert stages["image"]["status"] == "ok"
    assert stages["mesh"]["source"] == "external-edit"
    assert stages["rig"]["status"] == "stale"


def test_plan_after_a_mesh_edit_lists_rig_only(stage, tmp_path):
    _through_rig(stage, tmp_path)
    (tmp_path / "assets" / "glb" / "01_fire.glb").write_bytes(b"edited")
    stage.main(["refresh", str(tmp_path)])

    project = stage.load(tmp_path)
    steps = stage.plan_asset(tmp_path, project, "01_fire", "rig")

    assert [s["stage"] for s in steps] == ["rig"]


def test_plan_emits_no_command_and_an_install_hint_without_siblings(stage, tmp_path, monkeypatch):
    _through_rig(stage, tmp_path)
    (tmp_path / "assets" / "glb" / "01_fire.glb").write_bytes(b"edited")
    stage.main(["refresh", str(tmp_path)])
    monkeypatch.setattr(stage, "RIG", tmp_path / "absent" / "skintokens_rig.py")

    project = stage.load(tmp_path)
    (step,) = stage.plan_asset(tmp_path, project, "01_fire", "rig")

    assert step["cmd"] == []
    assert "hermes skills install official/gaming/skintokens-rigging" in step["hint"]


def test_flux_script_turns_the_image_stage_into_a_command(stage, tmp_path, monkeypatch):
    stage.main(["add", str(tmp_path), "01_fire", "fire ant"])
    flux = _write(tmp_path / "gen.py", b"")
    monkeypatch.setattr(stage, "FLUX", flux)

    project = stage.load(tmp_path)
    (step,) = stage.plan_asset(tmp_path, project, "01_fire", "image")

    assert step["cmd"][1:] == [
        str(flux), "--prompt", "fire ant", "-o", str(tmp_path / "images" / "01_fire.png")
    ]


def test_adopt_matches_prompts_by_words_and_skips_comments(stage, tmp_path):
    colony = tmp_path / "colony"
    _write(colony / "images" / "01_fire_ant.png", b"a")
    _write(colony / "images" / "02_ice_ant.png", b"b")
    prompts = colony / "prompts" / "ants.txt"
    prompts.parent.mkdir(parents=True)
    prompts.write_text(
        "# this comment used to shift every prompt\n"
        "# another comment\n"
        "fire ant warrior, glowing ember carapace\n"
        "ice ant scout, frost carapace\n"
    )

    stage.main(["adopt", str(colony), "--name", "colony"])
    project = stage.load(colony)

    assert project["assets"]["01_fire_ant"]["prompt"] == "fire ant warrior, glowing ember carapace"
    assert project["assets"]["02_ice_ant"]["prompt"] == "ice ant scout, frost carapace"


def test_prompt_lines_drop_comments_and_blanks(stage, tmp_path):
    prompts = tmp_path / "p.txt"
    prompts.write_text("# header\n\nfire ant\n# mid\nice ant\n")
    assert stage.prompt_lines(prompts) == ["fire ant", "ice ant"]


def test_notes_land_in_project_json_without_a_leftover_temp_file(stage, tmp_path):
    stage.main(["add", str(tmp_path), "01_fire", "fire ant"])
    stage.main(["note", str(tmp_path), "01_fire", "waiting on quota"])

    project = json.loads((tmp_path / "project.json").read_text())
    assert project["assets"]["01_fire"]["notes"] == "waiting on quota"
    assert not (tmp_path / "project.json.tmp").exists()
