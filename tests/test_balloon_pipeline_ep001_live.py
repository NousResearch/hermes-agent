from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_EP001_DIR = REPO_ROOT / "docs/plans/orbi-live-webtoon-20260420/webtoon/ep001"
ANALYZER = LIVE_EP001_DIR / "analyze_balloon_zones.py"
RENDERER = LIVE_EP001_DIR / "render_balloons.py"
UTILS_PATH = LIVE_EP001_DIR / "balloon_layout_utils.py"
SOURCE_MANIFEST = LIVE_EP001_DIR / "generated_fal_manifest_live.json"
LETTERING = LIVE_EP001_DIR / "lettering_script.yaml"
SCROLL_PLAN = LIVE_EP001_DIR / "scroll_plan.yaml"
INPUT_DIR = LIVE_EP001_DIR / "generated_fal_live"
EXPECTED_SPEECH_IDS = {"l01", "l02", "l05"}


def _load_live_utils():
    spec = importlib.util.spec_from_file_location("ep001_live_balloon_layout_utils", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load live EP001 utils from {UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_live_renderer():
    renderer_dir = str(LIVE_EP001_DIR)
    if renderer_dir not in sys.path:
        sys.path.insert(0, renderer_dir)
    spec = importlib.util.spec_from_file_location("ep001_live_render_balloons", RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load live EP001 renderer from {RENDERER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LIVE_UTILS = _load_live_utils()
LIVE_RENDERER = _load_live_renderer()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp_dir = tmp_path_factory.mktemp("balloon-pipeline-live")
    analysis_path = tmp_dir / "balloon_analysis_ep001.yaml"
    output_dir = tmp_dir / "generated_fal_live_ballooned"
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--manifest",
            str(SOURCE_MANIFEST),
            "--lettering",
            str(LETTERING),
            "--scroll-plan",
            str(SCROLL_PLAN),
            "--output",
            str(analysis_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--input-dir",
            str(INPUT_DIR),
            "--analysis",
            str(analysis_path),
            "--lettering",
            str(LETTERING),
            "--scroll-plan",
            str(SCROLL_PLAN),
            "--output-dir",
            str(output_dir),
            "--compose-longscroll",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return {
        "analysis": analysis_path,
        "output_dir": output_dir,
        "placement_manifest": output_dir / "placement_manifest.json",
        "longscroll": output_dir / "ep001_ballooned_longscroll.png",
    }


@pytest.fixture(scope="module")
def analysis(generated_artifacts: dict[str, Path]) -> dict:
    return _load_yaml(generated_artifacts["analysis"])


@pytest.fixture(scope="module")
def placement_manifest(generated_artifacts: dict[str, Path]) -> dict:
    return _load_json(generated_artifacts["placement_manifest"])


@pytest.fixture(scope="module")
def placements_by_id(placement_manifest: dict) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for panel in placement_manifest["panels"]:
        for placement in panel["placements"]:
            mapping[placement["id"]] = placement
    return mapping


def test_live_modules_are_loaded_from_filesystem_paths() -> None:
    assert UTILS_PATH.exists()
    assert RENDERER.exists()
    assert Path(LIVE_UTILS.__file__).resolve() == UTILS_PATH.resolve()
    assert Path(LIVE_RENDERER.__file__).resolve() == RENDERER.resolve()


def test_live_generated_artifacts_resolve_to_live_lane(generated_artifacts: dict[str, Path], analysis: dict, placement_manifest: dict) -> None:
    assert analysis["episode"] == "ep001"
    assert analysis["source"]["manifest"].endswith("generated_fal_manifest_live.json")
    assert "generated_fal_live" in analysis["source"]["default_input_dir"]
    assert generated_artifacts["placement_manifest"].exists()
    assert generated_artifacts["longscroll"].exists()
    assert placement_manifest["analysis_source_manifest"].endswith("generated_fal_manifest_live.json")
    assert "generated_fal_live" in placement_manifest["input_dir"]
    assert placement_manifest["longscroll"].endswith("generated_fal_live_ballooned/ep001_ballooned_longscroll.png")


def test_live_analysis_contract_is_tail_less(analysis: dict) -> None:
    panels_by_id = {panel["panel_id"]: panel for panel in analysis["panels"]}
    assert {"p02", "p06", "p08"} <= panels_by_id.keys()
    for panel_id, panel in panels_by_id.items():
        render_hints = panel["render_hints"]
        assert "allow_tail" not in render_hints
        assert "tail_policy" not in render_hints
        assert "tail_overrides" not in render_hints
        for anchor in panel.get("speaker_anchors", []):
            assert "preferred_tail_side" not in anchor


def test_live_manifest_uses_tail_less_attachment_metadata(placements_by_id: dict[str, dict]) -> None:
    assert EXPECTED_SPEECH_IDS <= placements_by_id.keys()
    for placement_id in EXPECTED_SPEECH_IDS:
        placement = placements_by_id[placement_id]
        assert placement["template"] == "speech"
        assert placement["lines"]
        assert placement["box"]
        assert placement["inner_box"]
        assert "tail_points" not in placement
        assert "tail_entry_edge" not in placement
        assert "tail_cross_ratio" not in placement
        assert "tail_cross_penalty" not in placement["score_breakdown"]


def test_live_speech_attachment_uses_anchor_based_local_zones(analysis: dict, placements_by_id: dict[str, dict]) -> None:
    panels_by_id = {panel["panel_id"]: panel for panel in analysis["panels"]}
    expected = {
        "l01": ("p02", "p02_mother_upper_right", "speaker_local", "mother_mouth_primary"),
        "l02": ("p06", "p06_mother_upper_right", "speaker_local", "mother_mouth_primary"),
        "l05": ("p08", "p08_mother_upper_right", "speaker_local", "mother_mouth_primary"),
    }
    for placement_id, (panel_id, zone_id, candidate_class, anchor_id) in expected.items():
        panel = panels_by_id[panel_id]
        local_zone_ids = {zone["id"] for zone in panel.get("speaker_local_zones", []) if zone.get("item_id") == placement_id}
        assert zone_id in local_zone_ids
        placement = placements_by_id[placement_id]
        assert placement["zone_id"] == zone_id
        assert placement["candidate_class"] == candidate_class
        assert placement["resolved_anchor_id"] == anchor_id
        assert placement["fallback_reason"] is None
        assert not any(reason.endswith("missing_speaker_local_zone") for reason in placement["review_reasons"])


def test_live_speech_outputs_exist_and_remain_rendered(placement_manifest: dict) -> None:
    for panel in placement_manifest["panels"]:
        speech_placements = [placement for placement in panel["placements"] if placement["id"] in EXPECTED_SPEECH_IDS]
        if not speech_placements:
            continue
        rendered_image = Image.open(panel["output"]).convert("RGBA")
        for placement in speech_placements:
            x1, y1, x2, y2 = placement["box"]
            crop = rendered_image.crop((x1, y1, x2, y2))
            assert crop.getbbox() is not None, f"Speech balloon region for {placement['id']} should contain rendered pixels"
