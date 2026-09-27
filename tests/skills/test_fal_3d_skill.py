"""Invariants for optional-skills/creative/fal-3d/scripts/fal_3d.py."""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "fal-3d" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fal_3d  # noqa: E402

# Input keys each endpoint's fal OpenAPI schema declares (captured 2026-09-26). A payload key
# outside this set would be forwarded to an endpoint that never declared it.
DECLARED = {
    "tripo3d/p2/text-to-3d": {"prompt", "negative_prompt", "texture", "pbr", "texture_quality", "texture_version",
                              "delight", "quad", "face_limit", "model_seed", "texture_seed", "image_seed",
                              "export_uv", "export_orientation", "auto_size"},
    "tripo3d/p2/image-to-3d": {"image_url", "texture", "pbr", "texture_quality", "texture_version", "delight",
                               "quad", "face_limit", "model_seed", "texture_seed", "export_uv",
                               "export_orientation", "auto_size"},
    "meshy/v7.1/text-to-3d": {"prompt", "mode", "texture_prompt", "texture_image_url", "seed", "symmetry_mode",
                              "enable_safety_checker", "pose_mode", "enable_pbr", "ultra_mode", "model_type",
                              "should_remesh", "target_polycount", "enable_prompt_expansion", "is_a_t_pose",
                              "geometry_resolution", "enable_rigging", "rigging_height_meters", "topology",
                              "enable_animation", "animation_action_id"},
    "meshy/v7.1/image-to-3d": {"image_url", "should_texture", "texture_prompt", "texture_image_url", "symmetry_mode",
                               "enable_safety_checker", "pose_mode", "enable_pbr", "ultra_mode", "model_type",
                               "should_remesh", "target_polycount", "is_a_t_pose", "geometry_resolution",
                               "enable_rigging", "rigging_height_meters", "topology", "enable_animation",
                               "animation_action_id"},
    "fal-ai/hunyuan-3d/v3.1/pro/text-to-3d": {"prompt", "generate_type", "enable_pbr", "face_count"},
    "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d": {"input_image_url", "generate_type", "enable_pbr", "face_count",
                                                "back_image_url", "left_image_url", "right_image_url", "top_image_url",
                                                "bottom_image_url", "left_front_image_url", "right_front_image_url"},
    "fal-ai/hunyuan-3d/v3.1/rapid/text-to-3d": {"prompt", "enable_pbr", "enable_geometry"},
    "fal-ai/hunyuan-3d/v3.1/rapid/image-to-3d": {"input_image_url", "enable_pbr", "enable_geometry"},
    "fal-ai/trellis-2": {"image_url", "seed", "decimation_target", "texture_size", "resolution", "remesh",
                         "remesh_band", "remesh_project", "ss_sampling_steps", "ss_guidance_strength",
                         "ss_guidance_rescale", "ss_rescale_t", "ss_guidance_interval_start", "ss_guidance_interval_end",
                         "shape_slat_sampling_steps", "shape_slat_guidance_strength", "shape_slat_guidance_rescale",
                         "shape_slat_rescale_t", "shape_slat_guidance_interval_start", "shape_slat_guidance_interval_end",
                         "tex_slat_sampling_steps", "tex_slat_guidance_strength", "tex_slat_guidance_rescale",
                         "tex_slat_rescale_t", "tex_slat_guidance_interval_start", "tex_slat_guidance_interval_end",
                         "uv_unwrap_angle_threshold_deg", "uv_unwrap_smooth_strength", "uv_unwrap_refine_iterations",
                         "uv_unwrap_global_iterations"},
}

EVERY_KNOB = ["--no-texture", "--pbr", "--quad", "--faces", "999999999", "--seed", "7",
              "--negative-prompt", "blurry", "--texture-quality", "detailed", "--dry-run"]


@pytest.mark.parametrize("model", sorted(fal_3d.MODELS))
@pytest.mark.parametrize("modality", ["text", "image"])
def test_dry_run_payload_only_uses_declared_keys(model, modality, capsys):
    source = ["--prompt", "ceramic owl"] if modality == "text" else ["--image", "https://example.com/owl.png"]
    endpoint = fal_3d.MODELS[model][modality]
    if endpoint is None:
        with pytest.raises(SystemExit):
            fal_3d.main(["--model", model, *source, *EVERY_KNOB])
        return
    fal_3d.main(["--model", model, *source, *EVERY_KNOB])
    out = json.loads(capsys.readouterr().out)
    assert out["endpoint"] == endpoint
    assert set(out["payload"]) <= DECLARED[endpoint], f"{model}/{modality} leaks undeclared keys"
    # --no-texture must land as the endpoint's own texture-off marker, and a clamped face count
    # must stay inside the declared range.
    if model != "trellis-2":  # TRELLIS.2 always textures; the script says so on stderr instead
        assert any(out["payload"].get(k) == v for k, v in fal_3d.TEXTURE_OFF_MARKERS), f"{model} ignores --no-texture"
    rng = fal_3d.MODELS[model]["faces"]
    for key in fal_3d.FACE_KEYS:
        if key in out["payload"]:
            assert rng and rng[0] <= out["payload"][key] <= rng[1], f"{model} faces outside {rng}"


def test_pick_mesh_prefers_glb_across_declared_output_shapes():
    glb = {"url": "https://v3b.fal.media/files/x/model.glb", "content_type": "model/gltf-binary"}
    obj = {"url": "https://v3b.fal.media/files/x/model.obj", "content_type": "model/obj"}
    # meshy / hunyuan pro / trellis: model_glb; tripo: model_mesh; hunyuan rapid: model_urls + model_obj only
    assert fal_3d.pick_mesh({"model_glb": glb, "model_urls": {"fbx": obj}}) == (glb["url"], "glb")
    assert fal_3d.pick_mesh({"model_mesh": glb, "rendered_image": None}) == (glb["url"], "glb")
    assert fal_3d.pick_mesh({"model_urls": {"obj": obj}, "model_obj": obj, "texture": None}) == (obj["url"], "obj")
    with pytest.raises(SystemExit):
        fal_3d.pick_mesh({"thumbnail": glb})
