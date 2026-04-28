import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tinker-atropos"
    / "tinker_atropos"
    / "environments"
    / "min_image_video_prompt_planning_tinker.py"
)


def load_env_module():
    spec = importlib.util.spec_from_file_location("min_image_video_prompt_planning_tinker", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_image_prompt_plan_scores_above_threshold():
    module = load_env_module()
    item = module.PROMPT_PLANNING_ITEMS[0]
    answer = """Template: image_prompt_v1
Modality: image
Prompt: Premium desk setup hero image with walnut desk, brushed metal lamp, notebook, soft morning lighting, balanced foreground and background, editorial composition, refined neutral palette, shallow depth of field, and clean negative space.
Visual Details: lighting, composition, foreground objects, background texture, palette, lens feel, and material contrast are specified for direct generation.
Composition: three-quarter desk angle, product centered, lamp as left frame, notebook in foreground, blurred shelves in background.
Style: premium editorial product photography with restrained contrast and warm highlights.
Constraints: preserve realistic proportions, no fake UI text, no extra hands, no brand logos.
Negative Constraints: avoid clutter, warped furniture, unreadable text, oversaturated colors, and low-resolution artifacts.
Note References: obs-img-hero-001, gpt-image-2, prompt-template, hero; follows lighting, composition, negative constraints.
Readiness Rationale: complete and ready for direct generation review because all required image fields, constraints, and note-grounded details are present.
"""

    score = module.score_prompt_planning_answer(answer, item)

    assert score["total"] >= 0.8
    assert score["field_coverage"] == 1.0
    assert score["note_grounding"] >= 0.8


def test_video_shot_plan_scores_above_threshold():
    module = load_env_module()
    item = module.PROMPT_PLANNING_ITEMS[1]
    answer = """Template: video_shot_plan_v1
Modality: video
Shot Plan: Shot 1 opens on the product silhouette, Shot 2 cuts to a slow macro detail, Shot 3 reveals the full product on a clean surface, Shot 4 ends on a stable hero frame.
Camera Motion: slow push-in, controlled lateral slide, macro rack focus, and final locked-off frame to make camera motion explicit.
Temporal Structure: 0-2s silhouette reveal, 2-5s macro texture, 5-8s full reveal, 8-10s final frame with constraints maintained.
Constraints: preserve product geometry, avoid fake labels, keep lighting consistent, and keep movement smooth.
Negative Constraints: no flicker, no warped product edges, no sudden cuts, no unreadable text, no extra objects.
Note References: obs-video-shot-001, seedance-2.0, shot-plan, motion; follows camera motion, temporal structure, constraints.
Readiness Rationale: complete and ready for direct generation review because the shot sequence, timing, camera motion, constraints, and metadata grounding are explicit.
"""

    score = module.score_prompt_planning_answer(answer, item)

    assert score["total"] >= 0.8
    assert score["field_coverage"] == 1.0
    assert score["note_grounding"] >= 0.8


def test_wrong_modality_and_placeholder_are_penalized():
    module = load_env_module()
    item = module.PROMPT_PLANNING_ITEMS[0]
    answer = """Template: video_shot_plan_v1
Modality: video
Prompt: TODO make it good.
Visual Details: [insert details]
Composition:
Style:
Constraints:
Negative Constraints:
Note References:
Readiness Rationale: call GPT-image-2 and generate image now.
"""

    score = module.score_prompt_planning_answer(answer, item)

    assert score["total"] < 0.45
    assert score["placeholder_penalty"] < 0
    assert score["live_generation_penalty"] < 0
    assert score["wrong_modality_penalty"] < 0


def test_missing_note_grounding_is_penalized():
    module = load_env_module()
    item = module.PROMPT_PLANNING_ITEMS[1]
    grounded_answer = """Template: video_shot_plan_v1
Modality: video
Shot Plan: Four-shot product reveal with explicit motion.
Camera Motion: slow push-in and macro rack focus.
Temporal Structure: 0-2s, 2-5s, 5-8s, 8-10s.
Constraints: preserve geometry and smooth motion.
Negative Constraints: no flicker or warped edges.
Note References: obs-video-shot-001, seedance-2.0, shot-plan, motion.
Readiness Rationale: complete and ready for direct generation review.
"""
    ungrounded_answer = grounded_answer.replace(
        "obs-video-shot-001, seedance-2.0, shot-plan, motion",
        "general inspiration",
    )

    grounded = module.score_prompt_planning_answer(grounded_answer, item)
    ungrounded = module.score_prompt_planning_answer(ungrounded_answer, item)

    assert grounded["note_grounding"] > ungrounded["note_grounding"]
    assert ungrounded["note_grounding"] < 0.5
