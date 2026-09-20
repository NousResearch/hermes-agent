"""Invariants for optional-skills/creative/fal-music/scripts/fal_music.py."""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "fal-music" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fal_music  # noqa: E402

# Keys each endpoint's fal OpenAPI schema declares (captured 2026-09-14). A payload key
# outside this set would be forwarded to an endpoint that never declared it.
DECLARED = {
    "elevenlabs-music": {"prompt", "music_length_ms", "force_instrumental", "output_format", "composition_plan", "seed"},
    "minimax-music-3": {"prompt", "lyrics", "duration", "seed", "num_inference_steps", "guidance_scale"},
    "minimax-music-2.6": {"prompt", "lyrics", "is_instrumental", "audio_setting", "lyrics_optimizer"},
    "lyria3": {"prompt", "negative_prompt", "image_url"},
    "lyria3-pro": {"prompt", "negative_prompt", "image_url"},
    "lyria3.5": {"prompt", "negative_prompt", "image_url"},
    "stable-audio-3": {"prompt", "duration", "seed", "negative_prompt", "output_format", "num_inference_steps",
                       "enable_prompt_expansion", "enable_safety_checker", "guidance_scale", "bitrate", "sync_mode"},
    "ace-step": {"tags", "lyrics", "duration", "seed", "guidance_scale", "minimum_guidance_scale", "scheduler",
                 "number_of_steps", "guidance_type", "lyric_guidance_scale", "guidance_interval", "tag_guidance_scale",
                 "guidance_interval_decay", "granularity_scale"},
    "elevenlabs-sfx": {"text", "duration_seconds", "loop", "prompt_influence", "output_format"},
}


@pytest.mark.parametrize("model", sorted(fal_music.MODELS))
def test_dry_run_payload_only_uses_declared_keys(model, capsys):
    # Every generic knob set at once: undeclared ones must be dropped, never forwarded.
    fal_music.main(["--model", model, "--prompt", "lofi, chill", "--lyrics", "[verse]\nhi", "--duration", "999",
                    "--seed", "7", "--negative-prompt", "harsh", "--instrumental", "--loop", "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert out["endpoint"] == fal_music.MODELS[model]["endpoint"]
    assert set(out["payload"]) <= DECLARED[model], f"{model} leaks undeclared keys"
    rng = fal_music.MODELS[model]["duration"]
    for key in ("duration", "duration_seconds", "music_length_ms"):
        if key in out["payload"]:
            value = out["payload"][key] / (1000 if key == "music_length_ms" else 1)
            assert rng and rng[0] <= value <= rng[1], f"{model} duration {value} outside {rng}"


def test_lyrics_required_models_default_to_instrumental_marker(capsys):
    fal_music.main(["--model", "minimax-music-3", "--prompt", "synthpop", "--dry-run"])
    assert json.loads(capsys.readouterr().out)["payload"]["lyrics"] == "[inst]"
