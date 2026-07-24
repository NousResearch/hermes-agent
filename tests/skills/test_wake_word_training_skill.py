"""Tests for the wake-word-training optional skill.

Pure-logic + frontmatter contract only — no network, no heavy deps. The scripts'
synthesis paths (openai/soundfile/scipy) are import-lazy, so importing the
helpers here needs nothing beyond stdlib.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "productivity" / "wake-word-training"


def _load(module_name: str):
    path = SKILL_DIR / "scripts" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_wwt_{module_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def positives():
    return _load("generate_positives")


@pytest.fixture(scope="module")
def cfg_mod():
    return _load("make_training_config")


# ── Frontmatter contract ─────────────────────────────────────────────────


def _frontmatter() -> dict:
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md must open with a YAML frontmatter block"
    return yaml.safe_load(m.group(1))


def test_description_is_short_one_sentence():
    desc = _frontmatter()["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (max 60)"
    assert desc.endswith("."), "description must end with a period"


def test_platforms_declared():
    # POSIX-ish training tooling — must declare supported platforms.
    assert _frontmatter().get("platforms"), "platforms gating is required"


def test_author_credits_a_human_first():
    author = _frontmatter()["author"]
    assert not author.lower().startswith("hermes"), "credit the human first"


# ── generate_positives pure helpers ──────────────────────────────────────


@pytest.mark.parametrize(
    "phrase,expected",
    [("Hey Morgane!", "hey_morgane"), ("  Hey   Hermes  ", "hey_hermes"), ("!!!", "wake_word")],
)
def test_slugify(positives, phrase, expected):
    assert positives.slugify(phrase) == expected


def test_variations_are_unique_and_nonempty(positives):
    v = positives.build_variations("hey morgane")
    assert v and len(v) == len(set(v))
    assert all(x.lower().startswith("hey morgane") for x in v)


def test_plan_balances_voices_and_is_unique(positives):
    voices = positives.voice_list("openai")
    plan = positives.plan_clips("hey morgane", 25, voices)
    assert len(plan) == 25
    filenames = [name for _, _, name in plan]
    assert len(set(filenames)) == 25, "filenames must be unique"
    # Voices cycle evenly: with 25 clips over 10 voices, counts differ by <= 1.
    from collections import Counter

    counts = Counter(v for v, _, _ in plan)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_plan_empty_for_nonpositive_count(positives):
    assert positives.plan_clips("hey morgane", 0, ["alloy"]) == []


def test_plan_requires_a_voice(positives):
    with pytest.raises(ValueError):
        positives.plan_clips("hey morgane", 5, [])


def test_unknown_engine_rejected(positives):
    with pytest.raises(ValueError):
        positives.voice_list("piper")


# ── make_training_config pure helper ─────────────────────────────────────


def test_config_sets_phrase_and_derived_name(cfg_mod):
    cfg = cfg_mod.build_config("Hey Morgane", "/tmp/out")
    assert cfg["target_phrase"] == ["Hey Morgane"]
    assert cfg["model_name"] == "hey_morgane"
    assert cfg["output_dir"] == "/tmp/out"
    assert isinstance(cfg["n_samples"], int) and cfg["n_samples"] > 0


def test_config_uses_custom_positives_only_when_given(cfg_mod):
    without = cfg_mod.build_config("hey morgane", "/tmp/out")
    assert "custom_positive_samples_dir" not in without

    with_pos = cfg_mod.build_config("hey morgane", "/tmp/out", positives_dir="/tmp/pos")
    assert with_pos["custom_positive_samples_dir"] == "/tmp/pos"


def test_config_is_yaml_serializable(cfg_mod):
    cfg = cfg_mod.build_config("hey morgane", "/tmp/out", positives_dir="/tmp/pos")
    round_tripped = yaml.safe_load(yaml.safe_dump(cfg))
    assert round_tripped == cfg
