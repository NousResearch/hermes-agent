"""Tests for blog.blog_illustrator — art-directed image set via Codex CLI.

The illustrator now drives every image from a single art brief (art_director):
one style + locked palette/motif + shared direction, with a unique prompt for
the hero and each section. Generation backend is Codex CLI (no FAL/Pollinations).
"""
import hashlib
import json
from pathlib import Path

import pytest

import blog.blog_illustrator as bi
import blog.art_director as ad
import config


@pytest.fixture(autouse=True)
def isolated_rotation_state(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "ROTATION_STATE_PATH", tmp_path / "skill_rotation.json")


@pytest.fixture(autouse=True)
def reviewed_reference_catalog(monkeypatch, tmp_path):
    """Give illustrator tests a real, isolated P11 reference pack."""
    root = tmp_path / "refs"
    root.mkdir()
    rows = []
    core_rows = []
    for reference_id, role in (
        ("layout-fixture", "layout"),
        ("style-fixture", "style"),
        ("composition-fixture", "composition"),
    ):
        content = reference_id.encode("utf-8")
        rel = f"{reference_id}.png"
        (root / rel).write_bytes(content)
        row = {
            "record_schema_version": "2", "reference_id": reference_id,
            "path": rel, "sha256": hashlib.sha256(content).hexdigest(),
            "provenance_class": "sahil_curated",
            "ownership_or_usage_basis": "test fixture",
            "usage_classification": "review-required",
            "allowed_roles": [role], "parent_reference_id": None,
        }
        rows.append(row)
        core_rows.append({
            **row, "core_role": role, "core_tag": "fixture",
            "curation_status": "visually-reviewed-core-candidate-test",
            "blocked_roles": ["generation", "publication"],
            "visual_rationale": "isolated test reference",
        })
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "core-pack.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in core_rows), encoding="utf-8"
    )
    monkeypatch.setattr(config, "IMAGERY_ANCHORS_DIR", str(root))


@pytest.fixture(autouse=True)
def stub_art_brief(monkeypatch):
    """Force a deterministic brief so tests never hit the LLM."""
    def fake_brief(draft, headings, recent_styles=None, recent_concept_fingerprints=None, llm=None):
        return {
            "style": "technical-diorama",
            "palette": "stone grey, brass, warm amber",
            "motif": "a recurring archway",
            "art_direction": "vast, awe-of-scale, one warm focal light.",
            "layout": "architectural cross-section",
            "layout_variants": ["control hall", "vault map"],
            "text_policy": "labels",
            "hero_prompt": f"hero for {draft.get('title','')}",
            "section_prompts": {h: f"section image for {h}" for h in headings},
        }
    monkeypatch.setattr(bi, "build_art_brief", fake_brief)


_DRAFT = {
    "title": "Token-Maxing at the Edge",
    "description": "A counterintuitive claim about edges and inference.",
    "body_md": """# Token-Maxing at the Edge

A counterintuitive claim.

## The mechanism

The numbers tell a story.

## Worked example

Here is the code.

## What I'd try next

The takeaway.
""",
    "stream": "ai",
}


def test_illustrate_returns_hero_and_section_paths(monkeypatch, tmp_path):
    def fake_generate(prompt, out_path, **kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("png", encoding="utf-8")
        return out_path
    monkeypatch.setattr(bi, "_generate_codex_image", fake_generate)
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)

    images = bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=2)
    assert images["hero_path"] is not None
    assert Path(images["hero_path"]).exists()
    assert isinstance(images["section_paths"], dict)
    assert len(images["section_paths"]) <= 2


def test_illustrate_caps_at_max_sections(monkeypatch, tmp_path):
    body = "# T\n\nLede\n\n" + "\n\n".join(f"## Section {i}\n\nText." for i in range(10))
    draft = {**_DRAFT, "body_md": body}
    monkeypatch.setattr(bi, "_generate_codex_image",
                        lambda prompt, out_path, **kw: (Path(out_path).write_text("x", encoding="utf-8"), out_path)[1])
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    images = bi.illustrate(draft, out_dir=tmp_path, max_sections=1)
    assert len(images["section_paths"]) <= 1


def test_illustrate_never_generates_more_than_two_sections_even_if_requested(monkeypatch, tmp_path):
    body = "# T\n\nLede\n\n" + "\n\n".join(f"## Section {i}\n\nText." for i in range(10))
    draft = {**_DRAFT, "body_md": body}
    monkeypatch.setattr(bi, "_generate_codex_image",
                        lambda prompt, out_path, **kw: (Path(out_path).write_text("x", encoding="utf-8"), out_path)[1])
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)

    images = bi.illustrate(draft, out_dir=tmp_path, max_sections=10)

    # One hero plus two sections is the runner's global max_images=3 budget.
    assert len(images["section_paths"]) <= 2


def test_illustrate_hero_only_when_max_sections_zero(monkeypatch, tmp_path):
    writes = []
    def fake_generate(prompt, out_path, **kw):
        writes.append(out_path)
        Path(out_path).write_text("x", encoding="utf-8")
        return out_path
    monkeypatch.setattr(bi, "_generate_codex_image", fake_generate)
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    images = bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=0)
    assert images["hero_path"] is not None
    assert images["section_paths"] == {}
    assert len(writes) == 1


def test_illustrate_section_paths_keyed_by_h2_heading(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "_generate_codex_image",
                        lambda prompt, out_path, **kw: (Path(out_path).write_text("x", encoding="utf-8"), out_path)[1])
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    images = bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=3)
    for key in images["section_paths"]:
        assert "## " not in key
        assert key.strip()


def test_illustrate_handles_failed_hero(monkeypatch, tmp_path):
    def fake_generate(prompt, out_path, **kw):
        if "hero.png" in out_path:
            return None
        Path(out_path).write_text("x", encoding="utf-8")
        return out_path
    monkeypatch.setattr(bi, "_generate_codex_image", fake_generate)
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    images = bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=1)
    assert images["hero_path"] is None


def test_all_images_share_one_style(monkeypatch, tmp_path):
    """Hero and every section use the same style/palette (consistency)."""
    prompts = []
    def fake_generate(prompt, out_path, **kw):
        prompts.append(prompt)
        Path(out_path).write_text("x", encoding="utf-8")
        return out_path
    monkeypatch.setattr(bi, "_generate_codex_image", fake_generate)
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=2)
    assert len(prompts) == 3  # hero + 2 sections
    # The Technical Diorama label and the locked palette appear in every prompt.
    assert all("Technical Diorama" in p for p in prompts)
    assert all("brass" in p for p in prompts)


def test_prompts_are_unique_per_image(monkeypatch, tmp_path):
    """Consistent style, but each image depicts its own concept."""
    prompts = []
    def fake_generate(prompt, out_path, **kw):
        prompts.append(prompt)
        Path(out_path).write_text("x", encoding="utf-8")
        return out_path
    monkeypatch.setattr(bi, "_generate_codex_image", fake_generate)
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=2)
    assert len(set(prompts)) == 3  # all distinct


def test_default_max_sections_from_config(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "_generate_codex_image",
                        lambda prompt, out_path, **kw: (Path(out_path).write_text("x", encoding="utf-8"), out_path)[1])
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    monkeypatch.setattr(config, "BLOG_MAX_SECTION_IMAGES", 2)
    images = bi.illustrate(_DRAFT, out_dir=tmp_path)
    assert len(images["section_paths"]) <= 2


def test_hard_fails_when_art_director_unavailable(monkeypatch, tmp_path):
    """When the LLM brief returns None, illustration hard-stops (no fallback).

    The fallback_brief was removed because it produced generic, article-
    disconnected images with no palette/motif/per-section art direction.
    The illustrator must now refuse to generate instead of shipping bad images.
    """
    monkeypatch.setattr(bi, "build_art_brief",
                        lambda draft, headings, recent_styles=None, recent_concept_fingerprints=None, llm=None: None)
    prompts = []
    def fake_generate(prompt, out_path, **kw):
        prompts.append(prompt)
        Path(out_path).write_text("x", encoding="utf-8")
        return out_path
    monkeypatch.setattr(bi, "_generate_codex_image", fake_generate)
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    images = bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=1)
    assert images["hero_path"] is None
    assert len(prompts) == 0  # no images generated



def test_art_brief_log_includes_seed_and_layout(monkeypatch, tmp_path, capsys):
    def fake_brief(draft, headings, recent_styles=None, recent_concept_fingerprints=None, llm=None):
        return {
            "style": "baoyu-infographic",
            "selection_seed": "abc123seed000000",
            "layout": "bento-grid comparison matrix",
            "palette": "cream, teal, black",
            "motif": "numbered cards",
            "art_direction": "dense information design.",
            "hero_prompt": "hero",
            "section_prompts": {},
            "text_policy": "labels",
            "text_elements": ["A"],
        }
    monkeypatch.setattr(bi, "build_art_brief", fake_brief)
    monkeypatch.setattr(bi, "_generate_codex_image",
                        lambda prompt, out_path, **kw: (Path(out_path).write_text("x", encoding="utf-8"), out_path)[1])
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)

    bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=0)
    out = capsys.readouterr().out
    assert "style=baoyu-infographic" in out
    assert "seed=abc123seed000000" in out
    assert "layout='bento-grid comparison matrix'" in out

def test_records_style_to_rotation_state(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "_generate_codex_image",
                        lambda prompt, out_path, **kw: (Path(out_path).write_text("x", encoding="utf-8"), out_path)[1])
    monkeypatch.setattr(bi, "_generate_webp", lambda p: p)
    bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=0)
    assert (tmp_path / "skill_rotation.json").exists()
    assert "technical-diorama" in bi._load_recent_styles()


def test_rotation_state_retains_recent_concept_fingerprint():
    bi._record_selection("technical-diorama", ["clockwork theatre", "origami futures", "deep sea"])

    assert bi._load_recent_concept_fingerprints() == ["clockwork theatre|origami futures|deep sea"]


def test_no_fal_imports():
    import inspect
    src = inspect.getsource(bi)
    import_lines = [l.strip() for l in src.splitlines()
                    if l.strip().startswith(("import ", "from "))]
    joined = "\n".join(import_lines)
    assert "fal_client" not in joined
    assert "draft_media" not in joined
    assert "Pollinations" not in joined


def test_illustrate_persists_provider_free_plan_before_generator(monkeypatch, tmp_path):
    calls = []

    def fake_generate(prompt, out_path, **kwargs):
        plan_path = tmp_path / "visual-plan.json"
        manifest_path = tmp_path / "asset-manifest.json"
        assert plan_path.exists()
        assert manifest_path.exists()
        assert json.loads(plan_path.read_text())["assets"]
        records = json.loads(manifest_path.read_text())["records"]
        assert records and all(record["state"] == "planned" for record in records)
        assert all(record["provider"] is None for record in records)
        calls.append((prompt, out_path))
        Path(out_path).write_text("png", encoding="utf-8")
        return out_path

    monkeypatch.setattr(bi, "_generate_codex_image", fake_generate)
    monkeypatch.setattr(bi, "_generate_webp", lambda path: path)
    monkeypatch.setattr(bi.subprocess, "run", lambda *args, **kwargs: pytest.fail("provider subprocess invoked"))

    result = bi.illustrate(_DRAFT, out_dir=tmp_path, max_sections=1)

    assert calls
    assert result["visual_plan_path"] == str(tmp_path / "visual-plan.json")
    assert result["asset_manifest_path"] == str(tmp_path / "asset-manifest.json")
