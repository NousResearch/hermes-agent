"""Tests for blog.blog_illustrator — art-directed image set via Codex CLI.

The illustrator now drives every image from a single art brief (art_director):
one style + locked palette/motif + shared direction, with a unique prompt for
the hero and each section. Generation backend is Codex CLI (no FAL/Pollinations).
"""
import hashlib
import json
import threading
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


def test_codex_cli_uses_isolated_pool_accounts_and_fails_over(monkeypatch, tmp_path):
    primary = tmp_path / "primary-codex"
    primary.mkdir()
    (primary / "auth.json").write_text('{"primary": true}')
    out = tmp_path / "out.png"
    calls = []

    class Entry:
        def __init__(self, ident, access, refresh):
            self.id, self.access_token, self.refresh_token = ident, access, refresh

    entries = [Entry("one", "access-1", "refresh-1"), Entry("two", "access-2", "refresh-2")]

    class Pool:
        def __init__(self):
            self.index = 0
            self.released = []

        def acquire_lease(self, _credential_id=None):
            return entries[self.index].id if self.index < len(entries) else None

        def entries(self):
            return entries

        def release_lease(self, ident):
            self.released.append(ident)

        def mark_exhausted_and_rotate(self, **kwargs):
            self.index += 1
            return entries[self.index] if self.index < len(entries) else None

    pool = Pool()
    monkeypatch.setenv("CODEX_HOME", str(primary))
    monkeypatch.setattr(bi, "_load_codex_pool", lambda: pool)

    def run(_argv, **kwargs):
        auth = Path(kwargs["env"]["CODEX_HOME"]) / "auth.json"
        payload = json.loads(auth.read_text())
        calls.append((Path(kwargs["env"]["CODEX_HOME"]), payload))
        if len(calls) == 1:
            return type("Result", (), {"returncode": 1, "stdout": "usage_limit_reached", "stderr": ""})()
        generated = tmp_path / "generated.png"
        generated.write_bytes(b"image")
        monkeypatch.setattr(bi, "_find_latest_codex_image", lambda **_kw: str(generated))
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(bi.subprocess, "run", run)

    assert bi._generate_codex_image("prompt", str(out), raise_on_cap=True) == str(out)
    assert [payload["tokens"]["access_token"] for _, payload in calls] == ["access-1", "access-2"]
    assert all(home != primary for home, _ in calls)
    assert all(not home.exists() for home, _ in calls)
    assert (primary / "auth.json").read_text() == '{"primary": true}'
    assert pool.released == ["one", "two"]


def test_codex_image_is_found_inside_isolated_codex_home(monkeypatch, tmp_path):
    out = tmp_path / "out.png"

    class Entry:
        id = "one"
        access_token = "access"
        refresh_token = "refresh"

    class Pool:
        def acquire_lease(self, _credential_id=None): return "one"
        def release_lease(self, _ident): pass
        def entries(self): return [Entry()]
        def has_available(self): return True

    monkeypatch.setattr(bi, "_load_codex_pool", Pool)

    def run(_argv, **kwargs):
        images = Path(kwargs["env"]["CODEX_HOME"]) / "generated_images" / "session"
        images.mkdir(parents=True)
        (images / "generated.png").write_bytes(b"isolated-image")
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(bi.subprocess, "run", run)
    assert bi._generate_codex_image("prompt", str(out)) == str(out)
    assert out.read_bytes() == b"isolated-image"


def test_codex_attempt_budget_covers_every_pool_account(monkeypatch, tmp_path):
    entries = [type("Entry", (), {
        "id": str(i), "access_token": f"access-{i}", "refresh_token": f"refresh-{i}",
    })() for i in range(3)]

    class Pool:
        def __init__(self): self.index = 0
        def acquire_lease(self, credential_id=None):
            entry = entries[self.index]
            assert credential_id == entry.id
            self.index += 1
            return entry.id
        def release_lease(self, _ident): pass
        def entries(self): return entries
        def has_available(self): return True

    calls = []
    monkeypatch.setattr(bi, "_load_codex_pool", Pool)

    def timeout(_argv, **kwargs):
        calls.append(kwargs["timeout"])
        raise bi.subprocess.TimeoutExpired(_argv, kwargs["timeout"])

    monkeypatch.setattr(bi.subprocess, "run", timeout)
    assert bi._generate_codex_image("prompt", str(tmp_path / "out.png"), timeout=1, retry_timeout=2) is None
    assert calls == [1, 2, 2]


def test_concurrent_codex_runs_never_share_or_mutate_primary_auth(monkeypatch, tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "auth.json").write_text("primary")
    homes = []
    homes_lock = threading.Lock()

    class Entry:
        id = "one"
        access_token = "access"
        refresh_token = "refresh"

    class Pool:
        def acquire_lease(self, _credential_id=None): return "one"
        def release_lease(self, _ident): pass
        def entries(self): return [Entry()]
        def has_available(self): return True

    monkeypatch.setenv("CODEX_HOME", str(primary))
    monkeypatch.setattr(bi, "_load_codex_pool", Pool)

    def run(_argv, **kwargs):
        home = Path(kwargs["env"]["CODEX_HOME"])
        assert json.loads((home / "auth.json").read_text())["tokens"]["access_token"] == "access"
        with homes_lock:
            homes.append(home)
        generated = tmp_path / f"generated-{threading.get_ident()}.png"
        generated.write_bytes(b"image")
        return type("Result", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(bi.subprocess, "run", run)
    monkeypatch.setattr(
        bi, "_find_latest_codex_image",
        lambda **_kw: str(tmp_path / f"generated-{threading.get_ident()}.png"),
    )
    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(
        bi._generate_codex_image("prompt", str(tmp_path / f"out-{i}.png"))
    )) for i in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    assert len(set(homes)) == 2
    assert all(not home.exists() for home in homes)
    assert (primary / "auth.json").read_text() == "primary"
    assert len(results) == 2 and all(results)


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
