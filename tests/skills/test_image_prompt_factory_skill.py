"""Standards + behavior tests for the image-prompt-factory optional skill.

Two layers, all offline (no network, no LLM):
  - SKILL.md pinned to the hardline authoring standards in AGENTS.md.
  - The shipped deterministic scripts behave per their contracts:
    pack_validate.py rejects every violation class; style_corpus.py fails
    closed on a cold cache and ranks deterministically on a fixture corpus.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "creative" / "image-prompt-factory"

MARKETING_WORDS = ("powerful", "comprehensive", "seamless", "advanced")

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def _load(name: str):
    # Register in sys.modules under a unique key: dataclass machinery resolves
    # string annotations via sys.modules[cls.__module__].
    mod_name = f"image_prompt_factory_{name}"
    spec = importlib.util.spec_from_file_location(
        mod_name, SKILL_DIR / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    m = re.search(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def pack_validate():
    return _load("pack_validate")


@pytest.fixture(scope="module")
def style_corpus():
    return _load("style_corpus")


# ── SKILL.md standards ──────────────────────────────────────────────────────

def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_name_matches_dir(frontmatter: dict) -> None:
    assert frontmatter["name"] == "image-prompt-factory"


def test_description_hardline(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    assert isinstance(desc, str), "description must be a plain string"
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"
    assert ". " not in desc, "description must be a single sentence"
    lowered = desc.lower()
    assert not any(w in lowered for w in MARKETING_WORDS)
    assert "image-prompt-factory" not in lowered, "must not repeat the skill name"


def test_platforms_all_three(frontmatter: dict) -> None:
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}


def test_author_credits_contributor(frontmatter: dict) -> None:
    assert "TheSmokeDev" in frontmatter["author"]


def test_license_mit(frontmatter: dict) -> None:
    assert frontmatter["license"] == "MIT"


def test_related_skills_exist_in_repo(frontmatter: dict) -> None:
    for related in frontmatter["metadata"]["hermes"]["related_skills"]:
        matches = list(REPO_ROOT.glob(f"skills/**/{related}/SKILL.md")) + list(
            REPO_ROOT.glob(f"optional-skills/**/{related}/SKILL.md")
        )
        assert matches, f"related skill does not exist in repo: {related!r}"


def test_modern_section_order(skill_text: str) -> None:
    positions = [skill_text.find(h) for h in REQUIRED_SECTIONS]
    missing = [h for h, p in zip(REQUIRED_SECTIONS, positions) if p == -1]
    assert not missing, f"missing required sections: {missing}"
    assert positions == sorted(positions), "sections out of the AGENTS.md order"


def test_no_direct_pytest_invocation(skill_text: str) -> None:
    assert "python -m pytest" not in skill_text
    assert "scripts/run_tests.sh" in skill_text


def test_line_budget(skill_text: str) -> None:
    assert len(skill_text.splitlines()) <= 220


def test_scripts_are_pure_ascii_and_stdlib() -> None:
    # Pure ASCII: the scripts are read and piped by cp1252-defaulting toolchains.
    for name in ("style_corpus.py", "pack_validate.py"):
        data = (SKILL_DIR / "scripts" / name).read_bytes()
        assert all(b <= 127 for b in data), f"{name} contains non-ASCII bytes"


# ── pack_validate.py behavior ───────────────────────────────────────────────

def _write_workdir(tmp_path, *, brief=None, grounding=None, pack=None):
    (tmp_path / "brief.json").write_text(json.dumps(brief or {}), encoding="utf-8")
    (tmp_path / "grounding.local.json").write_text(json.dumps(grounding or {}), encoding="utf-8")
    (tmp_path / "prompt-pack.json").write_text(json.dumps(pack or {}), encoding="utf-8")
    return tmp_path


def _valid_grounded_inputs():
    brief = {"count": 1, "subject_mode": "generic"}
    grounding = {
        "grounded": True,
        "resolved_case_ids": [101, 205],
        "prompt_engine": "gpt-image-2-style-library",
        "corpus_pin": "pin",
        "corpus_sha256": "sha",
        "license": "MIT",
    }
    pack = {
        "prompt_count": 1,
        "example_case_ids": [101],
        "prompt_engine": "gpt-image-2-style-library",
        "corpus_pin": "pin",
        "corpus_sha256": "sha",
        "license": "MIT",
        "concepts": [
            {
                "concept_id": "concept-01",
                "baked_prompt": "a product hero shot",
                "overlay_prompt": "a text-free product scene, no text, no words",
                "copy": {"headline": "x"},
            }
        ],
    }
    return brief, grounding, pack


def test_valid_grounded_pack_passes(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    summary = pack_validate.validate_pack(wd)
    assert summary["pack_valid"] is True
    assert summary["grounded"] is True
    assert summary["cited_case_ids"] == [101]


def test_hollow_citation_rejected(pack_validate, tmp_path) -> None:
    brief, _grounding, pack = _valid_grounded_inputs()
    wd = _write_workdir(tmp_path, brief=brief, grounding={"grounded": False}, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any("HOLLOW CITATION" in v for v in exc.value.violations)


def test_ungrounded_needs_self_authored(pack_validate, tmp_path) -> None:
    brief, _g, _p = _valid_grounded_inputs()
    pack = {"concepts": [{"baked_prompt": "x", "overlay_prompt": "y", "copy": {}}]}
    wd = _write_workdir(tmp_path, brief=brief, grounding={"grounded": False}, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any("self_authored" in v for v in exc.value.violations)


def test_cited_id_outside_resolved_set_rejected(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    pack["example_case_ids"] = [101, 999]
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any("never resolved" in v for v in exc.value.violations)


def test_provenance_mismatch_rejected(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    pack["corpus_pin"] = "different-pin"
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any("provenance mismatch" in v for v in exc.value.violations)


def test_concept_cap_enforced(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    concept = pack["concepts"][0]
    pack["concepts"] = [dict(concept, concept_id=f"c-{i}") for i in range(9)]
    pack["prompt_count"] = 9
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any("exceeds the cap" in v for v in exc.value.violations)


def test_empty_variant_rejected(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    pack["concepts"][0]["overlay_prompt"] = "   "
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any("empty overlay_prompt" in v for v in exc.value.violations)


def test_placeholder_sentinel_required(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    brief["subject_mode"] = "placeholder"
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any(pack_validate.SUBJECT_SENTINEL in v for v in exc.value.violations)


def test_placeholder_sentinel_satisfies(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    brief["subject_mode"] = "placeholder"
    tok = pack_validate.SUBJECT_SENTINEL
    pack["concepts"][0]["baked_prompt"] = f"Subject: {tok} centered"
    pack["concepts"][0]["overlay_prompt"] = f"Subject: {tok} left, no text, no words"
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    assert pack_validate.validate_pack(wd)["subject_mode"] == "placeholder"


def test_local_path_rejected_but_urls_allowed(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    pack["corpus_source"] = "https://github.com/freestylefly/awesome-gpt-image-2"
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    pack_validate.validate_pack(wd)  # URL must not trip the drive-letter regex
    pack["concepts"][0]["baked_prompt"] = r"see C:\Users\someone\art.png"
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any("absolute local path" in v for v in exc.value.violations)


def test_missing_copy_object_rejected(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    del pack["concepts"][0]["copy"]
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any("missing copy object" in v for v in exc.value.violations)


# ── style_corpus.py behavior ────────────────────────────────────────────────

def test_cold_cache_fails_closed(style_corpus, tmp_path) -> None:
    with pytest.raises(style_corpus.CorpusMissing) as exc:
        style_corpus.require_corpus(cache_dir=tmp_path / "empty")
    assert "prime" in str(exc.value), "the error must tell the operator how to fix it"


def test_cold_cache_cli_exit_1(style_corpus, tmp_path, capsys) -> None:
    rc = style_corpus.main(["--cache-dir", str(tmp_path / "empty"), "verify"])
    assert rc == 1
    assert "not provisioned" in capsys.readouterr().err


def _fixture_corpus(style_corpus):
    mk = style_corpus.Case
    cases = {
        1: mk(1, "en product", "studio product shot", "commerce", ("studio",), ("product",), True, ""),
        2: mk(2, "en poster", "bold poster layout", "poster", (), (), False, ""),
        3: mk(3, "cjk case", "中文提示词", "commerce", ("studio",), (), False, ""),
        4: mk(4, "en scene", "street scene photo", "photo", (), ("street",), False, ""),
    }
    templates = {
        "tpl-commerce": style_corpus.Template(
            "tpl-commerce", "commerce", "tpl-commerce", ("studio",), ("product",), (1,)
        )
    }
    return style_corpus.Corpus(root=Path("."), pin="testpin", cases=cases, templates=templates)


def test_select_deterministic_ranking(style_corpus) -> None:
    corpus = _fixture_corpus(style_corpus)
    g = style_corpus.select(corpus, template_id="tpl-commerce", category="commerce",
                            styles=["studio"], scenes=["product"], k=3)
    assert g.grounded is True
    # case 1: category 3 + style 2 + scene 1 + example 4 = 10; case 3: 3 + 2 = 5.
    assert g.resolved_case_ids[0] == 1
    assert g.provenance["prompt_engine"] == "gpt-image-2-style-library"


def test_select_lang_en_filters_cjk(style_corpus) -> None:
    corpus = _fixture_corpus(style_corpus)
    g = style_corpus.select(corpus, category="commerce", lang="en", k=5)
    assert 3 not in g.resolved_case_ids


def test_select_zero_match_is_honest_ungrounded(style_corpus) -> None:
    corpus = _fixture_corpus(style_corpus)
    g = style_corpus.select(corpus, category="no-such-category", k=5)
    assert g.grounded is False
    assert g.provenance == {}, "nothing to cite, nothing stamped"


def test_select_anchor_ids_and_unresolved(style_corpus) -> None:
    corpus = _fixture_corpus(style_corpus)
    g = style_corpus.select(corpus, case_ids=[2, 999], k=5)
    assert g.resolved_case_ids[0] == 2
    assert g.unresolved_case_ids == (999,)


def test_select_unknown_template_raises(style_corpus) -> None:
    corpus = _fixture_corpus(style_corpus)
    with pytest.raises(style_corpus.UsageError):
        style_corpus.select(corpus, template_id="nope")


def test_ground_cli_reads_selection_offline(style_corpus, tmp_path, monkeypatch, capsys) -> None:
    # No network: point ground at a fixture corpus via require_corpus monkeypatch.
    corpus = _fixture_corpus(style_corpus)
    monkeypatch.setattr(style_corpus, "require_corpus", lambda **kw: corpus)
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"template_id": "tpl-commerce", "category": "commerce",
                               "example_case_ids": [1]}), encoding="utf-8")
    rc = style_corpus.main(["ground", "--selection", str(sel)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["grounded"] is True
    grounding_file = tmp_path / "grounding.local.json"
    assert grounding_file.is_file()
    full = json.loads(grounding_file.read_text(encoding="utf-8"))
    assert full["exemplars"][0]["id"] == 1


def test_http_get_never_called_in_this_suite(style_corpus, monkeypatch) -> None:
    def boom(url):  # pragma: no cover - tripwire
        raise AssertionError(f"network attempted: {url}")
    monkeypatch.setattr(style_corpus, "_http_get", boom)
    corpus = _fixture_corpus(style_corpus)
    style_corpus.select(corpus, category="commerce", k=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
