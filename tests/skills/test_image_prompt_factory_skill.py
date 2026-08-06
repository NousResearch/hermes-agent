"""Standards + behavior tests for the image-prompt-factory optional skill.

Two layers, all offline (no network, no LLM):
  - SKILL.md pinned to the hardline authoring standards in AGENTS.md.
  - The shipped deterministic scripts behave per their contracts:
    pack_validate.py rejects every violation class; style_corpus.py fails
    closed on a cold cache and ranks deterministically on a fixture corpus.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

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
def frontmatter(skill_text: str) -> str:
    m = re.search(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return m.group(1)


def _frontmatter_value(frontmatter: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", frontmatter, re.MULTILINE)
    assert match, f"frontmatter missing {key!r}"
    return match.group(1).strip()


def _frontmatter_list(frontmatter: str, key: str) -> list[str]:
    value = _frontmatter_value(frontmatter, key)
    assert value.startswith("[") and value.endswith("]")
    return [item.strip() for item in value[1:-1].split(",") if item.strip()]


@pytest.fixture(scope="module")
def pack_validate():
    return _load("pack_validate")


@pytest.fixture(scope="module")
def style_corpus():
    return _load("style_corpus")


# ── SKILL.md standards ──────────────────────────────────────────────────────


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_name_matches_dir(frontmatter: str) -> None:
    assert _frontmatter_value(frontmatter, "name") == "image-prompt-factory"


def test_description_hardline(frontmatter: str) -> None:
    desc = _frontmatter_value(frontmatter, "description")
    assert len(desc) <= 60, (
        f"description is {len(desc)} chars (hardline <=60): {desc!r}"
    )
    assert desc.endswith("."), "description must end with a period"
    assert ". " not in desc, "description must be a single sentence"
    lowered = desc.lower()
    assert not any(w in lowered for w in MARKETING_WORDS)
    assert "image-prompt-factory" not in lowered, "must not repeat the skill name"


def test_platforms_all_three(frontmatter: str) -> None:
    assert set(_frontmatter_list(frontmatter, "platforms")) == {
        "linux",
        "macos",
        "windows",
    }


def test_author_credits_contributor(frontmatter: str) -> None:
    assert "TheSmokeDev" in _frontmatter_value(frontmatter, "author")


def test_license_mit(frontmatter: str) -> None:
    assert _frontmatter_value(frontmatter, "license") == "MIT"


def test_related_skills_exist_in_repo(frontmatter: str) -> None:
    for related in _frontmatter_list(frontmatter, "related_skills"):
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


def test_placeholder_docs_match_mechanical_field_grammar(skill_text: str) -> None:
    schema = (SKILL_DIR / "references" / "prompt-schema.md").read_text(encoding="utf-8")
    for text in (skill_text, schema):
        assert "closed field grammar" in text
        assert "identity/appearance vocabulary" in text


# ── pack_validate.py behavior ───────────────────────────────────────────────


def _write_workdir(tmp_path, *, brief=None, grounding=None, pack=None):
    (tmp_path / "brief.json").write_text(json.dumps(brief or {}), encoding="utf-8")
    (tmp_path / "grounding.local.json").write_text(
        json.dumps(grounding or {}), encoding="utf-8"
    )
    (tmp_path / "prompt-pack.json").write_text(json.dumps(pack or {}), encoding="utf-8")
    return tmp_path


def _valid_grounded_inputs():
    brief = {"count": 1, "subject_mode": "generic"}
    grounding = {
        "grounded": True,
        "resolved_case_ids": [101, 205],
        "prompt_engine": "gpt-image-2-style-library",
        "corpus_pin": "pin",
        "corpus_source": "https://example.test/corpus",
        "corpus_sha256": "sha",
        "license": "MIT",
    }
    pack = {
        "prompt_count": 1,
        "example_case_ids": [101],
        "prompt_engine": "gpt-image-2-style-library",
        "corpus_pin": "pin",
        "corpus_source": "https://example.test/corpus",
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


def test_nested_grounding_case_id_fails_validation_not_type_error(
    pack_validate, tmp_path
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    grounding["resolved_case_ids"] = [[]]
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert (
        "grounding resolved_case_ids must be an array of integers"
        in exc.value.violations
    )


def test_nested_pack_case_id_fails_validation_not_type_error(
    pack_validate, tmp_path
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    pack["example_case_ids"] = [[101]]
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert "pack example_case_ids must be an array of integers" in exc.value.violations


@pytest.mark.parametrize(
    "field",
    ["prompt_engine", "corpus_pin", "corpus_source", "corpus_sha256", "license"],
)
def test_each_documented_provenance_mismatch_rejected(
    pack_validate, tmp_path, field
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    pack[field] = f"different-{field}"
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)
    assert any(f"provenance mismatch on {field}" in v for v in exc.value.violations)


@pytest.mark.parametrize(
    "field",
    ["prompt_engine", "corpus_pin", "corpus_source", "corpus_sha256", "license"],
)
def test_matching_non_scalar_provenance_is_rejected(
    pack_validate, tmp_path, field
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    grounding[field] = {"nested": field}
    pack[field] = {"nested": field}
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert f"grounding {field} must be a string" in exc.value.violations
    assert f"pack {field} must be a string" in exc.value.violations


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda _g, p: p.update(prompt_count=[]),
            "pack prompt_count must be an integer",
        ),
        (
            lambda g, _p: g.update(unresolved_case_ids=[{}]),
            "grounding unresolved_case_ids must be an array of integers",
        ),
        (
            lambda g, _p: g.update(exemplars={}),
            "grounding exemplars must be an array of objects",
        ),
        (
            lambda _g, p: p["concepts"][0].update(concept_id=[]),
            "concept 1: concept_id must be a string",
        ),
        (
            lambda _g, p: p["concepts"][0]["copy"].update(headline=[]),
            "concept 1: copy values must be strings",
        ),
    ],
)
def test_sibling_nested_shapes_fail_deterministically(
    pack_validate, tmp_path, mutation, message
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    mutation(grounding, pack)
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert message in exc.value.violations


@pytest.mark.parametrize(
    ("artifact", "value"),
    [
        ("brief.json", []),
        ("grounding.local.json", "not-an-object"),
        ("prompt-pack.json", ["not", "an", "object"]),
    ],
)
def test_non_object_artifacts_fail_as_pack_invalid(
    pack_validate, tmp_path, artifact, value
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    (wd / artifact).write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert f"{artifact} must contain a JSON object" in exc.value.violations


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
    subject = f"Subject: {tok}; {pack_validate.SUBJECT_PRESERVATION_DIRECTIVE}"
    pack["concepts"][0]["baked_prompt"] = subject
    pack["concepts"][0]["overlay_prompt"] = (
        f"{subject}\nText handling: no text, no words"
    )
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)
    assert pack_validate.validate_pack(wd)["subject_mode"] == "placeholder"


def test_placeholder_sentinel_outside_subject_field_is_rejected(
    pack_validate, tmp_path
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    brief["subject_mode"] = "placeholder"
    tok = pack_validate.SUBJECT_SENTINEL
    for key in ("baked_prompt", "overlay_prompt"):
        pack["concepts"][0][key] = (
            f"Subject: a red-haired woman\nConstraints: replace with {tok}"
        )
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert any("Subject: field" in v for v in exc.value.violations)


def test_placeholder_subject_field_rejects_invented_traits(
    pack_validate, tmp_path
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    brief["subject_mode"] = "placeholder"
    tok = pack_validate.SUBJECT_SENTINEL
    for key in ("baked_prompt", "overlay_prompt"):
        pack["concepts"][0][key] = (
            f"Subject: {tok}, a red-haired woman; "
            f"{pack_validate.SUBJECT_PRESERVATION_DIRECTIVE}"
        )
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert any("canonical placeholder directive" in v for v in exc.value.violations)


@pytest.mark.parametrize("field", ["Composition/framing", "Lighting/mood"])
def test_placeholder_rejects_reviewer_trait_probes_outside_subject(
    pack_validate, tmp_path, field
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    brief["subject_mode"] = "placeholder"
    subject = (
        f"Subject: {pack_validate.SUBJECT_SENTINEL}; "
        f"{pack_validate.SUBJECT_PRESERVATION_DIRECTIVE}"
    )
    for key in ("baked_prompt", "overlay_prompt"):
        pack["concepts"][0][key] = (
            f"{subject}\n{field}: blue-eyed, red-haired woman in a medium shot"
        )
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert any(
        f"forbidden subject identity/appearance language in {field}" in violation
        for violation in exc.value.violations
    )


@pytest.mark.parametrize(
    "field",
    [
        "Use case",
        "Template",
        "Primary request",
        "Input references",
        "Scene/backdrop",
        "Style/medium",
        "Composition/framing",
        "Lighting/mood",
        "Color palette",
        "Text handling",
        "Constraints",
        "Avoid",
    ],
)
def test_placeholder_audits_every_non_subject_prompt_field(
    pack_validate, tmp_path, field
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    brief["subject_mode"] = "placeholder"
    subject = (
        f"Subject: {pack_validate.SUBJECT_SENTINEL}; "
        f"{pack_validate.SUBJECT_PRESERVATION_DIRECTIVE}"
    )
    probe = f"{subject}\n{field}: red-haired woman"
    pack["concepts"][0]["baked_prompt"] = probe
    pack["concepts"][0]["overlay_prompt"] = probe
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    with pytest.raises(pack_validate.PackInvalid) as exc:
        pack_validate.validate_pack(wd)

    assert any(
        "forbidden subject identity/appearance language" in v
        for v in exc.value.violations
    )


def test_placeholder_allows_useful_scene_pose_camera_lighting_and_style(
    pack_validate, tmp_path
) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    brief["subject_mode"] = "placeholder"
    subject = (
        f"Subject: {pack_validate.SUBJECT_SENTINEL}; "
        f"{pack_validate.SUBJECT_PRESERVATION_DIRECTIVE}"
    )
    useful = "\n".join([
        "Use case: editorial campaign hero",
        "Template: realistic-photography",
        "Scene/backdrop: quiet modern studio with a seamless backdrop",
        subject,
        "Style/medium: editorial photography with restrained film grain",
        "Composition/framing: three-quarter standing pose, centered, eye-level 50 mm camera, medium shot",
        "Lighting/mood: soft window light from camera left, calm cinematic mood",
        "Color palette: navy, cream, and warm amber accents",
        "Text handling: no text, no words",
        "Constraints: preserve reference-locked wardrobe and expression",
    ])
    pack["concepts"][0]["baked_prompt"] = useful
    pack["concepts"][0]["overlay_prompt"] = useful
    wd = _write_workdir(tmp_path, brief=brief, grounding=grounding, pack=pack)

    assert pack_validate.validate_pack(wd)["subject_mode"] == "placeholder"


def test_local_path_rejected_but_urls_allowed(pack_validate, tmp_path) -> None:
    brief, grounding, pack = _valid_grounded_inputs()
    source = "https://github.com/freestylefly/awesome-gpt-image-2"
    grounding["corpus_source"] = source
    pack["corpus_source"] = source
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

PINNED_TEMPLATE_ANCHORS = (
    ("ui-screenshot-system", "tpl-ui"),
    ("infographic-engine", "tpl-infographic"),
    ("scientific-scale-diagram", "tpl-infographic"),
    ("poster-layout-system", "tpl-poster"),
    ("sports-campaign-poster", "tpl-poster"),
    ("conceptual-typography-poster", "tpl-poster"),
    ("ink-double-exposure-poster", "tpl-poster"),
    ("nature-science-poster", "tpl-poster"),
    ("product-commerce-visual", "tpl-product"),
    ("personalized-beauty-report", "tpl-product"),
    ("brand-identity-package", "tpl-brand"),
    ("brand-touchpoint-board", "tpl-brand"),
    ("architecture-space", "tpl-architecture"),
    ("realistic-photography", "tpl-photo"),
    ("street-accident-moment", "tpl-photo"),
    ("illustration-art-style", "tpl-illustration"),
    ("character-design-sheet", "tpl-character"),
    ("3d-collectible-toy", "tpl-character"),
    ("scene-storytelling", "tpl-scene"),
    ("history-classical-themes", "tpl-history"),
    ("document-publishing", "tpl-document"),
    ("concept-product-breakdown", "tpl-other"),
)


def _template_record(template_id: str, anchor: str) -> dict:
    return {
        "id": template_id,
        "anchor": anchor,
        "cover": "/images/fixture.jpg",
        "title": {"en": "fixture", "zh": "fixture"},
        "description": {"en": "fixture", "zh": "fixture"},
        "category": "fixture",
        "styles": ["fixture-style"],
        "scenes": ["fixture-scene"],
        "tags": ["fixture-tag"],
        "useWhen": {"en": "fixture", "zh": "fixture"},
        "guidance": {"en": ["fixture"], "zh": ["fixture"]},
        "pitfalls": {"en": ["fixture"], "zh": ["fixture"]},
        "exampleCases": [1],
    }


def _load_template_fixture(style_corpus, tmp_path, templates):
    anchors = dict.fromkeys(t["anchor"] for t in templates)
    doc = "\n".join(f'<a name="{anchor}"></a>\nBODY FOR {anchor}' for anchor in anchors)
    return style_corpus._load(
        tmp_path,
        style_corpus.UPSTREAM_PIN,
        {
            "cases.json": b'{"cases": []}',
            "style-library.json": json.dumps({"templates": templates}).encode(),
            "templates.md": doc.encode(),
            "LICENSE": b"fixture license",
        },
    )


def test_cold_cache_fails_closed(style_corpus, tmp_path) -> None:
    with pytest.raises(style_corpus.CorpusMissing) as exc:
        style_corpus.require_corpus(cache_dir=tmp_path / "empty")
    assert "prime" in str(exc.value), "the error must tell the operator how to fix it"


def test_cold_cache_cli_exit_1(style_corpus, tmp_path, capsys) -> None:
    rc = style_corpus.main(["--cache-dir", str(tmp_path / "empty"), "verify"])
    assert rc == 1
    assert "not provisioned" in capsys.readouterr().err


def test_default_cache_uses_active_hermes_home(
    style_corpus, tmp_path, monkeypatch
) -> None:
    profile_home = tmp_path / "profiles" / "work"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.delenv(style_corpus.CACHE_ENV, raising=False)
    assert style_corpus.cache_root() == profile_home / "cache" / "image-prompt-factory"


def test_alternate_pin_is_rejected_before_network(
    style_corpus, tmp_path, monkeypatch
) -> None:
    def network_must_not_run(_url):
        raise AssertionError("unsupported pins must fail before fetching")

    monkeypatch.setattr(style_corpus, "_http_get", network_must_not_run)
    with pytest.raises(style_corpus.UsageError, match="unsupported corpus pin"):
        style_corpus.prime(pin="untrusted-pin", cache_dir=tmp_path)


def test_provenance_hashes_materialized_cases_file(style_corpus, tmp_path) -> None:
    cases_bytes = json.dumps(
        {
            "cases": [
                {
                    "id": 7,
                    "title": "fixture",
                    "prompt": "materialized prompt",
                    "category": "fixture",
                }
            ]
        },
        separators=(",", ":"),
    ).encode()
    (tmp_path / "cases.json").write_bytes(cases_bytes)
    (tmp_path / "style-library.json").write_text('{"templates": []}', encoding="utf-8")
    corpus = style_corpus._load(
        tmp_path,
        style_corpus.UPSTREAM_PIN,
        {
            "cases.json": cases_bytes,
            "style-library.json": b'{"templates": []}',
            "templates.md": b"",
            "LICENSE": b"fixture license",
        },
    )

    grounding = style_corpus.select(corpus, case_ids=[7])

    assert (
        grounding.provenance["corpus_sha256"] == hashlib.sha256(cases_bytes).hexdigest()
    )


def test_load_consumes_documented_template_anchor(style_corpus, tmp_path) -> None:
    corpus = _load_template_fixture(
        style_corpus,
        tmp_path,
        [_template_record("3d-collectible-toy", "tpl-character")],
    )

    assert "BODY FOR tpl-character" in style_corpus.template_body(
        corpus, "3d-collectible-toy"
    )


def test_all_22_pinned_templates_resolve_nonempty_bodies_and_toy_cli_succeeds(
    style_corpus, tmp_path, monkeypatch, capsys
) -> None:
    corpus = _load_template_fixture(
        style_corpus,
        tmp_path,
        [
            _template_record(template_id, anchor)
            for template_id, anchor in PINNED_TEMPLATE_ANCHORS
        ],
    )

    assert len(corpus.templates) == 22
    for template_id, _anchor in PINNED_TEMPLATE_ANCHORS:
        body = style_corpus.template_body(corpus, template_id)
        assert body.strip()
        assert "BODY FOR" in body

    monkeypatch.setattr(style_corpus, "require_corpus", lambda **_kwargs: corpus)
    assert style_corpus.main(["template", "3d-collectible-toy"]) == 0
    assert "BODY FOR tpl-character" in capsys.readouterr().out


def test_template_schema_rejects_unknown_keys(style_corpus, tmp_path) -> None:
    template = _template_record("3d-collectible-toy", "tpl-character")
    template["templateAnchor"] = template["anchor"]

    with pytest.raises(style_corpus.UsageError, match="unknown keys.*templateAnchor"):
        _load_template_fixture(style_corpus, tmp_path, [template])


@pytest.mark.parametrize(
    ("templates", "message"),
    [
        ([[]], "template 1 must be a JSON object"),
        (
            [dict(_template_record("toy", "tpl-character"), exampleCases=[[101]])],
            "exampleCases must be an array of integers",
        ),
        (
            [dict(_template_record("toy", "tpl-character"), styles=[[]])],
            "styles must be an array of strings",
        ),
    ],
)
def test_template_schema_nested_shapes_fail_deterministically(
    style_corpus, tmp_path, templates, message
) -> None:
    with pytest.raises(style_corpus.UsageError, match=message):
        style_corpus._load(
            tmp_path,
            style_corpus.UPSTREAM_PIN,
            {
                "cases.json": b'{"cases": []}',
                "style-library.json": json.dumps({"templates": templates}).encode(),
                "templates.md": b'<a name="tpl-character"></a>\nBODY\n',
                "LICENSE": b"fixture license",
            },
        )


def test_template_schema_rejects_empty_resolved_body(style_corpus, tmp_path) -> None:
    template = _template_record("toy", "tpl-character")
    with pytest.raises(style_corpus.UsageError, match="template body is empty"):
        style_corpus._load(
            tmp_path,
            style_corpus.UPSTREAM_PIN,
            {
                "cases.json": b'{"cases": []}',
                "style-library.json": json.dumps({"templates": [template]}).encode(),
                "templates.md": b'<a name="tpl-character"></a>\n',
                "LICENSE": b"fixture license",
            },
        )


def test_verified_cache_bytes_are_the_bytes_consumed(
    style_corpus, tmp_path, monkeypatch
) -> None:
    original = {
        "cases.json": json.dumps({
            "cases": [
                {
                    "id": 7,
                    "title": "verified",
                    "prompt": "verified prompt",
                    "category": "verified-category",
                }
            ]
        }).encode(),
        "style-library.json": json.dumps({
            "templates": [
                dict(
                    _template_record("verified-template", "tpl-verified"),
                    category="verified-category",
                    exampleCases=[7],
                )
            ]
        }).encode(),
        "templates.md": b'<a name="tpl-verified"></a>\nVERIFIED TEMPLATE BODY\n',
        "LICENSE": b"verified license",
    }
    replacement = {
        "cases.json": json.dumps({
            "cases": [
                {
                    "id": 8,
                    "title": "swapped",
                    "prompt": "swapped prompt",
                    "category": "swapped-category",
                }
            ]
        }).encode(),
        "style-library.json": json.dumps({
            "templates": [
                dict(
                    _template_record("swapped-template", "tpl-swapped"),
                    category="swapped-category",
                    exampleCases=[8],
                )
            ]
        }).encode(),
        "templates.md": b'<a name="tpl-swapped"></a>\nSWAPPED TEMPLATE BODY\n',
        "LICENSE": b"swapped license",
    }
    pin_dir = tmp_path / style_corpus.UPSTREAM_PIN
    pin_dir.mkdir()
    for name, data in original.items():
        (pin_dir / name).write_bytes(data)
    monkeypatch.setattr(
        style_corpus,
        "CORPUS_FILES",
        {
            name: (name, hashlib.sha256(data).hexdigest())
            for name, data in original.items()
        },
    )
    real_read_bytes = Path.read_bytes

    def read_bytes_then_swap(path):
        data = real_read_bytes(path)
        if path.name in original and data == original[path.name]:
            path.write_bytes(replacement[path.name])
        return data

    monkeypatch.setattr(Path, "read_bytes", read_bytes_then_swap)

    corpus = style_corpus.require_corpus(cache_dir=tmp_path)

    assert set(corpus.cases) == {7}
    assert set(corpus.templates) == {"verified-template"}
    assert "VERIFIED TEMPLATE BODY" in style_corpus.template_body(
        corpus, "verified-template"
    )


def test_ground_non_object_selection_fails_deterministically(
    style_corpus, tmp_path, capsys
) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("[]", encoding="utf-8")

    rc = style_corpus.main(["ground", "--selection", str(selection)])

    assert rc == 1
    assert "selection.json must contain a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            {"example_case_ids": [[101]]},
            "example_case_ids must be an array of integers",
        ),
        ({"style_tags": {"studio": True}}, "style_tags must be an array of strings"),
        ({"scene_tags": [[]]}, "scene_tags must be an array of strings"),
        ({"template_id": []}, "template_id must be a string"),
        ({"category": {}}, "category must be a string"),
    ],
)
def test_ground_selection_nested_shapes_fail_deterministically(
    style_corpus, tmp_path, monkeypatch, capsys, selection, message
) -> None:
    monkeypatch.setattr(
        style_corpus, "require_corpus", lambda **_kwargs: _fixture_corpus(style_corpus)
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")

    assert style_corpus.main(["ground", "--selection", str(path)]) == 1
    assert message in capsys.readouterr().err


def _fixture_corpus(style_corpus):
    mk = style_corpus.Case
    cases = {
        1: mk(
            1,
            "en product",
            "studio product shot",
            "commerce",
            ("studio",),
            ("product",),
            True,
            "",
        ),
        2: mk(2, "en poster", "bold poster layout", "poster", (), (), False, ""),
        3: mk(3, "cjk case", "中文提示词", "commerce", ("studio",), (), False, ""),
        4: mk(4, "en scene", "street scene photo", "photo", (), ("street",), False, ""),
    }
    templates = {
        "tpl-commerce": style_corpus.Template(
            "tpl-commerce", "commerce", "tpl-commerce", ("studio",), ("product",), (1,)
        )
    }
    return style_corpus.Corpus(
        root=Path("."),
        pin="testpin",
        cases=cases,
        templates=templates,
        cases_sha256="fixture-sha256",
    )


def test_select_deterministic_ranking(style_corpus) -> None:
    corpus = _fixture_corpus(style_corpus)
    g = style_corpus.select(
        corpus,
        template_id="tpl-commerce",
        category="commerce",
        styles=["studio"],
        scenes=["product"],
        k=3,
    )
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


def test_ground_cli_reads_selection_offline(
    style_corpus, tmp_path, monkeypatch, capsys
) -> None:
    # No network: point ground at a fixture corpus via require_corpus monkeypatch.
    corpus = _fixture_corpus(style_corpus)
    monkeypatch.setattr(style_corpus, "require_corpus", lambda **kw: corpus)
    sel = tmp_path / "selection.json"
    sel.write_text(
        json.dumps({
            "template_id": "tpl-commerce",
            "category": "commerce",
            "example_case_ids": [1],
        }),
        encoding="utf-8",
    )
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
