"""Tests for blog.art_director — one rich art brief per article."""
import json

import blog.art_director as ad


_DRAFT = {
    "title": "The crossover point: when local inference beats the API",
    "description": "API cost vs hardware amortisation.",
    "body_md": "# X\n\nFor two years the API won.\n\n## The mechanism\n\nHardware amortises.",
    "stream": "ai",
}
_HEADINGS = ["The mechanism"]


def test_blog_style_catalogue_excludes_rejected_visual_modes():
    assert "ninth-observatory" not in ad.STYLE_IDS
    assert "chromatic-institute" not in ad.STYLE_IDS
    assert "ninth-observatory" not in ad._styles_catalogue()
    assert "chromatic-institute" not in ad._styles_catalogue()


def _good_brief_json():
    return json.dumps({
        "style": "technical-diorama",
        "style_candidates": ["technical-diorama", "technical-diorama", "baoyu-infographic"],
        "layout": "architectural cross-section with labelled chambers",
        "layout_variants": ["architectural cross-section", "control hall", "vault map"],
        "text_policy": "labels",
        "text_elements": ["API", "LOCAL", "CROSSOVER"],
        "palette": "slate, brass, amber",
        "motif": "converging rails",
        "art_direction": "vast architectural space, one warm focal light.",
        "hero_prompt": "two cost curves crossing inside a vast engine hall",
        "section_prompts": [
            {"heading": "The mechanism", "prompt": "amortising hardware as stone foundations"}
        ],
        "concept_candidates": [{
            "candidate_id": "clockwork-theatre",
            "world": "A travelling clockwork theatre where futures are performed before they happen.",
            "fingerprint": ["clockwork theatre", "origami futures", "deep sea"],
            "creative_score": 92,
            "inspiration_category": "systems-as-worlds",
            "relevance_rationale": "The hero and supporting scene each bind to a supplied article target.",
            "scenes": [
                {
                    "asset_key": "hero",
                    "source_target_id": "hero",
                    "source_claim": "The crossover point compares API cost with hardware amortisation.",
                    "visual_translation": "Two physical routes race toward one answer, one rented and one owned.",
                    "scene": "A brass courier races across the stage with rented tickets while a permanent engine wakes below.",
                    "composition": "wide proscenium collision",
                    "creative_technique": "temporal-collision",
                },
                {
                    "asset_key": "section-01",
                    "source_target_id": "section-01",
                    "source_claim": "Hardware amortises.",
                    "visual_translation": "A durable machine pays for itself through repeated use.",
                    "scene": "Stagehands feed the same brass engine through repeated performances until its gears glow warm.",
                    "composition": "side-on process cutaway",
                    "creative_technique": "assembly-line",
                },
            ],
        }],
    })


def test_build_art_brief_parses_valid_llm_output():
    brief = ad.build_art_brief(_DRAFT, _HEADINGS, llm=lambda s, u: _good_brief_json())
    assert brief is not None
    assert brief["style"] == "technical-diorama"
    assert brief["layout"]
    assert brief["layout_variants"]
    assert brief["selection_seed"] == ad._selection_seed(_DRAFT)
    assert len(brief["selection_seed"]) == 16
    assert brief["style_candidates"]
    assert "physical mechanism" in brief["style_native_compiler"]
    assert brief["text_policy"] == "labels"
    assert brief["text_elements"] == ["API", "LOCAL", "CROSSOVER"]
    assert brief["palette"]
    assert brief["hero_prompt"]
    assert brief["section_prompts"]["The mechanism"]


def test_build_art_brief_handles_fenced_json():
    fenced = "Here you go:\n```json\n" + _good_brief_json() + "\n```\n"
    brief = ad.build_art_brief(_DRAFT, _HEADINGS, llm=lambda s, u: fenced)
    assert brief is not None
    assert brief["style"] == "technical-diorama"


def test_build_art_brief_accepts_provider_shape_with_rendering_fields_inside_candidate():
    payload = json.loads(_good_brief_json())
    candidate = payload["concept_candidates"][0]
    for field in ("style", "hero_prompt", "palette", "motif", "art_direction", "text_policy", "text_elements"):
        candidate[field] = payload.pop(field)

    brief = ad.build_art_brief(_DRAFT, _HEADINGS, llm=lambda s, u: json.dumps(payload))

    assert brief is not None
    assert brief["style"] == "technical-diorama"
    assert brief["hero_prompt"]


def test_build_art_brief_rejects_unknown_style():
    bad = _good_brief_json().replace("technical-diorama", "not-a-real-style")
    brief = ad.build_art_brief(_DRAFT, _HEADINGS, llm=lambda s, u: bad)
    assert brief is None


def test_build_art_brief_none_on_unparseable():
    brief = ad.build_art_brief(_DRAFT, _HEADINGS, llm=lambda s, u: "sorry, no JSON here")
    assert brief is None


def test_build_art_brief_none_on_llm_exception():
    def boom(s, u):
        raise RuntimeError("model down")
    assert ad.build_art_brief(_DRAFT, _HEADINGS, llm=boom) is None


def test_default_llm_retries_once_after_contract_invalid_response(monkeypatch):
    import llm_generate

    config = {
        "base": "https://example.invalid/v1",
        "model": "test-model",
        "key": "test-key",
    }
    responses = iter(["not valid JSON", _good_brief_json()])
    calls = []

    monkeypatch.setattr(llm_generate, "_llm_configs", lambda longform: [config])

    def fake_call(system, user, cfg, *, timeout, max_tokens):
        calls.append(cfg)
        return next(responses)

    monkeypatch.setattr(llm_generate, "_call_llm", fake_call)

    brief = ad.build_art_brief(_DRAFT, _HEADINGS)

    assert brief is not None
    assert brief["style"] == "technical-diorama"
    assert calls == [config, config]


def test_default_llm_stops_after_two_contract_invalid_responses(monkeypatch):
    import llm_generate

    config = {
        "base": "https://example.invalid/v1",
        "model": "test-model",
        "key": "test-key",
    }
    calls = []

    monkeypatch.setattr(llm_generate, "_llm_configs", lambda longform: [config])

    def fake_call(system, user, cfg, *, timeout, max_tokens):
        calls.append(cfg)
        return "still not valid JSON"

    monkeypatch.setattr(llm_generate, "_call_llm", fake_call)

    assert ad.build_art_brief(_DRAFT, _HEADINGS) is None
    assert calls == [config, config]


def test_prompt_allows_article_fit_rendering_variation():
    captured = {}
    def spy(system, user):
        captured["system"] = system
        return _good_brief_json()
    ad.build_art_brief(_DRAFT, _HEADINGS, recent_styles=["saga-noir", "pixel-art"], llm=spy)
    assert "Do not default to one house style" in captured["system"]


def test_prompt_names_recent_visual_choices_to_avoid_repeating():
    captured = {}

    def spy(system, user):
        captured["system"] = system
        return _good_brief_json()

    ad.build_art_brief(
        _DRAFT,
        _HEADINGS,
        recent_styles=["saga-noir", "pixel-art"],
        recent_concept_fingerprints=["clockwork theatre|deep sea|origami futures"],
        llm=spy,
    )

    assert "Recently used rendering styles" in captured["system"]
    assert "saga-noir" in captured["system"]
    assert "Recently used creative-world fingerprints" in captured["system"]
    assert "clockwork theatre|deep sea|origami futures" in captured["system"]


def test_core_references_rotate_from_article_seed():
    class Record:
        def __init__(self, reference_id):
            self.reference_id = reference_id
            self.allowed_roles = ["layout"]

    class Catalog:
        def records_for_contract(self):
            return [Record("a"), Record("b"), Record("c")]

    assert ad._core_record_ids(Catalog(), "layout", 2, selection_seed="0000000000000001") == ["b", "c"]
    assert ad._core_record_ids(
        Catalog(), "layout", 2, selection_seed="0000000000000001", excluded_ids={"b"}
    ) == ["c", "a"]


def test_full_article_in_user_prompt():
    captured = {}
    def spy(system, user):
        captured["user"] = user
        return _good_brief_json()
    ad.build_art_brief(_DRAFT, _HEADINGS, llm=spy)
    assert "Hardware amortises" in captured["user"]
    assert "The mechanism" in captured["user"]


def test_compose_prompt_includes_shared_direction_and_labels():
    brief = {
        "style": "technical-diorama",
        "layout": "architectural cross-section",
        "text_policy": "labels",
        "text_elements": ["API", "LOCAL"],
        "palette": "slate, brass",
        "motif": "rails",
        "art_direction": "vast hall.",
        "hero_prompt": "x",
        "section_prompts": {},
    }
    p = ad.compose_prompt("two curves crossing", brief)
    assert "Technical Diorama" in p
    assert "slate, brass" in p
    assert "rails" in p
    assert "two curves crossing" in p
    assert "Text is allowed" in p
    assert "API" in p and "LOCAL" in p
    assert "Creative Concept Direction" in p


def test_fallback_brief_picks_unused_style():
    recent = ["technical-diorama", "mythic-tech-codex"]
    brief = ad.fallback_brief(_DRAFT, _HEADINGS, recent_styles=recent)
    assert brief["style"] in ad.STYLE_IDS
    assert brief["style"] not in recent
    assert brief["selection_seed"] == ad._selection_seed(_DRAFT)
    assert brief["hero_prompt"]
    assert set(brief["section_prompts"].keys()) == set(_HEADINGS)


def test_section_prompts_mapped_by_order_tolerant_of_drift():
    payload = json.loads(_good_brief_json())
    payload["section_prompts"] = []
    brief = ad.build_art_brief(
        _DRAFT, ["The mechanism"], llm=lambda s, u: json.dumps(payload))
    assert brief is not None
    # Legacy prompt ordering is ignored once the candidate plan is present;
    # the supporting scene is compiled from the selected source-grounded world.
    assert "Stagehands" in brief["section_prompts"]["The mechanism"]


def test_style_catalogue_includes_missing_workflow_and_design_modes():
    expected = {
        "baoyu-article-illustrator",
        "baoyu-infographic",
        "baoyu-comic",
        "technical-diorama",
        "data-atlas",
        "typographic-poster-design",
        "vintage-print-atelier",
        "photographic-realism",
        "pixel-art",
        "signal-hud",
    }
    assert expected.issubset(ad.STYLE_IDS)


def test_art_director_system_prompt_requires_article_grounded_concept_candidates():
    captured = {}
    def spy(system, user):
        captured["system"] = system
        return _good_brief_json()

    ad.build_art_brief(_DRAFT, _HEADINGS, llm=spy)
    system = captured["system"]
    assert "controlled conceptual variation" in system
    assert "randomness is only the translation" in system
    assert "cinematic scene, comic, infographic" in system
    assert "inspiration_category" in system
    assert "systems-as-worlds" in system
    assert "Available rendering styles" in system
    assert "technical-diorama" in system


def test_text_capable_styles_do_not_inherit_blanket_text_ban():
    brief = {
        "style": "typographic-poster-design",
        "layout": "Swiss grid poster",
        "text_policy": "typography",
        "text_elements": ["EVALS BEFORE VIBES"],
        "palette": "black, cream, signal red",
        "motif": "red proof stamp",
        "art_direction": "poster-grade hierarchy.",
        "hero_prompt": "x",
        "section_prompts": {},
    }
    p = ad.compose_prompt("a poster built from the title phrase", brief)
    assert "Typography is the image" in p
    assert "EVALS BEFORE VIBES" in p
    assert "No text" not in p


def test_no_text_styles_still_ban_readable_text():
    brief = {
        "style": "photographic-realism",
        "layout": "documentary still",
        "text_policy": "none",
        "text_elements": ["SHOULD NOT APPEAR"],
        "palette": "muted office grey",
        "motif": "paper audit trail",
        "art_direction": "grounded.",
        "hero_prompt": "x",
        "section_prompts": {},
    }
    p = ad.compose_prompt("a real office scene", brief)
    assert "No readable text" in p
    assert "SHOULD NOT APPEAR" not in p



def test_system_prompt_requests_shared_world_candidates_with_bound_scenes():
    captured = {}
    def spy(system, user):
        captured["system"] = system
        return _good_brief_json()

    ad.build_art_brief(_DRAFT, _HEADINGS, llm=spy)
    system = captured["system"]
    assert "3-5 candidate shared worlds" in system
    assert "concept_candidates" in system
    assert "source_target_id" in system


def test_article_fit_style_is_retained_for_standard_articles():
    draft = {
        "title": "Bank of England reviews agentic AI rules for finance",
        "description": "Regulators test autonomous agents in financial markets",
        "body_md": "bank rules finance risk governance agentic ai capital controls",
        "stream": "pm",
    }
    payload = {
        "style": "mythic-tech-codex",
        "layout": "annotated plate",
        "text_policy": "labels",
        "text_elements": ["RISK"],
        "palette": "black, brass, red",
        "motif": "regulatory seal",
        "art_direction": "dense finance control system.",
        "hero_prompt": "finance agents inside a regulatory machine",
        "section_prompts": [],
        "concept_candidates": [{
            "candidate_id": "regulatory-courtroom",
            "world": "A brass courtroom where automated agents present evidence to a living ledger.",
            "fingerprint": ["courtroom", "brass ledger", "paper automata"],
            "creative_score": 92,
            "inspiration_category": "systems-as-worlds",
            "relevance_rationale": "The only scene binds to the article thesis target.",
            "scenes": [{
                "asset_key": "hero",
                "source_target_id": "hero",
                "source_claim": "Regulators test autonomous agents in financial markets.",
                "visual_translation": "Risk controls become visible evidence checks.",
                "scene": "A paper automaton presents a sealed risk ledger before a brass judge.",
                "composition": "asymmetrical courtroom reveal",
                "creative_technique": "puppet-master",
            }],
        }],
    }

    brief = ad.build_art_brief(draft, [], llm=lambda s, u: json.dumps(payload))

    assert brief is not None
    assert brief["style"] == "mythic-tech-codex"
    assert brief["style_candidates"][0] == "mythic-tech-codex"


def test_compose_prompt_enforces_the_assigned_asset_layout_not_all_variants():
    brief = {
        "style": "baoyu-infographic",
        "layout": "bento grid",
        "layout_variants": ["bento grid", "comparison matrix"],
        "style_native_compiler": ad._native_compiler_for("baoyu-infographic"),
        "text_policy": "labels",
        "text_elements": ["COST", "RISK"],
        "palette": "cream, teal, black",
        "motif": "numbered cards",
        "art_direction": "dense editorial information design.",
        "hero_prompt": "x",
        "section_prompts": {},
    }
    prompt = ad.compose_prompt("explain agent economics", brief, assigned_layout="side-on process cutaway")
    assert "Style-native prompt redraft rule" in prompt
    assert "dense infographic" in prompt
    assert "Assigned layout/composition for this asset: side-on process cutaway" in prompt
    assert "comparison matrix" not in prompt



def test_text_policy_normalisation_respects_style_defaults():
    assert ad._normalise_text_policy("ink-ember-studio", "labels") == "none"
    assert ad._normalise_text_policy("photographic-realism", "typography") == "none"
    assert ad._normalise_text_policy("typographic-poster-design", "labels") == "typography"
    assert ad._normalise_text_policy("baoyu-infographic", "none") == "labels"


def test_art_brief_compiles_a_grounded_shared_world_into_distinct_asset_layouts():
    payload = json.loads(_good_brief_json())
    payload["concept_candidates"] = [{
        "candidate_id": "clockwork-theatre",
        "world": "A travelling clockwork theatre where futures are performed before they happen.",
        "fingerprint": ["clockwork theatre", "origami futures", "deep sea"],
        "creative_score": 92,
        "inspiration_category": "systems-as-worlds",
        "relevance_rationale": "The hero shows prediction and verification; the supporting scene shows hardware amortisation.",
        "scenes": [
            {
                "asset_key": "hero",
                "source_target_id": "hero",
                "source_claim": "The crossover point compares API cost with hardware amortisation.",
                "visual_translation": "Two physical routes race toward the same answer, one rented and one owned.",
                "scene": "A brass courier races across the stage with rented tickets while a permanent engine wakes below.",
                "composition": "wide proscenium collision",
                "creative_technique": "temporal-collision",
            },
            {
                "asset_key": "section-01",
                "source_target_id": "section-01",
                "source_claim": "Hardware amortises.",
                "visual_translation": "A durable machine pays for itself through repeated use.",
                "scene": "Stagehands feed the same brass engine through repeated performances until its gears glow warm.",
                "composition": "side-on process cutaway",
                "creative_technique": "assembly-line",
            },
        ],
    }]

    brief = ad.build_art_brief(_DRAFT, _HEADINGS, llm=lambda s, u: json.dumps(payload))

    assert brief is not None
    assert brief["concept_plan"]["world"].startswith("A travelling clockwork theatre")
    assert brief["style"] == "technical-diorama"
    assert brief["asset_layouts"] == {
        "hero": "wide proscenium collision",
        "section-01": "side-on process cutaway",
    }
    assert "brass courier" in brief["hero_prompt"]
    assert "Stagehands" in brief["section_prompts"]["The mechanism"]
