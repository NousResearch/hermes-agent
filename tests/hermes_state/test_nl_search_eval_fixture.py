"""Contract checks for the privacy-safe, pack-aware NL evaluation corpus."""

import json
from pathlib import Path

from scripts.nl_search_eval import resolve_packs, select_cases


CORPUS = Path(__file__).parent / "fixtures" / "nl_search_eval_v1.json"


def _corpus():
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_eval_corpus_is_versioned_private_safe_and_multilingual():
    corpus = _corpus()
    assert corpus["version"] == 2
    mechanism = corpus["cases"]
    adversarial = corpus["adversarial_cases"]
    assert len(mechanism) >= 45
    assert len(adversarial) >= 10
    ids = [case["id"] for case in mechanism + adversarial]
    assert len(ids) == len(set(ids))
    languages = {case["lang"] for case in mechanism}
    assert len(languages) == 15
    for case in mechanism:
        assert set(case) == {"id", "lang", "packs", "scenario", "query", "target"}
        assert case["scenario"] == "morphology_stopwords"
        assert set(case["packs"]).issubset(languages)
    for case in adversarial:
        assert {"id", "lang", "packs", "scenario", "query", "relevant", "distractors"} <= set(case)
        assert case["scenario"] in {
            "exact_hit", "adjacent_turns", "absent", "fts_syntax", "lexical_near_miss",
            "routing", "cross_language",
        }
    for case in mechanism + adversarial:
        payload = " ".join(str(value) for value in case.values()).lower()
        assert not any(marker in payload for marker in ("http://", "https://", "@"))


def test_pack_selection_makes_each_stacked_layer_self_validating():
    corpus = _corpus()
    core = select_cases(corpus, {"default"})
    latin = select_cases(corpus, {"default", "es", "fr", "de", "pt", "it"})
    full = select_cases(
        corpus,
        {"default", "es", "fr", "de", "pt", "it", "ru", "be", "uk", "sr", "bg", "mk", "hr", "cs", "sk"},
    )
    assert core and len(core) < len(latin) < len(full)
    assert {case["lang"] for case in core} == {"default"}
    assert {"es", "fr", "de", "pt", "it"}.issubset({case["lang"] for case in latin})
    assert len({case["lang"] for case in full}) == 15


def test_eval_runner_is_portable_and_reports_relevance_metrics():
    runner = Path(__file__).parents[2] / "scripts" / "nl_search_eval.py"
    text = runner.read_text(encoding="utf-8")
    assert "TemporaryDirectory" in text
    assert "natural_language=natural_language" in text
    assert "precision_at_5" in text and "latency_p95_ms" in text
    assert "absent_query_accuracy" in text and "--packs" in text
    assert "SessionDB" in text and "state.db" in text
    assert "/home/" not in text
    assert "db_path=Path(\"/" not in text
    assert "http://" not in text and "https://" not in text
    assert "HERMES_SEARCH_NL_ROUTE_LOG" not in text
    assert "default" in resolve_packs("all")
    try:
        resolve_packs("does-not-exist")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown pack must fail fast")
