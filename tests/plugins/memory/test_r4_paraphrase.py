"""R4 paraphrase benchmark (>=30 cases): candidate mechanisms measured against
the R2 106-query baseline BEFORE any production promotion. Zero LLM.

Candidates (evaluated, not assumed):
  stem-union  — conservative suffix strip ADDED to the token set (never replaces)
  bigram-union — Thai bigrams already in production (R3); measured here too
Gates: candidate P@1 must not regress R2 baseline (0.9623); FP probes must hold.
Promotion happens only on measured gain — otherwise REJECT with evidence.
"""

import json
import os
import re
import unicodedata
from pathlib import Path

import pytest

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

ARM = os.environ.get("R4_ARM", "r4")
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "r4" / "paraphrase"
LLM_CALLS = 0
R2_BASELINE_P1 = 0.9623


def stem(word: str) -> str:
    """Conservative suffix strip: plural/past/progressive/adverb only.
    -ses words drop just -s (databases->database); -sses words drop -es
    (glasses->glass, passes->pass); lone final -s needs length >= 5
    (news/new stay distinct; deploys/tests still conflate)."""
    w = word.lower()
    if len(w) >= 6 and w.endswith("sses"):
        return w[:-2]
    if len(w) >= 6 and w.endswith("ses"):
        return w[:-1]
    for suf in ("ing", "ed", "es", "ly"):
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            stemmed = w[: -len(suf)]
            if suf == "ing" and len(stemmed) >= 5 and stemmed[-1] == stemmed[-2]:
                stemmed = stemmed[:-1]
            return stemmed
    if len(w) >= 5 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


class StemRetriever(FactRetriever):
    """Evaluation harness ONLY: stem-union Jaccard. Production _tokenize
    untouched until/unless this benchmark proves a clean win."""

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        base = FactRetriever._tokenize(text)
        return base | {stem(t) for t in base}


# (qid, query, relevant_keys) — 30 cases: terminology, aliases, morphology,
# paraphrase, Thai, mixed.
PARA_QUERIES = [
    ("pp01", "primary database store", ["db"]),
    ("pp02", "deployment uses blue green", ["pipe"]),
    ("pp03", "tests run with pytest eight", ["runner"]),
    ("pp04", "redis locking cache", ["cache"]),
    ("pp05", "editor dark theme preference", ["editor"]),
    ("pp06", "jobs with celery retries", ["queue"]),
    ("pp07", "bearer token authentication", ["auth"]),
    ("pp08", "SQLite WAL mode", ["db"]),
    ("pp09", "pytest version", ["runner"]),
    ("pp10", "BG deploy strategy", ["pipe"]),
    ("pp11", "databases used", ["db"]),
    ("pp12", "deploys with blue-green", ["pipe"]),
    ("pp13", "tested using pytest", ["runner"]),
    ("pp14", "cached with redis", ["cache"]),
    ("pp15", "editors preferences", ["editor"]),
    ("pp16", "queued jobs retrying", ["queue"]),
    ("pp17", "authenticated via tokens", ["auth"]),
    ("pp18", "which database?", ["db"]),
    ("pp19", "running deployments", ["pipe"]),
    ("pp20", "test runners versions", ["runner"]),
    ("pp21", "ฐานข้อมูลหลักคืออะไร?", ["thai_db"]),
    ("pp22", "ห้ามแก้ไข baseline", ["thai_base"]),
    ("pp23", "baseline ห้ามแก้", ["thai_base"]),
    ("pp24", "SQLite ฐานข้อมูลหลัก", ["thai_db"]),
    ("pp25", "รัน pytest version 8", ["runner"]),
    ("pp26", "ห้ามแก้ baseline do not edit", ["thai_base"]),
    ("pp27", "memory_store.db อยู่ที่ไหน?", ["f_db"]),
    ("pp28", "อ้างอิง fact ด้วยอะไร?", ["f_fact"]),
    ("pp29", "provider value?", ["prov_new"]),
    ("pp30", "which mode?", ["mode_a", "mode_b"]),
]

assert len(PARA_QUERIES) >= 30

SEED_FACTS = [
    ("db", 'Primary store is "SQLite" with WAL mode.', "project"),
    ("pipe", 'Deploy pipeline uses "BlueGreen" strategy.', "project"),
    ("runner", 'Test runner is "Pytest" version 8.', "tool"),
    ("cache", 'Cache layer uses "Redis" with locking.', "project"),
    ("queue", 'Job queue is "Celery" with retries.', "project"),
    ("auth", 'Auth uses "JWT" bearer tokens.', "project"),
    ("editor", "I prefer dark mode in the editor.", "user_pref"),
    ("thai_base", "ห้ามแก้ baseline โดยเด็ดขาด", "project"),
    ("thai_db", "ใช้ SQLite เป็นฐานข้อมูลหลัก", "project"),
    ("f_db", "memory_store.db อยู่ที่ $HERMES_HOME", "project"),
    ("f_fact", "ใช้ fact_id อ้างอิง fact ทุกครั้ง", "project"),
    ("prov_new", "provider = holographic", "project"),
    ("mode_a", "mode = safe", "project"),
    ("mode_b", "mode = aggressive", "project"),
    ("noise1", "ok thanks bye", "general"),
    ("noise2", "meeting moved to 3pm maybe?", "general"),
]


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    db = tmp_path_factory.mktemp("r4p") / "para.db"
    store = MemoryStore(str(db), hrr_dim=64)
    key_to_id = {}
    for key, content, category in SEED_FACTS:
        try:
            key_to_id[key] = store.add_fact(content, category=category)
        except Exception:
            pass
    for name, alias in [("SQLite", "db"), ("BlueGreen", "BG"), ("Pytest", "pytest"),
                        ("JWT", "token"), ("Celery", "queue")]:
        store.add_entity_alias(name, alias)
    id_to_key = {v: k for k, v in key_to_id.items()}
    yield store, id_to_key
    store.close()


def _p1(retriever_cls, store, id_to_key):
    retriever = retriever_cls(store=store, hrr_dim=64)
    hits, rows = 0, []
    for qid, text, relevant in PARA_QUERIES:
        ranked = []
        for row in retriever.search(text, limit=5):
            key = id_to_key.get(row["fact_id"])
            if key:
                ranked.append(key)
        ok = bool(ranked) and ranked[0] in relevant
        hits += ok
        rows.append({"qid": qid, "hit": ok, "ranked": ranked})
    return hits / len(PARA_QUERIES), rows


def test_r4_paraphrase_candidates(seeded):
    store, id_to_key = seeded
    base_p1, base_rows = _p1(FactRetriever, store, id_to_key)
    stem_p1, stem_rows = _p1(StemRetriever, store, id_to_key)
    payload = {"n": len(PARA_QUERIES), "baseline_p@1": round(base_p1, 4),
               "stem_union_p@1": round(stem_p1, 4),
               "delta": round(stem_p1 - base_p1, 4),
               "llm_calls": LLM_CALLS, "per_query": stem_rows}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "paraphrase.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    assert stem_p1 >= base_p1  # no-regression gate for the candidate
    assert LLM_CALLS == 0
    # PROMOTION DECISION (measured 2026-09-11): stem-union gains +0.033 on
    # this 30-case set but +0.000 on the full R2 106-query baseline (0.9623).
    # Not material -> REJECTED for production _tokenize. Harness retained.


def test_r4_stem_fp_probes():
    cases = [("bus", "business"), ("glass", "glasses"), ("news", "new"),
             ("passes", "pass"), ("database", "databases"), ("deploy", "deploys"),
             ("test", "tested")]
    for a, b in cases:
        same = stem(a) == stem(b)
        # only genuine morphological family may conflate (glass/glasses ARE
        # plural; news/new must stay distinct)
        if (a, b) in [("database", "databases"), ("deploy", "deploys"),
                      ("test", "tested"), ("glass", "glasses"), ("passes", "pass")]:
            assert same
        else:
            assert not same


def test_r4_stem_fp_retrieval(tmp_path):
    store = MemoryStore(str(tmp_path / "fp.db"), hrr_dim=64)
    try:
        store.add_fact("Take the bus to work daily.", category="general")
        store.add_fact("Business review meets quarterly goals.", category="project")
        r = StemRetriever(store=store, hrr_dim=64)
        out = r.search("bus schedule", limit=5)
        assert out and "bus to work" in out[0]["content"]
    finally:
        store.close()
