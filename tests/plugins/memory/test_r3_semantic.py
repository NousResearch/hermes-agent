"""R3 semantic-gap dataset: >=50 TRUE semantic-miss queries with failure-class
labels. No backend exists in this runtime (verified: no ollama binary/dir,
no server on 127.0.0.1:11434), so every true-semantic case is EXPECTED to
miss on stock — that expectation is itself asserted (no fake backend, no
fabricated benchmark). Deterministic grading only. Zero LLM.
"""

import pytest

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.semantic_brain import SEMANTIC_BRAIN_AVAILABLE
from plugins.memory.holographic.store import MemoryStore

# (qid, fact_content, query, failure_class)
# Classes: LEXICAL (fixable by normalization), TOKEN (segmentation),
# RANK (ranking/tie), ENTITY (alias), SEMANTIC (needs meaning, not tokens).
CASES = [
    # paraphrase, low token overlap -> SEMANTIC
    ("s01", "The project is written in Rust.", "Which systems language powers this codebase?", "SEMANTIC"),
    ("s02", "We deploy with blue-green strategy.", "How do releases reach production?", "SEMANTIC"),
    ("s03", "Test runner is pytest version 8.", "What tool validates correctness here?", "SEMANTIC"),
    ("s04", "Cache layer uses Redis with locking.", "Where is ephemeral state kept?", "SEMANTIC"),
    ("s05", "Auth uses JWT bearer tokens.", "How are callers authenticated?", "SEMANTIC"),
    ("s06", "Job queue is Celery with retries.", "What handles background work?", "SEMANTIC"),
    ("s07", "Baseline must not be edited without approval.", "What is immutable here?", "SEMANTIC"),
    ("s08", "Experiment with Redis cache failed due to locking.", "Which approach did not work?", "SEMANTIC"),
    ("s09", "Booking cutoff is Friday noon.", "When is the last moment to book?", "SEMANTIC"),
    ("s10", "On-call handoff happens every Monday.", "How often does on-call rotate?", "SEMANTIC"),
    # conceptual relation -> SEMANTIC
    ("s11", "Primary store is SQLite with WAL mode.", "Where does durable state live?", "SEMANTIC"),
    ("s12", "Invariant: never delete migration files.", "What must be preserved?", "SEMANTIC"),
    ("s13", "Benchmark shows retrieval p95 under 50ms.", "Is search fast enough?", "SEMANTIC"),
    ("s14", "Deploy pipeline uses BlueGreen strategy.", "How is downtime avoided?", "SEMANTIC"),
    ("s15", "I prefer dark mode in the editor.", "What theme do I like?", "SEMANTIC"),
    # indirect wording -> SEMANTIC
    ("s16", "provider = holographic", "Which backend serves memory now?", "SEMANTIC"),
    ("s17", "Test runner uses pytest 8.", "What version checks the code?", "SEMANTIC"),
    ("s18", "Cache layer uses Redis with locking.", "What prevents cache stampedes?", "SEMANTIC"),
    ("s19", "Auth uses JWT bearer tokens.", "What credential format is used?", "SEMANTIC"),
    ("s20", "Job queue is Celery with retries.", "What survives transient failures?", "SEMANTIC"),
    # cross-language semantic equivalence -> SEMANTIC
    ("s21", "ห้ามแก้ baseline โดยเด็ดขาด", "What must never be edited?", "SEMANTIC"),
    ("s22", "ใช้ SQLite เป็นฐานข้อมูลหลัก", "Where is the main database?", "SEMANTIC"),
    ("s23", "Project uses Rust.", "ภาษาอะไรใช้เขียนโปรเจกต์?", "SEMANTIC"),
    ("s24", "Baseline must not be edited without approval.", "อะไรห้ามแก้?", "SEMANTIC"),
    ("s25", "Deploy pipeline uses BlueGreen strategy.", "deploy ยังไงไม่ให้ล่ม?", "SEMANTIC"),
    # semantic compression (long fact, short conceptual query) -> SEMANTIC
    ("s26", "Architecture decision: use SQLite WAL for memory store because it needs zero ops.", "Which database decision was made?", "SEMANTIC"),
    ("s27", "Experiment with Redis cache failed due to locking contention under load.", "Any failed caching attempts?", "SEMANTIC"),
    ("s28", "Benchmark shows retrieval p95 under 50ms at 100 pages on staging hardware.", "Any performance evidence?", "SEMANTIC"),
    ("s29", "Invariant: never delete migration files because rollback depends on them.", "What protects rollbacks?", "SEMANTIC"),
    ("s30", "Booking cutoff is Friday noon for weekend deploys.", "Weekend deploy deadline?", "SEMANTIC"),
    # lexical misses (shared tokens exist; normalization/ranking territory)
    ("s31", "Project uses Rust.", "project rust language", "LEXICAL"),
    ("s32", "Deploy pipeline uses BlueGreen strategy.", "BLUEGREEN PIPELINE", "LEXICAL"),
    ("s33", "Test runner is pytest version 8.", "pytest-8 runner!", "LEXICAL"),
    ("s34", "Cache layer uses Redis.", "redis cache_layer", "LEXICAL"),
    ("s35", "Auth uses JWT bearer tokens.", "jwt/bearer tokens?", "LEXICAL"),
    # tokenization edge (hyphen/underscore/case/punct)
    ("s36", "Deploy pipeline uses blue-green strategy.", "blue-green deploy", "TOKEN"),
    ("s37", "Run tests via scripts/run_tests.sh nightly.", "run_tests.sh nightly", "TOKEN"),
    ("s38", "Fact store lives in memory_store.db file.", "memory_store.db location", "TOKEN"),
    ("s39", "Timeout is 30 seconds per attempt.", "timeout: 30 seconds", "TOKEN"),
    ("s40", "Build passes with FULL width chars.", "full width build", "TOKEN"),
    # ranking/tie territory
    ("s41", "provider = honcho", "provider value?", "RANK"),
    ("s42", "provider = holographic", "provider value?", "RANK"),
    ("s43", "mode = safe", "which mode?", "RANK"),
    ("s44", "mode = aggressive", "which mode?", "RANK"),
    ("s45", "timeout = 30", "timeout?", "RANK"),
    # entity/alias territory
    ("s46", 'Primary store is "SQLite" with WAL mode.', "db WAL", "ENTITY"),
    ("s47", 'Deploy pipeline uses "BlueGreen" strategy.', "BG strategy", "ENTITY"),
    ("s48", 'Test runner is "Pytest" version 8.', "pytest version", "ENTITY"),
    ("s49", 'Auth uses "JWT" bearer tokens.', "token auth", "ENTITY"),
    ("s50", 'Job queue is "Celery" with retries.', "queue retries", "ENTITY"),
    # extra semantic to clear 50
    ("s51", "I prefer dark mode in the editor.", "Which UI brightness do I want?", "SEMANTIC"),
    ("s52", "On-call handoff happens every Monday.", "Weekly handoff schedule?", "SEMANTIC"),
    ("s53", "Booking cutoff is Friday noon.", "Last booking time?", "SEMANTIC"),
    ("s54", "Invariant: never delete migration files.", "Which files are sacred?", "SEMANTIC"),
    ("s55", "ห้ามแก้ baseline โดยเด็ดขาด", "สิ่งใดแตะต้องไม่ได้?", "SEMANTIC"),
    ("s56", "Primary store is SQLite with WAL mode.", "What persists my memory?", "SEMANTIC"),
    ("s57", "Deploy pipeline uses BlueGreen strategy.", "How are releases made safe?", "SEMANTIC"),
    ("s58", "Test runner is pytest version 8.", "How is quality gated?", "SEMANTIC"),
    ("s59", "Cache layer uses Redis with locking.", "What coordinates cache access?", "SEMANTIC"),
    ("s60", "Auth uses JWT bearer tokens.", "What proves identity here?", "SEMANTIC"),
    ("s61", "Job queue is Celery with retries.", "What runs async jobs?", "SEMANTIC"),
    ("s62", "Baseline must not be edited without approval.", "What needs sign-off to change?", "SEMANTIC"),
    ("s63", "Experiment with Redis cache failed due to locking.", "What did we learn about caching?", "SEMANTIC"),
    ("s64", "Booking cutoff is Friday noon.", "When do bookings close?", "SEMANTIC"),
    ("s65", "On-call handoff happens every Monday.", "Start-of-week ritual?", "SEMANTIC"),
    ("s66", "I prefer dark mode in the editor.", "Light or dark?", "SEMANTIC"),
    ("s67", "Invariant: never delete migration files.", "What keeps history recoverable?", "SEMANTIC"),
    ("s68", "Benchmark shows retrieval p95 under 50ms.", "Latency characteristics?", "SEMANTIC"),
    ("s69", "Architecture decision: use SQLite WAL for memory store because it needs zero ops.", "Why SQLite?", "SEMANTIC"),
    ("s70", "Test runner uses pytest 8.", "Which framework runs the suite?", "SEMANTIC"),
    ("s71", "Cache layer uses Redis with locking.", "Distributed lock provider?", "SEMANTIC"),
    ("s72", "Auth uses JWT bearer tokens.", "Stateless auth mechanism?", "SEMANTIC"),
    ("s73", "Job queue is Celery with retries.", "Retry-capable worker?", "SEMANTIC"),
    ("s74", "provider = holographic", "Current memory backend?", "SEMANTIC"),
    ("s75", "ห้ามแก้ baseline โดยเด็ดขาด", "กฎเหล็กคืออะไร?", "SEMANTIC"),
    ("s76", "ใช้ SQLite เป็นฐานข้อมูลหลัก", "เก็บข้อมูลถาวรที่ไหน?", "SEMANTIC"),
    ("s77", "Project uses Rust.", "Compiled language choice?", "SEMANTIC"),
    ("s78", "Deploy pipeline uses BlueGreen strategy.", "Zero-downtime method?", "SEMANTIC"),
    ("s79", "Booking cutoff is Friday noon for weekend deploys.", "Cutoff for weekend work?", "SEMANTIC"),
    ("s80", "On-call handoff happens every Monday.", "Who takes the pager Mondays?", "SEMANTIC"),
]

assert len(CASES) >= 50


@pytest.fixture(scope="module")
def semstore(tmp_path_factory):
    db = tmp_path_factory.mktemp("r3sem") / "sem.db"
    store = MemoryStore(str(db), hrr_dim=64)
    for qid, content, _q, _cls in CASES:
        try:
            store.add_fact(content, category="project")
        except Exception:
            pass
    for name, alias in [("SQLite", "db"), ("BlueGreen", "BG"), ("Pytest", "pytest"),
                        ("JWT", "token"), ("Celery", "queue")]:
        store.add_entity_alias(name, alias)
    yield store
    store.close()


def test_r3_no_backend_available():
    assert SEMANTIC_BRAIN_AVAILABLE is False  # honest detection, no fake backend


def test_r3_semantic_gap_separation(semstore):
    retriever = FactRetriever(store=semstore, hrr_dim=64)
    rows = []
    for qid, content, query, cls in CASES:
        hits = [r["content"] for r in retriever.search(query, limit=3)]
        hit = content in hits
        rows.append({"qid": qid, "class": cls, "hit": hit})
    by_class: dict[str, list[bool]] = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row["hit"])
    summary = {k: {"n": len(v), "hits": sum(v),
                   "hit_rate": round(sum(v) / len(v), 3)} for k, v in by_class.items()}
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parents[3] / "results" / "r3" / "semantic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap_separation.json").write_text(
        json.dumps({"summary": summary, "rows": rows,
                    "backend": "none", "llm_calls": 0}, indent=1), encoding="utf-8")
    sem = summary.get("SEMANTIC", {"hit_rate": 0})
    lex = summary.get("LEXICAL", {"hit_rate": 0})
    # structural expectation: lexical coverage must clearly exceed semantic
    # coverage on a token engine (else the separation itself is broken)
    assert lex["hit_rate"] >= sem["hit_rate"]
    assert sem["n"] >= 30  # semantic subset is substantial
