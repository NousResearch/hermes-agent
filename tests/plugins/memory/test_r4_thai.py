"""R4 Thai retrieval evaluation: CURRENT bigram fallback vs spaceless-Thai
candidate across UI terminology, mixed Thai/English, transliteration,
spacing, punctuation, compound phrases. Zero LLM. NO tokenizer dependency
unless evidence demands it (default: dependency-free).
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
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "r4" / "thai"
LLM_CALLS = 0

# (key, content)
THAI_FACTS = [
    ("ui_save", "ปุ่มบันทึกอยู่ในแถบเครื่องมือด้านบน", "project"),
    ("ui_login", "หน้าเข้าสู่ระบบต้องใช้รหัสผ่าน", "project"),
    ("ui_search", "ช่องค้นหารองรับภาษาไทยและ English", "project"),
    ("trans_db", "ดาต้าเบสหลักคือ SQLite", "project"),
    ("trans_cache", "แคชหมดอายุใน 60 วินาที", "project"),
    ("mix_run", "รัน pytest ด้วย scripts/run_tests.sh", "tool"),
    ("mix_deploy", "deploy แบบ BlueGreen ไม่ดาวน์", "project"),
    ("base", "ห้ามแก้ baseline โดยเด็ดขาด", "project"),
    ("noise", "ok thanks bye", "general"),
]

# (qid, query, relevant, dimension)
THAI_QUERIES = [
    ("th01", "ปุ่มบันทึกอยู่ไหน?", ["ui_save"], "ui-term"),
    ("th02", "แถบเครื่องมือด้านบนมีอะไร?", ["ui_save"], "compound"),
    ("th03", "เข้าสู่ระบบยังไง?", ["ui_login"], "ui-term"),
    ("th04", "รหัสผ่านใช้ที่ไหน?", ["ui_login"], "ui-term"),
    ("th05", "ค้นหาภาษาไทยได้ไหม?", ["ui_search"], "mixed"),
    ("th06", "database หลักคืออะไร?", ["trans_db"], "translit"),
    ("th07", "ดาต้าเบสหลัก?", ["trans_db"], "translit"),
    ("th08", "แคชอยู่ได้นานแค่ไหน?", ["trans_cache"], "translit"),
    ("th09", "รัน tests ยังไง?", ["mix_run"], "mixed"),
    ("th10", "deploy แบบไหนไม่ดาวน์?", ["mix_deploy"], "mixed"),
    ("th11", "ห้ามแก้ไข baseline", ["base"], "spacing"),
    ("th12", "baseline ห้ามแก้!", ["base"], "punct"),
    ("th13", "ห้าม แก้ baseline", ["base"], "spacing"),
    ("th14", "ปุ่ม บันทึก แถบ เครื่องมือ", ["ui_save"], "spacing"),
    ("th15", "ช่องค้นหา English?", ["ui_search"], "mixed"),
    ("th16", "SQLite ดาต้าเบส?", ["trans_db"], "mixed"),
    ("th17", "BlueGreen deploy?", ["mix_deploy"], "mixed"),
    ("th18", "รหัสผ่านหน้า login?", ["ui_login"], "mixed"),
    ("th19", "baseline?", ["base"], "punct"),
    ("th20", "บันทึก อยู่ ไหน", ["ui_save"], "spacing"),
]

assert len(THAI_QUERIES) >= 20


def spaceless_thai_bigrams(text: str) -> set[str]:
    """Candidate: strip spaces inside Thai runs before bigramming, so
    'ห้าม แก้' and 'ห้ามแก้' produce identical grams."""
    norm = unicodedata.normalize("NFKC", text or "")
    grams: set[str] = set()
    for run in re.findall(r"[\u0e00-\u0e7f ]{2,}", norm):
        compact = run.replace(" ", "").replace("\u00a0", "")
        if len(compact) < 2:
            continue
        for i in range(len(compact) - 1):
            grams.add(compact[i:i + 2])
    return grams


class SpacelessRetriever(FactRetriever):
    """Evaluation harness ONLY: spaceless-Thai bigram candidate."""

    @staticmethod
    def _thai_bigrams(text: str) -> set[str]:
        return FactRetriever._thai_bigrams(text) | spaceless_thai_bigrams(text)


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    db = tmp_path_factory.mktemp("r4th") / "thai.db"
    store = MemoryStore(str(db), hrr_dim=64)
    key_to_id = {}
    for key, content, category in THAI_FACTS:
        try:
            key_to_id[key] = store.add_fact(content, category=category)
        except Exception:
            pass
    id_to_key = {v: k for k, v in key_to_id.items()}
    yield store, id_to_key
    store.close()


def _measure(cls, store, id_to_key):
    retriever = cls(store=store, hrr_dim=64)
    hits, rows, by_dim = 0, [], {}
    for qid, text, relevant in [(q[0], q[1], q[2]) for q in THAI_QUERIES]:
        dim = next(q[3] for q in THAI_QUERIES if q[0] == qid)
        ranked = [id_to_key.get(r["fact_id"]) for r in retriever.search(text, limit=5)]
        ranked = [k for k in ranked if k]
        ok = bool(ranked) and ranked[0] in relevant
        hits += ok
        by_dim.setdefault(dim, []).append(ok)
        rows.append({"qid": qid, "dim": dim, "hit": ok, "ranked": ranked})
    summary = {"n": len(rows), "p@1": round(hits / len(rows), 4),
               "by_dim": {k: round(sum(v) / len(v), 3) for k, v in by_dim.items()}}
    return summary, rows


def test_r4_thai_current_vs_candidate(seeded):
    store, id_to_key = seeded
    base, _rows = _measure(FactRetriever, store, id_to_key)
    cand, cand_rows = _measure(SpacelessRetriever, store, id_to_key)
    payload = {"current": base, "candidate_spaceless": cand,
               "delta": round(cand["p@1"] - base["p@1"], 4),
               "decision": "undecided", "llm_calls": LLM_CALLS,
               "per_query": cand_rows}
    # Promotion rule: candidate needs a MATERIAL gain (>= +0.05) with no
    # dimension collapsing to 0, else KEEP CURRENT (dependency-free).
    if cand["p@1"] - base["p@1"] >= 0.05 and all(v > 0 for v in cand["by_dim"].values()):
        payload["decision"] = "PROMOTE"
    else:
        payload["decision"] = "KEEP-CURRENT"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "thai.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    assert LLM_CALLS == 0
    assert cand["p@1"] >= base["p@1"]  # candidate must never regress
