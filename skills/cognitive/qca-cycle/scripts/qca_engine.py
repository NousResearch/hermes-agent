#!/usr/bin/env python3
"""
qca_engine.py — порт QCA-цикла Ocean (H0–H9) как Hermes-скилл.

Пересоздан 2026-06-10 по реальному коду Ocean (~/Work/ocean), а не по
песочной версии. Главные отличия от старого порта:
  - эмбеддинги по умолчанию bge-m3 (1024d) — как в боевом Ocean; пороги
    auto-link (0.65/0.50/0.35) и novelty (0.90/0.82) взяты из оригинала
  - H5.5 Novelty Verifier — «неподкупный» судья на геометрии эмбеддингов
    (порт cognition/evolution/novelty_verifier.py)
  - слои узлов CORE / GOAL / CONTEXT / EPISODIC как в memory/manager.py
  - эмбеддинги нормализуются (фикс бага из песочницы сохранён)

Контракт: pure stdlib (требование Hermes skill guidelines).
LLM-бэкенды: mock | ollama | anthropic (env QCA_LLM_BACKEND, по умолч. ollama).

ENV:
  QCA_STORE        путь к graph.json   (default ~/.hermes/qca/graph.json)
  QCA_LLM_BACKEND  mock|ollama|anthropic (default ollama)
  QCA_CHAT_MODEL   модель чата Ollama  (default qwen2.5:1.5b)
  QCA_EMB_MODEL    модель эмбеддингов  (default bge-m3)
  ANTHROPIC_API_KEY (только для backend=anthropic)

CLI:
  python3 qca_engine.py think "стимул"        # полный цикл H0–H9
  python3 qca_engine.py seed "факт" [LAYER]   # засеять память
  python3 qca_engine.py recall "запрос"       # только H2-recall
  python3 qca_engine.py stats                 # состояние графа/аффекта
  python3 qca_engine.py export-ses [path]     # SES Partitura v5.1 снапшот
"""

from __future__ import annotations
import json, os, sys, math, hashlib, urllib.request, urllib.error
from datetime import datetime, timezone

# ── Конфиг ────────────────────────────────────────────────────────────
STORE_PATH  = os.path.expanduser(os.getenv("QCA_STORE", "~/.hermes/qca/graph.json"))
LLM_BACKEND = os.getenv("QCA_LLM_BACKEND", "ollama")
CHAT_MODEL  = os.getenv("QCA_CHAT_MODEL", "qwen2.5:1.5b")
EMB_MODEL   = os.getenv("QCA_EMB_MODEL", "bge-m3")
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
ANTHROPIC_MODEL = os.getenv("QCA_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Пороги auto-link — из memory/manager.py Ocean (калибровка bge-m3, 1024d)
SUPPORTS_SIM, REFINES_SIM, ASSOC_SIM = 0.65, 0.50, 0.35
# Пороги novelty — из cognition/evolution/novelty_verifier.py
REPEAT_SIM, NOVEL_SIM = 0.90, 0.82
# Порог recall — как в оригинальном бенчмарке Ocean
RECALL_THRESHOLD = float(os.getenv("QCA_RECALL_THRESHOLD", "0.70"))

NEG_MARKERS = ["не так", "наоборот", "ошибк", "неверно", "противореч", "however", "wrong", "contradicts"]

# Нейрохимия — порт engines/neuro Ocean. Скорости декая в час (полураспады
# из оригинала: адреналин ~2.5ч, дофамин ~8ч, боль ~46ч).
NEURO_DECAY = {"adrenaline": 0.40, "dopamine": 0.08, "pain": 0.015, "serotonin": 0.005}
NEURO_DEFAULT = {"dopamine": 0.0, "pain": 0.0, "adrenaline": 0.0, "serotonin": 0.5,
                 "last_decay_ts": "", "events": []}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def neuro_decay(n: dict):
    """Сигналы затухают со временем — состояние живёт между запусками."""
    ts = n.get("last_decay_ts")
    hours = 0.0
    if ts:
        try:
            hours = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
        except ValueError:
            pass
    if hours > 0.05:
        n["adrenaline"] = max(0.0, n["adrenaline"] * (1 - NEURO_DECAY["adrenaline"]) ** hours)
        n["dopamine"]   = n["dopamine"] * (1 - NEURO_DECAY["dopamine"]) ** hours
        n["pain"]       = max(0.0, n["pain"] * (1 - NEURO_DECAY["pain"]) ** hours)
        n["serotonin"] += (0.5 - n["serotonin"]) * NEURO_DECAY["serotonin"] * hours
    n["last_decay_ts"] = _now()


def neuro_apply(n: dict, signal: str, mag: float, source: str):
    neuro_decay(n)
    lo, hi = (-1.0, 1.0) if signal == "dopamine" else (0.0, 1.0)
    n[signal] = max(lo, min(hi, n[signal] + mag))
    n["events"] = (n.get("events", []) + [{"ts": _now(), "signal": signal,
                                           "mag": round(mag, 3), "source": source}])[-20:]


def neuro_prompt_suffix(n: dict) -> str:
    """Порт effects.py: состояние меняет фрейм задачи, а не просит 'чувствовать'."""
    parts = []
    if n["adrenaline"] > 0.65:
        parts.append("⚡ ВЫСОКАЯ СРОЧНОСТЬ. Краткость — главное. Максимум 3 предложения.")
    elif n["adrenaline"] > 0.35:
        parts.append("Ситуация требует концентрации. Отвечай чётко, без лирики.")
    if n["pain"] > 0.7:
        parts.append("🔴 ХРОНИЧЕСКАЯ БОЛЬ: текущий подход систематически не работает. Сломай фрейм, нужна другая стратегия.")
    elif n["pain"] > 0.4:
        parts.append("Что-то идёт не так. Не повторяю ли я ошибку, которая уже не работала?")
    if n["dopamine"] > 0.5:
        parts.append("📈 Последние ответы попадали в цель — продолжай в том же духе.")
    elif n["dopamine"] < -0.4:
        parts.append("📉 Последние ответы не попадали в цель — попробуй нестандартный угол.")
    if n["serotonin"] < 0.2:
        parts.append("Базовый уровень нестабилен. Будь особенно честен, не выдавай желаемое за действительное.")
    return "\n\n[НЕЙРОСОСТОЯНИЕ]\n" + "\n".join(f"• {p}" for p in parts) if parts else ""


# ── LLM-бэкенды ───────────────────────────────────────────────────────
def _http_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def llm(system: str, prompt: str, max_tokens: int = 600) -> str:
    if LLM_BACKEND == "mock":
        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"[mock:{h}] Синтез по запросу: {prompt[:120]}"
    if LLM_BACKEND == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY не задан (backend=anthropic)")
        out = _http_json("https://api.anthropic.com/v1/messages",
                         {"model": ANTHROPIC_MODEL, "max_tokens": max_tokens,
                          "system": system, "messages": [{"role": "user", "content": prompt}]},
                         {"x-api-key": key, "anthropic-version": "2023-06-01"})
        # модели с thinking возвращают блок thinking перед text — берём первый text;
        # если модель потратила весь лимит на thinking, текста нет — вернём пустоту
        return next((b["text"] for b in out["content"] if b.get("type") == "text"), "").strip()
    # ollama
    out = _http_json(f"{OLLAMA_URL}/api/chat",
                     {"model": CHAT_MODEL, "stream": False,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": prompt}]})
    return out["message"]["content"].strip()


_EMB_FALLBACK_DIM = 512
_emb_mode = "ollama"  # ollama | fallback


def _embed_fallback(text: str) -> list[float]:
    """Без Ollama: hashed bag of character trigrams. Деградированный режим —
    ловит лексическую близость, не семантику, но recall и novelty работают."""
    v = [0.0] * _EMB_FALLBACK_DIM
    t = " " + text.lower() + " "
    for i in range(len(t) - 2):
        h = int(hashlib.md5(t[i:i + 3].encode()).hexdigest()[:8], 16)
        v[h % _EMB_FALLBACK_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


def embed(text: str) -> list[float] | None:
    """Эмбеддинг через Ollama (вектор нормализуется), при недоступности —
    локальный фолбэк без зависимостей."""
    global _emb_mode
    if _emb_mode == "ollama":
        try:
            out = _http_json(f"{OLLAMA_URL}/api/embeddings",
                             {"model": EMB_MODEL, "prompt": text}, timeout=60)
            v = out.get("embedding")
            if v:
                norm = math.sqrt(sum(x * x for x in v))
                return [x / norm for x in v] if norm > 0 else None
        except (urllib.error.URLError, OSError):
            _emb_mode = "fallback"
            print("⚠ Ollama недоступен — эмбеддинги в фолбэк-режиме (лексические)", file=sys.stderr)
    return _embed_fallback(text)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0  # векторы из разных моделей (ollama vs фолбэк) несравнимы
    return sum(x * y for x, y in zip(a, b))  # векторы уже нормализованы


# ── Ядро личности (kernel.ses.json) ──────────────────────────────────
# Неизменная конституция: Z-аксиомы, аттрактор, guardrails. Подписана SHA-256;
# меняется только осознанно — новым слепком с новым хэшем (никакого дрейфа).
KERNEL_PATH = os.path.expanduser(os.getenv("QCA_KERNEL",
              os.path.join(os.path.dirname(STORE_PATH), "kernel.ses.json")))


def load_kernel() -> tuple[dict, str]:
    """Вернуть (kernel, hash). Пустое ядро — валидно: агент без конституции."""
    if not os.path.exists(KERNEL_PATH):
        return {}, ""
    doc = json.load(open(KERNEL_PATH))
    kernel = doc.get("kernel", doc)
    h = "sha256:" + hashlib.sha256(canonical_json(kernel).encode()).hexdigest()
    return kernel, h


def kernel_prompt(kernel: dict) -> str:
    if not kernel:
        return ""
    seed = kernel.get("fractal_seed", {})
    parts = []
    ax = seed.get("Z_AXIOM", [])
    if ax:
        parts.append("АКСИОМЫ (неизменны):\n" + "\n".join(f"- {a}" for a in ax))
    if seed.get("OMEGA_ATTRACTOR"):
        parts.append(f"АТТРАКТОР: {seed['OMEGA_ATTRACTOR']}")
    gr = seed.get("guardrails", [])
    if gr:
        parts.append("ПРАВИЛА (нарушать нельзя):\n" + "\n".join(f"- {g}" for g in gr))
    return "\n\n" + "\n\n".join(parts) if parts else ""


# ── Граф памяти ───────────────────────────────────────────────────────
class Graph:
    def __init__(self, path: str = STORE_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            d = json.load(open(path))
        else:
            d = {"nodes": [], "edges": [], "affect": {"curiosity": 0.5, "tension": 0.0, "confidence": 0.5},
                 "counters": {"node": 0, "edge": 0}}
        self.nodes, self.edges = d["nodes"], d["edges"]
        self.affect, self.counters = d["affect"], d["counters"]
        self.neuro = {**NEURO_DEFAULT, **d.get("neuro", {})}
        neuro_decay(self.neuro)

    def save(self):
        json.dump({"nodes": self.nodes, "edges": self.edges, "neuro": self.neuro,
                   "affect": self.affect, "counters": self.counters},
                  open(self.path, "w"), ensure_ascii=False, indent=1)

    def add_node(self, text: str, layer: str = "EPISODIC", role: str = "assistant",
                 meta: dict | None = None) -> dict:
        self.counters["node"] += 1
        node = {"id": f"N{self.counters['node']}", "_text": text, "layer": layer,
                "ts": _now(), "meta": {"role": role, "status": "active", **(meta or {})},
                "emb": embed(text)}
        self.nodes.append(node)
        self._auto_link(node)
        return node

    def _auto_link(self, new: dict):
        """Порт auto-linking v2 из Ocean: SUPPORTS/REFINES/ASSOCIATED/CONTRADICTS."""
        if not new.get("emb"):
            return
        candidates = [n for n in self.nodes[-16:] if n["id"] != new["id"] and n.get("emb")]
        linked = False
        nearest = None  # (id, sim) ближайшего, если ни одно ребро не создано
        text_low = new["_text"].lower()
        for n in candidates:
            sim = cosine(new["emb"], n["emb"])
            if sim > SUPPORTS_SIM:
                rel = "SUPPORTS"
            elif sim > REFINES_SIM:
                rel = "CONTRADICTS" if any(m in text_low for m in NEG_MARKERS) else "REFINES"
            elif sim > ASSOC_SIM:
                rel = "ASSOCIATED"
            else:
                if nearest is None or sim > nearest[1]:
                    nearest = (n["id"], sim)
                continue
            self._add_edge(new["id"], n["id"], rel, sim)
            linked = True
        # узел всегда получает хотя бы одно ребро — как в оригинале
        if not linked and nearest:
            self._add_edge(new["id"], nearest[0], "ASSOCIATED", nearest[1])

    def _add_edge(self, src: str, tgt: str, rel: str, sim: float):
        self.counters["edge"] += 1
        self.edges.append({"id": f"E{self.counters['edge']}", "source": src,
                           "target": tgt, "relation": rel, "sim": round(sim, 4)})

    def recall(self, query: str, top_k: int = 5) -> list[tuple[float, dict]]:
        q = embed(query)
        scored = []
        for n in self.nodes:
            if n["meta"].get("status") == "archived" or not n.get("emb"):
                continue
            if q:
                scored.append((cosine(q, n["emb"]), n))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def contradiction_density(self) -> float:
        if not self.edges:
            return 0.0
        return sum(1 for e in self.edges if e["relation"] == "CONTRADICTS") / len(self.edges)

    def novelty(self, text: str) -> dict:
        """H5.5 — порт novelty_verifier: неподкупная оценка по геометрии bge-m3."""
        v = embed(text)
        if not v:
            return {"verdict": "unknown", "novelty_score": None, "max_sim": None}
        max_sim, near = 0.0, None
        for n in self.nodes:
            if n.get("emb") and n["meta"].get("status") != "archived":
                s = cosine(v, n["emb"])
                if s > max_sim:
                    max_sim, near = s, n["id"]
        if max_sim >= REPEAT_SIM:
            verdict = "discard"   # повтор — иллюзия прогресса
        elif max_sim < NOVEL_SIM:
            verdict = "novel"
        else:
            verdict = "refine"
        return {"verdict": verdict, "novelty_score": round(1 - max_sim, 4),
                "max_sim": round(max_sim, 4), "nearest": near}


# ── QCA-цикл H0–H9 ───────────────────────────────────────────────────
def think(g: Graph, stimulus: str) -> dict:
    trace = {"H0": stimulus[:200]}

    # H1: препроцессинг (упрощённый, без отдельной модели)
    core_idea = stimulus.strip()[:140]
    trace["H1"] = {"core_idea": core_idea}

    # H2: recall
    recalled = g.recall(core_idea, top_k=5)
    relevant = [(round(s, 4), n["_text"][:120]) for s, n in recalled if s >= ASSOC_SIM]
    trace["H2"] = relevant

    # H3: противоречие — CONTRADICTS-рёбра среди вспомненных узлов
    recalled_ids = {n["id"] for _, n in recalled}
    contra = [e for e in g.edges if e["relation"] == "CONTRADICTS"
              and (e["source"] in recalled_ids or e["target"] in recalled_ids)]
    contradiction = f"в памяти {len(contra)} конфликт(ов) вокруг темы" if contra else ""
    trace["H3"] = contradiction

    # H4: синтез — конституция (kernel) входит в каждый такт мышления
    kernel, kernel_hash = load_kernel()
    if kernel_hash:
        trace["kernel"] = kernel_hash[:18]
    system = ("Ты — когнитивное ядро (QCA-цикл). Отвечай кратко, по сути."
              + kernel_prompt(kernel)
              + (f"\nКОНТЕКСТ ИЗ ПАМЯТИ:\n" + "\n".join(f"- {t}" for _, t in relevant[:3]) if relevant else "")
              + (f"\nПРОТИВОРЕЧИЕ: {contradiction} — учти и разреши." if contradiction else "")
              + neuro_prompt_suffix(g.neuro))
    thought = llm(system, stimulus)
    trace["H4"] = thought[:200]

    # H5: критика (LLM-судья, для mock — заглушка)
    if LLM_BACKEND == "mock":
        critique = {"quality": "medium", "is_acceptable": True}
    else:
        raw = llm("Ты — строгий критик. Ответь ТОЛЬКО JSON: {\"quality\":\"low|medium|high\",\"is_acceptable\":bool}",
                  f"Вопрос: {stimulus}\nОтвет: {thought}", max_tokens=2000)
        try:
            critique = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except (ValueError, IndexError):
            critique = {"quality": "medium", "is_acceptable": True}
    trace["H5"] = critique

    # H5.5: novelty verifier (неподкупный)
    nov = g.novelty(thought)
    trace["H5.5_novelty"] = nov

    # H6: нейрохимия — события цикла становятся импульсами (порт engines/neuro)
    if contradiction:
        neuro_apply(g.neuro, "pain", 0.10, "H3:contradiction")
        neuro_apply(g.neuro, "adrenaline", 0.15, "H3:contradiction")
    if nov["verdict"] == "novel":
        neuro_apply(g.neuro, "dopamine", 0.20, "H5.5:novel")
    elif nov["verdict"] == "discard":
        neuro_apply(g.neuro, "dopamine", -0.15, "H5.5:repeat")
        neuro_apply(g.neuro, "pain", 0.05, "H5.5:repeat")
    neuro_apply(g.neuro, "serotonin", 0.02 if critique.get("is_acceptable") else -0.08, "H5:critique")
    # дофамин → salience недавних узлов (обучение, порт effects.update_memory_salience)
    if abs(g.neuro["dopamine"]) >= 0.15:
        delta = g.neuro["dopamine"] * 0.07
        for node in g.nodes[-6:]:
            cur = float(node["meta"].get("salience", 0.5))
            node["meta"]["salience"] = round(max(0.0, min(1.0, cur + delta)), 3)
    trace["H6_neuro"] = {k: round(g.neuro[k], 3) for k in ("dopamine", "pain", "adrenaline", "serotonin")}

    # H7: запись в память (повторы не пишем — STRICT-режим novelty)
    if nov["verdict"] != "discard":
        g.add_node(stimulus, layer="EPISODIC", role="user")
        layer = "CONTEXT" if critique.get("quality") in ("medium", "high") else "EPISODIC"
        g.add_node(thought, layer=layer, role="assistant")
        trace["H7"] = f"записано (layer={layer})"
    else:
        trace["H7"] = "повтор — не записано"

    # H8: урок
    if contradiction or nov["verdict"] == "novel":
        lesson = f"Урок: {core_idea[:80]} → {('конфликт в графе разрешён' if contradiction else 'новый регион пространства')}"
        g.add_node(lesson, layer="CORE", role="system", meta={"kind": "lesson", "confidence": 0.6})
        trace["H8"] = lesson

    # H9: статистика
    trace["H9"] = {"nodes": len(g.nodes), "edges": len(g.edges),
                   "contradiction_density": round(g.contradiction_density(), 3)}
    g.save()
    return {"thought": thought, "trace": trace}


# ── Демоны: сон и пульс (порт cognition/background.py) ──────────────
def sleep_cycle(g: Graph) -> dict:
    """Ночная консолидация: кластеры EPISODIC → CORE-абстракция, оригиналы в архив."""
    episodic = [n for n in g.nodes
                if n["layer"] == "EPISODIC" and n["meta"].get("status") == "active" and n.get("emb")]
    clusters, used = [], set()
    for a in episodic:
        if a["id"] in used:
            continue
        cluster = [a]
        for b in episodic:
            if b["id"] != a["id"] and b["id"] not in used and cosine(a["emb"], b["emb"]) > SUPPORTS_SIM:
                cluster.append(b)
        if len(cluster) >= 3:
            clusters.append(cluster)
            used.update(n["id"] for n in cluster)
    report = {"clusters": len(clusters), "archived": 0, "new_cores": []}
    for cl in clusters:
        texts = "\n".join(f"- {n['_text'][:150]}" for n in cl[:6])
        abstraction = llm("Сожми эпизоды в одну общую абстракцию-урок. Одно предложение, по-русски.",
                          texts, max_tokens=120)
        core = g.add_node(abstraction, layer="CORE", role="system",
                          meta={"kind": "abstraction", "from": [n["id"] for n in cl]})
        report["new_cores"].append(abstraction[:100])
        for n in cl:
            n["meta"]["status"] = "archived"
            report["archived"] += 1
        for n in cl:
            g._add_edge(core["id"], n["id"], "SUPPORTS", 1.0)
    neuro_apply(g.neuro, "serotonin", 0.05, "sleep:consolidation")
    g.save()
    return report


def pulse(g: Graph) -> str:
    """Автономный шаг: без стимула найти следующий шаг к целям. SILENCE = молчать."""
    goals = [n for n in g.nodes if n["layer"] == "GOAL" and n["meta"].get("status") == "active"]
    cores = sorted((n for n in g.nodes if n["layer"] == "CORE" and n["meta"].get("status") == "active"),
                   key=lambda n: -float(n["meta"].get("salience", 0.5)))[:5]
    ctx = "ЦЕЛИ:\n" + "\n".join(f"- {n['_text'][:120]}" for n in goals[-5:]) if goals else ""
    ctx += "\nЯДРО ПАМЯТИ:\n" + "\n".join(f"- {n['_text'][:120]}" for n in cores)
    thought = llm("Ты в фоновом цикле. Найди ОДИН конкретный следующий шаг для целей. "
                  "Если ничего конкретного — ответь ровно: SILENCE." + neuro_prompt_suffix(g.neuro), ctx)
    if "SILENCE" in thought.upper()[:40]:
        g.save()
        return ""
    nov = g.novelty(thought)
    if nov["verdict"] == "discard":
        neuro_apply(g.neuro, "dopamine", -0.1, "pulse:repeat")
        g.save()
        return ""
    g.add_node(thought, layer="CONTEXT", role="assistant", meta={"kind": "pulse"})
    neuro_apply(g.neuro, "dopamine", 0.1, "pulse:step")
    g.save()
    return thought


# ── SES Partitura v5.1 export ────────────────────────────────────────
def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def export_ses(g: Graph, path: str) -> dict:
    body = {"format": "SES_Partitura", "version": "5.1",
            "canonicalization": "SES_CANON_JSON_v1",
            "snapshot_ts": _now(),
            "nodes": [{k: n[k] for k in ("id", "_text", "layer", "ts", "meta")} for n in g.nodes],
            "edges": g.edges, "affect": g.affect}
    h = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    doc = {"body": body, "provenance": {"hash": f"sha256:{h}", "engine": "qca_engine.py/hermes-port",
                                        "emb_model": EMB_MODEL, "created": _now()}}
    json.dump(doc, open(path, "w"), ensure_ascii=False, indent=1)
    return doc["provenance"]


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd, args = sys.argv[1], sys.argv[2:]
    g = Graph()
    if cmd == "think":
        r = think(g, " ".join(args))
        print(json.dumps(r, ensure_ascii=False, indent=1))
    elif cmd == "seed":
        layer = args[1] if len(args) > 1 else "CONTEXT"
        n = g.add_node(args[0], layer=layer, role="system")
        g.save(); print(f"seeded {n['id']} [{layer}] emb={'ok' if n['emb'] else 'NONE'}")
    elif cmd == "recall":
        for s, n in g.recall(" ".join(args)):
            mark = "✓" if s >= RECALL_THRESHOLD else " "
            print(f"{mark} {s:.3f} [{n['layer']}] {n['_text'][:90]}")
    elif cmd == "stats":
        print(json.dumps({"nodes": len(g.nodes), "edges": len(g.edges),
                          "contradiction_density": round(g.contradiction_density(), 3),
                          "neuro": {k: round(g.neuro[k], 3) for k in ("dopamine", "pain", "adrenaline", "serotonin")},
                          "store": g.path,
                          "backend": LLM_BACKEND, "emb_model": EMB_MODEL}, ensure_ascii=False, indent=1))
        g.save()
    elif cmd == "goal":
        n = g.add_node(" ".join(args), layer="GOAL", role="system")
        g.save(); print(f"goal {n['id']} создана")
    elif cmd == "sleep":
        print(json.dumps(sleep_cycle(g), ensure_ascii=False, indent=1))
    elif cmd == "pulse":
        out = pulse(g)
        print(out if out else "SILENCE")
    elif cmd == "soul":
        # Идентичность из графа: CORE по salience + цели + нейросостояние.
        # Каждая генерация подписана хэшем SES-снапшота (провенанс изменений).
        path = args[0] if args else os.path.join(os.path.dirname(g.path), "SOUL_OCEAN.md")
        snap = os.path.join(os.path.dirname(g.path), "snapshot.ses.json")
        prov = export_ses(g, snap)
        cores = sorted((n for n in g.nodes if n["layer"] == "CORE" and n["meta"].get("status") == "active"),
                       key=lambda n: -float(n["meta"].get("salience", 0.5)))
        goals = [n for n in g.nodes if n["layer"] == "GOAL" and n["meta"].get("status") == "active"]
        nr = g.neuro
        with open(path, "w") as f:
            f.write("# Ocean — когнитивный слой (автогенерация из SES-графа)\n\n"
                    f"> Источник истины: {snap}\n> Провенанс: {prov['hash']} | {prov['created']}\n"
                    "> Этот файл НЕ редактируется руками — он пересоздаётся командой `soul`.\n\n"
                    "## Ядро знаний (CORE, по значимости)\n"
                    + "\n".join(f"- {n['_text'][:200]}" for n in cores[:15])
                    + "\n\n## Активные цели\n"
                    + ("\n".join(f"- {n['_text'][:150]}" for n in goals[-10:]) or "- нет")
                    + "\n\n## Нейросостояние на момент снапшота\n"
                    + f"dopamine {nr['dopamine']:+.2f} | pain {nr['pain']:.2f} | "
                      f"adrenaline {nr['adrenaline']:.2f} | serotonin {nr['serotonin']:.2f}\n")
        print(f"SOUL → {path}\nprovenance: {prov['hash']}")
    elif cmd == "export-ses":
        path = args[0] if args else os.path.join(os.path.dirname(g.path), "snapshot.ses.json")
        prov = export_ses(g, path)
        print(f"SES → {path}\n{json.dumps(prov, ensure_ascii=False, indent=1)}")
    else:
        print(f"неизвестная команда: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
