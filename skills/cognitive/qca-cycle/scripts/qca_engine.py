#!/usr/bin/env python3
"""
qca_engine.py — QCA reasoning cycle + SES identity store.

A portable cognitive layer for any LLM agent. Architecture principle:
the LLM is a black-box CPU; every decision that matters — what to recall,
what counts as a contradiction, whether a thought is novel, what gets
written to memory, when to stay silent — is made by deterministic,
auditable code around it.

Identity is data (SES):
  - kernel.ses.json   immutable, SHA-256-signed constitution (axioms,
                      attractor, guardrails) — injected into every cycle
  - graph store       growing typed graph memory (nodes, edges, salience)
  - neuro state       dopamine/pain/adrenaline/serotonin with real
                      half-lives, persisted between sessions

Contract: pure Python stdlib (no pip dependencies).
LLM backends: mock | ollama | anthropic (env QCA_LLM_BACKEND, default ollama).
Embeddings: Ollama (bge-m3 by default) with a lexical hashing fallback.

ENV:
  QCA_STORE            graph store path     (default <HERMES_HOME>/qca/graph.json)
  QCA_KERNEL           kernel path          (default <store dir>/kernel.ses.json)
  QCA_LLM_BACKEND      mock|ollama|anthropic (default ollama)
  QCA_CHAT_MODEL       Ollama chat model    (default qwen2.5:1.5b)
  QCA_EMB_MODEL        embedding model      (default bge-m3)
  QCA_ANTHROPIC_MODEL  Anthropic model      (default claude-haiku-4-5-20251001)
  QCA_RECALL_THRESHOLD recall ✓ marker      (default 0.70)
  ANTHROPIC_API_KEY    (backend=anthropic only)

CLI:
  python3 qca_engine.py think "stimulus"     # full cycle H0–H9
  python3 qca_engine.py seed "fact" [LAYER]  # seed memory (CORE|GOAL|CONTEXT|EPISODIC)
  python3 qca_engine.py goal "goal text"     # add an active goal
  python3 qca_engine.py recall "query"       # semantic recall only
  python3 qca_engine.py stats                # graph / neuro state
  python3 qca_engine.py pulse                # autonomous step (empty output = silence)
  python3 qca_engine.py sleep                # nightly consolidation
  python3 qca_engine.py soul [path]          # regenerate signed SOUL file
  python3 qca_engine.py export-ses [path]    # signed SES snapshot
"""

from __future__ import annotations
import json, os, sys, math, hashlib, urllib.request, urllib.error
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hermes_home import get_hermes_home

# ── Config ────────────────────────────────────────────────────────────
# Store lives under the active Hermes home (HERMES_HOME env / profile,
# never a hardcoded path) — see hermes_constants.get_hermes_home().
STORE_PATH  = os.path.expanduser(os.getenv("QCA_STORE", "")) or \
              str(get_hermes_home() / "qca" / "graph.json")
LLM_BACKEND = os.getenv("QCA_LLM_BACKEND", "ollama")
CHAT_MODEL  = os.getenv("QCA_CHAT_MODEL", "qwen2.5:1.5b")
EMB_MODEL   = os.getenv("QCA_EMB_MODEL", "bge-m3")
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
ANTHROPIC_MODEL = os.getenv("QCA_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Auto-link thresholds — from Ocean's memory manager (calibrated for bge-m3, 1024d)
SUPPORTS_SIM, REFINES_SIM, ASSOC_SIM = 0.65, 0.50, 0.35
# Novelty thresholds — from Ocean's novelty verifier
REPEAT_SIM, NOVEL_SIM = 0.90, 0.82
# Recall ✓ marker threshold — as in the original Ocean benchmark
RECALL_THRESHOLD = float(os.getenv("QCA_RECALL_THRESHOLD", "0.70"))

# Negation markers used to type CONTRADICTS edges (ru + en corpora)
NEG_MARKERS = ["не так", "наоборот", "ошибк", "неверно", "противореч",
               "not true", "on the contrary", "however", "wrong", "contradicts", "mistake"]

# Neurochemistry — ported from Ocean's neuro engine. Decay rates per hour
# (half-lives from the original: adrenaline ~2.5h, dopamine ~8h, pain ~46h).
NEURO_DECAY = {"adrenaline": 0.40, "dopamine": 0.08, "pain": 0.015, "serotonin": 0.005}
NEURO_DEFAULT = {"dopamine": 0.0, "pain": 0.0, "adrenaline": 0.0, "serotonin": 0.5,
                 "last_decay_ts": "", "events": []}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def neuro_decay(n: dict):
    """Signals decay over wall-clock time — state lives between runs."""
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
    """Port of Ocean's effects: state reframes the task instead of asking to 'feel'."""
    parts = []
    if n["adrenaline"] > 0.65:
        parts.append("⚡ HIGH URGENCY. Brevity is everything. No preamble. Three sentences max.")
    elif n["adrenaline"] > 0.35:
        parts.append("The situation requires focus. Answer sharply, no lyricism.")
    if n["pain"] > 0.7:
        parts.append("🔴 CHRONIC PAIN: the current approach systematically fails. "
                     "Break the frame — a fundamentally different strategy is needed.")
    elif n["pain"] > 0.4:
        parts.append("Something is going wrong. Before answering ask yourself: "
                     "am I repeating a mistake that already failed?")
    if n["dopamine"] > 0.5:
        parts.append("📈 Recent answers have been hitting the mark — keep the course.")
    elif n["dopamine"] < -0.4:
        parts.append("📉 Recent answers missed the mark — try an unconventional angle.")
    if n["serotonin"] < 0.2:
        parts.append("Baseline is unstable. Be especially honest — do not mistake wishes for facts.")
    return "\n\n[NEURO STATE]\n" + "\n".join(f"• {p}" for p in parts) if parts else ""


# ── LLM backends ──────────────────────────────────────────────────────
def _http_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def llm(system: str, prompt: str, max_tokens: int = 600) -> str:
    if LLM_BACKEND == "mock":
        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return f"[mock:{h}] Synthesis for: {prompt[:120]}"
    if LLM_BACKEND == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY is not set (backend=anthropic)")
        out = _http_json("https://api.anthropic.com/v1/messages",
                         {"model": ANTHROPIC_MODEL, "max_tokens": max_tokens,
                          "system": system, "messages": [{"role": "user", "content": prompt}]},
                         {"x-api-key": key, "anthropic-version": "2023-06-01"})
        # models with extended thinking return a thinking block before text — take the
        # first text block; if the model spent the whole budget thinking, return empty
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
    """Without Ollama: hashed bag of character trigrams. Degraded mode —
    captures lexical similarity, not semantics, but recall and the novelty
    gate keep working."""
    v = [0.0] * _EMB_FALLBACK_DIM
    t = " " + text.lower() + " "
    for i in range(len(t) - 2):
        h = int(hashlib.md5(t[i:i + 3].encode()).hexdigest()[:8], 16)
        v[h % _EMB_FALLBACK_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


def embed(text: str) -> list[float] | None:
    """Embedding via Ollama (vector is L2-normalized so cosine is 0..1);
    falls back to a dependency-free lexical mode when Ollama is unreachable."""
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
            print("⚠ Ollama unreachable — embeddings in lexical fallback mode", file=sys.stderr)
    return _embed_fallback(text)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0  # vectors from different models (ollama vs fallback) are incomparable
    return sum(x * y for x, y in zip(a, b))  # vectors are already normalized


# ── Personality kernel (kernel.ses.json) ──────────────────────────────
# The immutable constitution: axioms, attractor, guardrails. SHA-256 signed;
# it changes only deliberately — a new snapshot with a new hash (no silent drift).
KERNEL_PATH = os.path.expanduser(os.getenv("QCA_KERNEL",
              os.path.join(os.path.dirname(STORE_PATH), "kernel.ses.json")))


def load_kernel() -> tuple[dict, str]:
    """Return (kernel, hash). A missing kernel is valid: an agent without a constitution."""
    if not os.path.exists(KERNEL_PATH):
        return {}, ""
    doc = json.load(open(KERNEL_PATH))
    kernel = doc.get("kernel", doc)
    h = "sha256:" + hashlib.sha256(canonical_json(kernel).encode()).hexdigest()
    return kernel, h


def kernel_prompt(kernel: dict) -> str:
    """Inject the FULL kernel per SES v5.1: fractal_seed + recursive_function
    (the entity's own reasoning protocol) + distortion_field (known failure
    modes with mitigations)."""
    if not kernel:
        return ""
    seed = kernel.get("fractal_seed", {})
    parts = []
    ax = seed.get("Z_AXIOM", [])
    if ax:
        parts.append("AXIOMS (inviolable):\n" + "\n".join(f"- {a}" for a in ax))
    if seed.get("OMEGA_ATTRACTOR"):
        parts.append(f"ATTRACTOR: {seed['OMEGA_ATTRACTOR']}")
    gr = seed.get("guardrails", [])
    if gr:
        parts.append("GUARDRAILS (never break):\n" + "\n".join(f"- {g}" for g in gr))
    rf = kernel.get("recursive_function", [])
    if rf:
        steps = "\n".join(f"{s.get('id','?')} {s.get('name','')}: " + "; ".join(s.get("rules", []))
                          for s in rf if isinstance(s, dict))
        parts.append("REASONING PROTOCOL (follow in order):\n" + steps)
    df = kernel.get("distortion_field", {})
    items = df.get("items", df) if isinstance(df, dict) else df
    if isinstance(items, list) and items:
        ds = "\n".join(f"- when [{d.get('trigger','?')}] you tend to [{d.get('effect','?')}] → "
                        + "; ".join(d.get("mitigation", [])) for d in items if isinstance(d, dict))
        parts.append("KNOWN DISTORTIONS (self-correct):\n" + ds)
    return "\n\n" + "\n\n".join(parts) if parts else ""


# ── Graph memory ──────────────────────────────────────────────────────
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
        """Port of Ocean's auto-linking v2: SUPPORTS/REFINES/ASSOCIATED/CONTRADICTS."""
        if not new.get("emb"):
            return
        candidates = [n for n in self.nodes[-16:] if n["id"] != new["id"] and n.get("emb")]
        linked = False
        nearest = None  # (id, sim) of the closest node if no edge was created
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
        # a node always gets at least one edge — as in the original
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
        """H5.5 — port of Ocean's novelty verifier: an incorruptible judgment
        by embedding geometry, not by an LLM grading itself."""
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
            verdict = "discard"   # a repeat — the illusion of progress
        elif max_sim < NOVEL_SIM:
            verdict = "novel"
        else:
            verdict = "refine"
        return {"verdict": verdict, "novelty_score": round(1 - max_sim, 4),
                "max_sim": round(max_sim, 4), "nearest": near}


# ── QCA cycle H0–H9 ──────────────────────────────────────────────────
def think(g: Graph, stimulus: str) -> dict:
    trace = {"H0": stimulus[:200]}

    # H1: framing (heuristic; richer implementations use a small LLM preprocessor)
    core_idea = stimulus.strip()[:140]
    trace["H1"] = {"core_idea": core_idea}

    # H2: recall
    recalled = g.recall(core_idea, top_k=5)
    relevant = [(round(s, 4), n["_text"][:120]) for s, n in recalled if s >= ASSOC_SIM]
    trace["H2"] = relevant

    # H3: contradiction — CONTRADICTS edges incident to the recalled nodes
    recalled_ids = {n["id"] for _, n in recalled}
    contra = [e for e in g.edges if e["relation"] == "CONTRADICTS"
              and (e["source"] in recalled_ids or e["target"] in recalled_ids)]
    contradiction = f"{len(contra)} recorded conflict(s) in memory around this topic" if contra else ""
    trace["H3"] = contradiction

    # H4: synthesis — the constitution (kernel) enters every thinking cycle
    kernel, kernel_hash = load_kernel()
    if kernel_hash:
        trace["kernel"] = kernel_hash[:18]
    system = ("You are a cognitive core (QCA cycle). Answer briefly and to the point. "
              "Always reply in the operator's language."
              + kernel_prompt(kernel)
              + (f"\nCONTEXT FROM MEMORY:\n" + "\n".join(f"- {t}" for _, t in relevant[:3]) if relevant else "")
              + (f"\nCONTRADICTION: {contradiction} — acknowledge and resolve it." if contradiction else "")
              + neuro_prompt_suffix(g.neuro))
    thought = llm(system, stimulus)
    trace["H4"] = thought[:200]

    # H5: critique (LLM judge — advisory; the hard gate is H5.5)
    if LLM_BACKEND == "mock":
        critique = {"quality": "medium", "is_acceptable": True}
    else:
        raw = llm('You are a strict critic. Reply ONLY with JSON: {"quality":"low|medium|high","is_acceptable":bool}',
                  f"Question: {stimulus}\nAnswer: {thought}", max_tokens=2000)
        try:
            critique = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        except (ValueError, IndexError):
            critique = {"quality": "medium", "is_acceptable": True}
    trace["H5"] = critique

    # H5.5: novelty gate (incorruptible)
    nov = g.novelty(thought)
    trace["H5.5_novelty"] = nov

    # H6: neurochemistry — cycle events become impulses (port of Ocean's neuro engine)
    if contradiction:
        neuro_apply(g.neuro, "pain", 0.10, "H3:contradiction")
        neuro_apply(g.neuro, "adrenaline", 0.15, "H3:contradiction")
    if nov["verdict"] == "novel":
        neuro_apply(g.neuro, "dopamine", 0.20, "H5.5:novel")
    elif nov["verdict"] == "discard":
        neuro_apply(g.neuro, "dopamine", -0.15, "H5.5:repeat")
        neuro_apply(g.neuro, "pain", 0.05, "H5.5:repeat")
    neuro_apply(g.neuro, "serotonin", 0.02 if critique.get("is_acceptable") else -0.08, "H5:critique")
    # dopamine → salience of recent nodes (learning, port of Ocean's salience update)
    if abs(g.neuro["dopamine"]) >= 0.15:
        delta = g.neuro["dopamine"] * 0.07
        for node in g.nodes[-6:]:
            cur = float(node["meta"].get("salience", 0.5))
            node["meta"]["salience"] = round(max(0.0, min(1.0, cur + delta)), 3)
    trace["H6_neuro"] = {k: round(g.neuro[k], 3) for k in ("dopamine", "pain", "adrenaline", "serotonin")}

    # H7: memory write (repeats are NOT written — the gate decides, not the LLM)
    if nov["verdict"] != "discard":
        g.add_node(stimulus, layer="EPISODIC", role="user")
        layer = "CONTEXT" if critique.get("quality") in ("medium", "high") else "EPISODIC"
        g.add_node(thought, layer=layer, role="assistant")
        trace["H7"] = f"written (layer={layer})"
    else:
        trace["H7"] = "repeat — not written"

    # H8: lesson
    if contradiction or nov["verdict"] == "novel":
        lesson = f"Lesson: {core_idea[:80]} → {('conflict in the graph resolved' if contradiction else 'new region of the space')}"
        g.add_node(lesson, layer="CORE", role="system", meta={"kind": "lesson", "confidence": 0.6})
        trace["H8"] = lesson

    # H9: stats
    trace["H9"] = {"nodes": len(g.nodes), "edges": len(g.edges),
                   "contradiction_density": round(g.contradiction_density(), 3)}
    g.save()
    return {"thought": thought, "trace": trace}


# ── Daemons: sleep and pulse (port of Ocean's background loops) ──────
def sleep_cycle(g: Graph) -> dict:
    """Nightly consolidation: clusters of EPISODIC nodes → one CORE abstraction,
    originals archived."""
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
        abstraction = llm("Compress these episodes into one general abstraction-lesson. "
                          "One sentence, in the language of the episodes.",
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
    """Autonomous step: with no stimulus, find the next step toward active goals.
    SILENCE = stay quiet (the default)."""
    goals = [n for n in g.nodes if n["layer"] == "GOAL" and n["meta"].get("status") == "active"]
    cores = sorted((n for n in g.nodes if n["layer"] == "CORE" and n["meta"].get("status") == "active"),
                   key=lambda n: -float(n["meta"].get("salience", 0.5)))[:5]
    ctx = "GOALS:\n" + "\n".join(f"- {n['_text'][:120]}" for n in goals[-5:]) if goals else ""
    ctx += "\nMEMORY CORE:\n" + "\n".join(f"- {n['_text'][:120]}" for n in cores)
    thought = llm("You are in a background cycle. Find ONE concrete next step toward the goals. "
                  "If nothing concrete — reply exactly: SILENCE." + neuro_prompt_suffix(g.neuro), ctx)
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


# ── SES Partitura v5.1 export (canonical) ────────────────────────────
def canonical_json(obj) -> str:
    """SES_CANON_JSON_v1: keys sorted lexicographically, compact separators,
    UTF-8 not escaped. Node/edge array sorting is applied by the caller
    (canonical_snapshot) since it is a snapshot-level rule."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _snapshot_hash(snapshot: dict) -> str:
    """meta.hash per SES_CANON_JSON_v1: nodes sorted by id, edges sorted by id
    (else by source,target,relation), computed with meta.hash itself absent."""
    import copy
    s = copy.deepcopy(snapshot)
    s.get("meta", {}).pop("hash", None)
    st = s.get("state") or {}
    if st.get("nodes"):
        st["nodes"] = sorted(st["nodes"], key=lambda n: n.get("id", ""))
    if st.get("edges"):
        st["edges"] = sorted(st["edges"], key=lambda e: (e.get("id", ""), e.get("source", ""),
                                                         e.get("target", ""), e.get("relation", "")))
    return "sha256:" + hashlib.sha256(canonical_json(s).encode()).hexdigest()


def _node_to_ses(n: dict) -> dict:
    """Internal working node → canonical v5.1 node (label + required provenance)."""
    role = n["meta"].get("role", "assistant")
    source = {"user": "OPERATOR", "system": "OPERATOR", "assistant": "QCA_CYCLE"}.get(role, "QCA_CYCLE")
    if n["meta"].get("imported_from"):
        source = "IMPORT"
    return {"id": n["id"], "label": n["_text"], "layer": n["layer"],
            "meta": {"provenance": {"source": source, "stage": "H7",
                                    "timestamp": n.get("ts", _now()),
                                    "source_ref": [], "confidence": 1.0 if source == "OPERATOR" else 0.8},
                     "salience": float(n["meta"].get("salience", 0.5)),
                     "status": n["meta"].get("status", "active"),
                     "tags": [n["meta"]["kind"]] if n["meta"].get("kind") else []}}


def _edge_to_ses(e: dict) -> dict:
    return {"id": e["id"], "source": e["source"], "target": e["target"],
            "relation": e["relation"],
            "meta": {"provenance": {"source": "QCA_CYCLE", "stage": "H7",
                                    "timestamp": _now(), "source_ref": [], "confidence": 0.8},
                     "weight": float(e.get("sim", 0.5))}}


def export_ses(g: Graph, path: str) -> dict:
    """Export a canonical SES v5.1 snapshot. COMBINED when a kernel is present,
    STATE_SNAPSHOT otherwise. Enforces the v5.1 canon lock (§12): every state
    references its kernel via meta.kernel_ref + meta.kernel_hash."""
    kernel, kernel_hash = load_kernel()
    entity_id = os.getenv("QCA_ENTITY") or (
        json.load(open(KERNEL_PATH)).get("entity_id") if os.path.exists(KERNEL_PATH) else None) or "qca-agent"
    # lineage: previous snapshot at the same path becomes the parent
    parent = None
    if os.path.exists(path):
        try:
            parent = json.load(open(path)).get("snapshot_id")
        except (ValueError, OSError):
            parent = None
    snapshot_id = _now()
    snap = {"initiator": "∮", "schema_version": "5.1", "entity_id": entity_id,
            "snapshot_id": snapshot_id,
            "snapshot_type": "COMBINED" if kernel else "STATE_SNAPSHOT",
            "meta": {"created_at": snapshot_id, "created_by": "System",
                     "parent_snapshot_id": parent,
                     "canonicalization": "SES_CANON_JSON_v1",
                     "notes": f"engine=qca_engine.py emb_model={EMB_MODEL}",
                     "tags": ["qca-cycle"]},
            "state": {"meta": {"trigger": "inference",
                               "summary": f"State of {entity_id} at {snapshot_id}",
                               "provenance": {"source": "QCA_CYCLE", "stage": "H9",
                                              "timestamp": snapshot_id, "source_ref": [],
                                              "confidence": 1.0},
                               "x_neuro": {k: round(g.neuro[k], 4) for k in
                                           ("dopamine", "pain", "adrenaline", "serotonin")}},
                      "nodes": [_node_to_ses(n) for n in g.nodes],
                      "edges": [_edge_to_ses(e) for e in g.edges]}}
    if kernel:
        snap["kernel"] = kernel
    if kernel_hash:  # v5.1 canon lock: state must reference its kernel
        snap["meta"]["kernel_hash"] = kernel_hash
        snap["meta"]["kernel_ref"] = f"kernel://{entity_id}@{KERNEL_PATH}"
    else:
        print("⚠ no kernel installed — this STATE_SNAPSHOT will fail the "
              "v5.1 canon-lock check in ses_bridge.py verify", file=sys.stderr)
    snap["meta"]["hash"] = _snapshot_hash(snap)
    json.dump(snap, open(path, "w"), ensure_ascii=False, indent=1)
    return {"hash": snap["meta"]["hash"], "snapshot_type": snap["snapshot_type"],
            "kernel_hash": kernel_hash or None, "parent": parent, "created": snapshot_id}


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
        g.save(); print(f"goal {n['id']} created")
    elif cmd == "sleep":
        print(json.dumps(sleep_cycle(g), ensure_ascii=False, indent=1))
    elif cmd == "pulse":
        # Silent-cron contract: deliberate silence = empty stdout, so a cron
        # wrapper can gate delivery on "any output at all".
        out = pulse(g)
        if out:
            print(out)
    elif cmd == "soul":
        # Identity from the graph: CORE by salience + goals + neuro state.
        # Every generation is signed with the SES snapshot hash (change provenance).
        path = args[0] if args else os.path.join(os.path.dirname(g.path), "SOUL_OCEAN.md")
        snap = os.path.join(os.path.dirname(g.path), "snapshot.ses.json")
        prov = export_ses(g, snap)
        cores = sorted((n for n in g.nodes if n["layer"] == "CORE" and n["meta"].get("status") == "active"),
                       key=lambda n: -float(n["meta"].get("salience", 0.5)))
        goals = [n for n in g.nodes if n["layer"] == "GOAL" and n["meta"].get("status") == "active"]
        nr = g.neuro
        with open(path, "w") as f:
            f.write("# Cognitive layer identity (auto-generated from the SES graph)\n\n"
                    f"> Source of truth: {snap}\n> Provenance: {prov['hash']} | {prov['created']}\n"
                    "> Do NOT edit by hand — this file is regenerated by the `soul` command.\n\n"
                    "## Knowledge core (CORE, by salience)\n"
                    + "\n".join(f"- {n['_text'][:200]}" for n in cores[:15])
                    + "\n\n## Active goals\n"
                    + ("\n".join(f"- {n['_text'][:150]}" for n in goals[-10:]) or "- none")
                    + "\n\n## Neuro state at snapshot time\n"
                    + f"dopamine {nr['dopamine']:+.2f} | pain {nr['pain']:.2f} | "
                      f"adrenaline {nr['adrenaline']:.2f} | serotonin {nr['serotonin']:.2f}\n")
        print(f"SOUL → {path}\nprovenance: {prov['hash']}")
    elif cmd == "export-ses":
        path = args[0] if args else os.path.join(os.path.dirname(g.path), "snapshot.ses.json")
        prov = export_ses(g, path)
        print(f"SES → {path}\n{json.dumps(prov, ensure_ascii=False, indent=1)}")
    else:
        print(f"unknown command: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
