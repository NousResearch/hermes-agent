"""Auto Router — per-task WORKER-model selection for Caduceus (Pass 2).

Ported from the UltraCode-Shim proxy's Auto Router. In the shim it lives at the
HTTP layer and rewrites the request's model field; here it is a *pure selection
core* that Caduceus uses to choose which configured model a WORKER (a
``delegate_task`` child / Loom leaf) runs on for a given subtask — cheap models
for easy leaves, strong models for hard ones — chosen from the models the user
already configured.

Design (unchanged from the shim; see UltraCode-Shim/docs/AUTO_ROUTER.md):
  * The classifier scores CAPABILITY only (0-1, probability of first-try
    success); it never sees price. Cost is applied afterward by the selector
    (cheapest-among-viable), so the classifier can't be biased toward expensive
    models.
  * Per-task caching: classify once per (tier, task), reuse across a task's
    tool-call round-trips.
  * Degrades safely at every step: no/failed classifier -> deterministic
    fallback (configured default, else cheapest); unknown candidates dropped;
    image tasks hard-zero image-incapable candidates. It NEVER raises — the
    caller always gets a usable model id or None ("leave routing unchanged").

Integration plan (Pass 2 — this module is the core; wiring is the next step):
  1. NEVER route the orchestrator. Only worker/leaf selection is routed, so the
     planning hot path keeps the user's strong session model. (A classifier call
     before every planning turn would re-add the startup latency we removed.)
  2. ``tools/delegate_tool.py``: when Caduceus is active and the worker model is
     not explicitly pinned, call :func:`select` to choose the child model from
     the configured candidates; fall back to solo (inherit parent) if the router
     is unconfigured.
  3. Classifier call: inject a Hermes auxiliary-model completion (chat
     completions, non-streaming, temperature 0, small max_tokens) as the
     ``classify`` callable. Pure core stays network-free + unit-testable.
  4. Config: ``caduceus.router`` { enabled, classifier, threshold, default,
     cache, candidates:[{model, provider, cost, supports_images, card}] }.
  5. Surface it as the worker default only — no new user-facing knobs (honors
     the "one switch" simplification).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

# Per-task decision cache: key -> chosen candidate id. Shared, bounded.
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 256


@dataclass
class Candidate:
    """A model the router may choose among."""
    id: str                       # model id the worker will run on
    provider: str = ""            # optional provider override (else inherit)
    cost: float = 1.0             # RELATIVE weight; only ordering matters
    supports_images: bool = False
    card: str = ""                # capability card — the single most important field


# ---------------------------------------------------------------------------
# Pure helpers (ported verbatim in behavior from the shim)
# ---------------------------------------------------------------------------

def clamp01(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def parse_scores(text: str, candidate_ids: List[str]) -> dict:
    """Pull the {"scores": {...}} object out of the classifier's reply, leniently
    (it may wrap JSON in prose or a code fence). Missing candidates -> 0.0."""
    if not text:
        return {}
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {}
    for end in (e, text.find("}", s)):  # greedy (last }) then first object
        if end == -1 or end <= s:
            continue
        try:
            obj = json.loads(text[s:end + 1])
        except Exception:
            continue
        scores = obj.get("scores") if isinstance(obj, dict) else None
        if isinstance(scores, dict):
            return {cid: clamp01(scores.get(cid, 0)) for cid in candidate_ids}
    return {}


def router_pick(scores: dict, candidates: List[Candidate], threshold: float,
                has_images: bool):
    """Cheapest candidate whose score clears the bar; image-incapable models are
    hard-zeroed on image tasks; if none clear the bar, take the highest score."""
    scored = []
    for c in candidates:
        sc = scores.get(c.id, 0.0)
        if has_images and not c.supports_images:
            sc = 0.0
        scored.append((c, sc))
    viable = [(c, sc) for (c, sc) in scored if sc >= threshold]
    if viable:
        winner = min(viable, key=lambda cs: (float(cs[0].cost or 0), -cs[1]))
        return winner[0].id, winner[1], "score>=%.2f, cheapest" % threshold
    best = max(scored, key=lambda cs: cs[1], default=None)
    if best and best[1] > 0:
        return best[0].id, best[1], "below bar; highest score"
    return None, 0.0, "no usable score"


def fallback_id(candidates: List[Candidate], default: Optional[str] = None) -> Optional[str]:
    """Deterministic, classifier-free choice: configured default, else cheapest."""
    if default and any(c.id == default for c in candidates):
        return default
    if candidates:
        return min(candidates, key=lambda c: float(c.cost or 0)).id
    return None


def cache_key(task: str, tier: str) -> str:
    return "%s|%s" % (tier, hash(task or ""))


def build_classifier_system_prompt(candidates: List[Candidate]) -> str:
    lines = [
        "You are a task-routing classifier for an AI coding agent.",
        "You are given a <session> describing the user's current task and a list",
        "of candidate models. For EACH candidate, output a score from 0.0 to 1.0:",
        "the probability that the model completes THIS task correctly on its first",
        "attempt, without errors or rework.",
        "",
        "You are NOT choosing a winner. A downstream system combines your scores",
        "with cost data you do not see to make the final pick. Be an accurate,",
        "well-calibrated, independent probability estimator for each model.",
        "",
        "Scoring guide:",
        "  0.0       cannot attempt (e.g. images required but unsupported) -- exact 0.0",
        "  0.1-0.3   will almost certainly fail; lacks the capability",
        "  0.4-0.6   real chance of failure; touches a known weakness or is uncertain",
        "  0.7-0.8   likely success; handles this category well",
        "  0.9-1.0   near-certain success; well within demonstrated ability",
        "Use the full range. A short prompt is NOT necessarily an easy task -- hidden",
        "complexity (multi-file edits, debugging, niche domains, strict correctness)",
        "should pull scores down for weaker models. Default to ~0.5-0.6 when unsure.",
        "",
        "Candidate models:",
    ]
    for c in candidates:
        card = (c.card or "").strip() or "General-purpose model. No capability card provided."
        lines.append("- id: %s" % c.id)
        lines.append("  images: %s" % ("yes" if c.supports_images else "no"))
        lines.append("  capability: %s" % card)
    schema = {"scores": {c.id: 0.0 for c in candidates}, "reasoning": "one short sentence"}
    lines += [
        "",
        "Respond with ONE JSON object, no prose, no code fence, exactly this shape:",
        json.dumps(schema, ensure_ascii=False),
        "Every candidate id above MUST appear in \"scores\". Each value in [0.0, 1.0].",
    ]
    return "\n".join(lines)


def build_classifier_user_content(task: str, candidates: List[Candidate], *,
                                  has_images: bool = False, surface: str = "worker/sub-agent",
                                  turns: int = 1, tool_count: int = 0) -> str:
    task = task or "(no explicit instruction; infer from context)"
    if len(task) > 6000:  # head+tail, keep the classifier call cheap
        task = task[:3000] + "\n...\n" + task[-3000:]
    return "\n".join([
        "<session>",
        "  surface: %s" % surface,
        "  images_present: %s" % ("yes" if has_images else "no"),
        "  user_turns: %d" % turns,
        "  tools_available: %d" % tool_count,
        "  current_task: |",
        "\n".join("    " + ln for ln in task.splitlines()) or "    (empty)",
        "</session>",
        "",
        "Score these candidate ids: %s" % ", ".join(c.id for c in candidates),
    ])


# ---------------------------------------------------------------------------
# Selection orchestration (pure core; I/O injected as `classify`)
# ---------------------------------------------------------------------------

def select(task: str, candidates: List[Candidate], *,
           classify: Optional[Callable[[str, str], str]] = None,
           threshold: float = 0.7, default: Optional[str] = None,
           has_images: bool = False, tier: str = "worker", use_cache: bool = True,
           surface: str = "worker/sub-agent", turns: int = 1, tool_count: int = 0,
           log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Choose a candidate model id for ``task``, or None to leave routing
    unchanged. NEVER raises.

    ``classify(system_prompt, user_content) -> str`` is the injected classifier
    call (a cheap aux-model completion); omit it to force the deterministic
    fallback. Mirrors the shim's ``resolve_auto`` but model-agnostic and pure.
    """
    try:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0].id

        key = cache_key(task, tier)
        if use_cache:
            with _CACHE_LOCK:
                cached = _CACHE.get(key)
            if cached and any(c.id == cached for c in candidates):
                if log:
                    log("[auto_router] tier=%s cache-hit -> %s" % (tier, cached))
                return cached

        if classify is None:
            return fallback_id(candidates, default)

        sys_p = build_classifier_system_prompt(candidates)
        usr = build_classifier_user_content(task, candidates, has_images=has_images,
                                            surface=surface, turns=turns, tool_count=tool_count)
        try:
            text = classify(sys_p, usr)
        except Exception:
            return fallback_id(candidates, default)

        scores = parse_scores(text, [c.id for c in candidates])
        if not scores:
            return fallback_id(candidates, default)
        pick, score, why = router_pick(scores, candidates, threshold, has_images)
        if pick is None:
            return fallback_id(candidates, default)

        if use_cache:
            with _CACHE_LOCK:
                if len(_CACHE) >= _CACHE_MAX:
                    _CACHE.clear()
                _CACHE[key] = pick
        if log:
            log("[auto_router] tier=%s -> %s (%.2f; %s) scores=%s"
                % (tier, pick, score, why, {c.id: round(scores.get(c.id, 0.0), 2) for c in candidates}))
        return pick
    except Exception:
        return fallback_id(candidates, default)


# The small/cheap/fast tier is matched on whole TOKENS (the model id split on
# non-alphanumerics), so "mini"/"flash" match e.g. "gpt-4o-mini" / "gemini-flash"
# but never the *substring* "mini" inside "gemini"/"minimax". Checked first.
_SMALL_TOKENS = {"mini", "flash", "haiku", "nano", "lite", "small", "tiny", "8b", "7b", "3b", "1b"}
_SMALL_CARD = (
    "Small/cheap/fast model. Good for trivial edits, formatting, classification, "
    "data extraction, and short codegen. Weak on hard reasoning, multi-file "
    "refactors, and subtle debugging."
)

# Larger families, matched by substring on the full id (first match wins).
_FAMILY_CARDS = [
    (("minimax",),
     "MiniMax-M3: strong agentic coder, long context, adaptive thinking. Good at "
     "multi-step coding, tool loops, and refactors. Mid cost."),
    (("mimo",),
     "MiMo: fast capable mid-tier coder. Good at standard edits, codegen, and data "
     "wrangling; weaker on the hardest multi-file debugging."),
    (("gpt-5", "codex"),
     "GPT-5.5 / Codex class: frontier reasoning + agentic coding. Best for the "
     "hardest work — large multi-file refactors, subtle debugging, architecture, "
     "long autonomous runs. Premium cost."),
    (("opus",),
     "Claude Opus: frontier reasoning + coding; excellent at hard multi-file work, "
     "careful edits, and verification. Premium cost."),
    (("sonnet",),
     "Claude Sonnet: strong fast coder; great for most coding/tool work. Mid cost."),
    (("gemini",),
     "Gemini Pro: capable generalist with very large context; good at multi-file "
     "reading and standard coding."),
    (("grok",),
     "Grok: strong reasoning + coding generalist; solid across most coding/tool tasks."),
    (("deepseek",),
     "DeepSeek: strong coder/reasoner, cost-effective; good at algorithms and "
     "standard multi-file work."),
    (("qwen",),
     "Qwen Coder: capable cost-effective coding model; good at standard edits/codegen."),
    (("llama",),
     "Llama: open generalist; decent at simple-to-moderate coding, weaker on the "
     "hardest tasks."),
    (("kimi", "glm", "arcee"),
     "Capable open/aux coding model; good at standard coding and tool work, "
     "uncertain on the hardest debugging — score moderately."),
]


def default_card_for(model_id: str) -> str:
    """A sensible capability card for a known model family, else a conservative
    generic card. Lets users list candidates by id alone and still route well.

    Small-tier match is token-based (so 'mini' matches 'gpt-4o-mini' but not the
    substring inside 'gemini'/'minimax'); families match by substring.
    """
    m = (model_id or "").lower()
    tokens = {t for t in re.split(r"[^a-z0-9]+", m) if t}
    if tokens & _SMALL_TOKENS:
        return _SMALL_CARD
    for subs, card in _FAMILY_CARDS:
        if any(s in m for s in subs):
            return card
    return ("General-purpose model; capability unknown — score conservatively "
            "(~0.5) unless the task is clearly simple.")


def candidates_from_config(router_cfg: Optional[dict]) -> List[Candidate]:
    """Build the candidate list from a ``caduceus.router`` config block.

    A candidate with no ``card`` gets a default card auto-filled from its model
    family (:func:`default_card_for`), so listing models by id alone still routes
    sensibly.
    """
    out: List[Candidate] = []
    for c in (router_cfg or {}).get("candidates") or []:
        if not isinstance(c, dict):
            continue
        mid = c.get("model") or c.get("id")
        if not mid:
            continue
        card = str(c.get("card") or "").strip() or default_card_for(str(mid))
        out.append(Candidate(
            id=str(mid),
            provider=str(c.get("provider") or ""),
            cost=float(c.get("cost", 1) or 0),
            supports_images=bool(c.get("supports_images", False)),
            card=card,
        ))
    return out


def reset_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
