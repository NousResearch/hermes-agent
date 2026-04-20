"""Deterministic role-text → toolsets/skills classifier.

This is intentionally **not** an LLM call. The user's local model may not be
capable enough, and our spec mandates determinism (NFR-9). Instead we use a
small bag-of-words classifier with hand-curated keyword vectors per
candidate (loaded from ``data/role_keywords.json``).

Algorithm (see design.md §4.2):

* For each candidate ``c``:

      score(c | text) = Σ_kw  w_kw · tf_kw(text)

  with ``tf_kw(text) = 1 + log(count(kw in text))`` when the keyword appears
  at least once, else 0.

* Pick toolsets with ``score >= θ_toolset`` and skills with ``score >=
  θ_skill``; ties broken alphabetically for determinism.

* ``confidence = sigmoid((best_score - θ) / θ)`` for matches, else 0.

* Optional fitted weights are loaded from ``$HERMES_HOME/office/weights.json``
  if present (set by ``hermes office optimize``).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

from .models import ResolvedRole

# ────────────────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────────────────


def _load_bundled_keywords() -> dict[str, Any]:
    pkg = files("hermes_office") / "data" / "role_keywords.json"
    return json.loads(pkg.read_text(encoding="utf-8"))


def _merge_weights(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    if not overrides:
        return base
    out = json.loads(json.dumps(base))  # cheap deep copy
    for section in ("toolsets", "skills", "model_hints"):
        for cand, weights in overrides.get(section, {}).items():
            if not isinstance(weights, dict):
                continue
            cur = out.setdefault(section, {}).setdefault(cand, {})
            cur.update({k: float(v) for k, v in weights.items()})
    if "thresholds" in overrides and isinstance(overrides["thresholds"], dict):
        out.setdefault("thresholds", {}).update(
            {k: float(v) for k, v in overrides["thresholds"].items()}
        )
    return out


# ────────────────────────────────────────────────────────────────────────────
# Math helpers
# ────────────────────────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]+|[\u4e00-\u9fff]+")


def _normalise(text: str) -> str:
    return text.lower().strip()


def _count_keyword(text_lc: str, kw: str) -> int:
    """Substring count for the keyword. Multi-token keywords (e.g. ``"pull
    request"``) are matched literally; the lowercase substring search suffices."""
    if not kw:
        return 0
    kw_lc = kw.lower()
    if not kw_lc:
        return 0
    count = 0
    start = 0
    while True:
        idx = text_lc.find(kw_lc, start)
        if idx == -1:
            return count
        count += 1
        start = idx + len(kw_lc)


def _tf(count: int) -> float:
    return 1.0 + math.log(count) if count >= 1 else 0.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_candidate(text_lc: str, weights: dict[str, float]) -> tuple[float, list[str]]:
    """Return ``(score, matched_keywords)`` for one candidate."""
    total = 0.0
    matched: list[tuple[str, float]] = []
    for kw, w in weights.items():
        c = _count_keyword(text_lc, kw)
        if c == 0:
            continue
        contribution = float(w) * _tf(c)
        if contribution > 0:
            total += contribution
            matched.append((kw, contribution))
    matched.sort(key=lambda p: (-p[1], p[0]))
    return total, [kw for kw, _ in matched]


# ────────────────────────────────────────────────────────────────────────────
# Public class
# ────────────────────────────────────────────────────────────────────────────


class SkillResolver:
    """Loads the keyword bundle (+ optional fitted weights) once and resolves
    role-text descriptions into :class:`ResolvedRole` objects.
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        bundled = _load_bundled_keywords()
        overrides: dict[str, Any] = {}
        if weights_path is not None and weights_path.exists():
            try:
                overrides = json.loads(weights_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                overrides = {}
        self._data = _merge_weights(bundled, overrides)
        thresholds = self._data.get("thresholds", {})
        self.threshold_toolset = float(thresholds.get("toolset", 0.7))
        self.threshold_skill = float(thresholds.get("skill", 1.2))
        self.threshold_model_hint = float(thresholds.get("model_hint", 1.0))

    # ── score sections ─────────────────────────────────────────────────────

    def _section_scores(self, section: str, text_lc: str) -> list[tuple[str, float, list[str]]]:
        out: list[tuple[str, float, list[str]]] = []
        for cand, weights in self._data.get(section, {}).items():
            score, matched = score_candidate(text_lc, weights)
            out.append((cand, score, matched))
        # Deterministic tie-breaking: alphabetical by id.
        out.sort(key=lambda t: t[0])
        return out

    # ── public API ─────────────────────────────────────────────────────────

    def resolve(self, text: str) -> ResolvedRole:
        text_lc = _normalise(text)
        if not text_lc:
            return ResolvedRole(
                recommended_toolsets=[],
                recommended_skills=[],
                model_hint=None,
                confidence=0.0,
                rationale_md="(empty input)",
                matched_keywords=[],
            )

        toolset_scores = self._section_scores("toolsets", text_lc)
        skill_scores = self._section_scores("skills", text_lc)
        model_scores = self._section_scores("model_hints", text_lc)

        toolsets = [c for c, s, _ in toolset_scores if s >= self.threshold_toolset]
        skills = [c for c, s, _ in skill_scores if s >= self.threshold_skill]

        # Model hint: pick the single best one above threshold (else None).
        best_model_pair = max(
            ((c, s) for c, s, _ in model_scores if s >= self.threshold_model_hint),
            key=lambda p: (p[1], p[0]),
            default=None,
        )
        model_hint = best_model_pair[0] if best_model_pair else None

        # Confidence: based on whichever was the strongest signal.
        candidates = [
            (s, self.threshold_toolset, "toolset", c)
            for c, s, _ in toolset_scores
        ] + [
            (s, self.threshold_skill, "skill", c)
            for c, s, _ in skill_scores
        ]
        if candidates:
            best_score, best_thr, _kind, _cand = max(candidates, key=lambda p: p[0])
            if best_score >= best_thr:
                confidence = _sigmoid((best_score - best_thr) / best_thr)
            else:
                confidence = 0.0
        else:
            confidence = 0.0

        # Always include `todo` if the user mentions multi-step work — small
        # pragmatic boost; keeps the assistant honest. (Captured in keywords.)
        toolsets = sorted(set(toolsets))
        skills = sorted(set(skills))

        # Aggregate matched keywords (top 12, deduplicated, by contribution).
        matched_all: dict[str, float] = defaultdict(float)
        toolsets_section = self._data.get("toolsets", {})
        skills_section = self._data.get("skills", {})
        for cand_id, score, matched in toolset_scores + skill_scores:
            if score <= 0:
                continue
            section = toolsets_section if cand_id in toolsets_section else skills_section
            weights = section.get(cand_id, {})
            for kw in matched:
                w = float(weights.get(kw, 0.0))
                contribution = w * _tf(_count_keyword(text_lc, kw))
                if contribution > matched_all[kw]:
                    matched_all[kw] = contribution
        matched_keywords = [
            kw for kw, _ in sorted(matched_all.items(), key=lambda p: (-p[1], p[0]))[:12]
        ]

        rationale = self._format_rationale(
            text_lc, toolset_scores, skill_scores, toolsets, skills, model_hint
        )

        return ResolvedRole(
            recommended_toolsets=toolsets,
            recommended_skills=skills,
            model_hint=model_hint,
            confidence=round(confidence, 4),
            rationale_md=rationale,
            matched_keywords=matched_keywords,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_rationale(
        text_lc: str,
        toolset_scores: list[tuple[str, float, list[str]]],
        skill_scores: list[tuple[str, float, list[str]]],
        toolsets: list[str],
        skills: list[str],
        model_hint: str | None,
    ) -> str:
        lines = ["**Why these picks?**", ""]
        if toolsets:
            lines.append("**Toolsets above threshold:**")
            for cand, score, matched in toolset_scores:
                if cand in toolsets:
                    kws = ", ".join(matched[:4]) or "(none)"
                    lines.append(f"- `{cand}` — score {score:.2f} — matched: {kws}")
            lines.append("")
        if skills:
            lines.append("**Skills above threshold:**")
            for cand, score, matched in skill_scores:
                if cand in skills:
                    kws = ", ".join(matched[:4]) or "(none)"
                    lines.append(f"- `{cand}` — score {score:.2f} — matched: {kws}")
            lines.append("")
        if model_hint:
            lines.append(f"**Model hint:** `{model_hint}`")
        if not toolsets and not skills:
            lines.append(
                "_No candidates scored above threshold. Try adding more details "
                "or pick a preset role._"
            )
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Optimisation loop (Story 4.14, design §4.2.3)
# ────────────────────────────────────────────────────────────────────────────


def optimize(
    telemetry_path: Path,
    out_path: Path,
    *,
    epochs: int = 50,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-fit per-(keyword, candidate) weights from telemetry and write
    ``weights.json``. Pure-Python, fixed-seed, monotonically improving on the
    training set when ``learning_rate`` is small.

    Telemetry file format (one JSON per line)::

        {"role_text": "...", "skills": [...], "toolsets": [...],
         "success": 1, ...}

    Returns a small report dict.
    """
    import random

    rng = random.Random(seed)
    bundled = _load_bundled_keywords()
    weights: dict[str, dict[str, dict[str, float]]] = {
        "toolsets": {k: dict(v) for k, v in bundled["toolsets"].items()},
        "skills": {k: dict(v) for k, v in bundled["skills"].items()},
    }

    # Read training samples.
    samples: list[dict[str, Any]] = []
    if telemetry_path.exists():
        with open(telemetry_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not samples:
        return {"epochs": 0, "samples": 0, "loss_first": 0.0, "loss_last": 0.0}

    def _loss(samples: list[dict[str, Any]]) -> float:
        total = 0.0
        for s in samples:
            text_lc = _normalise(s.get("role_text", ""))
            label = float(s.get("success", 0))
            # Use the average score across the candidates the operator
            # actually picked (skills + toolsets).  This is intentionally a
            # surrogate; perfect for monotonic improvement.
            picks = list(s.get("toolsets", [])) + list(s.get("skills", []))
            if not picks:
                continue
            scores: list[float] = []
            for cand in picks:
                section = "toolsets" if cand in weights["toolsets"] else "skills"
                w = weights.get(section, {}).get(cand, {})
                score, _ = score_candidate(text_lc, w)
                scores.append(score)
            if not scores:
                continue
            avg_score = sum(scores) / len(scores)
            pred = _sigmoid(avg_score - 1.0)
            total += (pred - label) ** 2
        return total / max(1, len(samples))

    loss_first = _loss(samples)

    for epoch in range(epochs):
        rng.shuffle(samples)
        for s in samples:
            text_lc = _normalise(s.get("role_text", ""))
            label = float(s.get("success", 0))
            for cand in list(s.get("toolsets", [])) + list(s.get("skills", [])):
                section = "toolsets" if cand in weights["toolsets"] else "skills"
                w = weights.get(section, {}).get(cand)
                if not w:
                    continue
                score, _ = score_candidate(text_lc, w)
                pred = _sigmoid(score - 1.0)
                err = label - pred
                # Update each keyword present in this text.
                for kw in list(w.keys()):
                    tf = _tf(_count_keyword(text_lc, kw))
                    if tf == 0:
                        continue
                    delta = learning_rate * err * tf
                    new_w = max(0.05, w[kw] + delta)
                    w[kw] = new_w

    loss_last = _loss(samples)

    out_payload = {
        "_about": "Fitted by `hermes office optimize`. Written automatically; safe to delete to revert to bundled defaults.",
        "thresholds": bundled.get("thresholds", {}),
        "toolsets": weights["toolsets"],
        "skills": weights["skills"],
        "model_hints": bundled.get("model_hints", {}),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "epochs": epochs,
        "samples": len(samples),
        "loss_first": round(loss_first, 6),
        "loss_last": round(loss_last, 6),
    }
