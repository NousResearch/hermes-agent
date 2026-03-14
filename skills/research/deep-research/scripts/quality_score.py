#!/usr/bin/env python3
"""
Deep Research Quality Scorer — Proof-Based Validation

The agent doesn't rate itself. It provides PROOF OBJECTS and this script
computes the score. If the proof is weak or missing, the score reflects that.

Usage:
  python3 quality_score.py assess --proof proof.json
  python3 quality_score.py checklist

Proof object format (JSON):
{
  "sub_questions": [
    {"question": "What is X?", "answered": true, "sources": ["url1", "url2"]},
    {"question": "Why does Y?", "answered": false, "sources": []}
  ],
  "source_types_used": ["web_search", "news", "scraping"],
  "source_tiers": {"tier1": 2, "tier2": 3, "tier3": 1, "tier4": 0, "tier5": 0},
  "contrarian_searches": [
    {"query": "why X is wrong", "found_evidence": true, "result_summary": "..."}
  ],
  "claims": [
    {"claim": "X costs Y", "status": "verified", "sources": ["url1", "url2"]},
    {"claim": "Z will happen", "status": "supported", "sources": ["url1"]},
    {"claim": "A is true", "status": "ai_inference", "sources": []},
    {"claim": "B is true", "status": "unverified", "sources": []}
  ],
  "hypothesis_revised": true,
  "hypothesis_original": "I think X because Y",
  "hypothesis_revised_text": "I now think X' because Z",
  "contradictions": [
    {"contradiction": "Source A says X, Source B says Y", "resolved": true, "resolution": "..."},
    {"contradiction": "...", "resolved": false, "resolution": ""}
  ],
  "gaps_identified": [
    {"gap": "Missing data on Z", "rank": "critical", "filled": true},
    {"gap": "Unclear timeline", "rank": "important", "filled": false},
    {"gap": "Nice to have detail", "rank": "nice_to_have", "filled": false}
  ],
  "search_rounds": 2,
  "ai_inference_labeled": true,
  "platform": "telegram"
}
"""

import json
import sys
import argparse
from typing import Optional


# Quality dimensions with weights (total = 1.0)
DIMENSIONS = {
    "coverage": {
        "name": "Sub-Question Coverage",
        "weight": 0.20,
        "description": "All sub-questions answered with at least 1 source each",
    },
    "diversity": {
        "name": "Source Diversity",
        "weight": 0.10,
        "description": "Multiple source types used (not just one search tool)",
    },
    "source_quality": {
        "name": "Source Quality",
        "weight": 0.10,
        "description": "Higher-tier sources (primary/analyst) outweigh lower-tier",
    },
    "contrarian": {
        "name": "Contrarian Evidence",
        "weight": 0.15,
        "description": "Actively searched for evidence against the hypothesis",
    },
    "verification": {
        "name": "Claim Verification",
        "weight": 0.15,
        "description": "Key claims traced to sources, AI inference labeled",
    },
    "revision": {
        "name": "Hypothesis Revision",
        "weight": 0.10,
        "description": "Hypothesis updated based on evidence",
    },
    "contradictions": {
        "name": "Contradiction Resolution",
        "weight": 0.10,
        "description": "Contradictions identified and resolved where possible",
    },
    "saturation": {
        "name": "Gap Analysis & Saturation",
        "weight": 0.10,
        "description": "Gaps identified, ranked, and critical gaps filled",
    },
}

MIN_SEARCH_ROUNDS = 2
MIN_CONTRARIAN_SEARCHES = 1
MIN_SOURCE_TYPES = 2
MIN_TIER1_TIER2_SOURCES = 1


def score_coverage(proof: dict) -> tuple[float, list[str]]:
    """Score sub-question coverage. Deductions for unanswered questions."""
    questions = proof.get("sub_questions", [])
    if not questions:
        return 0.0, ["No sub-questions defined — coverage cannot be assessed"]

    total = len(questions)
    answered = sum(1 for q in questions if q.get("answered") and q.get("sources"))
    unanswered = total - answered

    score = answered / total
    notes = []
    if unanswered > 0:
        unanswered_qs = [q["question"] for q in questions if not q.get("answered") or not q.get("sources")]
        for q in unanswered_qs:
            notes.append(f"UNANSWERED: {q}")
        score -= 0.05 * unanswered  # penalty per unanswered

    return max(0.0, min(1.0, score)), notes


def score_diversity(proof: dict) -> tuple[float, list[str]]:
    """Score source type diversity. Need at least MIN_SOURCE_TYPES different types."""
    types = proof.get("source_types_used", [])
    count = len(set(types))

    if count == 0:
        return 0.0, ["No source types recorded"]
    elif count == 1:
        return 0.3, [f"Only 1 source type used ({types[0]}). Add at least {MIN_SOURCE_TYPES - 1} more."]
    elif count == 2:
        return 0.7, [f"2 source types used: {', '.join(types)}"]
    elif count >= 3:
        return 1.0, [f"{count} source types used: {', '.join(types)}"]
    return 0.5, []


def score_source_quality(proof: dict) -> tuple[float, list[str]]:
    """Score based on source tier distribution. Higher tiers = better."""
    tiers = proof.get("source_tiers", {})
    t1 = tiers.get("tier1", 0)
    t2 = tiers.get("tier2", 0)
    t3 = tiers.get("tier3", 0)
    t4 = tiers.get("tier4", 0)
    t5 = tiers.get("tier5", 0)

    total = t1 + t2 + t3 + t4 + t5
    if total == 0:
        return 0.0, ["No sources recorded"]

    # Weighted: tier1=1.0, tier2=0.8, tier3=0.5, tier4=0.2, tier5=0.0
    quality_sum = t1 * 1.0 + t2 * 0.8 + t3 * 0.5 + t4 * 0.2 + t5 * 0.0
    score = quality_sum / total

    notes = []
    if t1 + t2 == 0:
        notes.append("WARNING: No Tier 1-2 sources (primary/analyst). Only reporting and secondary.")
        score -= 0.2
    if t5 > t1 + t2 + t3:
        notes.append("WARNING: More unverified sources than verified ones.")

    return max(0.0, min(1.0, score)), notes


def score_contrarian(proof: dict) -> tuple[float, list[str]]:
    """Score contrarian search. Must have explicit searches against hypothesis."""
    searches = proof.get("contradictions_found", [])  # legacy key
    contrarian = proof.get("contrarian_searches", [])

    if not contrarian:
        # Check if contradictions list has evidence
        if searches:
            return 0.3, ["Contradictions found but no explicit contrarian search performed"]
        return 0.0, ["No contrarian searches performed. Run searches against your hypothesis."]

    count = len(contrarian)
    with_evidence = sum(1 for s in contrarian if s.get("found_evidence"))

    if count < MIN_CONTRARIAN_SEARCHES:
        return 0.2, [f"Only {count} contrarian search(es). Minimum {MIN_CONTRARIAN_SEARCHES} required."]

    if with_evidence == 0:
        return 0.5, ["Contrarian searches run but no contradictory evidence found (or not searched properly)"]

    score = min(1.0, 0.6 + (with_evidence * 0.2))
    return score, [f"{count} contrarian searches, {with_evidence} found evidence against hypothesis"]


def score_verification(proof: dict) -> tuple[float, list[str]]:
    """Score claim verification. Claims must be rated and traced."""
    claims = proof.get("claims", [])
    if not claims:
        return 0.0, ["No claims recorded for verification"]

    total = len(claims)
    verified = sum(1 for c in claims if c.get("status") == "verified")
    supported = sum(1 for c in claims if c.get("status") == "supported")
    ai_inference = sum(1 for c in claims if c.get("status") == "ai_inference")
    unverified = sum(1 for c in claims if c.get("status") == "unverified")

    notes = []

    # Penalty for unverified claims that should be verified
    if unverified > 0:
        notes.append(f"{unverified} UNVERIFIED claim(s) — should be removed or sourced")

    # Penalty if AI inference not labeled
    if not proof.get("ai_inference_labeled", False):
        # Check if any claims look like inference but aren't labeled
        unlabeled_inference = sum(1 for c in claims if c.get("status") not in ["verified", "supported", "ai_inference", "unverified"])
        if unlabeled_inference > 0:
            notes.append(f"{unlabeled_inference} claims have unclear status — label as verified/supported/ai_inference/unverified")

    # Score: verified+sourced = good, unverified = bad, ai_inference labeled = fine
    good_claims = verified + supported
    score = (good_claims / total) * 0.8 + (0.2 if ai_inference > 0 and proof.get("ai_inference_labeled") else 0)

    # Hard penalty for unverified claims
    score -= 0.1 * unverified

    return max(0.0, min(1.0, score)), notes


def score_revision(proof: dict) -> tuple[float, list[str]]:
    """Score hypothesis revision. Must be revised if evidence contradicts original."""
    revised = proof.get("hypothesis_revised", False)
    original = proof.get("hypothesis_original", "")
    revised_text = proof.get("hypothesis_revised_text", "")

    if not original:
        return 0.0, ["No hypothesis stated"]

    if not revised:
        # Check if there's contradicting evidence that should have triggered revision
        contradictions = proof.get("contradictions", [])
        contrarian = proof.get("contrarian_searches", [])
        found_against = any(c.get("found_evidence") for c in contrarian) if contrarian else False

        if found_against or any(c.get("resolved") for c in contradictions):
            return 0.3, ["Evidence contradicts original hypothesis but hypothesis was NOT revised — this is a red flag"]
        return 0.7, ["Hypothesis stated but not revised (no strong contradicting evidence found)"]

    if revised and revised_text:
        # Good — revised and stated
        if revised_text == original:
            return 0.5, ["Hypothesis marked as revised but text is identical to original"]
        return 1.0, [f"Hypothesis revised: '{original}' -> '{revised_text}'"]

    return 0.5, ["Hypothesis marked as revised but revised text not provided"]


def score_contradictions(proof: dict) -> tuple[float, list[str]]:
    """Score contradiction handling. Identify AND resolve."""
    contradictions = proof.get("contradictions", [])
    if not contradictions:
        # No contradictions might mean none found OR not looked for
        contrarian = proof.get("contrarian_searches", [])
        if not contrarian:
            return 0.3, ["No contradictions recorded and no contrarian searches — may not have looked"]
        return 0.7, ["No contradictions found (contrarian searches were performed)"]

    total = len(contradictions)
    resolved = sum(1 for c in contradictions if c.get("resolved"))

    if resolved == 0:
        return 0.2, [f"{total} contradiction(s) identified but NONE resolved"]

    score = resolved / total
    return score, [f"{resolved}/{total} contradictions resolved"]


def score_saturation(proof: dict) -> tuple[float, list[str]]:
    """Score gap analysis and saturation. Must have multiple rounds for full score."""
    gaps = proof.get("gaps_identified", [])
    rounds = proof.get("search_rounds", 1)

    notes = []

    if not gaps:
        if rounds < MIN_SEARCH_ROUNDS:
            return 0.1, [f"No gaps recorded and only {rounds} search round(s). Run gap analysis."]
        return 0.4, ["No gaps recorded but multiple search rounds performed"]

    # Check gap ranking
    ranked = sum(1 for g in gaps if g.get("rank") in ["critical", "important", "nice_to_have"])
    if ranked < len(gaps):
        notes.append(f"{len(gaps) - ranked} gap(s) not ranked by impact")

    # Check critical gaps filled
    critical = [g for g in gaps if g.get("rank") == "critical"]
    critical_filled = sum(1 for g in critical if g.get("filled"))

    if critical and critical_filled < len(critical):
        notes.append(f"{len(critical) - critical_filled} CRITICAL gap(s) still unfilled")
        return 0.4, notes

    # Round bonus
    if rounds < MIN_SEARCH_ROUNDS:
        notes.append(f"Only {rounds} search round(s). Minimum {MIN_SEARCH_ROUNDS} recommended.")
        return 0.6, notes

    # All good
    score = 0.7 + (0.1 * min(rounds - 1, 3))  # bonus for extra rounds, capped at 3
    filled = sum(1 for g in gaps if g.get("filled"))
    if filled == len(gaps):
        score = min(1.0, score + 0.1)

    return score, notes


def compute_score(proof: dict) -> dict:
    """Compute overall quality score from proof object."""
    scorers = {
        "coverage": score_coverage,
        "diversity": score_diversity,
        "source_quality": score_source_quality,
        "contrarian": score_contrarian,
        "verification": score_verification,
        "revision": score_revision,
        "contradictions": score_contradictions,
        "saturation": score_saturation,
    }

    results = {}
    total_score = 0.0
    all_notes = []

    for key, scorer in scorers.items():
        dim = DIMENSIONS[key]
        score, notes = scorer(proof)
        weighted = score * dim["weight"]
        total_score += weighted
        results[key] = {
            "name": dim["name"],
            "weight": dim["weight"],
            "raw_score": round(score, 2),
            "weighted_score": round(weighted, 3),
            "notes": notes,
        }
        all_notes.extend(notes)

    # Determine status
    if total_score >= 0.90:
        status = "SATURATED"
        message = "Research is complete and high quality."
    elif total_score >= 0.70:
        status = "GOOD"
        message = "Research is solid but could be improved."
    elif total_score >= 0.50:
        status = "ADEQUATE"
        message = "Research has significant gaps. Address weak dimensions."
    else:
        status = "INSUFFICIENT"
        message = "Research is below acceptable quality. Major improvements needed."

    # Weak dimensions (below 0.5 raw)
    weak = [r["name"] for r in results.values() if r["raw_score"] < 0.5]

    return {
        "score": round(total_score, 3),
        "status": status,
        "message": message,
        "dimensions": results,
        "weak_dimensions": weak,
        "actionable_notes": all_notes,
        "search_rounds": proof.get("search_rounds", 1),
    }


def print_checklist():
    """Print the proof checklist the agent must fill."""
    print("""
=== DEEP RESEARCH QUALITY PROOF CHECKLIST ===

Before computing score, the agent must provide a proof JSON with:

1. SUB-QUESTIONS (Coverage)
   - List all sub-questions from Phase 1
   - Mark each as answered: true/false
   - List sources for each answered question

2. SOURCE TYPES (Diversity)
   - List all tool types used (e.g., "ddgs", "scrapling", "twitter", "browser")
   - Minimum 2 different types required

3. SOURCE TIERS (Quality)
   - Count sources by tier:
     tier1: Primary (gov data, official reports, direct quotes)
     tier2: Expert analysis (analyst reports, peer-reviewed)
     tier3: Credible reporting (Reuters, Bloomberg, BBC)
     tier4: Secondary (blogs, opinion, regional news)
     tier5: Unverified (social media, anonymous)

4. CONTRARIAN SEARCHES (Contrarian)
   - List searches run AGAINST your hypothesis
   - For each: query, whether evidence was found, summary
   - Minimum 1 required

5. CLAIMS (Verification)
   - List every specific claim in your draft answer
   - Rate each: "verified" (2+ sources), "supported" (1 source),
     "ai_inference" (your analysis, label it), "unverified" (remove it)
   - Set ai_inference_labeled: true

6. HYPOTHESIS (Revision)
   - State original hypothesis (from Phase 1)
   - Set hypothesis_revised: true/false
   - If revised, state revised hypothesis

7. CONTRADICTIONS
   - List contradictions found between sources
   - For each: description, resolved: true/false, resolution text

8. GAPS (Saturation)
   - List gaps identified in gap analysis
   - Rank each: "critical", "important", "nice_to_have"
   - Mark filled: true/false for each
   - Record search_rounds (how many iterations)

Run: python3 quality_score.py assess --proof your_proof.json
""")


def cmd_assess(args):
    """Assess research quality from proof file."""
    try:
        with open(args.proof, 'r', encoding='utf-8') as f:
            proof = json.load(f)
    except FileNotFoundError:
        print(f"Error: Proof file '{args.proof}' not found")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in proof file: {e}")
        return 1

    result = compute_score(proof)

    # Output
    print(json.dumps(result, indent=2))

    # Also print human-readable summary
    print(f"\n=== QUALITY SCORE: {result['score']:.3f} ({result['status']}) ===")
    print(f"{result['message']}\n")

    for key, dim in result["dimensions"].items():
        bar = "█" * int(dim["raw_score"] * 20) + "░" * (20 - int(dim["raw_score"] * 20))
        print(f"  {dim['name']:30s} {bar} {dim['raw_score']:.2f} (weight: {dim['weight']:.0%})")
        for note in dim["notes"]:
            print(f"    -> {note}")

    if result["weak_dimensions"]:
        print(f"\n  WEAK: {', '.join(result['weak_dimensions'])}")

    if result["score"] < 0.90:
        print(f"\n  NOT SATURATED — continue research loop")
    else:
        print(f"\n  SATURATED — proceed to synthesis")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Deep Research Quality Scorer")
    sub = parser.add_subparsers(dest="command")

    p_assess = sub.add_parser("assess", help="Assess quality from proof JSON")
    p_assess.add_argument("--proof", required=True, help="Path to proof JSON file")

    sub.add_parser("checklist", help="Print the proof checklist")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "checklist":
        print_checklist()
        return 0
    elif args.command == "assess":
        return cmd_assess(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
