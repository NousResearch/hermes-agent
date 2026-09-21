"""
SLOPESS — Deterministic Prose Quality Linting for AI Output

A zero-LLM heuristics engine that scores AI-generated text for detectable
slop patterns — the visual, structural, and lexical patterns that make prose
immediately identifiable as AI-authored.

Score: 0–10. Higher = more slop detected. Pass threshold: < 6.

Usage:
    from slopless import audit_slop

    result = audit_slop("Your draft text here", context="twitter")
    # => {"slop_score": 3, "issues": [...], "passed": True, "threshold": 6}

CLI:
    python slopless.py "Your draft text here" --context twitter
    python slopless.py --file draft.txt
"""

import re
import argparse
import sys
from typing import Dict, List


# ─── Pattern Registry ───────────────────────────────────────────────────────

BOILERPLATE_MANTRAS = [
    "Shipping apps. Breaking things.",
    "Tap to play.",
    "Make the call.",
    "Your move.",
    "Game on.",
    "Let's build.",
    "Built different.",
    "Say less.",
    "We move.",
    "Trust the process.",
]

AI_ISMS = re.compile(
    r"\b("
    r"delve into|"
    r"it's important to note|"
    r"let's be honest|"
    r"in today's digital age|"
    r"at the end of the day|"
    r"a testament to|"
    r"let's dive in|"
    r"dive into|"
    r"in today's (?:fast-paced|rapidly changing|ever-evolving) world|"
    r"when it comes to|"
    r"in the realm of|"
    r"in the world of|"
    r"whether you're|"
    r"a game-changer|"
    r"a double-edged sword|"
    r"navigating the (?:landscape|complexities|world)|"
    r"unlock the full potential|"
    r"tapestry of|"
    r"rich tapestry|"
    r"it's worth noting that|"
    r"it is worth noting that|"
    r"ultimately, it boils down to|"
    r"at the end of the day, "
    r")",
    re.IGNORECASE,
)

TEMPLATE_ITIS_PATTERNS = [
    r"Every AI tool now has a\s+['\"]?\w[\w\s]+['\"]?\s+feature",
    r"In the (?:fast-paced|rapidly changing|ever-evolving) (?:world|landscape) of",
    r"As a \w+\s*, I (?:often|frequently|always|sometimes)",
    r"The truth is, \w+",
    r"Whether you're a \w+\s*or a \w+",
    r"(?:Imagine|Picture) (?:this|a world where|having the ability to)",
    r"Gone are the days when",
    r"(?:In this|Throughout this) (?:article|guide|post|blog)",
    r"So, (?:what are you|why are you) waiting for\??",
]

TEMPLATE_ITIS_RE = re.compile(
    "|".join(TEMPLATE_ITIS_PATTERNS), re.IGNORECASE
)

SPECIFICITY_PATTERNS = {
    "number": re.compile(r"\b\d+%|\b\d+x|\b\d+\s*hours?|\b£[\d,.]+|\b\$\d[\d,.]*|\b\d+\s*(?:weeks?|months?|days?|years?)\b"),
    "tool_tech": re.compile(r"\b(Claude|Convex|Supabase|GitHub|Vercel|Postgres|Python|React|Node\.?js|Docker|AWS|GCP|Kubernetes|TensorFlow|PyTorch|llama\.cpp|vLLM|SGLang|Ollama|Hermes|KENSEI)\b", re.IGNORECASE),
    "app_product": re.compile(r"\b(Plenishd|MatchdayMaestro|CoachOS|Kick-tionary|Player Portfolio Builder|Kensei|Kinexio|SoftwareOne|Facilities Management)\b", re.IGNORECASE),
    "timeframe": re.compile(r"\b\d+\s*(weeks?|months?|days?|years?|Q\d|20\d{2})\b", re.IGNORECASE),
    "url": re.compile(r"https?://\S+"),
}

GENERIC_FILLER_PHRASES = re.compile(
    r"\b("
    r"game-changer|"
    r"groundbreaking|"
    r"revolutionary|"
    r"next-generation|"
    r"cutting-edge|"
    r"state-of-the-art|"
    r"unprecedented|"
    r"innovative|"
    r"powerful|"
    r"robust|"
    r"seamless|"
    r"intuitive|"
    r"comprehensive|"
    r"tailored|"
    r"customised|"
    r"bespoke|"
    r"transformative)"
    r"\b",
    re.IGNORECASE,
)


# ─── Context Profiles ──────────────────────────────────────────────────────

CONTEXT_CONFIG = {
    "general": {
        "short_line_threshold": 30,
        "short_line_minimum": 4,
        "hashtag_allowance": 2,
        "ai_ism_weight": 1,
        "ai_ism_cap": 3,
        "over_polished_weight": 1,
        "filler_zero_score": 3,
        "filler_one_score": 1,
        "filler_strict_word_count": None,
        "inspirational_quote_penalty": 0,
    },
    "twitter": {
        "short_line_threshold": 20,
        "short_line_minimum": 4,
        "hashtag_allowance": 1,
        "ai_ism_weight": 1,
        "ai_ism_cap": 3,
        "over_polished_weight": 0.5,
        "filler_zero_score": 3,
        "filler_one_score": 1,
        "filler_strict_word_count": None,
        "inspirational_quote_penalty": 0,
    },
    "linkedin": {
        "short_line_threshold": 30,
        "short_line_minimum": 4,
        "hashtag_allowance": 3,
        "ai_ism_weight": 2,
        "ai_ism_cap": 4,
        "over_polished_weight": 1,
        "filler_zero_score": 3,
        "filler_one_score": 1,
        "filler_strict_word_count": None,
        "inspirational_quote_penalty": 2,
    },
    "blog": {
        "short_line_threshold": 30,
        "short_line_minimum": 4,
        "hashtag_allowance": 1,
        "ai_ism_weight": 1,
        "ai_ism_cap": 3,
        "over_polished_weight": 0,  # relaxed
        "filler_zero_score": 3,
        "filler_one_score": 1,
        "filler_strict_word_count": 200,
        "inspirational_quote_penalty": 0,
    },
    "marketing": {
        "short_line_threshold": 30,
        "short_line_minimum": 4,
        "hashtag_allowance": 2,
        "ai_ism_weight": 1,
        "ai_ism_cap": 3,
        "over_polished_weight": 1,
        "filler_zero_score": 3,
        "filler_one_score": 1,
        "filler_strict_word_count": None,
        "inspirational_quote_penalty": 0,
    },
}


# ─── Core Audit ─────────────────────────────────────────────────────────────

def audit_slop(body_text: str, *, context: str = "general") -> Dict:
    """
    Evaluate prose for AI-detectable slop patterns.

    Args:
        body_text: The draft text to evaluate.
        context: One of "general", "twitter", "linkedin", "blog", "marketing".

    Returns:
        {
            "slop_score": int,      # 0–10 (capped)
            "issues": List[str],    # human-readable findings
            "passed": bool,         # True if slop_score < 6
            "threshold": int,       # always 6
            "breakdown": Dict,      # raw scores per category
        }
    """
    cfg = CONTEXT_CONFIG.get(context, CONTEXT_CONFIG["general"])
    issues: List[str] = []
    breakdown: Dict[str, int] = {}
    score = 0

    text = body_text.strip()
    if not text:
        return {
            "slop_score": 0,
            "issues": ["empty text"],
            "passed": True,
            "threshold": 6,
            "breakdown": {},
        }

    # 1. Boilerplate mantras
    mantra_score = 0
    for mantra in BOILERPLATE_MANTRAS:
        if mantra.lower() in text.lower():
            issues.append(f"boilerplate mantra: \"{mantra}\"")
            mantra_score += 2
    breakdown["boilerplate"] = mantra_score
    score += mantra_score

    # 2. Template-itis
    template_matches = TEMPLATE_ITIS_RE.findall(text)
    if template_matches:
        template_score = 2
        breakdown["template_itis"] = template_score
        score += template_score
        issues.append(f"template-itis: {len(template_matches)} template-like structure(s) detected")

    # 3. Specificity / Generic filler
    specificity_count = sum(
        1 for patt in SPECIFICITY_PATTERNS.values() if patt.search(text)
    )
    filler_words = set(m.lower() for m in GENERIC_FILLER_PHRASES.findall(text))

    if cfg["filler_strict_word_count"]:
        # For blog: check 200+ word blocks independently
        words = text.split()
        wc = cfg["filler_strict_word_count"]
        for i in range(0, len(words), wc):
            block = " ".join(words[i:i + wc])
            if len(block.split()) >= wc:
                if not any(p.search(block) for p in SPECIFICITY_PATTERNS.values()):
                    specificity_count = 0
                    break

    if specificity_count == 0:
        filler_score = cfg["filler_zero_score"]
    elif specificity_count == 1:
        filler_score = cfg["filler_one_score"]
    else:
        filler_score = 0

    if filler_words:
        issues.append(f"generic filler: {', '.join(sorted(filler_words))[:120]}")

    breakdown["generic_filler"] = filler_score
    score += filler_score

    # 4. Over-polished structure (stanza-like, all short lines)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= cfg["short_line_minimum"]:
        all_short = all(len(l) < cfg["short_line_threshold"] for l in lines)
        if all_short:
            w = cfg["over_polished_weight"]
            if w > 0:
                breakdown["over_polished"] = int(w)
                score += int(w)
                issues.append("over-polished structure: stanza-like, all lines short; vary length or use paragraph")

    # 5. Slant hashtags & adverb bloat
    hashtags = re.findall(r"#\w+", text)
    if len(hashtags) > cfg["hashtag_allowance"]:
        hashtag_score = 1
        breakdown["hashtags"] = hashtag_score
        score += hashtag_score
        issues.append(f"hashtag bloat: {len(hashtags)} hashtags (max {cfg['hashtag_allowance']} for {context})")

    # Adverb bloat: 4+ -ly adverbs in text
    adverbs = re.findall(r"\b\w+ly\b", text)
    if len(adverbs) >= 4:
        breakdown["adverb_bloat"] = 1
        score += 1
        issues.append(f"adverb bloat: {len(adverbs)} -ly adverbs")

    # 6. Corporate AI-isms
    ism_matches = AI_ISMS.findall(text)
    if ism_matches:
        isms_unique = set(m.lower() for m in ism_matches)
        ism_cap = int(cfg["ai_ism_cap"])
        ism_weight = cfg["ai_ism_weight"] * len(isms_unique)
        ism_score = min(ism_weight, ism_cap)
        breakdown["ai_isms"] = ism_score
        score += ism_score
        issues.append(f"AI-isms ({len(isms_unique)}): {', '.join(sorted(isms_unique)[:5])}")

    # Inspirational quote penalty (for LinkedIn)
    if cfg["inspirational_quote_penalty"] > 0:
        quote_patterns = [
            r"^(Remember|Never|Always|Believe|Dream)\b.*\.$",
            r"\b(success is not about|the journey|chase your dreams|believe in yourself)\b",
        ]
        for qp in quote_patterns:
            if re.search(qp, text, re.IGNORECASE | re.MULTILINE):
                breakdown["inspirational_quote"] = cfg["inspirational_quote_penalty"]
                score += cfg["inspirational_quote_penalty"]
                issues.append("generic inspirational quote pattern detected")
                break

    # Cap at 10
    score = min(score, 10)

    return {
        "slop_score": score,
        "issues": issues,
        "passed": score < 6,
        "threshold": 6,
        "breakdown": breakdown,
    }


# ─── Batch Lint ──────────────────────────────────────────────────────────────

def batch_lint(drafts: List[Dict]) -> List[Dict]:
    """
    Lint multiple drafts at once.

    Args:
        drafts: [{"id": str, "body": str, "context": str}, ...]

    Returns:
        [{"id", "slop_score", "issues", "passed"}, ...]
    """
    results = []
    for d in drafts:
        audit = audit_slop(d["body"], context=d.get("context", "general"))
        results.append({"id": d["id"], **audit})
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SLOPESS — Deterministic prose quality linting"
    )
    p.add_argument("text", nargs="?", help="Prose text to audit (or use --file)")
    p.add_argument("--file", "-f", help="Read text from file")
    p.add_argument("--context", "-c", default="general",
                    choices=["general", "twitter", "linkedin", "blog", "marketing"],
                    help="Content context (default: general)")
    p.add_argument("--json", action="store_true", help="Output JSON")
    args = p.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        p.error("Provide TEXT or use --file")

    result = audit_slop(text, context=args.context)

    if args.json:
        import json
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"SLOPESS audit [{args.context}] — {status}")
        print(f"Score: {result['slop_score']}/10 (threshold: {result['threshold']})")
        if result["issues"]:
            print("Issues:")
            for i in result["issues"]:
                print(f"  - {i}")
        if result["breakdown"]:
            print("Breakdown:")
            for k, v in result["breakdown"].items():
                print(f"  {k}: +{v}")


if __name__ == "__main__":
    main()
