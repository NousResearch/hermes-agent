"""Reply suggestion pipeline for X/Twitter.

Scans target accounts, identifies high-value posts to reply to, generates
thoughtful contextual replies, and queues them for approval.

The x-algorithm weights replies-with-author-interaction at ~150x — this is
the single highest-leverage activity for growth on X in 2026.
"""

import json
import random
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

# X accounts whose recent posts we scan for reply opportunities
TARGET_ACCOUNTS = [
    # Indie dev / build-in-public
    "marc_louvion",
    "levelsio",
    "tahseen_rahman",
    "shadcn",
    "rauchg",
    "theo",
    "swyx",
    "ryan_c_harris",
    "kentcdodds",
    "pk_hal",
    # AI tools / Claude Code ecosystem
    "AnthropicAI",
    "claude_code",
    "alexalbert__",
    "amasad",
    # Football / MUFC
    "TheAthleticFC",
    "utdreport",
    "ManUtd",
    "GaryLineker",
    # Tech / startups
    "naval",
    "paulg",
    "sweatystartup",
]

# Reply style patterns mapped to post categories
REPLY_PATTERNS = {
    "build_update": [
        "add_data_point",
        "share_parallel_experience",
        "ask_deepening_question",
    ],
    "technical_tip": [
        "affirm_and_extend",
        "offer_alternative",
        "share_own_experience",
    ],
    "opinion": [
        "dry_agreement_with_twist",
        "offer_counterpoint_respectfully",
        "share_relevant_data",
    ],
    "observation": [
        "add_specific_example",
        "dry_agreement_with_twist",
        "build_on_insight",
    ],
    "milestone": [
        "congratulate_specific",
        "ask_about_next_step",
        "share_parallel_milestone",
    ],
    "football": [
        "dry_observation",
        "build_on_moment",
        "historical_comparison",
    ],
}

REPLY_EXEMPLARS = {
    "add_data_point": "Veracode's data backs this — 86% of AI-generated code samples failed XSS defence in their latest eval.",
    "share_parallel_experience": "Hit this same issue last week. Fixed it by switching from sequential prompts to a sub-agent loop with parallel workers.",
    "ask_deepening_question": "How are you handling the edge case where the context window fills mid-conversation? That's where my setup breaks.",
    "affirm_and_extend": "Same pattern here. Adding: the sub-agent gets its own prompt file with explicit constraints — stops it drifting in long runs.",
    "offer_alternative": "Worth testing Convex for the backend. Supabase is great but the realtime layer was the bottleneck for me in this pattern.",
    "share_own_experience": "I ran a 4-app portfolio through this exact pipeline. The thing that broke first wasn't the code — it was the content generation quality gate.",
    "dry_agreement_with_twist": "True. Though I'd argue the real bottleneck shifts from model selection to context engineering once you hit production.",
    "offer_counterpoint_respectfully": "Interesting take. My experience has been different — Convex's schema validation caught more edge cases than my unit tests did.",
    "share_relevant_data": "I tracked this across 4 apps. Average time saved per debugging session with Claude Code: 47 minutes. That's 3+ hours/week reclaimed.",
    "add_specific_example": "This showed up in my pipeline too. The 'Every AI tool now has a memory feature' pattern was generating 7x more traffic than the technical deep dives.",
    "build_on_insight": "The next layer after this: template diversity matters more than template quality. Having 132 templates beats 10 perfect ones.",
    "congratulate_specific": "The 4-step decision framework you shared last week saved me from shipping a feature I'd have killed 2 weeks later. Appreciate the clarity.",
    "ask_about_next_step": "Solid milestone. What's the one thing that surprised you most about the build that you didn't expect going in?",
    "share_parallel_milestone": "Shipped my 4th app the same week. The feeling doesn't change — still that 'did I actually ship this?' moment every time.",
    "dry_observation": "One-nil at half time. Every United fan I know is already pacing. The trauma is the brand at this point.",
    "historical_comparison": "The 1999 treble team was 1-0 down at half time in the final. Tell your mates to calm down. It's not over till it's over.",
}

# ──────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────

def _categorize_post(post: Dict) -> str:
    """Guess which category a post belongs to for reply pattern selection.
    
    Uses keyword heuristics rather than an LLM call (cheaper, faster).
    """
    text = (post.get("text", "") + " " + " ".join(post.get("hashtags", []))).lower()
    
    if any(w in text for w in ["ship", "shipped", "launch", "milestone", "released", "v1", "v2"]):
        return "milestone"
    if any(w in text for w in ["how to", "step", "tutorial", "guide", "setup", "config", "workflow"]):
        return "technical_tip"
    if any(w in text for w in ["think", "opinion", "hot take", "unpopular", "believe"]):
        return "opinion"
    if any(w in text for w in ["noticed", "observ", "pattern", "trend", "every", "always"]):
        return "observation"
    if any(w in text for w in ["goal", "match", "united", "fc", "premier league", "half time", "final"]):
        return "football"
    
    return "build_update"


def _sample_reply_pattern(post_category: str) -> str:
    """Pick a reply pattern suited to the post category."""
    patterns = REPLY_PATTERNS.get(post_category, ["add_data_point", "share_parallel_experience"])
    return random.choice(patterns)


def _generate_reply(post: Dict, template_pattern: str, account_handle: str) -> Optional[str]:
    """Generate a reply from the exemplar bank, contextualised to the post.
    
    Returns None if no suitable template could be generated.
    """
    template = REPLY_EXEMPLARS.get(template_pattern)
    if not template:
        return None
    
    # Add @mention
    reply = f"@{account_handle} {template}"
    
    # Truncate to 280 chars per X limits
    if len(reply) > 280:
        reply = reply[:277] + "..."
    
    return reply


def suggest_replies_for_post(post: Dict, account_handle: str) -> List[Dict]:
    """Generate 1-2 reply suggestions for a given post.
    
    Returns list of dicts with 'reply_text', 'pattern', 'category'.
    Empty list if post isn't worth replying to.
    """
    # ── Anti-scripted guardrail ──
    # Without actual source tweet context, return no suggestion. Canned
    # facts/metrics in exemplars must not be used without real tweet content
    # to contextualise against.
    text = post.get("text", "")
    if len(text) < 10:
        return []
    
    # Require actual tweet content — not a placeholder or synthetic post.
    # Real tweets have substantive text; synthetic/placeholder posts don't.
    if len(text) < 40:
        return []
    
    # Skip engagement bait / bot-looking content
    bait_patterns = [
        r"RT if", r"retweet if", r"like if", r"comment below",
        r"follow for", r"tag someone", r"mention someone",
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in bait_patterns):
        return []
    
    category = _categorize_post(post)
    pattern = _sample_reply_pattern(category)
    reply_text = _generate_reply(post, pattern, account_handle)
    
    if not reply_text:
        return []
    
    suggestions = [{
        "reply_text": reply_text,
        "pattern": pattern,
        "category": category,
    }]
    
    # 30% chance of a second suggestion with a different pattern
    if random.random() < 0.3:
        second_pattern = _sample_reply_pattern(category)
        while second_pattern == pattern and len(REPLY_PATTERNS.get(category, [])) > 1:
            second_pattern = _sample_reply_pattern(category)
        
        second_reply = _generate_reply(post, second_pattern, account_handle)
        if second_reply:
            suggestions.append({
                "reply_text": second_reply,
                "pattern": second_pattern,
                "category": category,
            })
    
    return suggestions


def run_reply_suggestions(dry_run: bool = False) -> List[Dict]:
    """Main entry point: fetch recent posts from target accounts and suggest replies.
    
    Returns list of suggested replies with metadata.
    
    NOTE: Actual X API calls are NOT implemented here. This module provides
    the pipeline structure and reply generation logic. When xurl is activated,
    wire it here to fetch real posts.
    
    For now, returns placeholder output showing the pipeline works.
    """
    suggestions = []
    
    if dry_run:
        for account in TARGET_ACCOUNTS[:5]:
            print(f"  [{account}] Would fetch timeline (via xurl)")
            print(f"    Would generate 2-5 reply suggestions per high-value post")
        return []
    
    # Not yet wired. x_scout.py and engagement_suggester.py already use xurl
    # for timeline fetching — mirror that pattern here when this pipeline is
    # activated.
    
    print("Reply suggestion pipeline ready. Wire xurl to activate.")
    return suggestions


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    results = run_reply_suggestions(dry_run=dry)
    
    if not dry:
        print(f"Generated {len(results)} reply suggestions")
        for r in results[:5]:
            print(f"  [{r['category']}/{r['pattern']}] {r['reply_text'][:80]}")
