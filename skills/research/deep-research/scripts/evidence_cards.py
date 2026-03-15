#!/usr/bin/env python3
"""
Evidence Cards — Structured Evidence Tracking for Deep Research

Every piece of evidence becomes a card with metadata, verification status,
and bias detection. This prevents claims from floating unanchored.

Usage:
  python3 evidence_cards.py init                        # Create new card deck
  python3 evidence_cards.py add --claim "X" --source "url" --type "primary"
  python3 evidence_cards.py verify --card-id "card-001" --status "verified"
  python3 evidence_cards.py debate --card-id "card-001"  # Generate FOR/AGAINST
  python3 evidence_cards.py export                       # Export all cards as JSON
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)


def get_state_dir():
    """Return a writable state directory for evidence cards."""
    override = os.environ.get("HERMES_DEEP_RESEARCH_STATE_DIR")
    if override:
        return override
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return os.path.join(hermes_home, "state", "deep-research")
    return str(Path.home() / ".hermes" / "state" / "deep-research")


STATE_DIR = get_state_dir()
CARDS_DIR = os.path.join(STATE_DIR, "evidence_cards")


def timestamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def load_deck(deck_path: str) -> dict:
    """Load the evidence card deck."""
    if os.path.exists(deck_path):
        with open(deck_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"cards": [], "created": timestamp(), "session": None}


def save_deck(deck_path: str, deck: dict):
    """Save the evidence card deck."""
    os.makedirs(os.path.dirname(deck_path), exist_ok=True)
    with open(deck_path, "w", encoding="utf-8") as f:
        json.dump(deck, f, indent=2, ensure_ascii=False)


def classify_source_type(url: str, source_type: Optional[str] = None) -> str:
    """Classify source type from URL or explicit type."""
    if source_type:
        return source_type
    
    url_lower = url.lower()
    
    # Prediction markets
    if any(x in url_lower for x in ["polymarket", "metaculus", "manifold", "predictit"]):
        return "prediction_market"
    
    # Government/official
    if any(x in url_lower for x in [".gov", ".mil", "whitehouse", "state.gov", "congress"]):
        return "government"
    
    # Financial data
    if any(x in url_lower for x in ["bloomberg", "reuters", "ft.com", "wsj", "cnbc"]):
        return "financial_news"
    
    # Analyst reports
    if any(x in url_lower for x in ["goldman", "morgan", "jpmorgan", "analyst", "research"]):
        return "analyst_report"
    
    # Academic
    if any(x in url_lower for x in ["arxiv", "scholar", "acm.org", "ieee", "nature", "science.org"]):
        return "academic"
    
    # News
    if any(x in url_lower for x in ["news", "bbc", "cnn", "nytimes", "washingtonpost", "apnews"]):
        return "news"
    
    # Social
    if any(x in url_lower for x in ["twitter", "x.com", "reddit", "threads"]):
        return "social_media"
    
    # Think tanks
    if any(x in url_lower for x in ["rand", "brookings", "cfr.org", "csis", "heritage"]):
        return "think_tank"
    
    return "web"


def classify_source_tier(source_type: str, credibility_override: Optional[int] = None) -> str:
    """Classify source tier (1-5) based on type."""
    if credibility_override:
        return f"tier{credibility_override}"
    
    tier_map = {
        "government": "tier1",
        "prediction_market": "tier1",  # Real-money forecasts are high signal
        "analyst_report": "tier2",
        "academic": "tier2",
        "think_tank": "tier2",
        "financial_news": "tier3",
        "news": "tier3",
        "web": "tier4",
        "blog": "tier4",
        "social_media": "tier5",
    }
    return tier_map.get(source_type, "tier4")


def detect_bias_flags(url: str, claim: str, source_type: str) -> list:
    """Detect potential bias flags for a source."""
    flags = []
    url_lower = url.lower()
    claim_lower = claim.lower()
    
    # Financial conflict of interest
    if source_type == "analyst_report":
        if any(x in url_lower for x in ["goldman", "morgan", "jpmorgan"]):
            flags.append({
                "type": "financial_conflict",
                "note": "Investment bank may have trading position in related assets",
                "severity": "medium"
            })
    
    # Political bias
    if any(x in url_lower for x in ["heritage", "brookings", "csis"]):
        flags.append({
            "type": "institutional_bias",
            "note": "Think tank may have ideological or funding-driven perspective",
            "severity": "low"
            })
    
    # Prediction market caveats
    if source_type == "prediction_market":
        flags.append({
            "type": "market_caveat",
            "note": "Prediction market reflects crowd probability, not expert consensus. May be thin liquidity.",
            "severity": "info"
        })
    
    # Social media
    if source_type == "social_media":
        flags.append({
            "type": "unverified_source",
            "note": "Social media source - verify with primary source before relying",
            "severity": "high"
        })
    
    # Recency check (for claims about current events)
    if any(word in claim_lower for word in ["will", "forecast", "predict", "expected"]):
        flags.append({
            "type": "time_sensitive",
            "note": "Forecast claim - verify timestamp and check for updates",
            "severity": "info"
        })
    
    return flags


def create_evidence_card(
    claim: str,
    source_url: str,
    source_title: Optional[str] = None,
    source_type: Optional[str] = None,
    credibility: Optional[int] = None,
    excerpt: Optional[str] = None,
    context: Optional[str] = None,
) -> dict:
    """Create a structured evidence card."""
    card_id = f"card-{uuid.uuid4().hex[:8]}"
    detected_type = classify_source_type(source_url, source_type)
    tier = classify_source_tier(detected_type, credibility)
    bias_flags = detect_bias_flags(source_url, claim, detected_type)
    
    return {
        "card_id": card_id,
        "claim": claim,
        "source": {
            "url": source_url,
            "title": source_title or source_url,
            "type": detected_type,
            "tier": tier,
            "accessed": timestamp(),
        },
        "verification": {
            "status": "unverified",  # unverified, supported, verified, disputed
            "cross_references": [],
            "verification_notes": None,
        },
        "excerpt": excerpt,
        "context": context,
        "bias_flags": bias_flags,
        "debate": None,  # Populated by debate command
        "metadata": {
            "created": timestamp(),
            "last_updated": timestamp(),
        }
    }


def cmd_init(args):
    """Initialize a new evidence card deck."""
    deck_path = os.path.join(CARDS_DIR, "deck.json")
    
    if os.path.exists(deck_path) and not args.force:
        print(f"Deck already exists: {deck_path}")
        print("Use --force to overwrite")
        return 1
    
    deck = {
        "cards": [],
        "created": timestamp(),
        "session": args.session or None,
        "statistics": {
            "total_cards": 0,
            "verified": 0,
            "supported": 0,
            "unverified": 0,
            "disputed": 0,
            "tier_counts": {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0, "tier5": 0},
        }
    }
    
    save_deck(deck_path, deck)
    print(json.dumps({"deck": deck_path, "created": True, "cards": 0}, indent=2))
    return 0


def cmd_add(args):
    """Add a new evidence card."""
    deck_path = os.path.join(CARDS_DIR, "deck.json")
    deck = load_deck(deck_path)
    
    card = create_evidence_card(
        claim=args.claim,
        source_url=args.source,
        source_title=args.title,
        source_type=args.type,
        credibility=args.credibility,
        excerpt=args.excerpt,
        context=args.context,
    )
    
    deck["cards"].append(card)
    
    # Update statistics
    deck["statistics"]["total_cards"] = len(deck["cards"])
    tier = card["source"]["tier"]
    deck["statistics"]["tier_counts"][tier] = deck["statistics"]["tier_counts"].get(tier, 0) + 1
    
    save_deck(deck_path, deck)
    
    print(json.dumps({
        "added": True,
        "card_id": card["card_id"],
        "tier": card["source"]["tier"],
        "bias_flags": len(card["bias_flags"]),
        "total_cards": deck["statistics"]["total_cards"],
    }, indent=2))
    return 0


def cmd_verify(args):
    """Update verification status of a card."""
    deck_path = os.path.join(CARDS_DIR, "deck.json")
    deck = load_deck(deck_path)
    
    card = None
    for c in deck["cards"]:
        if c["card_id"] == args.card_id:
            card = c
            break
    
    if not card:
        print(f"Card not found: {args.card_id}")
        return 1
    
    old_status = card["verification"]["status"]
    card["verification"]["status"] = args.status
    card["verification"]["verification_notes"] = args.notes
    card["metadata"]["last_updated"] = timestamp()
    
    if args.cross_ref:
        card["verification"]["cross_references"].append(args.cross_ref)
    
    # Update statistics
    if old_status != args.status:
        deck["statistics"][old_status] = max(0, deck["statistics"].get(old_status, 0) - 1)
        deck["statistics"][args.status] = deck["statistics"].get(args.status, 0) + 1
    
    save_deck(deck_path, deck)
    
    print(json.dumps({
        "updated": True,
        "card_id": card["card_id"],
        "old_status": old_status,
        "new_status": args.status,
    }, indent=2))
    return 0


def cmd_debate(args):
    """Generate FOR/AGAINST debate for a card's claim."""
    deck_path = os.path.join(CARDS_DIR, "deck.json")
    deck = load_deck(deck_path)
    
    card = None
    for c in deck["cards"]:
        if c["card_id"] == args.card_id:
            card = c
            break
    
    if not card:
        print(f"Card not found: {args.card_id}")
        return 1
    
    # Generate debate structure (the agent will fill in the arguments)
    debate = {
        "claim": card["claim"],
        "analyst_for": {
            "argument": None,  # Agent fills this
            "evidence_needed": [],
            "strength": None,
        },
        "analyst_against": {
            "argument": None,  # Agent fills this
            "evidence_needed": [],
            "strength": None,
        },
        "analyst_alternative": {
            "alternative_hypothesis": None,
            "argument": None,
            "evidence_needed": [],
        },
        "synthesis": {
            "winner": None,  # "for", "against", "inconclusive"
            "confidence": None,
            "key_evidence": [],
            "remaining_gaps": [],
        },
        "generated": timestamp(),
    }
    
    card["debate"] = debate
    save_deck(deck_path, deck)
    
    print(json.dumps({
        "debate_created": True,
        "card_id": card["card_id"],
        "claim": card["claim"],
        "instruction": "Agent must fill in analyst_for, analyst_against, analyst_alternative, and synthesis fields",
        "fields_to_complete": [
            "debate.analyst_for.argument",
            "debate.analyst_for.evidence_needed",
            "debate.analyst_for.strength",
            "debate.analyst_against.argument",
            "debate.analyst_against.evidence_needed",
            "debate.analyst_against.strength",
            "debate.analyst_alternative.alternative_hypothesis",
            "debate.analyst_alternative.argument",
            "debate.synthesis.winner",
            "debate.synthesis.confidence",
            "debate.synthesis.key_evidence",
        ]
    }, indent=2))
    return 0


def cmd_export(args):
    """Export all evidence cards as JSON."""
    deck_path = os.path.join(CARDS_DIR, "deck.json")
    deck = load_deck(deck_path)
    
    # Calculate additional statistics
    tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0, "tier5": 0}
    status_counts = {"verified": 0, "supported": 0, "unverified": 0, "disputed": 0}
    bias_flag_count = 0
    
    for card in deck["cards"]:
        tier_counts[card["source"]["tier"]] = tier_counts.get(card["source"]["tier"], 0) + 1
        status_counts[card["verification"]["status"]] = status_counts.get(card["verification"]["status"], 0) + 1
        bias_flag_count += len(card.get("bias_flags", []))
    
    export = {
        "deck": deck,
        "analysis": {
            "total_cards": len(deck["cards"]),
            "tier_distribution": tier_counts,
            "verification_distribution": status_counts,
            "total_bias_flags": bias_flag_count,
            "quality_score": calculate_quality_score(tier_counts, status_counts),
        },
        "exported": timestamp(),
    }
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"Exported to: {args.output}")
    else:
        print(json.dumps(export, indent=2))
    
    return 0


def calculate_quality_score(tier_counts: dict, status_counts: dict) -> float:
    """Calculate a quality score based on tier and verification distribution."""
    total = sum(tier_counts.values())
    if total == 0:
        return 0.0
    
    # Tier quality: tier1=1.0, tier2=0.8, tier3=0.5, tier4=0.3, tier5=0.1
    tier_quality = (
        tier_counts.get("tier1", 0) * 1.0 +
        tier_counts.get("tier2", 0) * 0.8 +
        tier_counts.get("tier3", 0) * 0.5 +
        tier_counts.get("tier4", 0) * 0.3 +
        tier_counts.get("tier5", 0) * 0.1
    ) / total
    
    # Verification quality: verified=1.0, supported=0.7, unverified=0.3, disputed=0.0
    total_verified = sum(status_counts.values())
    if total_verified == 0:
        verification_quality = 0.0
    else:
        verification_quality = (
            status_counts.get("verified", 0) * 1.0 +
            status_counts.get("supported", 0) * 0.7 +
            status_counts.get("unverified", 0) * 0.3 +
            status_counts.get("disputed", 0) * 0.0
        ) / total_verified
    
    # Combined score (60% verification, 40% source quality)
    return round(0.6 * verification_quality + 0.4 * tier_quality, 3)


def cmd_summary(args):
    """Show summary of evidence cards."""
    deck_path = os.path.join(CARDS_DIR, "deck.json")
    deck = load_deck(deck_path)
    
    total = len(deck["cards"])
    if total == 0:
        print("No evidence cards found. Use 'add' command to create cards.")
        return 0
    
    # Count by tier
    tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0, "tier5": 0}
    status_counts = {"verified": 0, "supported": 0, "unverified": 0, "disputed": 0}
    
    for card in deck["cards"]:
        tier_counts[card["source"]["tier"]] = tier_counts.get(card["source"]["tier"], 0) + 1
        status_counts[card["verification"]["status"]] = status_counts.get(card["verification"]["status"], 0) + 1
    
    quality = calculate_quality_score(tier_counts, status_counts)
    
    summary = {
        "total_cards": total,
        "tiers": tier_counts,
        "verification": status_counts,
        "quality_score": quality,
        "quality_interpretation": (
            "HIGH" if quality >= 0.7 else
            "MEDIUM" if quality >= 0.5 else
            "LOW"
        ),
    }
    
    print(json.dumps(summary, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evidence Cards — Structured Evidence Tracking")
    sub = parser.add_subparsers(dest="command")
    
    # init
    p_init = sub.add_parser("init", help="Initialize new evidence card deck")
    p_init.add_argument("--session", help="Session ID to link cards to")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing deck")
    
    # add
    p_add = sub.add_parser("add", help="Add new evidence card")
    p_add.add_argument("--claim", required=True, help="The claim or finding")
    p_add.add_argument("--source", required=True, help="Source URL")
    p_add.add_argument("--title", help="Source title")
    p_add.add_argument("--type", help="Source type (auto-detected if not provided)")
    p_add.add_argument("--credibility", type=int, help="Credibility score 1-5")
    p_add.add_argument("--excerpt", help="Relevant excerpt from source")
    p_add.add_argument("--context", help="Additional context")
    
    # verify
    p_verify = sub.add_parser("verify", help="Update verification status")
    p_verify.add_argument("--card-id", required=True, help="Card ID to update")
    p_verify.add_argument("--status", required=True, 
                          choices=["unverified", "supported", "verified", "disputed"],
                          help="Verification status")
    p_verify.add_argument("--notes", help="Verification notes")
    p_verify.add_argument("--cross-ref", help="Cross-reference URL")
    
    # debate
    p_debate = sub.add_parser("debate", help="Generate debate structure for a claim")
    p_debate.add_argument("--card-id", required=True, help="Card ID to debate")
    
    # export
    p_export = sub.add_parser("export", help="Export all evidence cards")
    p_export.add_argument("--output", help="Output file path")
    
    # summary
    sub.add_parser("summary", help="Show evidence card summary")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "init": cmd_init,
        "add": cmd_add,
        "verify": cmd_verify,
        "debate": cmd_debate,
        "export": cmd_export,
        "summary": cmd_summary,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())