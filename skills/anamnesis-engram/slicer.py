#!/usr/bin/env python3
"""
Module 4: Memory Slicer Runner
Ties together digester, scorer, and evolver.

Usage:
    python3 slicer.py <conversation_file> [--max-chars 300] [--slices 3] [--verbose]
    
    Or pipe text:
    echo "conversation text" | python3 slicer.py - 

Input: raw conversation text (or JSON with {"turns": [{"role":"user","content":"..."}]})
Output: optimized memory slices ready to save
"""

import sys
import json
import argparse
from digester import digest, digest_turns
from scorer import score_slice, rank_slices
from evolver import evolve_slices


def extract_facts_from_digest(d: dict) -> list[str]:
    """
    Pull candidate facts from a digest result.
    Combines extracted facts, entity mentions, and key terms
    into concise fact strings.
    """
    facts = []

    # Direct facts from the digester
    for f in d.get("facts", []):
        # Clean up whitespace but preserve intentional content
        cleaned = " ".join(f.split())
        if cleaned and len(cleaned) > 5:
            facts.append(cleaned)

    # Entity-based facts
    seen_entities = set()
    for text, label in d.get("entities", []):
        if text.lower() not in seen_entities:
            seen_entities.add(text.lower())
            # Don't add bare entities, they'll be in the facts already

    return facts


def deduplicate_facts(facts: list[str], threshold: float = 0.6) -> list[str]:
    """
    Remove near-duplicate facts using simple token overlap.
    """
    if not facts:
        return []

    unique = [facts[0]]
    for fact in facts[1:]:
        fact_tokens = set(fact.lower().split())
        is_dup = False
        for existing in unique:
            existing_tokens = set(existing.lower().split())
            if not fact_tokens or not existing_tokens:
                continue
            overlap = len(fact_tokens & existing_tokens) / min(len(fact_tokens), len(existing_tokens))
            if overlap > threshold:
                # Keep the longer one
                if len(fact) > len(existing):
                    unique.remove(existing)
                    unique.append(fact)
                is_dup = True
                break
        if not is_dup:
            unique.append(fact)

    return unique


def run(
    text: str,
    max_chars: int = 300,
    n_slices: int = 3,
    is_turns: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Full pipeline: text -> digest -> facts -> evolve -> ranked slices.
    
    Args:
        text: raw text or JSON turns
        max_chars: char limit per slice
        n_slices: how many slices to produce
        is_turns: if True, parse as JSON turns
        verbose: print intermediate steps
    
    Returns:
        {
            "facts": list of extracted facts,
            "slices": list of optimized slice dicts,
            "digest": the raw digest output,
        }
    """
    # Step 1: Digest
    if is_turns:
        turns = json.loads(text) if isinstance(text, str) else text
        d = digest_turns(turns)
        # Focus on user facts + combined facts
        user_facts = extract_facts_from_digest(d["user"])
        combined_facts = extract_facts_from_digest(d["combined"])
        all_facts = user_facts + [f for f in combined_facts if f not in user_facts]
        raw_digest = d
    else:
        d = digest(text)
        all_facts = extract_facts_from_digest(d)
        raw_digest = d

    if verbose:
        print("=== RAW FACTS (%d) ===" % len(all_facts))
        for f in all_facts:
            print("  - %s" % f)
        print()

    # Step 2: Deduplicate
    facts = deduplicate_facts(all_facts)

    if verbose:
        print("=== DEDUPED FACTS (%d) ===" % len(facts))
        for f in facts:
            print("  - %s" % f)
        print()

    if not facts:
        return {"facts": [], "slices": [], "digest": raw_digest}

    # Step 3: Evolve
    evolved = evolve_slices(
        facts,
        max_chars=max_chars,
        population_size=60,
        generations=50,
        verbose=verbose,
    )

    # Step 4: Return top N
    slices = evolved[:n_slices]

    return {
        "facts": facts,
        "slices": slices,
        "digest": raw_digest,
    }


def main():
    parser = argparse.ArgumentParser(description="Memory Slicer: optimize conversation into memory slices")
    parser.add_argument("input", help="File path or '-' for stdin")
    parser.add_argument("--max-chars", type=int, default=300, help="Max chars per slice (default: 300)")
    parser.add_argument("--slices", type=int, default=3, help="Number of slices to produce (default: 3)")
    parser.add_argument("--turns", action="store_true", help="Input is JSON turns format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show intermediate steps")

    args = parser.parse_args()

    # Read input
    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input, "r") as f:
            text = f.read()

    # Run pipeline
    result = run(
        text,
        max_chars=args.max_chars,
        n_slices=args.slices,
        is_turns=args.turns,
        verbose=args.verbose,
    )

    # Output
    print("=" * 60)
    print("MEMORY SLICER OUTPUT")
    print("=" * 60)
    print()
    print("Facts extracted: %d" % len(result["facts"]))
    print("Slices produced: %d" % len(result["slices"]))
    print()

    for i, s in enumerate(result["slices"]):
        print("--- Slice %d (score: %s, chars: %s/%s, facts: %s) ---" % (
            i + 1, s["composite"], s["char_count"], args.max_chars, s["facts_count"]
        ))
        print(s["text"])
        print()


if __name__ == "__main__":
    main()
