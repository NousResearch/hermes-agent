#!/usr/bin/env python3
"""Backfill quality scores for routing decisions using LLM-as-judge.

Reads unscored routing decisions from state.db, scores them via an LLM,
and writes the scores back. This is a batch operation — run periodically
(e.g., weekly) to build training data for the routing classifier.

Usage:
    python scripts/backfill_routing_quality.py [--limit 100] [--model google/gemini-2.5-flash]

The scoring rubric:
    1.0 = Perfect response, no issues
    0.8 = Good response, minor issues
    0.5 = Adequate but could be better
    0.3 = Poor response, significant issues
    0.0 = Wrong or harmful response
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))

SCORING_PROMPT = """You are a quality evaluator for an AI assistant's responses.

Given a user message and the model that was used to generate the response, rate how likely a response from this model would be adequate for this type of message.

Consider:
- Simple greetings, factual questions, and casual chat → any model works (score 0.8-1.0)
- Code generation, debugging, multi-step reasoning → needs strong model (score 0.2-0.4 for cheap model)
- Ambiguous — could go either way (score 0.5-0.6)

User message: {message}
Model used: {model}

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief explanation>"}}"""


def main():
    parser = argparse.ArgumentParser(description="Backfill routing quality scores")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max decisions to score per run (default: 100)")
    parser.add_argument("--model", type=str, default="google/gemini-2.5-flash",
                        help="Model for scoring (default: google/gemini-2.5-flash)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be scored without writing")
    args = parser.parse_args()

    from hermes_state import SessionDB

    db = SessionDB()
    rows = db.get_routing_decisions(limit=args.limit, scored_only=False)

    # Filter to only unscored rows
    unscored = [r for r in rows if r.get("quality_score") is None]
    logger.info("Found %d unscored routing decisions (of %d total)", len(unscored), len(rows))

    if not unscored:
        logger.info("Nothing to score.")
        db.close()
        return

    if args.dry_run:
        for row in unscored[:5]:
            text = (row.get("message_text") or "")[:80]
            logger.info("  Would score: [%s] %s...", row.get("routed_model"), text)
        logger.info("  ... and %d more", max(0, len(unscored) - 5))
        db.close()
        return

    # Try to use OpenAI-compatible API
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package required: pip install openai")
        db.close()
        sys.exit(1)

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        logger.error("Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        db.close()
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    scored = 0
    failed = 0

    for row in unscored:
        message_text = row.get("message_text", "")
        routed_model = row.get("routed_model", "unknown")
        row_id = row.get("id")

        prompt = SCORING_PROMPT.format(message=message_text[:500], model=routed_model)

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content)
            score = float(result.get("score", 0.5))
            score = max(0.0, min(1.0, score))

            db.update_routing_quality_score(row_id, score)
            scored += 1

            if scored % 10 == 0:
                logger.info("  Scored %d/%d...", scored, len(unscored))

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.warning("  Failed to score row %d: %s", row_id, e)
            failed += 1
            time.sleep(1.0)

    db.close()
    logger.info("Done. Scored: %d, Failed: %d", scored, failed)


if __name__ == "__main__":
    main()
