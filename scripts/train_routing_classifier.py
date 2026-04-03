#!/usr/bin/env python3
"""Train the routing classifier from collected routing decisions.

Reads labeled routing decisions from state.db (quality_score must be
backfilled first), trains a LogisticRegression model, and saves it to
~/.hermes/models/routing_classifier.pkl.

Usage:
    python scripts/train_routing_classifier.py [--min-samples 200] [--threshold 0.7]

Requirements:
    pip install scikit-learn
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add hermes-agent root to path for imports
HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train routing classifier")
    parser.add_argument("--min-samples", type=int, default=200,
                        help="Minimum scored samples required to train (default: 200)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Quality score threshold for 'good enough for cheap' (default: 0.7)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for .pkl model (default: ~/.hermes/models/routing_classifier.pkl)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for testing (default: 0.2)")
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
    except ImportError:
        logger.error("scikit-learn is required: pip install scikit-learn")
        sys.exit(1)

    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from agent.routing_features import feature_names

    # Load data
    db = SessionDB()
    rows = db.get_routing_decisions(limit=10000, scored_only=True)
    db.close()

    if len(rows) < args.min_samples:
        logger.error(
            "Only %d scored routing decisions found (need %d). "
            "Run backfill_routing_quality.py first.",
            len(rows), args.min_samples
        )
        sys.exit(1)

    logger.info("Loaded %d scored routing decisions", len(rows))

    # Build feature matrix and labels
    names = feature_names()
    X = []
    y = []

    for row in rows:
        features = [
            float(row.get("message_char_count") or 0),
            float(row.get("message_word_count") or 0),
            float(row.get("message_text", "").count("\n") + 1),
            float(row.get("has_code_block") or 0),
            float(row.get("has_url") or 0),
            1.0 if "`" in (row.get("message_text") or "") else 0.0,
            1.0 if row.get("complex_keyword_hit") else 0.0,
            0.0,  # tool_mention_count not stored in DB — use 0
            1.0 if "?" in (row.get("message_text") or "") else 0.0,
            0.0,  # avg_word_length — recompute
            float(row.get("conversation_depth") or 0),
        ]

        # Recompute avg_word_length from message text
        text = (row.get("message_text") or "").strip()
        words = text.split()
        if words:
            features[9] = sum(len(w) for w in words) / len(words)

        X.append(features)

        # Label: 1 = safe to route cheap, 0 = needs primary
        quality = row.get("quality_score", 0) or 0
        is_cheap_safe = quality >= args.threshold
        y.append(1 if is_cheap_safe else 0)

    X = np.array(X)
    y = np.array(y)

    # Report class balance
    n_cheap = int(y.sum())
    n_primary = len(y) - n_cheap
    logger.info("Class balance: cheap-safe=%d, needs-primary=%d", n_cheap, n_primary)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Train
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=["primary", "cheap"]))
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    # Feature importances
    logger.info("\nFeature Importances:")
    for name, coef in sorted(zip(names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        logger.info("  %-25s %+.4f", name, coef)

    # Save
    output_path = Path(args.output) if args.output else get_hermes_home() / "models" / "routing_classifier.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("\nModel saved to %s", output_path)

    # Save metadata
    meta_path = output_path.with_suffix(".json")
    meta = {
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "quality_threshold": args.threshold,
        "feature_names": names,
        "accuracy": float((y_pred == y_test).mean()),
        "class_balance": {"cheap_safe": n_cheap, "needs_primary": n_primary},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)


if __name__ == "__main__":
    main()
