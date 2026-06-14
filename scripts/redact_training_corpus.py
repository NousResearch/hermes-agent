"""Redact local identifiers and secrets from a Hermes training corpus JSONL."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hermes_training_corpus import cli_redact


if __name__ == "__main__":
    raise SystemExit(cli_redact())
