"""Build SFT JSONL from a redacted Hermes operator corpus."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hermes_training_corpus import cli_build_sft


if __name__ == "__main__":
    raise SystemExit(cli_build_sft())
