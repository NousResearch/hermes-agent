"""Export Hermes operator training corpus from state.db and optional logs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hermes_training_corpus import cli_export


if __name__ == "__main__":
    raise SystemExit(cli_export())
