#!/usr/bin/env python3
"""
cosmos Module Entry Point

This enables running cosmos as a Python module:
    python -m cosmos [options]

See `python -m cosmos --help` for available options.
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and run main
from main import main

if __name__ == "__main__":
    main()
