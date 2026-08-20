"""Main pipeline entrypoint."""
import os
import sys

from src.helpers import load_assets


def main():
    """Run the mixed-docs pipeline."""
    root = os.path.dirname(os.path.abspath(__file__))
    assets = load_assets(os.path.join(root, ".."))
    print(f"Loaded {len(assets)} assets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
