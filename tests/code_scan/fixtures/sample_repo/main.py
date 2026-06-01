#!/usr/bin/env python3
"""Sample repository entry point for UA-005 mode testing."""
import os
import sys

from utils import helper


def main():
    """Application entry point."""
    print(f"Hello from {os.path.basename(__file__)}")
    helper()
    return 0


if __name__ == "__main__":
    sys.exit(main())
