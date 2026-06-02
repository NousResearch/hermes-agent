#!/usr/bin/env python3
"""Main CLI entrypoint."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="A CLI tool")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("Running CLI tool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
