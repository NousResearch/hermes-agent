#!/usr/bin/env python3
"""Compatibility launcher for the Hermes classic CLI package."""

from cli.app import main


if __name__ == "__main__":
    import fire

    fire.Fire(main)
