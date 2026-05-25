"""Direct module runner for environments where the plugin is not enabled."""

from __future__ import annotations

import argparse
import sys

from .cli import register_cli, trend_discovery_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m plugins.trend_discovery.cli_runner")
    register_cli(parser)
    return trend_discovery_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
