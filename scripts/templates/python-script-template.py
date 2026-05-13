#!/usr/bin/env python3
"""Safe starter template for IT automation lab Python scripts."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    target: str
    dry_run: bool


def parse_args(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser(description="IT automation lab script template")
    parser.add_argument("--target", required=True, help="Target host, path, service, or identifier")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing state")
    args = parser.parse_args(argv)
    return Config(target=args.target, dry_run=args.dry_run)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"required command not found: {name}")


def run_command(command: list[str], *, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        logging.info("DRY-RUN: %s", " ".join(command))
        return None
    logging.info("RUN: %s", " ".join(command))
    return subprocess.run(command, check=True, text=True, capture_output=True)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s")
    config = parse_args(argv)
    require_command("python3")

    logging.info("Starting automation for target: %s", config.target)
    logging.info("Replace this template action with a safe, idempotent operation.")
    run_command(["python3", "--version"], dry_run=config.dry_run)
    logging.info("Completed. Add verification commands before using this in production.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
