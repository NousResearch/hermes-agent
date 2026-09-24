#!/usr/bin/env python3
"""Run the fixed selected-PostgreSQL validation manifest.

The manifest deliberately names every test file.  Do not replace it with a
marker selection: the repository default excludes integration tests, while a
broad integration selection reaches unrelated ACP/aiohttp optional suites.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_tests.sh"
SUMMARY = re.compile(
    r"^=== Summary: (?P<files>\d+) files, (?P<tests>\d+) tests passed, "
    r"(?P<failed>\d+) failed .*===$",
    re.MULTILINE,
)

PARTITIONS = (
    (
        "public-runtime",
        (
            "tests/agent/test_postgresql_compression_rotation_oracle.py",
            "tests/integration/test_postgresql_cli_session_store.py",
            "tests/test_state_store_runtime_readiness.py",
            "tests/test_state_store_config.py",
            "tests/test_state_store_factory.py",
            "tests/tools/test_react_to_message_tool.py",
        ),
    ),
    (
        "rotation-faults",
        (
            "tests/integration/test_postgresql_compression_rotation_acceptance.py",
            "tests/integration/test_postgresql_compression_coordination.py",
            "tests/integration/test_postgresql_phase12_fault_harness.py",
            "tests/integration/test_postgresql_session_runtime_ownership.py",
        ),
    ),
    (
        "delivery-import-operations",
        (
            "tests/integration/test_postgresql_delivery_ledger.py",
            "tests/integration/test_postgresql_state_store_sqlite_import.py",
            "tests/integration/test_postgresql_state_store_operations.py",
            "tests/integration/test_postgresql_owned_family_safety.py",
            "tests/integration/test_postgresql_test_target.py",
        ),
    ),
    (
        "store-search-context",
        (
            "tests/integration/test_postgresql_state_store_slice.py",
            "tests/integration/test_postgresql_state_store_fixture.py",
            "tests/test_postgresql_state_store_search_grammar.py",
            "tests/test_contextual_session_search_store.py",
        ),
    ),
)
MANIFEST = tuple(path for _name, paths in PARTITIONS for path in paths)


def validate_manifest() -> None:
    if len(MANIFEST) != 19 or len(set(MANIFEST)) != len(MANIFEST):
        raise RuntimeError("selected-PostgreSQL manifest must contain 19 unique files")
    for path in MANIFEST:
        if not (ROOT / path).is_file():
            raise RuntimeError(f"manifest file is missing: {path}")


def run_partition(name: str, paths: tuple[str, ...]) -> tuple[str, int, int]:
    environment = os.environ.copy()
    environment["HERMES_TEST_WORKERS"] = "1"
    environment["HERMES_TEST_FILE_RETRIES"] = "0"
    command = [str(RUNNER), "-o", "addopts=", *paths, "-v", "--tb=short"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sys.stdout.write(f"\n===== {name} =====\n{completed.stdout}")
    match = SUMMARY.search(completed.stdout)
    if completed.returncode or match is None:
        raise RuntimeError(f"{name} failed (exit={completed.returncode})")
    if int(match.group("failed")):
        raise RuntimeError(f"{name} reported failed tests")
    return name, int(match.group("files")), int(match.group("tests"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("serial", "parallel"), required=True)
    args = parser.parse_args()
    validate_manifest()

    if args.mode == "serial":
        _name, files, tests = run_partition("serial", MANIFEST)
        if files != len(MANIFEST):
            raise RuntimeError(f"serial executed {files}, expected {len(MANIFEST)} files")
        print(f"MANIFEST_OK mode=serial files={files} tests={tests}")
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PARTITIONS)) as executor:
        futures = [executor.submit(run_partition, name, paths) for name, paths in PARTITIONS]
        results = [future.result() for future in futures]
    files = sum(result[1] for result in results)
    tests = sum(result[2] for result in results)
    if files != len(MANIFEST):
        raise RuntimeError(f"parallel executed {files}, expected {len(MANIFEST)} files")
    print(f"MANIFEST_OK mode=parallel files={files} tests={tests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
