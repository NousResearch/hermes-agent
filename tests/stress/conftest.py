"""pytest config for the stress/ subdirectory.

These tests are slow (30s+), spawn subprocesses, and are not run by
default. Enable via `pytest --run-stress` or by running the scripts
directly.

The scripts are primarily __main__-executable entry points; pytest
isn't expected to collect individual test functions from them.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-stress", default=False):
        return
    skip_stress = pytest.mark.skip(
        reason="stress test (opt-in via --run-stress or run script directly)"
    )
    for item in items:
        if "tests/stress" in str(item.fspath):
            item.add_marker(skip_stress)


def pytest_addoption(parser):
    parser.addoption(
        "--run-stress",
        action="store_true",
        default=False,
        help="Run the stress/battle-test suite (slow, spawns subprocesses).",
    )


collect_ignore_glob = [
    # The stress scripts have top-level code and hard-coded paths; they're
    # meant to run as `python tests/stress/<name>.py`, not as pytest modules.
    "*.py",
]
