import os
import sys

# Make helpers importable without packaging the tests dir.
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "agent: hits the live hermes container + model — slow, costs tokens, "
        "non-deterministic. Run a case ~3x and require a pass-rate threshold.",
    )
