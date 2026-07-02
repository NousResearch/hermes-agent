#!/usr/bin/env python3
"""run.py — grouped test suites for the information-gain skill.

  basic  (DEFAULT)  every mocked class across the 3 test files — offline, no Ollama, ~seconds.
  live              only the model-calling classes (TestLive, TestEvalLive) — needs Ollama.
  all               both.

  python3 tests/run.py            # basic
  python3 tests/run.py live -v    # live, verbose
  python3 tests/run.py all

Live classes are gated on INFOGAIN_TEST_LIVE (set here for live/all, so direct
`python3 tests/test_*.py` runs stay basic-by-default). Exits non-zero on failure (CI gate).
"""

import argparse
import os
import sys
import unittest

SUITES = ("basic", "live", "all")
TEST_MODULES = ("test_infogain", "test_evals", "test_eval_families")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suite", nargs="?", choices=SUITES, default="basic")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.suite in ("live", "all"):
        # must precede the imports below — the gate decorator is evaluated at class-decoration time
        os.environ["INFOGAIN_TEST_LIVE"] = "1"

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    suite, picked = unittest.TestSuite(), []
    loader = unittest.TestLoader()
    for mod_name in TEST_MODULES:
        mod = __import__(mod_name)
        for name in dir(mod):
            cls = getattr(mod, name)
            if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)):
                continue
            is_live = "Live" in cls.__name__
            if (args.suite == "basic" and is_live) or (args.suite == "live" and not is_live):
                continue
            suite.addTests(loader.loadTestsFromTestCase(cls))
            picked.append(f"{mod_name}.{cls.__name__}")

    print(f"suite: {args.suite} — {len(picked)} classes, {suite.countTestCases()} tests")
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
