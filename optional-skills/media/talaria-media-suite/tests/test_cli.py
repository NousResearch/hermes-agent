#!/usr/bin/env python3
"""TP1 — CLI sweep: --help for prog + all groups + all
68 subs exits 0 with ZERO mock requests logged; unknown group/sub exit 2;
missing-key and missing-host invocations exit 1 naming the missing var
and send zero requests. Network-free under the dead-proxy trap."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
from harness import MOCK_LOG, make_env, run_cli  # noqa: E402

SUB_TOTAL = 68          # 9+9+2+9+4+5+14+10+5+1
GROUPS_WITH_SUBS = ["sonarr", "radarr", "lidarr", "prowlarr", "bazarr",
                    "overseerr", "jellyfin", "library", "ops"]
TOP_LEAVES = ["dashboard", "status", "setup", "discover", "diagnose",
              "queue", "version"]


def _subcommands():
    """(group, sub) pairs for every group-carried sub, walked from the
    live argparse tree so the sweep can never drift from registration."""
    import sys
    sys.path.insert(0, str(harness.PY))
    from talaria.__main__ import build_parser
    tree = build_parser()._subparsers._group_actions[0].choices
    pairs = []
    for group, parser in sorted(tree.items()):
        inner = getattr(parser, "_subparsers", None)
        if inner is not None:
            for sub in sorted(inner._group_actions[0].choices):
                pairs.append((group, sub))
    return pairs


class TestHelpSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        harness.setup_sandbox()
        # no mock is started for this tier; prove zero requests at the end
        if MOCK_LOG.is_file():
            MOCK_LOG.unlink()

    @classmethod
    def tearDownClass(cls):
        harness.teardown_sandbox()

    def test_prog_help(self):
        proc = run_cli(["--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for group in GROUPS_WITH_SUBS + TOP_LEAVES:
            self.assertIn(group, proc.stdout)

    def test_all_groups_and_subs_help(self):
        pairs = _subcommands()
        self.assertEqual(len(pairs) + 1, SUB_TOTAL)  # + dashboard leaf
        for group in GROUPS_WITH_SUBS:
            proc = run_cli([group, "--help"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
        for group, sub in pairs:
            proc = run_cli([group, sub, "--help"])
            self.assertEqual(proc.returncode, 0,
                             "%s %s: %s" % (group, sub, proc.stderr))
        for leaf in TOP_LEAVES:
            proc = run_cli([leaf, "--help"])
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_help_sweep_sent_zero_requests(self):
        _subcommands()                      # force a fresh walk too
        run_cli(["--help"])
        self.assertFalse(MOCK_LOG.is_file() and MOCK_LOG.read_text().strip(),
                         "help sweep must not touch the network")

    def test_version(self):
        proc = run_cli(["version"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("1.0.0", proc.stdout)


class TestUsageErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        harness.setup_sandbox()

    @classmethod
    def tearDownClass(cls):
        harness.teardown_sandbox()

    def test_unknown_group_exits_2(self):
        proc = run_cli(["nosuch"])
        self.assertEqual(proc.returncode, 2)

    def test_unknown_sub_exits_2(self):
        proc = run_cli(["sonarr", "nosuch"])
        self.assertEqual(proc.returncode, 2)


class TestEnvGating(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        harness.setup_sandbox()
        if MOCK_LOG.is_file():
            MOCK_LOG.unlink()

    @classmethod
    def tearDownClass(cls):
        harness.teardown_sandbox()

    def test_missing_key_names_var_and_sends_nothing(self):
        proc = run_cli(["sonarr", "queue"], env=make_env(SONARR_KEY=None))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("SONARR_KEY", proc.stderr)
        self.assertIn("setup", proc.stderr)
        self.assertFalse(MOCK_LOG.is_file() and MOCK_LOG.read_text().strip(),
                         "missing key must send zero requests")

    def test_missing_jellyfin_key_names_var(self):
        proc = run_cli(["jellyfin", "sessions"],
                       env=make_env(JELLYFIN_KEY=None))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("JELLYFIN_KEY", proc.stderr)

    def test_missing_host_names_setup_hint(self):
        env = make_env(TALARIA_HOST=None)
        proc = run_cli(["sonarr", "queue"], env=env)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("host not set", proc.stderr)
        self.assertIn("setup", proc.stderr)
        self.assertFalse(MOCK_LOG.is_file() and MOCK_LOG.read_text().strip(),
                         "missing host must send zero requests")


if __name__ == "__main__":
    unittest.main()
