#!/usr/bin/env python3
"""Static checks for the takeover kit. No live VNC or Camoufox required."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "templates" / "takeover_view.sh"
INSTALL = ROOT / "scripts" / "install.sh"
SOCKS = ROOT / "scripts" / "wg_socks5.py"
MESH = ".".join(("10", "66", "67"))
LEAK_RES = [
    re.compile(re.escape(MESH)),
    re.compile(r"136\.62\."),
    re.compile(r"23\.230\."),
    re.compile(r"facebook", re.I),
    re.compile(r"sitrep", re.I),
    re.compile(r"silverfish", re.I),
    re.compile(r"phan\.cx", re.I),
    re.compile(r"grokbot-takeover", re.I),
    re.compile(r"cookies\.sqlite", re.I),
    re.compile(r"ugreen", re.I),
]


def _tree_files() -> list[Path]:
    skip = {".pyc"}
    out = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix not in skip and "__pycache__" not in p.parts:
            out.append(p)
    return out


class TakeoverKitTests(unittest.TestCase):
    def test_launcher_rejects_public_bind(self):
        env = os.environ.copy()
        env["BIND_IP"] = "0.0.0.0"
        env["TAKEOVER_BASE"] = tempfile.mkdtemp(prefix="takeover-test-")
        r = subprocess.run(
            ["bash", str(LAUNCHER), "start"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Never 0.0.0.0", r.stderr)

    def test_socks_rejects_public_bind(self):
        env = os.environ.copy()
        env["PEER_WG_IP"] = "0.0.0.0"
        r = subprocess.run(
            ["python3", str(SOCKS)],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Never 0.0.0.0", r.stderr)

    def test_install_usage(self):
        r = subprocess.run(["bash", str(INSTALL)], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("check|vps|peer", r.stderr)

    def test_check_mode_no_sudo(self):
        env = os.environ.copy()
        env["TAKEOVER_DEST"] = tempfile.mkdtemp(prefix="takeover-check-")
        r = subprocess.run(
            ["bash", str(INSTALL), "check"],
            env=env,
            capture_output=True,
            text=True,
        )
        out = r.stdout + r.stderr
        self.assertIn("MISSING=", out)
        self.assertIn("READY=", out)
        self.assertNotIn("apt-get", out)

    def test_shell_syntax(self):
        for script in (LAUNCHER, INSTALL, ROOT / "scripts" / "verify.sh"):
            r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=f"{script}: {r.stderr}")

    def test_skill_has_e2e_and_peer_docs(self):
        skill = (ROOT / "SKILL.md").read_text()
        self.assertIn("fresh-vps.md", skill)
        self.assertIn("peer-egress.md", skill)
        self.assertIn("hermes-agent-install.md", skill)
        self.assertIn("install.sh", skill)
        self.assertTrue((ROOT / "references" / "fresh-vps.md").exists())
        self.assertTrue((ROOT / "references" / "peer-egress.md").exists())
        self.assertTrue((ROOT / "references" / "hermes-agent-install.md").exists())
        self.assertTrue((ROOT / "scripts" / "camoufox_server.py").exists())
        self.assertTrue((ROOT / "templates" / "env.example").exists())

    def test_no_leaked_host_data(self):
        hits = []
        for path in _tree_files():
            if path.name == "test_takeover.py":
                continue
            text = path.read_text(errors="ignore")
            for cre in LEAK_RES:
                if cre.search(text):
                    hits.append(f"{path.relative_to(ROOT)}:{cre.pattern}")
        self.assertEqual(hits, [], msg="leaks: " + ", ".join(hits))

    def test_env_example_has_empty_ips(self):
        text = (ROOT / "templates" / "env.example").read_text()
        self.assertIn("BIND_IP=", text)
        self.assertNotRegex(text, r"BIND_IP=\d")
        self.assertIn("HERMES_BROWSER_PROXY=", text)


if __name__ == "__main__":
    unittest.main()
