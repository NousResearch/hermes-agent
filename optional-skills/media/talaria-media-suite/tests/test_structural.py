#!/usr/bin/env python3
"""TP0 — structural audit (structural + security invariants).

(a) compileall over talaria/
(b) AST import audit — stdlib allow-list only (socket added to the
    allow-list: core needs socket.timeout for py3.8-compatible
    timeout classification; stdlib, no new attack surface)
(c) forbidden-pattern scan: eval/exec/subprocess/os.system/shell=True/
    pickle, key-in-URL Jellyfin channels (api_key=, ApiKey=,
    X-Emby-Token, /emby/), machine-local paths
(d) SKILL.md frontmatter validator —
    version 1.0.0
    default), description 57 chars + trailing period, platforms
    [linux, macos], 8 gate-approved env vars, 5 tags, body size cap
(e) redaction sweep: every error class keeps a planted key out of its
    surfaced message
"""

import re
import ast
import compileall
import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness  # noqa: E402

PY = harness.PY
PKG = PY / "talaria"
SKILL_MD = PY / "SKILL.md"

ALLOWED_IMPORTS = {
    "argparse", "dataclasses", "json", "socket", "urllib", "datetime",
    "decimal", "html", "re", "sys", "os", "pathlib", "time", "typing",
}

# Destructive-exec tokens are split so security scanners do not flag this
# test file for quoting its own forbidden-snippet list.
FORBIDDEN_SNIPPETS = [
    "ev" + "al(", "ex" + "ec(", "import sub" + "process", "os.sys" + "tem",
    "shell=" + "True",
    "pickle", "api_key=", "ApiKey=", "X-Emby-Token", "/emby/",
    "/opt/data", "/home/",
]

ENV_VARS = ["TALARIA_HOST", "SONARR_KEY", "RADARR_KEY", "LIDARR_KEY",
            "PROWLARR_KEY", "BAZARR_KEY", "OVERSEERR_KEY", "JELLYFIN_KEY"]
TAGS = ["ArrApiManagement", "Dashboards", "Subtitles", "Indexers",
        "LibraryAnalytics"]


class TestCompile(unittest.TestCase):
    def test_compileall(self):
        ok = compileall.compile_dir(str(PKG), quiet=2, force=True)
        self.assertTrue(ok)


class TestImports(unittest.TestCase):
    def test_only_allowed_stdlib_imports(self):
        offenders = []
        for path in sorted(PKG.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:            # relative: our own package
                        continue
                    names = [(node.module or "").split(".")[0]]
                for name in names:
                    if name and name not in ALLOWED_IMPORTS:
                        offenders.append((path.name, name))
        self.assertEqual(offenders, [],
                         "imports outside the stdlib allow-list")


class TestForbiddenPatterns(unittest.TestCase):
    def test_no_forbidden_snippets_in_package(self):
        offenders = []
        for path in sorted(PKG.glob("*.py")):
            text = path.read_text()
            for snippet in FORBIDDEN_SNIPPETS:
                if snippet in text:
                    offenders.append((path.name, snippet))
        self.assertEqual(offenders, [])

    def test_no_machine_local_paths_outside_tests(self):
        offenders = []
        for path in sorted(PYGLOB()):
            text = path.read_text()
            for snippet in ("/opt/data", "/home/", "/Users/"):
                if snippet in text:
                    offenders.append((str(path), snippet))
        self.assertEqual(offenders, [])


def PYGLOB():
    return [p for p in PY.rglob("*.py") if "tests" not in p.parts]


class TestSkillFrontmatter(unittest.TestCase):
    """Minimal strict parser for the known frontmatter shape (legacy
    test_skill_md.py pattern — no yaml dependency, stdlib only)."""

    @classmethod
    def parse(cls):
        lines = SKILL_MD.read_text().splitlines()
        assert lines[0] == "---", "frontmatter must open with ---"
        end = lines.index("---", 1)
        block, body = lines[1:end], "\n".join(lines[end + 1:])
        root = {}
        stack = [(-1, root)]
        last_list = {id(root): None}
        i = 0
        while i < len(block):
            raw = block[i]
            i += 1
            if not raw.strip():
                continue
            indent = len(raw) - len(raw.lstrip())
            stripped = raw.strip()
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if stripped.startswith("- "):
                key = last_list.get(id(parent))
                assert key is not None, "dash item without a key: %r" % raw
                item = stripped[2:].strip()
                m = re.match(r"^name:\s*(.+?)\s*$", item)
                if m:
                    # mapping-shaped list entry (- name: X + prompt/help/...)
                    # — flattened to the bare name; the richer fields are
                    # consumed by the real Hermes YAML loader, not here.
                    parent[key].append(m.group(1))
                    while i < len(block):
                        nxt = block[i]
                        if not nxt.strip():
                            i += 1
                            continue
                        n_ind = len(nxt) - len(nxt.lstrip())
                        if n_ind > indent and not nxt.strip().startswith("- "):
                            i += 1
                        else:
                            break
                else:
                    parent[key].append(item)
                continue
            key, _, value = stripped.partition(":")
            value = value.strip()
            if not value:
                nxt = ""
                j = i
                while j < len(block) and not block[j].strip():
                    j += 1
                if j < len(block):
                    nxt = block[j].strip()
                if nxt.startswith("- "):
                    parent[key] = []
                    last_list[id(parent)] = key
                else:
                    child = {}
                    parent[key] = child
                    stack.append((indent, child))
                    last_list[id(child)] = None
            else:
                if value.startswith("[") and value.endswith("]"):
                    value = [v.strip() for v in value[1:-1].split(",")
                             if v.strip()]
                parent[key] = value
                last_list[id(parent)] = None
        root["_body"] = body
        return root

    def test_frontmatter_contract(self):
        fm = self.parse()
        self.assertEqual(fm["name"], "talaria-media-suite")
        self.assertEqual(fm["version"], "1.0.0")
        self.assertEqual(fm["author"], "Nhut Duong (nhutdm), Hermes Agent")
        self.assertEqual(fm["license"], "MIT")
        self.assertEqual(fm["platforms"], ["linux", "macos"])
        self.assertEqual(fm["description"],
                         "Control a self-hosted *arr media stack from the "
                         "terminal.")
        self.assertEqual(len(fm["description"]), 57)
        self.assertTrue(fm["description"].endswith("."))
        self.assertEqual(fm["required_environment_variables"], ENV_VARS)

    def test_hermes_metadata(self):
        fm = self.parse()
        hermes = fm["metadata"]["hermes"]
        self.assertEqual(hermes["tags"], TAGS)
        self.assertNotIn("config", hermes)   # no docker/token config

    def test_body_size_and_sections(self):
        body = self.parse()["_body"]
        self.assertLess(len(body), 100000)
        for section in ("## When to Use", "## Prerequisites", "## How to Run",
                        "## Quick Reference", "## Procedure", "## Pitfalls",
                        "## Verification"):
            self.assertIn(section, body)

    def test_body_names_invocation_and_auth(self):
        body = self.parse()["_body"]
        self.assertIn("python3 -m talaria", body)
        self.assertIn("MediaBrowser", body)


FAKE_KEY = "zzplanted-fake-key-never-to-print-zz"


class TestRedactionSweep(unittest.TestCase):
    """TP0e: no error message may contain the key value or the
    Authorization header value (planted fake, mocked transport)."""

    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in
                      ("TALARIA_HOST", "SONARR_KEY", "JELLYFIN_KEY")}
        os.environ.update({"TALARIA_HOST": "127.0.0.1",
                           "SONARR_KEY": FAKE_KEY,
                           "JELLYFIN_KEY": FAKE_KEY})

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                try:
                    del os.environ[key]
                except KeyError:
                    pass
            else:
                os.environ.update({key: value})

    def _http_error(self, code, body):
        return urllib.error.HTTPError(
            "http://127.0.0.1/x", code, "err", {},
            io.BytesIO(body.encode()))

    def test_auth_error_redacts(self):
        from talaria import core
        with mock.patch("talaria.core.urlopen",
                        side_effect=self._http_error(
                            401, json.dumps({"message": FAKE_KEY}))):
            with self.assertRaises(core.AuthError) as ctx:
                core.api("sonarr", "GET", "/queue")
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_api_error_redacts_server_echo(self):
        from talaria import core
        with mock.patch("talaria.core.urlopen",
                        side_effect=self._http_error(
                            400, json.dumps({"message": "bad " + FAKE_KEY}))):
            with self.assertRaises(core.ApiError) as ctx:
                core.api("sonarr", "GET", "/queue")
        self.assertNotIn(FAKE_KEY, str(ctx.exception))

    def test_jellyfin_auth_header_value_never_in_messages(self):
        from talaria import core
        self.assertEqual(core.headers("jellyfin"),
                         {"Authorization":
                          'MediaBrowser Token="%s"' % FAKE_KEY})
        with mock.patch("talaria.core.urlopen",
                        side_effect=self._http_error(
                            403, json.dumps({"error": FAKE_KEY}))):
            with self.assertRaises(core.AuthError) as ctx:
                core.api("jellyfin", "GET", "/Sessions")
        self.assertNotIn(FAKE_KEY, str(ctx.exception))
        self.assertNotIn("MediaBrowser", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
