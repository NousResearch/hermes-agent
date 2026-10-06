"""Long secret-keyword runs must not stall the line-anchored config/YAML key scans.

_CFG_ANCHORED_RE and _YAML_ASSIGN_RE had a backtrackable ``<class>*`` on both sides of the
keyword, so a line like ``"token" * N`` retried every keyword occurrence with a fresh scan to
the end of the run: quadratic per line (~1.8 s for ``_CFG_ANCHORED_RE`` on a 50 KB line). The
redactor runs on every log line and outbound gateway message. Like the dotted-key ReDoS tests
in test_redact.py, these exercise the patterns directly; each case runs in a subprocess whose
timeout bounds a regression without hanging pytest.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

_KEYWORDS = ["token", "secret", "password", "passwd", "credential", "auth",
             "api_key", "apikey", "API KEY", "api-key"]

# name -> (pattern, python expression over ``run`` building one line)
_CASES = {
    "anchored_spaced": ("_CFG_ANCHORED_RE", "run + ' = x'"),
    "anchored_no_value": ("_CFG_ANCHORED_RE", "run + '= '"),
    "anchored_export": ("_CFG_ANCHORED_RE", "'export ' + run + ' = x'"),
    "yaml_spaced": ("_YAML_ASSIGN_RE", "'  ' + run + ' : x'"),
    "yaml_no_value": ("_YAML_ASSIGN_RE", "run + ':'"),
    "yaml_quoted_open": ("_YAML_ASSIGN_RE", "run + ': \"x'"),
    "yaml_gutter": ("_YAML_ASSIGN_RE", "'5|  ' + run + ' : x'"),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_50kb_keyword_run_is_linear(case):
    pattern, expr = _CASES[case]
    code = f'''
import time
from agent import redact
for kw in {_KEYWORDS!r}:
    run = kw * (50_000 // len(kw))
    text = {expr}
    start = time.perf_counter()
    list(redact.{pattern}.finditer(text))
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, (kw, elapsed)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p)
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60, cwd=_REPO_ROOT, env=env)
