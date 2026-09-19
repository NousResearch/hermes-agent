"""discrimination-guard must catch probes that cannot fail, and stay quiet
on probes that can.

The second half carries the weight: a guard that flags sound work is noise,
and noise gets muted within a day.
"""
import importlib.util
from pathlib import Path

import pytest

PLUGIN = (Path(__file__).resolve().parents[2]
          / "plugins" / "discrimination-guard" / "__init__.py")


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("discrimination_guard", PLUGIN)
    assert spec is not None and spec.loader is not None, f"cannot load {PLUGIN}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flagged(guard, content, path="/tmp/audits/probe.py"):
    out = guard._on_pre_tool_call(tool_name="write_file",
                                  args={"path": path, "content": content})
    return bool(out and out.get("action") in ("warn", "block"))


# --------------------------------------------------------------- must flag

TAUTOLOGY = '''
"""Verify the guard refuses a dirty tree."""
rc = cmd_branch("feature2", "master", d)
on = current_branch(d)
check("dirty tree: reported honestly", True, (rc == 0) == (on == "feature2"))
print("verdict: guard ok")
'''

UNCHECKED_SUBPROCESS = '''
"""Probe whether the branch switch worked."""
import subprocess
subprocess.run("git checkout -B fix/new upstream/main", shell=True)
subprocess.run("git apply /tmp/x.patch", shell=True)
print("branch switched: ok")
'''

ZERO_FAILURES = '''
"""Baseline the suite before judging the patch."""
import subprocess
out = subprocess.run("pytest tests/ -q", shell=True, capture_output=True,
                     text=True).stdout
fails = [l for l in out.splitlines() if l.startswith("FAILED")]
print("baseline: no failures, patch is clean" if not fails else fails)
'''

GREP_AS_PROOF = '''
"""Check collect_files rejects non-regular files."""
src = open("graphify/extract.py").read()
block = src[src.find("def collect_files"):][:3000]
has_guard = "is_file" in block
print("guard present:", has_guard, "-> verdict: safe")
'''


@pytest.mark.parametrize("label,content", [
    ("tautological assertion", TAUTOLOGY),
    ("unchecked subprocess", UNCHECKED_SUBPROCESS),
    ("zero failures, nothing ran", ZERO_FAILURES),
    ("grep as proof of behaviour", GREP_AS_PROOF),
])
def test_a_probe_that_cannot_fail_is_flagged(guard, label, content):
    assert flagged(guard, content), f"{label} should have been flagged"


# -------------------------------------------------------------- must be quiet

WIRE_PROBE_WITH_CONTROL = '''
"""Does the webhook leak the key on the wire?"""
import secrets
key = "sk-" + secrets.token_hex(24)
body = wh._serialize_payload("pre_tool_call", {"args": {"cmd": key}}, "id").decode()
print("key present:", key in body)
control = "see file.txt for details"
assert control in wh._serialize_payload(
    "pre_tool_call", {"args": {"cmd": control}}, "id").decode(), "prose untouched"
'''

RETURNCODE_CHECKED = '''
"""Measure the failure rate on a pristine worktree."""
import subprocess
r = subprocess.run(["pytest", "tests/x.py", "-q"], capture_output=True, text=True)
if r.returncode not in (0, 1):
    raise SystemExit(f"UNKNOWN: pytest could not run (rc={r.returncode})")
print("failed:", "FAILED" in r.stdout, "| baseline unchanged:", r.returncode == 0)
'''

MUTATION_VERIFIED = '''
"""Prove the fix is not vacuous by reverting it."""
mod = load_real_module()
before = mod.file_hash(f, root)
f.write_text("different")
after = mod.file_hash(f, root)
print("digest changed:", before != after)
print("control, untouched file stable:", mod.file_hash(g, root) == g_digest)
'''

ORDINARY_SOURCE = '''
def render_summary(rows):
    """Format a table for the CLI."""
    width = max(len(r.name) for r in rows)
    return "\\n".join(f"{r.name:<{width}}  {r.value}" for r in rows)
'''


@pytest.mark.parametrize("label,content,path", [
    ("wire probe with a control", WIRE_PROBE_WITH_CONTROL, "/tmp/audits/p.py"),
    ("returncode checked", RETURNCODE_CHECKED, "/tmp/audits/p.py"),
    ("mutation verified", MUTATION_VERIFIED, "/tmp/audits/p.py"),
    ("ordinary source file", ORDINARY_SOURCE, "/tmp/project/render.py"),
])
def test_a_sound_probe_is_left_alone(guard, label, content, path):
    assert not flagged(guard, content, path), f"{label} should not be flagged"


def test_short_writes_are_ignored(guard):
    assert not flagged(guard, "print('ok')")


def test_non_write_tools_are_ignored(guard):
    out = guard._on_pre_tool_call(tool_name="terminal",
                                  args={"command": "echo ok, verdict clean"})
    assert out is None
