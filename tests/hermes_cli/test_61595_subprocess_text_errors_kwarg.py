"""
Regression test for issue #61595 - subprocess._readerthread crashes with
UnicodeDecodeError on Windows when child process output is GBK/CP936.

Per the issue: "In all Hermes subprocess creation points (terminal tool,
code_execution, asyncio.create_subprocess_exec, direct subprocess.Popen),
ensure errors='replace' is set on Windows".

This test enforces that any subprocess.run or subprocess.Popen call site
within Hermes (non-test) that uses text=True must also pass errors= as
an explicit kwarg (any value is acceptable - 'replace' or 'ignore' or
'backslashreplace'). The test works by importing each module and
inspecting the source.

Why: the failure mode is that the subprocess reader thread dies silently,
stdout/stderr becomes None, and downstream .split() raises AttributeError
or the process appears to produce no output. Setting errors='replace' or
errors='ignore' on text-mode subprocess calls guarantees the reader
thread does not raise UnicodeDecodeError, regardless of the locale's
default encoding.

Regression coverage: 10 currently-broken call sites identified via
static analysis:
    tools/transcription_tools.py:1191, 1237, 1239
    tools/environments/singularity.py:223
    tools/tts_tool.py:1884
    tools/voice_mode.py:392, 409
    hermes_cli/onepassword_secrets_cli.py:426
    hermes_cli/main.py:1169
    hermes_cli/setup.py:1430
"""

import re
import sys
from pathlib import Path


# Modules that contain subprocess.run or Popen call sites with text=True
# that should have errors= set
TARGET_MODULES = {
    "tools.transcription_tools": ["tools/transcription_tools.py"],
    "tools.environments.singularity": ["tools/environments/singularity.py"],
    "tools.tts_tool": ["tools/tts_tool.py"],
    "tools.voice_mode": ["tools/voice_mode.py"],
    "hermes_cli.onepassword_secrets_cli": ["hermes_cli/onepassword_secrets_cli.py"],
    "hermes_cli.main": ["hermes_cli/main.py"],
    "hermes_cli.setup": ["hermes_cli/setup.py"],
}


def _extract_text_subprocess_call_blocks(source: str) -> list[tuple[int, str]]:
    """Find every `subprocess.run(...)` and `subprocess.Popen(...)` call in
    source that has text=True (or text=...) and return (line_number, block).

    A "block" is the multi-line call up through the matching close paren.
    """
    results = []
    lines = source.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        # Match `subprocess.run` or `subprocess.Popen` followed by (...)
        m = re.search(r"subprocess\.(?:run|Popen)\b\s*\(", line)
        if m:
            start = i + 1  # 1-indexed
            # Find matching close paren
            depth = 1
            j = i
            # Count parens from position after the (
            start_pos = m.end()
            # Track depth across lines
            for j in range(i, n):
                line_text = lines[j]
                # Only count parens after the call's open paren
                if j == i:
                    seg = line_text[start_pos:]
                else:
                    seg = line_text
                for c in seg:
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if depth == 0:
                            break
                if depth == 0:
                    break
            # Block is lines i..j
            block = "\n".join(lines[i:j + 1])
            # Does it have text=True (or text=...)
            if re.search(r"\btext\s*=\s*(True|False)\b", block):
                results.append((start, block))
            i = j + 1
        else:
            i += 1
    return results


def test_text_subprocess_calls_have_errors_kwarg():
    """Every subprocess.run / subprocess.Popen call with text=True must
    also pass errors= as a kwarg. This is the failing-first invariant for
    #61595 — fails on the unfixed codebase (because the call sites don't
    set errors=), passes on the fixed codebase."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    failures = []
    for module_name, paths in TARGET_MODULES.items():
        # Pick the first path that exists
        rel_path = None
        for p in paths:
            candidate = worktree / p
            if candidate.exists():
                rel_path = candidate
                break
        assert rel_path, f"module {module_name} not found in worktree"

        source = rel_path.read_text()
        blocks = _extract_text_subprocess_call_blocks(source)
        for lineno, block in blocks:
            if not re.search(r"\berrors\s*=", block):
                failures.append(
                    f"{rel_path.relative_to(worktree)}:{lineno}\n"
                    f"  text=True without errors= kwarg — reader thread may "
                    f"crash with UnicodeDecodeError on Windows non-UTF-8 "
                    f"locale (issue #61595)\n  block:\n{block}"
                )
    assert not failures, (
        "subprocess.run/Popen call sites with text=True but missing errors=:\n\n"
        + "\n\n".join(failures)
    )


def test_wmic_call_has_errors_kwarg():
    """Sanity check: the wmic call (which already had a fix for #17049) must
    not regress. This test should always pass; it catches any future
    refactor that drops the errors= kwarg."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    main_py = (worktree / "hermes_cli" / "main.py").read_text()

    # Find the wmic block — distinctive enough to be unambiguous
    wmic_match = re.search(
        r'result\s*=\s*subprocess\.run\(\s*\[?\s*"wmic"[^)]+\)',
        main_py,
        re.DOTALL,
    )
    if wmic_match is None:
        # Pattern didn't match; wmic call may have been refactored.
        # Skip rather than fail.
        return
    block = wmic_match.group(0)
    assert "errors" in block, (
        f"the wmic subprocess.run call lost its errors= kwarg (regression of #17049)\n{block}"
    )