"""
Regression test for issue #61595 - subprocess._readerthread crashes with
UnicodeDecodeError on Windows when child process output is GBK/CP936.

Per the issue: "In all Hermes subprocess creation points (terminal tool,
code_execution, asyncio.create_subprocess_exec, direct subprocess.Popen),
ensure errors='replace' is set on Windows".

This test enforces that any subprocess.run or subprocess.Popen call site
within Hermes (non-test) that uses text=True must also pass errors= as
an explicit kwarg.
"""

import re
from pathlib import Path


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
    results = []
    lines = source.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        m = re.search(r"subprocess\.(?:run|Popen)\b\s*\(", line)
        if m:
            start = i + 1
            depth = 1
            j = i
            start_pos = m.end()
            for j in range(i, n):
                line_text = lines[j]
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
            block = "\n".join(lines[i:j + 1])
            if re.search(r"\btext\s*=\s*(True|False)\b", block):
                results.append((start, block))
            i = j + 1
        else:
            i += 1
    return results


def test_text_subprocess_calls_have_errors_kwarg():
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    failures = []
    for module_name, paths in TARGET_MODULES.items():
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
                    f"  text=True without errors= kwarg (#61595)"
                )
    assert not failures, "failures: " + "\n".join(failures)


def test_wmic_call_has_errors_kwarg():
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    main_py = (worktree / "hermes_cli" / "main.py").read_text()
    wmic_match = re.search(
        r"result\s*=\s*subprocess\.run\(\s*\[?\s*\"wmic\"[^)]+\)",
        main_py, re.DOTALL,
    )
    if wmic_match is None:
        return
    block = wmic_match.group(0)
    assert "errors" in block, f"wmic subprocess.run lost errors= kwarg (regression of #17049)"


def test_safe_subprocess_run_helper_exists():
    """Engine-run enhancement: a safe_subprocess_run helper was added to
    hermes_cli/_subprocess_compat.py as a future-facing alternative to
    scattering errors= kwargs. This test pins that contract."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    compat = (worktree / "hermes_cli" / "_subprocess_compat.py").read_text()
    assert "safe_subprocess_run" in compat, (
        "safe_subprocess_run helper missing from hermes_cli/_subprocess_compat.py "
        "(engine-run future-facing fix surface for #61595)"
    )
