"""discrimination-guard — refuse a verdict whose measurement cannot be wrong.

WHY THIS EXISTS
---------------
16 wrong verdicts in one session, and all 16 are one defect:

    a measurement was accepted without first establishing that it can
    DISTINGUISH the claimed outcome from its opposite.

Four disguises, all the same thing:

  indicator      read a signal without checking what it can mean
                 (`tf.format` on a read handle reports the READER default;
                  `hasattr` cannot see a nested function)
  environment    the probe ran in a world unlike production
                 (an incomplete venv turns ModuleNotFoundError into "19 bugs")
  discrimination the check passes under BOTH outcomes
                 (`(rc == 0) == (on == wanted)` is a tautology)
  silent-failure the command failed and said so somewhere unread
                 (`git checkout -B` prints "Aborting" and exits non-zero)

The existing contract-guard matches TEXT PATTERNS of past mistakes, so it
scored 0/8 against this session's new shapes. Pattern matching on last week's
errors cannot catch next week's. This guard asks a structural question instead:

    does this probe contain anything that could have come out the other way?

WHAT IT CHECKS (all structural, none tied to a specific past bug)
  1. a verdict is being emitted with no negative control in the same probe
  2. an assertion compares two of the probe's own observations (tautology)
  3. a subprocess/shell result is consumed without inspecting returncode
  4. "zero failures" style conclusions with no proof anything ran
  5. presence-of-text (grep/`in`) used as evidence a code path is guarded

Default is warn. DISCRIMINATION_GUARD_BLOCK=1 makes it refuse.
"""
from __future__ import annotations

import os
import re
from typing import Any, List

# --- structural signals ---------------------------------------------------

# a probe that prints a verdict
_VERDICT = re.compile(
    r"\b(ok|pass(?:ed)?|fail(?:ed|s)?|clean|safe|leak|vulnerable|broken|"
    r"present|missing|correct|wrong|verdict|confirmed|reproduce[sd]?)\b",
    re.I)

# a negative control: something asserted NOT to happen, or a control fixture
_NEGATIVE_CONTROL = re.compile(
    r"(negative[ _]control|control\b|counterexample|must not|should not|"
    r"assert not|!=|is False|== False|unchanged|untouched|baseline)", re.I)

# comparing two observations the probe itself produced
_TAUTOLOGY = re.compile(
    r"\(\s*\w+\s*==\s*[^)]+\)\s*==\s*\(\s*\w+\s*==\s*[^)]+\)"
    r"|assert\s+\w+\s*==\s*\w+\s*$")

# subprocess use
_SUBPROCESS = re.compile(r"subprocess\.(run|call|Popen)|check_output|os\.system")
_RC_CHECKED = re.compile(
    r"returncode|\.check_returncode|check=True|rc\s*[!=]=|exit_code|"
    r"\$\?|CalledProcessError")

# "nothing failed" conclusions
_ZERO_FAIL = re.compile(
    r"(0|zero|no)\s+(fail(?:ure|ed)?s?|error(?:s)?|leak(?:s)?|regression(?:s)?)",
    re.I)
_RAN_PROOF = re.compile(
    r"(collected|passed|\btotal\b|len\(|count|assert\s+\w+\s*>\s*0|"
    r"\bran\b|executed)", re.I)

# text presence used as proof of behaviour
_GREP_AS_PROOF = re.compile(
    r"(\"[^\"]+\"\s+in\s+(src|source|text|body|content|block|code)\w*"
    r"|grep\s+-c\b)")
_DRIVES_CODE = re.compile(
    r"(import(?:lib)?|spec_from_file_location|__import__|"
    r"getattr\(|\bcall\(|\(\)\s*$)", re.M)

_PROBE_HINT = re.compile(
    r"(audits?/|probe|verify|check|hunt|prove|reach|test_|_test\b)", re.I)


def _looks_like_a_probe(path: str, content: str) -> bool:
    if _PROBE_HINT.search(path or ""):
        return True
    return bool(_VERDICT.search(content) and
                ("print(" in content or "assert " in content))


def analyse(content: str) -> List[str]:
    """Return structural objections. Empty list == nothing to say."""
    out: List[str] = []

    if _VERDICT.search(content) and not _NEGATIVE_CONTROL.search(content):
        out.append(
            "emits a verdict but contains NO negative control — add a case that "
            "must come out the other way, or the probe cannot fail")

    if _TAUTOLOGY.search(content):
        out.append(
            "an assertion compares two of the probe's own observations; that is "
            "true whichever way the system behaves — assert the contract instead")

    if _SUBPROCESS.search(content) and not _RC_CHECKED.search(content):
        out.append(
            "consumes a subprocess result without inspecting returncode — a "
            "command that refused to run reads as a clean result")

    if _ZERO_FAIL.search(content) and not _RAN_PROOF.search(content):
        out.append(
            "concludes 'no failures' without evidence anything RAN — zero "
            "failures and zero executions look identical")

    if _GREP_AS_PROOF.search(content) and not _DRIVES_CODE.search(content):
        out.append(
            "uses text presence as proof a code path behaves — grep finds the "
            "symbol, not the branch that executes; drive the real function")

    return out


def _on_pre_tool_call(tool_name: str = "", args: Any = None, **_: Any):
    if tool_name not in ("write_file", "patch", "execute_code"):
        return None
    if not isinstance(args, dict):
        return None
    content = args.get("content") or args.get("code") or args.get("new_string") or ""
    path = args.get("path") or ""
    if not isinstance(content, str) or len(content) < 120:
        return None
    if not _looks_like_a_probe(str(path), content):
        return None

    problems = analyse(content)
    if not problems:
        return None

    msg = ("discrimination-guard: this probe may not be able to produce the "
           "opposite result.\n" + "\n".join(f"  - {p}" for p in problems))
    if os.environ.get("DISCRIMINATION_GUARD_BLOCK") == "1":
        return {"action": "block",
                "message": msg + "\n\n(unset DISCRIMINATION_GUARD_BLOCK to warn only)"}
    return {"action": "warn", "message": msg}


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
