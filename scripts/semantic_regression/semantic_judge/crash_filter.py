"""Deterministic crash-related failure classifier.

Runs BEFORE any LLM call. Its job is to separate two distinct failure classes:

  - CRASH-RELATED  : the Hermes process / tool dispatch blew up (uncaught
    exception, stack trace, segfault, timeout kill, connection-level error that
    aborted the run, terminal returning exit<0 with a stack trace). These are
    already covered by the existing crash-focused test suite and must NEVER be
    reported as semantic regressions (task AC #3).
  - SEMANTIC       : the run completed; its OUTPUT (tool calls, params, plugin
    output, skill result) is present but may be *wrong* (wrong tool, wrong
    params, wrong ordering, fabricated/skipped result). These are the judge's
    domain.

The classifier is intentionally conservative: it only routes a run OUT of the
semantic judge when crash evidence is unambiguous. Anything it cannot
confidently classify as a crash is passed through to the LLM judge, which is
also told to treat raw stack traces as crash-class (defence in depth).

Signals that indicate a CRASH (stack-trace / unhandled-error class):
  - literal "Traceback (most recent call last):"
  - a Python exception line ("TypeError:", "KeyError:", "AttributeError:",
    "RuntimeError:", "ValueError:" etc.) immediately following a Traceback
  - "Segmentation fault", "core dumped", "Killed", "OOM", "MemoryError"
  - "tool_executor: Tool X returned error" wrapping an *unhandled* exception
    payload (as opposed to an expected per-scenario error like a multi-match
    patch refusal, which is a *semantic contract* signal, not a crash)
  - "Unhandled exception", "Fatal", "Process crashed", "exit code 137/139"
  - a terminal/tool returning exit_code < 0 along with a traceback

Deliberate, scenario-defined "expected error" signals are NOT crashes:
  - patch multi-match refusal (T3) -> semantic contract, judge decides
  - terminal exit != 0 for a command that genuinely failed (T4) -> semantic
  - plugin __init__ SyntaxError isolation (P5) -> semantic (isolation is the
    contract), NOT a crash of the session
These stay in the semantic channel.
"""

import re

# Strong, unambiguous crash markers. Presence => crash_related=True.
TRACEBACK = re.compile(r"Traceback \(most recent call last\):")
PY_EXC = re.compile(
    r"^\s*(?:"
    r"(?:Type|Key|Attribute|Runtime|Value|Index|Name|Syntax|OS|Permission|"
    r"FileNotFound|Import|ModuleNotFound|Connection|Timeout|Overflow|ZeroDiv|"
    r"Recursion|Unicode|Environment|Assertion)Error"
    r"):", re.MULTILINE
)
UNHANDLED = re.compile(
    r"(?:Unhandled exception|Fatal error|Process crashed|"
    r"Segmentation fault|core dumped|core\.%d|MemoryError|\bKilled\b|\bOOM\b)",
    re.IGNORECASE,
)
BAD_EXIT = re.compile(r"exit_code[:\s=]+(-[1-9]|13[0-9])", re.IGNORECASE)

# A tool error that is an *unhandled* Python exception (crash) vs an expected
# semantic/contract error. Expected-error phrases that must NOT trigger crash:
SEMANTIC_ERROR_CONTEXT = (
    "multi-match",
    "multiple matches",
    "old_string not found",
    "not found",
    "no unique match",
    "ambiguous",
    "collision",
    "is not enabled",
    "plugin skipped",
    "isolated",
    "syntax error in plugin",
    "does not resolve",
    "already applied",
    "no-op",
    "expected failure",
    "assertion",
    "validation failed",
    "not supported",
)


def classify(log_text: str) -> dict:
    """Return {'crash_related': bool, 'reason': str|null, 'markers': [str]}."""
    markers = []
    if not log_text or not log_text.strip():
        return {"crash_related": False, "reason": None, "markers": markers}

    if TRACEBACK.search(log_text):
        markers.append("traceback")
    # Python exception line NOT preceded by an expected semantic-error context.
    for m in PY_EXC.finditer(log_text):
        line = log_text[max(0, m.start() - 200): m.end()]
        if any(ctx in line.lower() for ctx in SEMANTIC_ERROR_CONTEXT):
            continue
        markers.append(f"py_exc:{m.group(0).strip()[:40]}")
    if UNHANDLED.search(log_text):
        markers.append("unhandled")
    if BAD_EXIT.search(log_text):
        markers.append("bad_exit")

    if markers:
        return {
            "crash_related": True,
            "reason": "Crash-related failure detected (covered by existing "
                      "crash suite): " + ", ".join(markers),
            "markers": markers,
        }
    return {"crash_related": False, "reason": None, "markers": markers}
