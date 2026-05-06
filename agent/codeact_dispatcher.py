"""CodeAct response dispatcher.

Handles the two jobs that sit between the model's raw API response and the
HermesKernel:

1. **Parsing** — extract the Python code block from the model response.
   Supports three formats in priority order:
   a. Structured envelope  {"thoughts": "...", "code": "..."}
   b. Markdown fence        ```python\\n...\\n```
   c. Bare code             (any response that looks like Python)

2. **Dispatching** — hand the extracted code to ``HermesKernel.execute()``
   and return a result string suitable for appending to the message history.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.codeact_kernel import HermesKernel

logger = logging.getLogger(__name__)

# Regex for ```python ... ``` or ``` ... ``` code fences.
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Heuristic: a response "looks like Python" if it starts with common keywords.
_PYTHON_STARTS = (
    "import ", "from ", "def ", "class ", "if ", "for ", "while ",
    "with ", "try:", "return ", "pass", "print(", "#", "result", "output", "data",
)


class CodeActParseError(ValueError):
    """Raised when no valid code block can be extracted from the model response."""


def extract_code(response: str, envelope_mode: bool = True) -> tuple[str, str]:
    """Return ``(thoughts, code)`` extracted from the model response.

    Parameters
    ----------
    response:
        Raw string returned by the model (tool call ``arguments`` field, or the
        full text response depending on how the caller feeds it in).
    envelope_mode:
        When True, attempt JSON envelope parse first.  When False, go straight
        to markdown/bare-code parsing.

    Raises
    ------
    CodeActParseError
        If no code block can be found by any strategy.
    """
    response = response.strip()

    if envelope_mode:
        thoughts, code = _try_envelope(response)
        if code is not None:
            return thoughts, code
        logger.debug("CodeAct envelope parse failed; falling back to fence/bare parse")

    # Markdown fence fallback.
    code = _try_fence(response)
    if code is not None:
        return "", code

    # Bare code fallback.
    if _looks_like_python(response):
        return "", response

    raise CodeActParseError(
        f"Could not extract code from model response "
        f"(first 200 chars): {response[:200]!r}"
    )


def dispatch(
    response: str,
    kernel: "HermesKernel",
    envelope_mode: bool = True,
) -> str:
    """Parse *response* and execute the extracted code in *kernel*.

    Returns the execution result string (stdout + last value + errors) that
    should be stored as the tool result in the message history.
    """
    try:
        thoughts, code = extract_code(response, envelope_mode=envelope_mode)
    except CodeActParseError as exc:
        # Return the parse error as the tool result so the model can recover.
        error_msg = (
            f"[CodeAct parse error] {exc}\n\n"
            "Please respond with valid Python code inside a JSON envelope:\n"
            '{"thoughts": "your reasoning", "code": "your Python code"}'
        )
        logger.warning("CodeAct parse error: %s", exc)
        return error_msg

    if thoughts:
        logger.debug("CodeAct thoughts: %s", thoughts[:120])
    logger.debug("CodeAct code (%d chars): %s...", len(code), code[:80].replace("\n", "↵"))

    return kernel.execute(code)


# ---------------------------------------------------------------------------
# Private parse helpers
# ---------------------------------------------------------------------------

def _try_envelope(response: str) -> tuple[str, str | None]:
    """Attempt to parse the JSON envelope.  Returns (thoughts, code|None)."""
    # The model may prepend text before the JSON — find the first '{'.
    brace_idx = response.find("{")
    if brace_idx == -1:
        return "", None

    candidate = response[brace_idx:]
    # Find the matching closing brace (handle nested braces).
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i, ch in enumerate(candidate):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return "", None

    json_str = candidate[: end + 1]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return "", None

    code = parsed.get("code")
    if not isinstance(code, str) or not code.strip():
        return "", None

    thoughts = parsed.get("thoughts", "")
    if not isinstance(thoughts, str):
        thoughts = ""

    return thoughts, code.strip()


def _try_fence(response: str) -> str | None:
    """Extract the first Python code fence.  Returns None if not found."""
    match = _CODE_FENCE_RE.search(response)
    if match:
        return match.group(1).strip()
    return None


def _looks_like_python(text: str) -> bool:
    """Heuristic: does this text look like standalone Python code?"""
    stripped = text.lstrip()
    if any(stripped.startswith(kw) for kw in _PYTHON_STARTS):
        return True
    return bool(re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*=", stripped))
