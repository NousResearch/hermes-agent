"""Test helper for decoding the UNTRUSTED_DOCUMENT JSON fence.

Completes the incomplete merge b1576ef3c: 8 content-trust consumer
tests import ``loads_fenced_json`` from this module to extract and
parse the JSON payload that web tools (web_search, web_extract) wrap
in the ``UNTRUSTED_DOCUMENT`` fence defined in
``hermes_cli/content_trust.py``.

Fence format (two variants produced by production code):

    <<<UNTRUSTED_DOCUMENT>>>
    {json}
    <<<END_UNTRUSTED_DOCUMENT>>>

    <<<UNTRUSTED_DOCUMENT>>> source="web_search" tier="external-untrusted">>>
    {json}
    <<<END_UNTRUSTED_DOCUMENT>>>

This helper is test-only. It does NOT change production web-tool
behaviour — it simply makes the fence decodable for assertions.
"""

from __future__ import annotations

import json
import re

# Fence markers — must match hermes_cli/content_trust.py.
_UNTRUSTED_BEGIN = "<<<UNTRUSTED_DOCUMENT>>>"
_UNTRUSTED_END = "<<<END_UNTRUSTED_DOCUMENT>>>"

# Regex: capture everything between the opening fence line (which may
# carry source=/tier= attributes) and the closing fence line.
# DOTALL so the payload can span multiple lines.
_FENCE_RE = re.compile(
    re.escape(_UNTRUSTED_BEGIN) + r"[^\n]*\n(.*?)" + re.escape(_UNTRUSTED_END),
    re.DOTALL,
)


def loads_fenced_json(text: str) -> dict | list:
    """Decode the JSON payload inside a UNTRUSTED_DOCUMENT fence.

    Args:
        text: A string containing exactly one UNTRUSTED_DOCUMENT fence
              with a JSON payload between the begin/end markers.

    Returns:
        The parsed JSON (dict or list).

    Raises:
        ValueError: If the fence markers are absent or the fenced
                    content is not valid JSON.
    """
    match = _FENCE_RE.search(text or "")
    if match is None:
        raise ValueError(
            "UNTRUSTED_DOCUMENT fence not found in input — "
            "expected <<<UNTRUSTED_DOCUMENT>>> ... <<<END_UNTRUSTED_DOCUMENT>>>"
        )
    payload = match.group(1).strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Fenced content is not valid JSON: {exc}"
        ) from exc
