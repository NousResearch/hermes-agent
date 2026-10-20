"""Parse authoritative context windows from provider overflow messages."""

import re
from typing import Optional


def parse_context_limit_from_error(error_msg: str) -> Optional[int]:
    """Context limit quoted in a provider error ("maximum context length is 32768 tokens"), if any.

    A message about only an OUTPUT cap ("... model output limit of 16384") never says "context";
    bail out so the generic "limit ... of N" pattern can't cache the output cap as the window."""
    error_lower = error_msg.lower()
    if ("output limit" in error_lower or "output tokens" in error_lower or "output token" in error_lower) and "context" not in error_lower:
        return None
    patterns = (
        r'available\s+context\s+size\s*\(?\s*(\d{4,})',
        r'max_model_len\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM: "max_model_len 32768", "=32768", ": 32768", "(32768)", "is 32768"
        r'maximum model length\s*(?:is\s*)?[:=(]?\s*(\d{4,})',  # vLLM alt: "maximum model length 131072", "... is 131072"
        r'(?:max(?:imum)?|limit)\s*(?:context\s*)?(?:length|size|window)?\s*(?:is|of|:)?\s*(\d{4,})',
        r'context\s*(?:length|size|window)\s*(?:is|of|:)?\s*(\d{4,})',
        r'(\d{4,})\s*(?:token)?\s*(?:context|limit)',
        r'>\s*(\d{4,})\s*(?:max|limit|token)',  # "250000 tokens > 200000 maximum"
        r'(\d{4,})\s*(?:max(?:imum)?)\b',  # "200000 maximum"
        # Gemini: "input token count is 32825 but model only supports up to
        # 32768" — anchor on the phrase so the input count isn't captured.
        r'supports?\s+(?:only\s+)?up\s+to\s+(\d{4,})',
    )
    for match in filter(None, (re.search(pattern, error_lower) for pattern in patterns)):
        limit = int(match.group(1))
        if 1024 <= limit <= 10_000_000:  # sanity: must be a plausible window
            return limit
    return None

