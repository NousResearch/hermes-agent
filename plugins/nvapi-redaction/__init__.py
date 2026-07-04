"""nvapi-redaction — a vendor token format as a plugin, not a core PR.

NVIDIA API keys (``nvapi-...``) authenticate NIM endpoints and
build.nvidia.com — common in self-hosted Hermes stacks running local
NIM backends. The format is not in the core ``_PREFIX_PATTERNS`` list
in ``agent/redact.py``, so today an ``nvapi-`` key that lands in a
transport error or an ``env``-dump would be masked only if a generic
pattern happens to catch it.

Historically the fix was a one-line core PR appending to
``_PREFIX_PATTERNS`` (that's how fw_, retaindb_, hsk-, mem0_, and brv_
got there). This plugin is the same one-liner shipped through
``ctx.register_redaction_patterns()`` instead — the reference
implementation for the redaction-registry plugin interface.

Registered patterns are additive-only: they extend what gets masked and
cannot weaken built-in redaction.
"""

from __future__ import annotations

# NVIDIA API keys: "nvapi-" followed by the token body. Real keys are
# 60+ chars; the 20-char floor mirrors the conservative floors used for
# other vendors in agent/redact.py while avoiding prose false-positives.
NVAPI_PATTERN = r"nvapi-[A-Za-z0-9_-]{20,}"


def register(ctx) -> None:
    ctx.register_redaction_patterns([NVAPI_PATTERN])
