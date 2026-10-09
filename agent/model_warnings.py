"""Application-owned warnings about model suitability."""

from __future__ import annotations

import re


NOUS_HERMES_NON_AGENTIC_WARNING = (
    "Nous Research Hermes 3 & 4 models are NOT agentic and are not designed "
    "for use with Hermes Agent. They lack the tool-calling capabilities "
    "required for agent workflows. Consider using an agentic model instead "
    "(Claude, GPT, Gemini, DeepSeek, etc.)."
)

# Match only the real Nous Research Hermes 3 / 4 chat families; a bare
# substring check false-positives on unrelated local Modelfile names.
_NOUS_HERMES_NON_AGENTIC_RE = re.compile(
    r"(?:^|[/:])hermes[-_ ]?[34](?:[-_.:]|$)", re.IGNORECASE
)


def is_nous_hermes_non_agentic(model_name: str) -> bool:
    """Return whether *model_name* is a Nous Hermes 3/4 chat model."""
    return bool(model_name and _NOUS_HERMES_NON_AGENTIC_RE.search(model_name))


def nous_hermes_non_agentic_warning(model_name: str) -> str:
    """Return the Hermes 3/4 suitability warning, or an empty string."""
    return (
        NOUS_HERMES_NON_AGENTIC_WARNING
        if is_nous_hermes_non_agentic(model_name)
        else ""
    )


__all__ = [
    "NOUS_HERMES_NON_AGENTIC_WARNING",
    "is_nous_hermes_non_agentic",
    "nous_hermes_non_agentic_warning",
]
