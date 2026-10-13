"""Compatible Responses recovery for auxiliary callers."""

from typing import Any, Optional
from types import SimpleNamespace


def _recover_aux_response_message(response: Any) -> Optional[SimpleNamespace]:
    """Recover compatible Responses output through the complete native normaliser."""
    from agent.auxiliary_client import _field
    from agent.auxiliary_codex_response import _parse_codex_final_response

    output_text = _field(response, "output_text")
    if not isinstance(_field(response, "output"), list) and not (
        isinstance(output_text, str) and output_text.strip()
    ):
        return None
    native = SimpleNamespace(**response) if isinstance(response, dict) else response
    recovered = _parse_codex_final_response(native)
    message = recovered.choices[0].message
    if message.content or message.tool_calls or message.refusal or message.reasoning:
        return recovered
    return None
