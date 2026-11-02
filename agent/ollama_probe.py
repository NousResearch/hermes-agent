"""The Ollama ``/api/show`` probe: the payload's shape and the context it advertises.

A local gateway that proxies models can answer ``/api/tags`` and ``/api/show`` in an
Ollama-ish shape. Believing that payload made Hermes send Ollama's own request extension
(``options.num_ctx``) to a backend that validates its schema strictly and rejects the field
(#142239) - so the shape is checked before any of it is used.
"""

from __future__ import annotations

from typing import Any, Optional

# Keys only Ollama's ShowResponse carries. A real Ollama always has at least one of them:
# the Modelfile text, the parameter block, the chat template, or the 0.6+ capability list.
_OLLAMA_SHOW_MARKERS = ("modelfile", "parameters", "template", "capabilities")


def is_ollama_show_payload(data: Any) -> bool:
    """True when a ``/api/show`` payload is Ollama's own shape.

    A compatibility layer answers with its own summary (``details.context_length`` and a
    ``model_info`` block) and none of the markers above. That is not a model card: the window
    it advertises is the gateway's, and sending it on as Ollama's ``num_ctx`` gets the whole
    request rejected by a strict upstream (#142239)."""
    if not isinstance(data, dict):
        return False
    if any(key in data for key in _OLLAMA_SHOW_MARKERS):
        return True
    details = data.get("details")
    return isinstance(details, dict) and "format" in details


def ollama_show_context(
    data: dict[str, Any], *, gguf_first: bool, minimum: Optional[int] = None
) -> Optional[int]:
    """Context length from an Ollama ``/api/show`` payload. ``parameters.num_ctx`` is the RUNTIME
    window (Modelfile override), ``model_info.*.context_length`` the GGUF training max. Local users
    control num_ctx (prefer it); hosted operators may cap it arbitrarily (``gguf_first``)."""
    def _num_ctx():
        for line in data.get("parameters", "").split("\n"):
            parts = line.strip().split()
            if "num_ctx" in line and len(parts) >= 2:
                try:
                    yield int(parts[-1])
                except ValueError:
                    continue
    def _gguf():
        for key, value in data.get("model_info", {}).items():
            if "context_length" in key and isinstance(value, (int, float)):
                yield int(value)
    for reader in ((_gguf, _num_ctx) if gguf_first else (_num_ctx, _gguf)):
        for ctx in reader():
            if minimum is None or ctx >= minimum:
                return ctx
    return None
