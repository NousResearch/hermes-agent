"""Supported gateway response envelopes for auxiliary calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class AuxiliaryEnvelopeError(RuntimeError):
    """A failed gateway envelope with its native provider error facts."""

    def __init__(self, error: Any, task: str | None = None) -> None:
        fields = error if isinstance(error, dict) else {}
        self.body = {"error": error}
        self.status_code = fields.get("status_code", fields.get("status"))
        self.code = fields.get("code")
        self.type = fields.get("type")
        message = fields.get("message")
        super().__init__(
            f"Auxiliary {task or 'call'}: provider returned error envelope: "
            f"{message or error}"
        )


def _unwrap_data_envelope(response: Any, task: str | None = None) -> Any:
    """Unwrap gateway envelopes like {"data": {<chat completion>}, "success": true}.

    Some OpenAI-compatible gateways (e.g. api.cline.bot) wrap non-streaming
    JSON bodies in a data/success envelope.  The SDK leniently parses this
    into a ChatCompletion with choices=None and keeps the envelope keys as
    extra fields, so the real completion is reachable at ``response.data``.
    Error envelopes ({"success": false, "data": {"error": ...}}) raise the
    provider's actual error instead of a generic invalid-response error.
    """
    from agent.auxiliary_client import _field

    if _field(response, "choices"):
        return response
    data = _field(response, "data")
    if not isinstance(data, dict):
        return response
    if _field(response, "success") is False:
        err = data.get("error") or data
        raise AuxiliaryEnvelopeError(err, task)
    if not data.get("choices"):
        return response
    try:
        return type(response).model_validate(data)
    except (AttributeError, TypeError, ValueError):
        # No model_validate (SimpleNamespace) or the inner payload fails
        # strict validation (pydantic ValidationError is a ValueError).
        pass
    choices = []
    for ch in data["choices"]:
        msg = ch.get("message") if isinstance(ch, dict) else None
        if isinstance(msg, dict):
            choices.append(
                SimpleNamespace(
                    message=SimpleNamespace(**msg),
                    finish_reason=ch.get("finish_reason"),
                )
            )
    if not choices:
        return response
    return SimpleNamespace(
        id=data.get("id"),
        model=data.get("model", ""),
        object=data.get("object", "chat.completion"),
        choices=choices,
        usage=data.get("usage"),
    )
