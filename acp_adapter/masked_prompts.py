"""Hermes' masked prompts in an ACP editor that opts in: the sudo password, a skill's missing
credential, and the browser vault's unlock, save-login and one-time-code panels.

For a local viewer's turn the gateway sends these as its own server→client requests (``srq-`` ids,
``gateway/session_server_requests.py``) and waits for a ``ValueResult``. The CLI, TUI and Desktop
answer them with their own masked panels. ACP has no masked-input primitive, and form elicitation
is not meant to carry sensitive values, so an editor that advertises
``clientCapabilities._meta.hermes.maskedPrompts: true`` in ``initialize`` gets them as the
``_hermes/masked_prompt`` extension request. Any other editor is sent nothing, as before: the
prompt waits for another viewer or for its own timeout.

Request params::

    {"sessionId", "promptId", "kind", "title", "message",
     "fields": [{"name", "label", "masked"}], "details": {...the gateway's params...}}

Response: ``{"action": "accept", "values": {<field name>: <string>}}``. Anything else (a
``decline``, a ``cancel``, an error) is the skip every other viewer sends.

``_hermes/masked_prompt_cancel`` (a notification, ``{"sessionId", "promptId", "reason"}``) withdraws
a request the gateway settled first: another viewer answered it, the turn was interrupted, or it
timed out. The values travel only in the editor's response and the gateway's response frame; they
are never logged here.
"""
from __future__ import annotations

import json
from typing import Any

METHOD = "_hermes/masked_prompt"
CANCEL_METHOD = "_hermes/masked_prompt_cancel"


def client_supports_masked_prompts(initialize_params: Any) -> bool:
    """True only for an explicit ``maskedPrompts: true`` under ``_meta.hermes`` of the v1
    ``clientCapabilities`` or v2 ``capabilities``."""
    if not isinstance(initialize_params, dict):
        return False
    for key in ("clientCapabilities", "capabilities"):
        capabilities = initialize_params.get(key)
        meta = capabilities.get("_meta") if isinstance(capabilities, dict) else None
        hermes = meta.get("hermes") if isinstance(meta, dict) else None
        if isinstance(hermes, dict) and hermes.get("maskedPrompts") is True:
            return True
    return False


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _field(name: str, label: str, masked: bool = True) -> dict[str, Any]:
    return {"name": name, "label": label, "masked": masked}


def _sudo(params: dict) -> tuple[str, str, list]:
    command = _str(params.get("command"))
    message = f"Hermes needs your sudo password to run:\n{command}" if command else \
        "Hermes needs your sudo password to run a command."
    return "sudo password required", message, [_field("value", "Password")]


def _secret(params: dict) -> tuple[str, str, list]:
    env_var = _str(params.get("env_var"))
    metadata = params.get("metadata") if isinstance(params.get("metadata"), dict) else {}
    skill = _str(metadata.get("skill_name"))
    message = _str(params.get("prompt")) or f"Enter a value for {env_var}."
    help_text = _str(metadata.get("help"))
    if help_text:
        message = f"{message}\n{help_text}"
    message += "\nIt is saved to your Hermes profile; the model never sees it."
    title = f"Skill setup required: {skill}" if skill else "Skill setup required"
    return title, message, [_field("value", env_var or "Value")]


def _vault_unlock(params: dict) -> tuple[str, str, list]:
    name = _str(params.get("display_name")) or _str(params.get("backend")) or "your password manager"
    return f"Unlock {name}", f"Hermes needs your master password to unlock {name}.", [_field("value", "Master password")]


def _vault_save_login(params: dict) -> tuple[str, str, list]:
    origin, site = _str(params.get("origin")), _str(params.get("site"))
    message = f"Hermes stores this login in your vault for {origin or site} and fills it in; the model never sees it."
    return (f"Save a login for {site or origin}", message,
            [_field("identifier", "Username or email", masked=False), _field("password", "Password")])


def _vault_code(params: dict) -> tuple[str, str, list]:
    site, hint = _str(params.get("site")), _str(params.get("hint"))
    title = f"Enter the code {site} sent you" if site else "Enter your one-time code"
    return title, hint or "Enter the one-time code you were sent.", [_field("value", "Code", masked=False)]


# kind -> (title, message, fields) for the editor's card.
_CARDS = {"sudo": _sudo, "secret": _secret, "vault.unlock_prompt": _vault_unlock,
          "vault.save_login": _vault_save_login, "vault.code": _vault_code}
KINDS = frozenset(_CARDS)


def build_request(editor_session_id: str, prompt_id: str, kind: str, params: dict) -> dict[str, Any]:
    """The ``_hermes/masked_prompt`` params for one gateway request (``kind`` is in ``KINDS``)."""
    card = _CARDS.get(kind)
    if card is None:
        raise ValueError(f"not a masked prompt: {kind}")
    params = params if isinstance(params, dict) else {}
    title, message, fields = card(params)
    details = {key: value for key, value in params.items() if key != "session_id"}
    return {"sessionId": editor_session_id, "promptId": prompt_id, "kind": kind, "title": title,
            "message": message, "fields": fields, "details": details}


def answer(kind: str, response: Any) -> str:
    """The ``ValueResult`` value for the editor's response; ``""`` (skip) for anything but an accept.
    ``vault.save_login`` goes back as the one JSON string the gateway's cards send."""
    if not isinstance(response, dict) or response.get("action") != "accept":
        return ""
    values = response.get("values")
    if not isinstance(values, dict):
        return ""
    if kind == "vault.save_login":
        password, identifier = values.get("password"), values.get("identifier")
        if not isinstance(password, str) or not password:
            return ""
        return json.dumps({"identifier": identifier if isinstance(identifier, str) else "", "password": password})
    value = values.get("value")
    return value if isinstance(value, str) else ""
