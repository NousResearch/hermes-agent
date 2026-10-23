"""A canonical ``clarify`` prompt in an ACP editor, answered through ``clarify.respond``.

A client that advertises ``elicitation.form`` in ``initialize`` gets the question as an
``elicitation/create`` form (free text or the offered choices); any other client gets a permission
card whose options are the choices plus Skip, the one interaction every ACP client renders.
Form schema and capability detection adapted from #126313 (Utku Bakir).
"""
from __future__ import annotations

import json
from typing import Any

ELICITATION_METHOD = "elicitation/create"
SKIP_OPTION = "skip"


def client_supports_form_elicitation(initialize_params: Any) -> bool:
    """ACP names an elicitation mode as supported only by its presence (``"form": {}``), under
    v1 ``clientCapabilities`` or v2 ``capabilities``. Read from the raw initialize params:
    the pinned SDK's ``ClientCapabilities`` model drops the field."""
    if not isinstance(initialize_params, dict):
        return False
    for key in ("clientCapabilities", "capabilities"):
        capabilities = initialize_params.get(key)
        elicitation = capabilities.get("elicitation") if isinstance(capabilities, dict) else None
        if isinstance(elicitation, dict) and elicitation.get("form") is not None:
            return True
    return False


def build_clarify_elicitation(session_id: str, prompt: dict) -> dict[str, Any]:
    choices = list(prompt.get("choices") or [])
    schema: dict[str, Any] = {"type": "string"}
    if choices:
        schema = {"type": "string", "enum": choices}
        if prompt.get("multi_select"):
            schema = {"type": "array", "items": schema}
    return {"sessionId": session_id, "mode": "form", "message": prompt["question"],
            "requestedSchema": {"type": "object", "properties": {"answer": schema}, "required": ["answer"]}}


def elicited_answer(response: Any) -> str | None:
    """Accept -> the answer (a multi-select list as the JSON array clarify decodes); decline or
    cancel -> ``""``, the skip every surface sends; anything else -> ``None`` (not answered)."""
    if not isinstance(response, dict) or response.get("action") not in {"accept", "decline", "cancel"}:
        return None
    if response["action"] != "accept":
        return ""
    value = (response.get("content") or {}).get("answer")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return json.dumps(value, ensure_ascii=False)
    return value if isinstance(value, str) else None


def build_clarify_permission(prompt: dict):
    """``(tool_call, options, answers)``: one option per offered choice plus Skip; ``answers``
    maps each option id to the exact ``clarify.respond`` answer."""
    import acp
    from acp.schema import PermissionOption

    choices = list(prompt.get("choices") or [])
    options = [PermissionOption(option_id=f"choice-{index}", kind="allow_once", name=choice)
               for index, choice in enumerate(choices)]
    options.append(PermissionOption(option_id=SKIP_OPTION, kind="reject_once", name="Skip"))
    answers = {f"choice-{index}": choice for index, choice in enumerate(choices)}
    answers[SKIP_OPTION] = ""
    tool_call = acp.update_tool_call(
        f"clarify-{prompt['prompt_id']}", title=prompt["question"], kind="other", status="pending",
        content=[acp.tool_content(acp.text_block(prompt["question"]))],
        raw_input={"question": prompt["question"], "choices": choices})
    return tool_call, options, answers
