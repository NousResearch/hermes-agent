"""Materialize server-side Responses images at conversation intake, never normalization."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.image_gen_provider import save_native_b64_image


def _append_output_lines(message: Any, output_lines: list[str]) -> None:
    content = message.content or ""
    lines = [line for line in dict.fromkeys(output_lines) if line not in content.splitlines()]
    if not lines:
        return
    block = "\n".join(lines)
    message.content = f"{content}\n{block}" if content else block
    # Exact Codex message replay otherwise shadows the newly appended content. Only this
    # response's local references get a follower; past context and native items stay intact.
    pd = getattr(message, "provider_data", None) or {}
    if pd.get("codex_message_items"):
        pd["codex_message_items"] = [*pd["codex_message_items"], {
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": block}],
        }]


def materialize_response_images(agent: Any, message: Any) -> None:
    """Consume the transient image carrier before hooks or durable message construction."""
    outputs = getattr(message, "responses_image_outputs", None)
    if not outputs:
        return
    artifacts = getattr(agent, "_responses_image_artifacts", None)
    if artifacts is None:
        artifacts = agent._responses_image_artifacts = {}
    local = []
    for item in outputs:
        key = (item.get("response_identity", item.get("response_id")), item.get("item_id") or item.get("output_index"))
        if key not in artifacts:
            artifacts[key] = {k: v for k, v in item.items() if k not in ("result", "output_format", "status")}
            try:
                artifacts[key]["path"] = item.get("path") or str(save_native_b64_image(
                    item.get("result"), output_format=item.get("output_format"),
                ))
            except ValueError as exc:
                artifacts[key]["error"] = str(exc)
            except OSError:
                artifacts[key]["error"] = "Could not write generated image."
        if artifacts[key] not in local:
            local.append(artifacts[key])
    message.provider_data["responses_image_outputs"] = local
    _append_output_lines(message, _artifact_lines(local))


def _artifact_lines(artifacts: Any) -> list[str]:
    lines = []
    for item in artifacts:
        if item.get("path") and not Path(item["path"]).is_file():
            item.pop("path")
            item["error"] = "Generated image artifact is missing."
        lines.append(f"MEDIA:{item['path']}" if item.get("path") else f"Image output error: {item['error']}")
    return lines


def append_pending_response_images(agent: Any, message: Any) -> None:
    """Carry undelivered images from function/incomplete rounds into the final reply."""
    pending = [item for item in (getattr(agent, "_responses_image_artifacts", None) or {}).values() if not item.get("delivered")]
    _append_output_lines(message, _artifact_lines(pending))


def preserve_recovery_response_images(
    agent: Any, messages: Any, final_response: str, finish_reason: str,
) -> tuple[str, bool, Any]:
    """Settle recovery output before persistence; return text and public transform metadata."""
    from types import SimpleNamespace
    from agent.message_metadata import append_message, stamp_message_timestamp
    from agent.context_compressor import _DB_PERSISTED_MARKER

    # A settled transform may deliberately remove or withhold these attachments.
    # Missing interim delivery is not permission to undo that decision.
    from agent.turn_finalizer import apply_llm_output_transform, public_pre_transform

    recorded = getattr(agent, "_llm_output_transform", None)
    turn_id = getattr(agent, "_current_turn_id", "") or ""
    message = SimpleNamespace(content=final_response, provider_data={})
    if isinstance(recorded, tuple) and len(recorded) == 3 and recorded[0] == turn_id:
        # Execution is settled, but the authoritative final may not have a row yet.
        transformed, pre_transform = recorded[1:]
    else:
        append_pending_response_images(agent, message)
        transformed, pre_transform = False, None
        if getattr(agent, "_hold_stream", False) is True or (
            finish_reason != "interrupted" and not getattr(agent, "_interrupt_requested", False)
        ):
            # These exits never reach normal finalization. Ordinary and held transforms
            # must both precede the first durable write; interrupted ordinary output stays raw.
            message.content, transformed, pre_transform = apply_llm_output_transform(agent, message.content, turn_id=turn_id)
    tail = messages[-1] if messages else None
    # Stripping just the appended MEDIA can equal the input notice while no final
    # exists. Publication depends on a visible final row, not on a text delta.
    if not (
        isinstance(tail, dict) and tail.get("role") == "assistant"
        and tail.get("content") == message.content and not tail.get("tool_calls")
        and tail.get("display_kind") != "hidden"
        and (finish_reason != "interrupted" or tail.get("display_metadata", {}).get("interrupted") is True)
    ):
        if (
            isinstance(tail, dict) and tail.get("role") == "assistant"
            and tail.get("content") == final_response and not tail.get(_DB_PERSISTED_MARKER)
            and not tail.get("tool_calls") and tail.get("display_kind") != "hidden"
        ):
            # A fresh budget summary is not durable yet: augment it, don't duplicate it.
            tail["content"] = message.content
            if finish_reason == "interrupted":
                tail.setdefault("display_metadata", {})["interrupted"] = True
            stamp_message_timestamp(tail)
        else:
            row = {"role": "assistant", "content": message.content, "finish_reason": finish_reason}
            if finish_reason == "interrupted":
                row["display_metadata"] = {"interrupted": True}
            append_message(messages, row)
    return message.content, transformed, public_pre_transform(agent, pre_transform)


def interrupted_image_result(agent: Any, messages: Any, api_call_count: int, text: str) -> dict[str, Any]:
    """Settle a pending native final before the interrupt owner closes its tool tail."""
    result = {
        "final_response": text, "messages": messages, "api_calls": api_call_count,
        "completed": False, "interrupted": True,
    }
    if any(not item.get("delivered") for item in (getattr(agent, "_responses_image_artifacts", None) or {}).values()):
        from agent.turn_finalizer import settle_held_final
        settle_held_final(agent, result)
    return result


def delivered_response_image_refs(agent: Any) -> str:
    """References already shown live that a refusal owner must retain without resending."""
    delivered = [item for item in (getattr(agent, "_responses_image_artifacts", None) or {}).values() if item.get("delivered")]
    return "\n".join(_artifact_lines(delivered))


def note_response_images_delivered(agent: Any, text: str) -> None:
    """Record successful interim delivery before the generic text dedupe folds newlines."""
    lines = set(text.splitlines())
    for item in (getattr(agent, "_responses_image_artifacts", None) or {}).values():
        if item.get("path") and f"MEDIA:{item['path']}" in lines:
            item["delivered"] = True
