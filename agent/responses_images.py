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
    if not hasattr(agent, "_responses_image_artifacts"):
        agent._responses_image_artifacts = {}
    artifacts = agent._responses_image_artifacts
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
    pending = [item for item in getattr(agent, "_responses_image_artifacts", {}).values() if not item.get("delivered")]
    _append_output_lines(message, _artifact_lines(pending))


def preserve_recovery_response_images(
    agent: Any, messages: Any, final_response: str, finish_reason: str,
) -> str:
    """Persist undelivered native artifacts on recovery exits that bypass normal finalization."""
    from types import SimpleNamespace
    from agent.message_metadata import append_message, stamp_message_timestamp
    from agent.context_compressor import _DB_PERSISTED_MARKER

    message = SimpleNamespace(content=final_response, provider_data={})
    append_pending_response_images(agent, message)
    if message.content != final_response:
        tail = messages[-1] if messages else None
        if (
            isinstance(tail, dict) and tail.get("role") == "assistant"
            and tail.get("content") == final_response and not tail.get(_DB_PERSISTED_MARKER)
        ):
            # A fresh budget summary is not durable yet: augment it, don't duplicate it.
            tail["content"] = message.content
            stamp_message_timestamp(tail)
        else:
            append_message(messages, {"role": "assistant", "content": message.content, "finish_reason": finish_reason})
    return message.content


def note_response_images_delivered(agent: Any, text: str) -> None:
    """Record successful interim delivery before the generic text dedupe folds newlines."""
    lines = set(text.splitlines())
    for item in getattr(agent, "_responses_image_artifacts", {}).values():
        if item.get("path") and f"MEDIA:{item['path']}" in lines:
            item["delivered"] = True
