"""Route bulk request payloads around the OpenAI SDK's ``maybe_transform`` walk.

Both wire families pay the same cost. ``responses.create`` and ``chat.completions.create``
re-walk the whole request body against their typed param union with the GIL held, so a
multi-MB conversation can wedge the process pre-network, where no watchdog socket kill helps.
The SDK merges ``extra_body`` AFTER the transform, so moving wire-format bulk fields there
yields a byte-identical request without the walk.

Shared by ``agent.codex_runtime`` (Responses: ``input``/``tools``) and
``agent.chat_completion_helpers`` (Chat Completions: ``messages``/``tools``).
"""

from __future__ import annotations

import os
from typing import Any

# Bulk fields carrying the conversation payload; the rest is scalar config the transform handles fast.
RESPONSES_BYPASS_FIELDS = ("input", "tools")
CHAT_COMPLETIONS_BYPASS_FIELDS = ("messages", "tools")


def _is_plain_json_data(value: Any) -> bool:
    """True when ``value`` is purely JSON wire types; pydantic models / generators must keep the typed SDK path."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_plain_json_data(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_is_plain_json_data(item) for item in value)
    return False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def bypass_sdk_request_transform(
    request_kwargs: dict, *, fields: tuple[str, ...], escape_hatch_env: str, required_empty: str | None = None
) -> dict:
    """Move ``fields`` into ``extra_body`` so the SDK transform never walks them.

    ``escape_hatch_env`` restores the typed path. ``required_empty`` names a field the SDK
    declares required (``@required_args``): it must stay present, so it is passed as an empty
    container of its own type rather than removed.
    """
    if _env_flag(escape_hatch_env):
        return request_kwargs
    moved = {
        f: request_kwargs[f] for f in fields
        if isinstance(request_kwargs.get(f), (dict, list)) and _is_plain_json_data(request_kwargs[f])
    }
    if not moved:
        return request_kwargs
    bypassed = {key: value for key, value in request_kwargs.items() if key not in moved}
    if required_empty is not None and required_empty in moved:
        # ``@required_args`` is checked before the transform, so the parameter has to be there.
        bypassed[required_empty] = type(moved[required_empty])()
    extra_body = bypassed.get("extra_body")
    merged = dict(extra_body) if isinstance(extra_body, dict) else {}
    # An explicit caller-provided extra_body entry keeps precedence (SDK post-transform merge).
    bypassed["extra_body"] = {**merged, **{f: v for f, v in moved.items() if f not in merged}}
    return bypassed


def is_openai_sdk_completions(client: Any) -> bool:
    """True only for a real ``openai`` SDK chat-completions surface.

    The MoA facade and the suite's stand-in clients expose the same attribute path but do no
    transform, and moving their payload into ``extra_body`` would silently drop it.
    """
    completions = getattr(getattr(client, "chat", None), "completions", None)
    if completions is None:
        return False
    return type(completions).__module__.startswith("openai.")


def bypass_chat_sdk_request_transform(request_kwargs: dict, client: Any) -> dict:
    """``bypass_sdk_request_transform`` for Chat Completions; a no-op off the real OpenAI SDK.

    ``messages`` is ``@required_args`` on ``chat.completions.create``, so it rides along as an
    empty list while its real value travels in ``extra_body``.
    """
    if not is_openai_sdk_completions(client):
        return request_kwargs
    return bypass_sdk_request_transform(
        request_kwargs,
        fields=CHAT_COMPLETIONS_BYPASS_FIELDS,
        escape_hatch_env="HERMES_CHAT_SDK_TRANSFORM",
        required_empty="messages",
    )
