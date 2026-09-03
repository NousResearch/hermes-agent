"""Route bulk request payloads around the OpenAI SDK's request transform (#93650).

Both the Responses API (``responses.create``) and chat-completions
(``chat.completions.create``) re-walk the entire request body against their
TypedDict/union param graph client-side, before any byte leaves the process.
That walk holds the GIL, and #93650 documents it wedging for 12+ hours on a
~1.4 MB conversation — starving every other thread, including the TTFB and
stale-call watchdogs whose whole job is to rescue this exact call. Because the
hang is client-side and pre-network, no socket kill can unblock it.

Hermes assembles these payloads from JSON round-trips, so they are already in
wire format and the walk has nothing to convert. The SDK merges ``extra_body``
into the JSON body *after* the transform
(``_base_client._build_request``), so moving the already-wire-format bulk
fields there skips the walk and produces a byte-identical request.

This module is the shared home for the helpers; ``agent.codex_runtime``
re-exports them so existing import paths keep working.
"""

from __future__ import annotations

import os
from typing import Any

# Bulk request fields that carry the conversation payload, per API family.
# Everything else in a request is scalar configuration the SDK transform
# handles in microseconds.
RESPONSES_BYPASS_FIELDS = ("input", "tools")
CHAT_COMPLETIONS_BYPASS_FIELDS = ("messages", "tools")


def _is_plain_json_data(value: Any) -> bool:
    """True when ``value`` is composed purely of JSON wire types.

    The SDK's request transform exists to convert typed params (TypedDict
    key aliases, pydantic models, ``PropertyInfo`` formats) into wire
    format.  Hermes assembles these payloads from JSON round-trips, so they
    are already wire format — but that is only provable when every node is
    a plain JSON type.  Anything else must keep the typed SDK path.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_plain_json_data(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_is_plain_json_data(item) for item in value)
    return False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def bypass_sdk_request_transform(
    request_kwargs: dict,
    *,
    fields: tuple = RESPONSES_BYPASS_FIELDS,
    escape_hatch_env: str = "HERMES_CODEX_SDK_TRANSFORM",
    required_empty: tuple = (),
) -> dict:
    """Move wire-format bulk ``fields`` into ``extra_body``.

    Returns ``request_kwargs`` unchanged when there is nothing safe to move,
    so a caller can always use the result unconditionally. Fields holding
    anything that is not plain JSON data (pydantic models, generators) stay
    on the typed path, which still needs the transform.

    ``required_empty`` names fields the SDK declares ``@required_args`` and
    would reject as missing: those are kept in the typed kwargs as an empty
    list, and the ``extra_body`` copy overwrites them in the JSON body. Set
    ``escape_hatch_env`` to restore the pre-fix behaviour.
    """
    if _env_flag(escape_hatch_env):
        return request_kwargs

    moved = {
        field: request_kwargs[field]
        for field in fields
        if isinstance(request_kwargs.get(field), (dict, list))
        and _is_plain_json_data(request_kwargs[field])
    }
    if not moved:
        return request_kwargs

    bypassed = {
        key: value for key, value in request_kwargs.items() if key not in moved
    }
    for field in required_empty:
        if field in moved:
            # The SDK rejects the call outright if a @required_args parameter
            # is absent; an empty list satisfies the signature and the
            # extra_body entry replaces it in the body the server sees.
            bypassed[field] = []
    extra_body = bypassed.get("extra_body")
    merged = dict(extra_body) if isinstance(extra_body, dict) else {}
    for field, value in moved.items():
        # An explicit caller-provided extra_body entry keeps precedence,
        # matching what the SDK's post-transform merge would have done.
        merged.setdefault(field, value)
    bypassed["extra_body"] = merged
    return bypassed


def is_openai_sdk_completions(client: Any) -> bool:
    """True when ``client.chat.completions`` is the real OpenAI SDK object.

    Only the SDK performs the transform this bypass exists to skip, and only
    the SDK merges ``extra_body`` into the body afterwards. Hermes also drives
    chat-completions-shaped facades that are NOT the SDK — the in-process MoA
    aggregator (``agent/moa_loop.py``) most importantly, plus the stand-ins
    the test suite injects — and handing those an ``extra_body`` they never
    merge would silently drop the conversation. Gate on the real thing.
    """
    completions = getattr(getattr(client, "chat", None), "completions", None)
    if completions is None:
        return False
    return type(completions).__module__.startswith("openai.")


def bypass_chat_sdk_request_transform(request_kwargs: dict, client: Any) -> dict:
    """``bypass_sdk_request_transform`` for chat-completions, SDK-gated."""
    if not is_openai_sdk_completions(client):
        return request_kwargs
    return bypass_sdk_request_transform(
        request_kwargs,
        fields=CHAT_COMPLETIONS_BYPASS_FIELDS,
        escape_hatch_env="HERMES_CHAT_SDK_TRANSFORM",
        # ``messages`` is @required_args on chat.completions.create.
        required_empty=("messages",),
    )
