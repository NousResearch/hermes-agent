"""Defensive guard for openai SDK ``parse_response`` on null ``response.output``.

The ChatGPT consumer Codex backend (``chatgpt.com/backend-api/codex``)
intermittently emits ``response.completed`` events whose ``response.output``
is ``null`` instead of an empty list — observed on ``gpt-5.5`` in May 2026.
The high-level ``responses.stream(...)`` / ``responses.parse(...)`` helpers
in ``openai==2.24.0`` then invoke
``openai.lib._parsing._responses.parse_response``, whose first line iterates
``for output in response.output:`` without a ``None`` guard, raising
``TypeError: 'NoneType' object is not iterable`` mid-stream.

Hermes's primary codex path (``agent/codex_runtime.py``) is already
structurally immune — it consumes the raw event stream from
``responses.create(stream=True)`` and never reads
``response.completed.response.output`` for content reconstruction. This
shim covers the remaining callers (any provider profile that still uses
the high-level helpers, plus third-party plugins) and any future openai
version that re-introduces the unguarded iteration.

Upstream PRs tracking the fix: openai/openai-python#3322, #3316, #3286.
This shim becomes a no-op once a Hermes release pins an openai version
that ships the upstream guard, since the wrapped function will simply
re-enter a body that already handles ``response.output is None``.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_GUARD_FLAG = "_hermes_or_empty_guard"


def _install() -> bool:
    try:
        from openai.lib._parsing import _responses as _src
    except ImportError:
        return False

    original = getattr(_src, "parse_response", None)
    if original is None:
        return False
    if getattr(original, _GUARD_FLAG, False):
        return False

    def parse_response(*, text_format, input_tools, response):
        if response is not None and getattr(response, "output", None) is None:
            try:
                response.output = []
            except Exception:
                try:
                    object.__setattr__(response, "output", [])
                except Exception:
                    pass
        return original(
            text_format=text_format,
            input_tools=input_tools,
            response=response,
        )

    setattr(parse_response, _GUARD_FLAG, True)
    parse_response.__wrapped__ = original  # type: ignore[attr-defined]
    parse_response.__doc__ = (
        "Hermes wrapper that coerces ``response.output`` from ``None`` to "
        "``[]`` before delegating to the original openai parse_response. "
        "See agent/_openai_compat.py."
    )

    _src.parse_response = parse_response

    for modname in (
        "openai.lib.streaming.responses._responses",
        "openai.resources.responses.responses",
    ):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        if getattr(mod, "parse_response", None) is original:
            mod.parse_response = parse_response

    logger.debug("openai parse_response guard installed (hermes _openai_compat)")
    return True


_INSTALLED = _install()
