"""Adapter-side runtime shim for resuming a turn from a tool result.

Hermes' ``run_conversation`` always appends a synthetic user turn at the start
of a turn (``turn_context.build_turn_context``). On an AG-UI *resume* — where a
client-side tool has just returned its result and there is no new user message
— that trailing user turn is wrong: the model should continue directly from the
tool result, and OpenAI-style backends (and aimock's fixture matcher) require
the request to *end* with the tool message.

Rather than change Hermes core, this installs a narrow wrapper around
``build_turn_context`` that — ONLY while the ``_RESUME`` context flag is set
(i.e. only for AG-UI resume runs) — drops a trailing ``user`` turn that
immediately follows a ``tool`` message. With the flag off (the default, and
every non-AG-UI code path) the wrapper is a pure pass-through, so core behavior
is unchanged.

This is a deliberate, fully-adapter-side workaround. The clean alternative is a
~15-line gated ``continue_from_history`` flag in Hermes core; this shim keeps
core untouched at the cost of a runtime monkeypatch confined to this package.
"""

from __future__ import annotations

import contextvars
import logging

logger = logging.getLogger(__name__)

_RESUME: contextvars.ContextVar[bool] = contextvars.ContextVar("agui_resume", default=False)
_installed = False


def set_resume(flag: bool) -> contextvars.Token:
    return _RESUME.set(flag)


def reset_resume(token: contextvars.Token) -> None:
    _RESUME.reset(token)


def install() -> None:
    """Idempotently wrap ``conversation_loop.build_turn_context``."""
    global _installed
    if _installed:
        return
    import agent.conversation_loop as cl

    original = cl.build_turn_context

    def _wrapped(agent, *args, **kwargs):
        ctx = original(agent, *args, **kwargs)
        if not _RESUME.get():
            return ctx
        try:
            msgs = ctx.messages
            if (
                len(msgs) >= 2
                and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "user"
                and isinstance(msgs[-2], dict) and msgs[-2].get("role") == "tool"
            ):
                msgs.pop()
                idx = max(
                    (i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "user"),
                    default=-1,
                )
                ctx.current_turn_user_idx = idx
                try:
                    agent._persist_user_message_idx = idx
                except Exception:
                    pass
        except Exception:
            logger.debug("AG-UI resume shim failed; leaving turn context unmodified", exc_info=True)
        return ctx

    cl.build_turn_context = _wrapped
    _installed = True
