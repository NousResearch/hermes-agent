"""Gateway sender context for opt-in plugin slash commands."""

import inspect

from gateway.session import SessionSource


def invoke_plugin_command_handler(handler, raw_args: str, source: SessionSource):
    """Preserve the PR's original required-parameter / variadic context contract.

    A variadic handler receives context, as does a handler with at least two
    required positional parameters. Inspect before calling so a handler's TypeError
    cannot cause a second invocation.
    """
    try:
        params = list(inspect.signature(handler).parameters.values())
        positional = [
            p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
            and p.default is p.empty
        ]
        has_varargs = any(p.kind is p.VAR_POSITIONAL for p in params)
        wants_context = has_varargs or len(positional) >= 2
    except (TypeError, ValueError):
        wants_context = False

    if not wants_context:
        return handler(raw_args)

    return handler(raw_args, {
        "user_id": getattr(source, "user_id", None),
        "user_name": getattr(source, "user_name", None),
        "chat_id": getattr(source, "chat_id", None),
        "chat_type": getattr(source, "chat_type", None),
        "platform": getattr(getattr(source, "platform", None), "value", None),
    })
