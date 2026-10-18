"""``@`` references of one turn: expanded once, in the session's workspace, on the text the model reads.

The inbound prepare step, the classic CLI's API-only text (``cli_turn_v1.api_text``: voice prefix,
model-switch and skills notes) and managed workers (``--safe-mode`` / ``--ignore-user-config``,
which never pass the inbound step) all expand here. A local session's refusal is the turn's reply:
its adapter delivers through settlement, so a side ``send`` would never reach the viewer.
"""
import logging
import os
from contextlib import suppress
from functools import partial

from gateway.config import Platform

logger = logging.getLogger(__name__)


async def expand_references(text, *, cwd, context_length, plugins=True, source_slices=None):
    """``(text, refusal)``: the expanded text, or the refusal when the injection was refused. A failed
    expansion leaves the text as typed."""
    from agent.i18n import t
    try:
        from agent.context_references import preprocess_context_references_async
        expand = partial(preprocess_context_references_async, cwd=cwd, context_length=context_length, allowed_root=cwd)
        result = await (expand(text) if plugins else expand(text, plugins=False))
    except Exception as exc:
        logger.warning("@ context reference expansion failed: %s", exc, exc_info=True)
        return text, None
    if result.blocked:
        return text, "\n".join(result.warnings) or t("gateway.notify.context_injection_refused")
    if source_slices is not None:
        source_slices.extend(result.source_slices)
    return (result.message if result.expanded else text), None


def _inbound_cwd(runner, source):
    from gateway.session_policy import policy_for_source
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        terminal_env = os.environ.get
    # A local session's references resolve in ITS workspace: this runs before the turn binds
    # the policy scope, so TERMINAL_CWD is still the daemon's own.
    if (policy := policy_for_source(runner, source)) is not None:
        return policy.cwd
    return terminal_env("TERMINAL_CWD", os.path.expanduser("~"))


async def inbound_references(runner, source, session_key, text, *, source_slices=None):
    """``expand_references`` for an inbound turn: a local session's policy cwd, else the terminal cwd."""
    try:
        cwd = _inbound_cwd(runner, source)
        context_length = await runner._inbound_model_context_length(source, session_key)
    except Exception as exc:
        logger.warning("@ context reference expansion failed: %s", exc, exc_info=True)
        return text, None
    return await expand_references(
        text, cwd=cwd, context_length=context_length, source_slices=source_slices,
    )


async def turn_references(runner, event, source, session_key, text):
    """An inbound turn's text with the references of the text the model reads expanded; None when
    refused. The classic CLI's committed ``api_text`` replaces ``text`` at the turn
    (``TurnRunner._prepare_turn_message``), so ITS references are expanded and rebound for this
    execution, and ``text`` stays the stored user row as typed. A refused local turn leaves its
    refusal on ``event`` for the prepare step to answer with."""
    from gateway.session_cli_turn import cli_turn, rebind_api_text
    api_text = cli_turn(event.message_id).get('api_text')
    target = text if api_text is None else api_text
    if "@" not in target:
        return text
    source_slices = getattr(event, "_gateway_source_slices", None)
    if source.platform == Platform.LOCAL:
        expanded, refusal = await inbound_references(
            runner, source, session_key, target, source_slices=source_slices,
        )
        if refusal is not None:
            event._gateway_turn_refusal = refusal
            return None
    elif (expanded := await runner._expand_inbound_context_references(
            source, session_key, target, source_slices=source_slices)) is None:
        return None
    if api_text is None:
        return expanded
    rebind_api_text(event.message_id, expanded)
    return text


async def managed_turn_text(authority, row, policy, *, source_slices=None):
    """``(text, refusal)`` for a managed worker's admitted text, expanded against the session's
    workspace. Built-in kinds only for a bypass session, which loads no plugins; the context length
    is the frozen policy's, never the profile file a bypass session must not read. ``None`` text means
    the admitted text as committed."""
    text = (row.get('payload') or {}).get('text')
    if not isinstance(text, str) or "@" not in text:
        return None, None
    try:
        from agent.model_metadata import get_model_context_length_async
        model_config = policy.config(authority).get('model') or {}
        pinned = model_config.get('context_length') if isinstance(model_config, dict) else None
        with suppress(TypeError, ValueError):
            pinned = int(pinned) if pinned is not None else None
        context_length = await get_model_context_length_async(
            policy.current_model(authority) or '', base_url=policy.base_url or '', api_key='',
            provider=policy.provider or '', config_context_length=pinned if isinstance(pinned, int) else None)
    except Exception as exc:
        logger.warning("@ context reference expansion failed: %s", exc, exc_info=True)
        return text, None
    return await expand_references(text, cwd=policy.cwd, context_length=context_length,
                                   plugins=not policy.ignore_user_config, source_slices=source_slices)


def refused_turn_result(row, refusal):
    """A turn that never ran because its references were refused: the refusal is its answer, and a
    finite (``chat -q`` / ``-z``) prompt fails with it, as ``execute_admission`` settles a notice."""
    if row['payload'].get('finite') is True:
        return {'final_response': '', 'messages': [], 'failed': True, 'completed': False, 'error': refusal}
    return {'final_response': refusal, 'messages': []}
