import logging
import os
import sys

# Hermes loads directory plugins by file location WITHOUT adding the plugin
# dir to sys.path, so absolute `from subagent_handles.* import` would fail at
# runtime (it only works under pytest, where conftest inserts this dir). Make
# the internal `subagent_handles` package importable from both contexts.
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from subagent_handles.registry import SubagentHandle, registry
from subagent_handles.persister import default_persist_root, SessionPersister

logger = logging.getLogger(__name__)


def _get_persister():
    return SessionPersister(default_persist_root())


def _on_subagent_start(**kwargs: object) -> None:
    child_subagent_id = kwargs.get("child_subagent_id")
    child_session_id = kwargs.get("child_session_id")
    child_goal = kwargs.get("child_goal")
    parent_subagent_id = kwargs.get("parent_subagent_id")

    child_role = kwargs.get("child_role")

    if not child_subagent_id or not child_session_id or child_goal is None:
        logger.debug("subagent_start missing required kwargs, skipping")
        return

    try:
        handle = SubagentHandle(
            subagent_id=str(child_subagent_id),
            session_id=str(child_session_id),
            goal=str(child_goal),
            parent_subagent_id=str(parent_subagent_id) if parent_subagent_id else None,
            role=str(child_role) if child_role else "",
        )
        registry.register(handle)
        try:
            _get_persister().checkpoint(handle)
        except Exception:
            logger.debug("subagent_start checkpoint failed", exc_info=True)
    except ValueError:
        pass
    except Exception:
        logger.debug("subagent_start registry registration failed", exc_info=True)


def _on_subagent_stop(**kwargs: object) -> None:
    child_session_id = kwargs.get("child_session_id")
    if not child_session_id:
        logger.debug("subagent_stop missing child_session_id, skipping")
        return

    try:
        target = str(child_session_id)
        for handle in registry:
            if handle.session_id == target:
                registry.set_state(handle.subagent_id, "done")
                try:
                    _get_persister().checkpoint(handle)
                except Exception:
                    logger.debug("subagent_stop checkpoint failed", exc_info=True)
                break
    except Exception:
        logger.debug("subagent_stop registry update failed", exc_info=True)


def register(ctx) -> None:
    ctx.register_hook("subagent_start", _on_subagent_start)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    try:
        from subagent_handles.status import register_tools as _register_tools

        _register_tools(ctx)
    except Exception:
        logger.debug("subagent_handles tool registration failed", exc_info=True)

    try:
        _get_persister().restore(registry)
    except Exception:
        logger.debug("subagent_start restore from disk failed", exc_info=True)
