"""Profile-scoped user-message overrides, snapshotted before each review fork."""

import logging
import stat
from pathlib import Path

from agent.system_prompt import _agent_home
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

logger = logging.getLogger(__name__)
_FILE_MAX_BYTES = 64 * 1024
_KINDS = {"_MEMORY_REVIEW_PROMPT": "memory", "_SKILL_REVIEW_PROMPT": "skill",
          "_COMBINED_REVIEW_PROMPT": "combined"}


def _file_prompt(raw_path, home):
    if not isinstance(raw_path, str):
        raise ValueError("expected a path string")
    path = Path(raw_path.strip()).expanduser()
    if not path.is_absolute():
        path = home / path
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > _FILE_MAX_BYTES:
        raise ValueError("expected a regular UTF-8 file of at most 64 KiB")
    with path.open("rb") as stream:
        text = stream.read(_FILE_MAX_BYTES + 1)
    if len(text) > _FILE_MAX_BYTES:
        raise ValueError("file exceeds 64 KiB")
    prompt = text.decode("utf-8").strip()
    if not prompt:
        raise ValueError('empty file; use inline "" to disable automatic reviews')
    return prompt


def _configured_prompt(config, kind, home, default, explicit):
    inline = config.get(kind)
    if inline is not None:
        if not isinstance(inline, str):
            raise ValueError("expected an inline string")
        if inline == "":
            return default if explicit else None
        if not inline.strip():
            raise ValueError('whitespace-only inline prompt; use "" to disable')
        return inline.strip()
    raw_path = config.get(f"{kind}_file")
    if raw_path is None or (isinstance(raw_path, str) and not raw_path.strip()):
        return default
    return _file_prompt(raw_path, home)


def resolve_review_prompt(agent, name, default, explicit=False):
    """Agent override → inline → file → default; invalid auto overrides skip."""
    override = getattr(agent, name, None)
    # AIAgent inherits the constants; inherited defaults must not shadow config.
    if isinstance(override, str) and override and override != default:
        return override
    home = _agent_home(agent) or get_hermes_home()
    token = set_hermes_home_override(home)
    try:
        from hermes_cli.config import load_config_readonly

        agent_config = load_config_readonly().get("agent", {})
        config = agent_config.get("review_prompts") if isinstance(agent_config, dict) else None
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("Background review config unreadable (%s); using default", type(exc).__name__)
        return default
    finally:
        reset_hermes_home_override(token)
    if not isinstance(config, dict):
        return default
    kind = _KINDS[name]
    try:
        return _configured_prompt(config, kind, home, default, explicit)
    except (OSError, ValueError) as exc:
        logger.warning("Invalid agent.review_prompts.%s override (%s); %s. Fix or remove the override",
                       kind, type(exc).__name__, "using default for explicit review" if explicit else "skipping review")
        return default if explicit else None
