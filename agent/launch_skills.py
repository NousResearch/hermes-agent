"""Persist only the rendered launch skills, independently of the rest of the prompt."""

import json
import logging
import re

logger = logging.getLogger(__name__)
_HEADER = re.compile(r'<hermes_launch_skills chars="(\d+)">\n')
_END = "\n</hermes_launch_skills>"
_LOCATOR = re.compile(r'\n<!-- hermes-launch-skills-offset:([0-9]{1,12}) -->\Z')

_METADATA = "<!-- hermes-launch-skills-v1 "


def annotate_launch_skills(prompt: str, names: list[str], paths: list[str]) -> str:
    """Carry builder-owned canonical names without interpreting any skill prose."""
    if not prompt:
        return prompt
    metadata = json.dumps({"names": names, "paths": paths, "chars": len(prompt)}, separators=(",", ":"))
    return f"{_METADATA}{metadata} -->\n{prompt}"


def _preloaded_metadata(prompt: str | None) -> dict | None:
    """Read only a complete constructor-owned envelope; legacy raw prompts have none."""
    if not isinstance(prompt, str) or not prompt.startswith(_METADATA):
        return None
    header, separator, body = prompt.partition("\n")
    if not separator or not header.endswith(" -->"):
        return None
    try:
        metadata = json.loads(header[len(_METADATA):-4])
    except (ValueError, TypeError):
        return None
    if (not isinstance(metadata, dict) or set(metadata) != {"names", "paths", "chars"}
            or type(metadata["chars"]) is not int or metadata["chars"] != len(body)
            or not isinstance(metadata["names"], list) or not isinstance(metadata["paths"], list)
            or not all(isinstance(name, str) and name for name in metadata["names"])
            or not all(isinstance(path, str) and path for path in metadata["paths"])):
        return None
    return metadata


def preloaded_skill_names(prompt: str | None) -> list[str] | None:
    metadata = _preloaded_metadata(prompt)
    return metadata["names"] if metadata is not None else None


def preloaded_skill_paths(prompt: str | None) -> set[str]:
    metadata = _preloaded_metadata(prompt)
    return set(metadata["paths"]) if metadata is not None else set()


def launch_skills_locator(offset: int) -> str:
    """Renderer-owned trailer records the frame's absolute character offset.

    It ends the renderer-owned identity paragraph; no-skill prompts need no record.
    """
    return f'<!-- hermes-launch-skills-offset:{offset} -->'


def frame_launch_skills(prompt: str | None) -> str | None:
    """Length framing preserves even skill text containing the closing delimiter."""
    if not isinstance(prompt, str) or not prompt:
        return None
    return f'<hermes_launch_skills chars="{len(prompt)}">\n{prompt}{_END}'


def persisted_launch_skills(stored_prompt: str | None) -> str | None:
    """Read only the renderer-owned frame, never examples elsewhere in the prompt."""
    if not isinstance(stored_prompt, str):
        return None
    from agent.surface_switch import split_runtime_boundary
    identity = split_runtime_boundary(stored_prompt)[0]
    locator = _LOCATOR.search(identity)
    if locator is None:
        return None
    offset = int(locator.group(1))
    match = _HEADER.match(stored_prompt, offset)
    if match is None:
        return None
    start = match.end()
    end = start + int(match.group(1))
    if end + len(_END) <= locator.start() and stored_prompt[end:end + len(_END)] == _END:
        return stored_prompt[start:end]
    return None


def restore_launch_skills(agent, stored_prompt: str | None = None) -> None:
    """Recover retained bytes without reloading files or overriding explicit launch skills.

    Review can run before the first resumed turn, so it also needs the DB path.
    Only the renderer's final locator identifies state; a frame found elsewhere
    could be a quoted example. Historical prompts without a locator are ambiguous.
    """
    explicit = getattr(agent, "preloaded_skills_prompt", None)
    if isinstance(explicit, str) and explicit:
        return
    if stored_prompt is None:
        stored_prompt = getattr(agent, "_cached_system_prompt", None)
        db = getattr(agent, "_session_db", None)
        if (not isinstance(stored_prompt, str) or not stored_prompt) and db is not None:
            try:
                row = db.get_session(agent.session_id)
                stored_prompt = row.get("system_prompt") if row else None
            except Exception:
                logger.warning("Unable to restore launch skills for %s", agent.session_id, exc_info=True)
    retained = persisted_launch_skills(stored_prompt)
    if retained is not None:
        agent.preloaded_skills_prompt = retained
