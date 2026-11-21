"""Opt-in instruction-file refresh, called once under the session's turn lease.

Receipts live in the persisted prompt so recreating an agent cannot resurrect an
older snapshot. No watcher mutates an agent while a model/tool call is running.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_RECEIPT = re.compile(r"\n<!-- hermes-instructions:([0-9a-f]{64}) -->\s*$")


def _snapshot(agent: Any, paths: list[str]) -> tuple[str, str]:
    from agent import prompt_builder as pb
    from agent.system_prompt import _agent_home, _context_files_part

    length = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    length = length if isinstance(length, int) and length > 0 else None
    wants_soul = agent.load_soul_identity or not agent.skip_context_files
    soul = pb.load_soul_md(length, home_override=_agent_home(agent)) if wants_soul else None
    project = _context_files_part(agent, length, bool(soul))
    extra = []
    if not agent.skip_context_files:
        for raw in paths:
            path = Path(raw).expanduser().resolve()
            # Explicit operator-owned sources are required: a failed read must not
            # silently remove a standing instruction or advance its receipt.
            content = pb._read_text_with_timeout(path)
            if content is None:
                raise TimeoutError(f"Instruction file read timed out: {path}")
            extra.append(pb._context_section(content, str(path), str(path), path, length))
    supplement = "\n\n".join(extra)
    encoded = json.dumps([str(_agent_home(agent)), soul, project, supplement], ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest(), supplement


def refresh_instructions(agent: Any, system_message: str | None = None) -> bool:
    """Refresh changed instructions BEFORE request assembly; unchanged prompts stay byte-identical.

    The caller owns the session turn lease. Configuration is profile-scoped and
    read on each turn so enabling this also migrates legacy stored prompts once.
    Histories, pinned toolsets, clients, workers and session IDs are untouched.
    """
    from hermes_cli.config import load_config_readonly
    from agent.safe_worker_policy import safe_worker_enabled

    config = load_config_readonly().get("instruction_refresh", {})
    if not isinstance(config, dict) or config.get("enabled") is not True or safe_worker_enabled():
        return False
    paths = config.get("paths", [])
    if not isinstance(paths, list) or any(not isinstance(p, str) or not p.strip() for p in paths):
        raise ValueError("instruction_refresh.paths must be a list of non-empty file paths")
    digest, supplement = _snapshot(agent, paths)
    old = agent._cached_system_prompt or ""
    receipt = _RECEIPT.search(old)
    if receipt and receipt.group(1) == digest:
        return False

    # Build with the existing renderer, including its context precedence, scanning,
    # caps and profile isolation. Do not invoke session-start hooks or reset tools.
    prompt = agent._build_system_prompt(system_message)
    if supplement:
        prompt += "\n\n# Additional operator instruction files\n\n" + supplement
    prompt += f"\n<!-- hermes-instructions:{digest} -->"
    agent._cached_system_prompt = prompt
    if agent._session_db:
        agent._session_db.update_system_prompt(agent.session_id, prompt)
    logger.info("Instruction files refreshed at turn boundary (session=%s)", agent.session_id)
    return True
