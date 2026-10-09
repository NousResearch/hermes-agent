"""Load-time priority ordering for credential-pool entries."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential


_ANTHROPIC_SOURCE_RANK = {
    "env:ANTHROPIC_TOKEN": 0,
    "env:CLAUDE_CODE_OAUTH_TOKEN": 1,
    "hermes_pkce": 2,
    "claude_code": 3,
    "env:ANTHROPIC_API_KEY": 4,
}


def _normalize_pool_priorities(provider: str, entries: list[PooledCredential]) -> bool:
    if provider != "anthropic":
        return False
    from agent.credential_pool import _is_manual_source

    manual_entries = sorted(
        (entry for entry in entries if _is_manual_source(entry.source)),
        key=lambda entry: entry.priority,
    )
    seeded_entries = sorted(
        (entry for entry in entries if not _is_manual_source(entry.source)),
        key=lambda entry: (
            _ANTHROPIC_SOURCE_RANK.get(entry.source, len(_ANTHROPIC_SOURCE_RANK)),
            entry.priority,
            entry.label,
        ),
    )
    id_to_idx = {entry.id: idx for idx, entry in enumerate(entries)}
    changed = False
    for new_priority, entry in enumerate([*manual_entries, *seeded_entries]):
        if entry.priority != new_priority:
            entries[id_to_idx[entry.id]] = replace(entry, priority=new_priority)
            changed = True
    return changed
