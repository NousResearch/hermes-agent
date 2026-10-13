"""Projection of published MCP skill catalogs into the model-facing skills index.

Only routing information crosses into the prompt: the qualified load key and the
description. Frontmatter, manifests and file lists stay in the registry — they are
the host's evidence for validating a read later, not content the model should be
reasoning about, and injecting them would spend the context window on facts that
change the routing decision not at all.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Remote descriptions are third-party text on a per-row budget in a system-prompt
# block that must stay byte-stable for the life of a conversation.
MAX_REMOTE_DESCRIPTION = 300
_TRUNCATION_SUFFIX = "…"
MAIN_FILE_NAME = "SKILL.md"


def locator_of(uri: str) -> str:
    """``skill://builtin/del/SKILL.md`` → ``builtin/del`` — the URI's own path.

    The key is built from this, never from the frontmatter ``name``. A skill's
    identity is its URI; the name is a label it chose, and two skills on one server
    are free to share one. A name-built key would collapse those two into a single
    entry: one row, one disable, one load — and the model told there is one skill
    where the server published two.

    The ``SKILL.md`` tail is dropped because it is in every skill's URI and so
    distinguishes nothing; what is left is what a person reads to tell two skills
    apart.
    """
    parts = urlsplit(uri if isinstance(uri, str) else "")
    path = parts.path[: -len(MAIN_FILE_NAME)] if parts.path.endswith("/" + MAIN_FILE_NAME) else parts.path
    return f"{parts.netloc}{path}".strip("/")


def qualified_name(server: str, uri: str) -> str:
    """The load key the index advertises, ``skill_view`` resolves, and the enable/disable
    list names this skill by.

    Double colon, deliberately not plugin's single-colon ``plugin:skill``: the two
    namespaces must stay distinguishable in one string, and a bare ``uri`` never
    silently means "the remote one" — the server label is always in front of it.
    """
    return f"{server}::{locator_of(uri)}"


def split_qualified(name: str) -> Optional[tuple[str, str]]:
    """``("skills", "builtin/del")`` for a well-formed qualified key, else None.

    Split on the FIRST ``::``: a locator may contain colons, while the server label
    (a host-assigned config key) cannot — and a single colon means the plugin
    namespace, which this never claims.
    """
    if not isinstance(name, str) or "::" not in name:
        return None
    server, locator = name.split("::", 1)
    if not server or not locator:
        return None
    return server, locator


def remote_entry_disabled(server: str, uri: str, disabled: Optional[set[str]] = None) -> bool:
    """Whether this remote skill is switched off by the profile's skills config.

    Matched on the qualified key only. ``skills.disabled`` is a list of *names*, and
    a remote skill's stable name is this key — the server label and the SKILL.md URI
    locate it, they are not a second authorization dimension. A bare local name is
    deliberately not matched: an entry written for a local skill must not silently
    switch off an unrelated remote skill that happens to share a name.
    """
    if disabled is None:
        from agent.skill_utils import get_disabled_skill_names
        disabled = get_disabled_skill_names()
    return qualified_name(server, uri) in disabled


def mcp_skill_rows(disabled: set[str], available_tools: Optional[set[str]],
                   available_toolsets: Optional[set[str]],
                   session_platform: Optional[str]) -> tuple[list[tuple[str, str, str]], tuple]:
    """``([(category, qualified_name, description)], cache_token)`` for every published remote skill.

    The cache token is the set of ``(server, snapshot_revision)`` pairs, so the
    rendered-index LRU is invalidated by a catalog republish — otherwise a session
    would keep serving the index built before the server's skills changed, which is
    the whole failure mode this feature exists to remove.

    Never raises: a prompt build must not fail because an optional remote catalog
    is in an odd state, and a missing catalog is simply no rows.
    """
    try:
        from agent.prompt_builder import _skill_should_show, extract_skill_conditions
        from agent.skill_utils import skill_matches_apps, skill_matches_environment, skill_matches_platform
        from tools.mcp_skills_manager import current_scope_id, get_manager
    except Exception:
        logger.debug("MCP skill rows unavailable: imports failed", exc_info=True)
        return [], ()

    manager = get_manager()
    scope_id = current_scope_id()
    try:
        snapshots = manager.published(scope_id)
    except Exception:
        logger.debug("MCP skill rows unavailable: catalog read failed", exc_info=True)
        return [], ()

    rows: list[tuple[str, str, str]] = []
    token: list[tuple[str, int]] = []
    for snapshot in sorted(snapshots, key=lambda s: s.origin_id):
        if not snapshot.is_fresh():
            # Past its TTL the metadata still routes (the rows below are served), but the
            # host stops claiming it is current — so ask for a refresh on the way past.
            # Debounced inside the manager; a prompt build never waits for the result.
            manager.request_refresh(snapshot.origin_id, scope_id, reason="stale-metadata")
        token.append((snapshot.origin_id, snapshot.snapshot_revision))
        for entry in sorted(snapshot.entries.values(),
                            key=lambda e: qualified_name(snapshot.origin_id, e.identity.locator)):
            key = qualified_name(snapshot.origin_id, entry.identity.locator)
            if remote_entry_disabled(snapshot.origin_id, entry.identity.locator, disabled):
                continue
            frontmatter: Any = entry.frontmatter
            try:
                if not (skill_matches_platform(frontmatter) and skill_matches_environment(frontmatter)
                        and skill_matches_apps(frontmatter)):
                    continue
                if not _skill_should_show(extract_skill_conditions(frontmatter), available_tools,
                                          available_toolsets, session_platform):
                    continue
            except Exception:
                logger.debug("Skipping remote skill %s: frontmatter gating failed", key, exc_info=True)
                continue
            rows.append((f"mcp/{snapshot.origin_id}", key, _trim(entry.description)))
    return rows, tuple(token)


def _trim(description: str) -> str:
    text = " ".join(str(description).split())
    if len(text) <= MAX_REMOTE_DESCRIPTION:
        return text
    return text[: MAX_REMOTE_DESCRIPTION - 1] + _TRUNCATION_SUFFIX
