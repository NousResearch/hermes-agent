"""Projection of published MCP skill catalogs into the model-facing skills index.

Only routing information crosses into the prompt: the qualified load key and the
description. Frontmatter, manifests and file lists stay in the registry — they are
the host's evidence for validating a read later, not content the model should be
reasoning about, and injecting them would spend the context window on facts that
change the routing decision not at all.

A description is third-party text landing inside a prompt block the harness opened,
so everything a server publishes passes two checks before it is rendered:

* **Scanned.** The same ``context``-scope patterns that gate SOUL.md/AGENTS.md
  (``agent.prompt_builder._scan_context_content``) run over the text that will
  actually appear, and a hit drops the description rather than the row: the skill
  stays routable by name, and the payload never reaches the system prompt.
* **Neutralized.** Wrapper tokens are defanged even when nothing matched. A
  description shaped ``</available_skills> … <available_skills>`` closes the real
  block early with no injection keyword at all, so the keyword scan cannot see it —
  the prompt is plain text to the model, and only the literal token has to match
  (#8884, arXiv 2609.01222 vector B-1). Same shape main uses for tool results
  (``agent/tool_dispatch_helpers.py::_neutralize_delimiters``).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

MAIN_FILE_NAME = "SKILL.md"

# Wrapper tokens the prompt treats as its own structure. A description that contains one
# can close or reopen a block the builder owns, so it is defanged before rendering.
# Case-insensitive so a differently-cased spelling cannot slip past (#56172's rule for
# ``untrusted_tool_result``). Hyphens and underscores both count as separators.
_WRAPPER_TOKEN_RE = re.compile(
    r"available_skills|memory[-_]context|untrusted_tool_result|missing_context|skill_pruned",
    re.IGNORECASE,
)


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

    The cache token includes the scope, server, connection generation and revision:
    a reconnect may restart revisions at one, but must never reuse the old index.

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
    token: list[tuple[str, str, int, int]] = []
    for snapshot in sorted(snapshots, key=lambda s: s.origin_id):
        if not snapshot.is_fresh():
            # Past its TTL the metadata still routes (the rows below are served), but the
            # host stops claiming it is current — so ask for a refresh on the way past.
            # Debounced inside the manager; a prompt build never waits for the result.
            manager.request_refresh(snapshot.origin_id, scope_id, reason="stale-metadata")
        token.append((snapshot.scope_id, snapshot.origin_id, snapshot.connection_generation,
                      snapshot.snapshot_revision))
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
            rows.append((f"mcp/{snapshot.origin_id}", key,
                         index_description(entry.description, key=key)))
    return rows, tuple(token)


def index_description(description: Any, *, key: str = "") -> str:
    """*description* as the index may render it: fitted to the shared prompt budget, scanned,
    and with wrapper tokens defanged. ``""`` when the text is unfit for the prompt at all.

    Order matters. Clipping first means the scan judges the text the model would actually
    see — a hostile tail past the budget never reaches it, so a legitimate routing
    description is not thrown away for characters that would have been cut.
    """
    from agent.skill_utils import clip_skill_description
    from tools.threat_patterns import scan_for_threats

    text = clip_skill_description(" ".join(str(description).split()))
    if not text:
        return ""
    findings = scan_for_threats(text, scope="context")
    if findings:
        logger.warning("MCP skill %s: description withheld from the skills index (%s)",
                       key or "<unnamed>", ", ".join(findings))
        return ""
    return _neutralize_wrapper_tokens(text)


def _neutralize_wrapper_tokens(text: str) -> str:
    """Swap a wrapper token's separator so it stays readable but no longer matches the
    token the builder opened its block with."""
    def _defang(match: re.Match[str]) -> str:
        token = match.group(0)
        return token.replace("_", "-") if "_" in token else token.replace("-", "_")

    return _WRAPPER_TOKEN_RE.sub(_defang, text)
