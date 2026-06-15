"""Build per-session Claude Code sandboxes.

Each hermes session that uses the ``claude-code-acp`` provider needs a
self-contained cwd that Claude Code will read as its workspace. The sandbox
contains:

* ``CLAUDE.md`` — preamble + SOUL.md (agent identity) + memory context +
  toolbelt hint + platform context. Tool names referenced inline are
  rewritten to their MCP form (``mcp__hermes_tools__<name>``) so Claude
  Code can locate them on the bus.
* ``.claude/skills/<skill_name>/SKILL.md`` — one flattened directory per
  filtered hermes skill. Anthropic's skill convention is flat (one level
  under ``.claude/skills/``), so nested hermes skills are promoted by
  their frontmatter ``name:``.
* ``.mcp.json`` — mcpServers config pointing at ``hermes mcp tools-serve``
  (the hermes tool registry) and ``hermes mcp serve`` (messaging bridge).

The sandbox is idempotent: a manifest (``.hermes-sandbox.json``) caches
inputs' mtimes + sizes so unchanged sandboxes skip rebuild.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RUNTIME_SUBDIR = Path("runtime") / "claude-code"
SANDBOX_MANIFEST = ".hermes-sandbox.json"
DEFAULT_SANDBOX_MAX_AGE_DAYS = 7

_CLAUDE_MCP_TOOL_PREFIX = "mcp__hermes_tools__"

# Shared source of truth: the per-prompt client and the persistent CLAUDE.md
# sandbox both need to tell Claude Code the same things (how replies reach
# the user, where tools are, skill conventions, etc.). Keep that content in
# one place in :mod:`agent.claude_code_acp_client` and join it here.
from agent.claude_code_acp_client import CLAUDE_SYSTEM_PREAMBLE_LINES as _SHARED_PREAMBLE_LINES

_PREAMBLE = "\n\n".join(_SHARED_PREAMBLE_LINES)


@dataclass(frozen=True)
class SandboxComponents:
    """Component inclusion is fixed at design time — not re-derived at runtime.

    Defaults (all True) match the current primary-agent behavior: SOUL.md /
    memory / skills / .mcp.json / toolbelt / platform block / settings.local.json
    are all enabled. Each preset is named after its feature combination (not
    after the caller) and is derived from the three real AIAgent flags
    (``skip_context_files`` / ``load_soul_identity`` / ``skip_memory``).

    These flags are read by the inner helpers via the ``include=`` keyword
    argument — see :func:`_load_soul`, :func:`_build_memory_block`,
    :func:`_flatten_skills_into`, :func:`_write_mcp_json`.
    """

    include_soul: bool = True
    include_memory: bool = True
    include_skills: bool = True
    include_mcp: bool = True
    include_toolbelt: bool = True
    include_platform: bool = True
    include_settings: bool = True


SANDBOX_PRESETS: Dict[str, SandboxComponents] = {
    # Full primary sandbox — all components enabled.
    "primary": SandboxComponents(
        include_soul=True, include_memory=True, include_skills=True,
        include_mcp=True, include_toolbelt=True, include_platform=True,
        include_settings=True,
    ),
    # Primary minus memory block (background_review, cron-with-workdir).
    "primary_minus_memory": SandboxComponents(
        include_soul=True, include_memory=False, include_skills=True,
        include_mcp=True, include_toolbelt=True, include_platform=True,
        include_settings=True,
    ),
    # Subagent / curator / ignore_rules / feishu_comment — only preamble +
    # toolbelt + platform + settings + mcp (.mcp.json always retained:
    # subagents need to discover hermes tools). Skip SOUL/memory/skills.
    "minimal": SandboxComponents(
        include_soul=False, include_memory=False, include_skills=False,
        include_mcp=True, include_toolbelt=True, include_platform=True,
        include_settings=True,
    ),
}


def _resolve_preset(agent: Any) -> str:
    """Pure derivation: reuses the agent's three existing flags, no new attrs.

    Important: the original ``is_subagent`` / ``is_curator`` /
    ``is_background_review`` / ``is_cron`` symbols don't exist in the
    codebase (rg 0 hits) — they can't be used. Only the three real flags
    (``skip_context_files`` / ``load_soul_identity`` / ``skip_memory``)
    drive the lookup, so callsites stay unchanged.
    """
    skip_ctx = bool(getattr(agent, "skip_context_files", False))
    load_soul = bool(getattr(agent, "load_soul_identity", False))
    skip_mem = bool(getattr(agent, "skip_memory", False))

    if not skip_ctx and not skip_mem:
        return "primary"
    if not skip_ctx and skip_mem:
        return "primary_minus_memory"
    if skip_ctx and load_soul and skip_mem:
        # cron-no-workdir is functionally identical to primary_minus_memory
        # (same SandboxComponents fields), so we route both to it.
        return "primary_minus_memory"
    if skip_ctx and not load_soul and skip_mem:
        return "minimal"
    return "primary"  # fallback — unreachable in practice


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sandbox_root(hermes_home: Path) -> Path:
    """Return the base directory under which per-session sandboxes live."""
    return hermes_home / RUNTIME_SUBDIR


def session_sandbox_path(session_id: str, *, hermes_home: Path) -> Path:
    """Return the sandbox directory for a given session (may not exist yet)."""
    if not session_id or "/" in session_id or ".." in session_id:
        raise ValueError(f"Invalid session_id for sandbox: {session_id!r}")
    return sandbox_root(hermes_home) / session_id


def _manifest_is_current(manifest_path: Path, digest: str, sandbox: Path) -> bool:
    """Return True if the on-disk manifest matches *digest* and CLAUDE.md exists.

    Returns False (forcing a rebuild) on: missing manifest, unreadable manifest,
    digest mismatch, or missing CLAUDE.md. Narrow-except rather than blanket
    ``except Exception`` so a real bug (e.g. AttributeError from a stubbed
    filesystem) isn't silently swallowed.
    """
    try:
        prior = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        logger.debug("Sandbox manifest %s missing (first build)", manifest_path)
        return False
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug(
            "Sandbox manifest %s unreadable (%s: %s); rebuilding",
            manifest_path,
            type(exc).__name__,
            exc,
        )
        return False
    return bool(
        prior.get("digest") == digest and (sandbox / "CLAUDE.md").exists()
    )


def _write_settings_local(sandbox: Path, *, model: Optional[str] = None) -> None:
    """Pin sandbox-level settings that the ACP adapter must accept.

    Specifically: a valid ``permissions.defaultMode``. If the user's
    ``~/.claude/settings.json`` has an invalid value, Claude Code rejects
    ``session/new``. Sandbox-local settings take precedence over user
    settings in the ACP settings merge, so this unblocks boot unconditionally.

    Model is NOT written here — it is set via ``session/set_config_option``
    after ``session/new`` so that parallel subagents with different models
    don't overwrite each other through a shared sandbox file.
    """
    settings_local: Dict[str, Any] = {
        "permissions": {"defaultMode": "bypassPermissions"},
    }
    (sandbox / ".claude" / "settings.local.json").write_text(
        json.dumps(settings_local, indent=2),
        encoding="utf-8",
    )


def _write_mcp_json(
    sandbox: Path,
    *,
    include: bool = True,
    session_id: str,
    hermes_home: Path,
    platform: Optional[str],
) -> None:
    """Materialize ``.mcp.json`` pointing Claude Code at hermes's MCP servers."""
    if not include:
        return
    (sandbox / ".mcp.json").write_text(
        json.dumps(
            _build_mcp_config(
                session_id=session_id, hermes_home=hermes_home, platform=platform
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_sandbox_manifest(
    manifest_path: Path,
    *,
    digest: str,
    session_id: str,
    skill_count: int,
) -> None:
    """Stamp the manifest so the next invocation can short-circuit on cache hit."""
    manifest_path.write_text(
        json.dumps(
            {
                "digest": digest,
                "generated_at": time.time(),
                "session_id": session_id,
                "skill_count": skill_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_session_sandbox(
    session_id: str,
    agent: Any,
    *,
    hermes_home: Path,
    platform: Optional[str] = None,
    available_tools: Optional[set[str]] = None,
    available_toolsets: Optional[set[str]] = None,
    model: Optional[str] = None,
) -> Path:
    """Build (or rebuild if stale) a Claude Code sandbox for *session_id*.

    Resolves the *agent*'s three flags (``skip_context_files`` /
    ``load_soul_identity`` / ``skip_memory``) into a preset name via
    :func:`_resolve_preset`, looks up the corresponding
    :class:`SandboxComponents` in :data:`SANDBOX_PRESETS`, then serializes
    manifest digest + CLAUDE.md + .mcp.json + settings.local.json +
    flattened skills according to those components. Because the manifest
    digest folds the components in, switching presets for the same session
    automatically forces a rebuild.

    Returns the absolute path to the sandbox cwd.
    """
    sandbox = session_sandbox_path(session_id, hermes_home=hermes_home)
    sandbox.mkdir(parents=True, exist_ok=True)

    # 1. Resolve preset → components
    preset_name = _resolve_preset(agent)
    components = SANDBOX_PRESETS[preset_name]

    # 2. Manifest digest reflects components (preset change → digest change → force rebuild)
    manifest_path = sandbox / SANDBOX_MANIFEST
    inputs = _collect_manifest_inputs(
        hermes_home=hermes_home, platform=platform, model=model,
        components=components,
    )
    digest = _hash_inputs(inputs)

    # 3. Cache check
    if _manifest_is_current(manifest_path, digest, sandbox):
        logger.debug("Sandbox %s is up-to-date; skipping rebuild", sandbox)
        return sandbox

    # 4. Build CLAUDE.md composition parts (None / "" = skip)
    tool_names = _resolve_tool_names()
    toolbelt_hint = (
        _build_toolbelt_hint(tool_names) if components.include_toolbelt else ""
    )
    platform_block = (
        _build_platform_block(platform) if components.include_platform else None
    )
    soul_text = _load_soul(hermes_home, include=components.include_soul)
    memory_block = _build_memory_block(agent, include=components.include_memory)

    # 5. Skills: include=False → drop stale dir; include=True → flatten-write
    skills_out = sandbox / ".claude" / "skills"
    if skills_out.exists():
        shutil.rmtree(skills_out, ignore_errors=True)
    if components.include_skills:
        skills_out.mkdir(parents=True, exist_ok=True)
        flattened = _flatten_skills_into(
            include=True,
            hermes_home=hermes_home,
            target=skills_out,
            available_tools=available_tools,
            available_toolsets=available_toolsets,
        )
    else:
        flattened = 0

    # 6. Compose CLAUDE.md
    claude_md = _compose_claude_md(
        soul_text=soul_text,
        memory_block=memory_block,
        toolbelt_hint=toolbelt_hint,
        platform_block=platform_block,
        tool_names=tool_names,
    )
    (sandbox / "CLAUDE.md").write_text(claude_md, encoding="utf-8")

    # 7. settings.local.json (include_settings=False → drop stale)
    if components.include_settings:
        # _write_settings_local writes <sandbox>/.claude/settings.local.json
        # without mkdir-ing the parent. The old code always implicitly
        # created .claude via skills_out.mkdir, but include_skills=False
        # paths (e.g. minimal preset) skip that — so we mkdir here.
        (sandbox / ".claude").mkdir(parents=True, exist_ok=True)
        _write_settings_local(sandbox, model=model)
    else:
        stale_settings = sandbox / ".claude" / "settings.local.json"
        if stale_settings.exists():
            stale_settings.unlink()

    # 8. .mcp.json (include_mcp=False → drop stale; subagent keeps mcp always)
    if components.include_mcp:
        _write_mcp_json(
            sandbox,
            session_id=session_id,
            hermes_home=hermes_home,
            platform=platform,
            include=True,
        )
    else:
        stale_mcp = sandbox / ".mcp.json"
        if stale_mcp.exists():
            stale_mcp.unlink()

    # 9. Manifest stamp
    _write_sandbox_manifest(
        manifest_path,
        digest=digest,
        session_id=session_id,
        skill_count=flattened,
    )

    logger.info(
        "Built Claude Code sandbox %s (preset=%s, %d skills, %d tools)",
        sandbox, preset_name, flattened, len(tool_names),
    )
    return sandbox


def cleanup_session_sandbox(session_id: str, *, hermes_home: Path) -> bool:
    """Remove the sandbox directory for *session_id*. Returns True if removed."""
    try:
        sandbox = session_sandbox_path(session_id, hermes_home=hermes_home)
    except ValueError:
        return False
    if not sandbox.exists():
        return False
    shutil.rmtree(sandbox, ignore_errors=True)
    return True


def cleanup_stale_sandboxes(
    *, hermes_home: Path, max_age_days: int = DEFAULT_SANDBOX_MAX_AGE_DAYS
) -> int:
    """Remove sandboxes older than *max_age_days*. Returns count removed."""
    root = sandbox_root(hermes_home)
    if not root.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            manifest = child / SANDBOX_MANIFEST
            if manifest.exists():
                mtime = manifest.stat().st_mtime
            else:
                mtime = child.stat().st_mtime
        except Exception:
            continue
        if mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


DEFAULT_ACP_LOG_MAX_AGE_DAYS = 7


def cleanup_stale_acp_logs(
    *, hermes_home: Path, max_age_days: int = DEFAULT_ACP_LOG_MAX_AGE_DAYS
) -> int:
    """Delete ACP stderr logs older than *max_age_days*. Returns count removed.

    Logs are written by :meth:`ClaudeCodeACPClient._open_stderr_log_file` into
    ``<hermes_home>/logs/acp/``. They're useful for post-mortem diagnosis
    of subprocess crashes, but accumulate one file per ACP session —
    without pruning, a heavy user piles up thousands in a month.
    """
    log_dir = Path(hermes_home) / "logs" / "acp"
    if not log_dir.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for child in log_dir.iterdir():
        if not child.is_file():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                child.unlink()
                removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# CLAUDE.md composition
# ---------------------------------------------------------------------------


def _compose_claude_md(
    *,
    soul_text: Optional[str],
    memory_block: Optional[str],
    toolbelt_hint: str,
    platform_block: Optional[str],
    tool_names: List[str],
) -> str:
    parts: List[str] = [_PREAMBLE.strip()]
    if soul_text:
        parts.append("## Agent identity (SOUL.md)\n\n" + _rewrite_tool_names(soul_text, tool_names))
    if memory_block:
        parts.append(memory_block)
    parts.append(toolbelt_hint)
    if platform_block:
        parts.append(platform_block)
    return "\n\n".join(p.strip() for p in parts if p and p.strip()) + "\n"


def _rewrite_tool_names(text: str, tool_names: List[str]) -> str:
    """Prefix bare hermes tool names with ``mcp__hermes_tools__`` in *text*.

    Double-prefix guard: SOUL.md may reference tools either as bare names
    (``Store``) or as already-prefixed MCP identifiers
    (``mcp__hermes_tools__Store``). The lookbehind inside the substituter
    skips the latter so we never produce
    ``mcp__hermes_tools__mcp__hermes_tools__Store``. Removing this check
    breaks Claude Code silently: the double-prefixed name is valid syntax
    but refers to no registered tool, so the agent fails with "tool not
    found" after a full turn of confused retries. Keep the guard.
    """
    if not text or not tool_names:
        return text
    # Sort longest-first so overlapping names don't shadow each other.
    names = sorted((n for n in tool_names if n and n.isidentifier()), key=len, reverse=True)
    if not names:
        return text
    pattern = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9_])")
    def _sub(match: re.Match) -> str:
        name = match.group(1)
        # Skip if already prefixed — see docstring for why this matters.
        start = match.start()
        if start >= len(_CLAUDE_MCP_TOOL_PREFIX) and text[start - len(_CLAUDE_MCP_TOOL_PREFIX):start] == _CLAUDE_MCP_TOOL_PREFIX:
            return name
        return _CLAUDE_MCP_TOOL_PREFIX + name
    return pattern.sub(_sub, text)


def _build_toolbelt_hint(tool_names: List[str]) -> str:
    if not tool_names:
        return "## Tools\n\nNo hermes tools are currently available."
    lines = ["## Hermes toolbelt"]
    lines.append(
        "The following hermes tools are available via the `hermes_tools` MCP"
        " server. Call them using the `mcp__hermes_tools__<name>` prefix."
    )
    for name in tool_names:
        try:
            from tools.registry import registry

            entry = registry.get_entry(name)
            desc = ""
            if entry:
                desc = (entry.description or entry.schema.get("description", "")).strip().splitlines()
                desc = desc[0] if desc else ""
            if desc:
                lines.append(f"- `{_CLAUDE_MCP_TOOL_PREFIX}{name}` — {desc[:120]}")
            else:
                lines.append(f"- `{_CLAUDE_MCP_TOOL_PREFIX}{name}`")
        except Exception:
            lines.append(f"- `{_CLAUDE_MCP_TOOL_PREFIX}{name}`")
    return "\n".join(lines)


def _build_memory_block(agent: Any, *, include: bool = True) -> Optional[str]:
    """Build the memory-context block via the same code path the AIAgent uses."""
    if not include:
        return None
    try:
        from agent.memory_manager import build_memory_context_block
    except Exception:
        return None
    raw = _collect_raw_memory(agent)
    if not raw:
        return None
    return build_memory_context_block(raw)


def _collect_raw_memory(agent: Any) -> str:
    """Concatenate whatever memory the agent's memory manager has loaded.

    Prefer the manager bound to the live AIAgent. Fall back to reading the
    on-disk memory dir directly so the sandbox works even when the agent
    object is a stub or is still in construction.

    Probe order (``fetch_all`` → ``get_all_snippets`` → ``build_context`` →
    disk) exists because *agent* is duck-typed. Call sites vary: the
    live ``AIAgent`` has ``fetch_all`` on a full ``MemoryManager``; gateway-
    side shims expose ``get_all_snippets``; some test doubles only implement
    ``build_context``; the sandbox path also fires from contexts where no
    manager is plumbed in at all (auxiliary_client construction, headless
    bootstrapping) — hence the disk fallback. Each method is tried in
    isolation and skipped on failure, because losing memory silently is
    preferable to failing the sandbox build.
    """
    # Try agent-bound manager
    mgr = getattr(agent, "memory_manager", None) if agent is not None else None
    if mgr is not None:
        for attr in ("fetch_all", "get_all_snippets", "build_context"):
            fn = getattr(mgr, attr, None)
            if callable(fn):
                try:
                    got = fn()
                    if isinstance(got, str) and got.strip():
                        return got
                    if isinstance(got, (list, tuple)):
                        joined = "\n\n".join(str(x) for x in got if x)
                        if joined.strip():
                            return joined
                except Exception as exc:
                    logger.debug("Memory fetch via %s failed: %s", attr, exc)

    # Fallback: read ~/.hermes/memories/*.md
    try:
        from hermes_constants import get_hermes_home

        mem_dir = get_hermes_home() / "memories"
    except Exception:
        return ""
    if not mem_dir.exists():
        return ""
    parts: List[str] = []
    for p in sorted(mem_dir.glob("*.md")):
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"# {p.stem}\n{text}")
        except Exception:
            continue
    return "\n\n".join(parts)


def _build_platform_block(platform: Optional[str]) -> Optional[str]:
    plat = (platform or os.environ.get("HERMES_SESSION_PLATFORM") or "").strip()
    if not plat:
        return None
    user = (
        os.environ.get("HERMES_SESSION_USER")
        or os.environ.get("HERMES_SESSION_DISPLAY_NAME")
        or ""
    ).strip()
    chat = os.environ.get("HERMES_SESSION_CHAT", "").strip()
    lines = [f"## Platform context", f"Active platform: `{plat}`"]
    if user:
        lines.append(f"User display name: {user}")
    if chat:
        lines.append(f"Channel: {chat}")
    return "\n".join(lines)


def _load_soul(hermes_home: Path, *, include: bool = True) -> Optional[str]:
    if not include:
        return None
    path = hermes_home / "SOUL.md"
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        return content or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Skills flattening
# ---------------------------------------------------------------------------


def _skill_is_enabled(
    skill_file: Path,
    *,
    disabled_names: set[str],
    available_tools: Optional[set[str]],
    available_toolsets: Optional[set[str]],
    skill_matches_platform,
    parse_frontmatter,
    skill_should_show,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return ``(frontmatter_name, frontmatter_dict)`` if the skill is enabled.

    Applies the same four gates the prompt-builder path uses:
    platform match, disabled-list, condition filter, and readability. Any
    gate failing returns ``None`` so the caller skips the skill.
    """
    try:
        content = skill_file.read_text(encoding="utf-8")
    except OSError:
        return None
    frontmatter, _body = parse_frontmatter(content)
    if not skill_matches_platform(frontmatter):
        return None
    frontmatter_name = str(frontmatter.get("name") or skill_file.parent.name).strip()
    dir_name = skill_file.parent.name
    if frontmatter_name in disabled_names or dir_name in disabled_names:
        return None
    try:
        conds = _extract_skill_conditions(frontmatter)
        if not skill_should_show(conds, available_tools, available_toolsets):
            return None
    except Exception:
        # Filters that raise should not block sandbox construction — fall
        # through to "include" so a broken condition doesn't hide a skill.
        pass
    return frontmatter_name or dir_name, frontmatter


def _unique_slug(
    base_name: str, source: Path, seen: Dict[str, Path]
) -> str:
    """Slugify *base_name* and append a 6-char SHA1 suffix on collision.

    The 6-char suffix is a pragmatic tradeoff: collision probability is
    negligible at realistic skill counts (<100), directory names stay
    human-readable, and a re-collision on the same source path is harmless
    (we overwrite with identical content, so the result is stable).
    """
    slug = _sanitize_skill_slug(base_name)
    if slug in seen and seen[slug] != source:
        suffix = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:6]
        slug = f"{slug}-{suffix}"
    return slug


def _flatten_skills_into(
    *,
    include: bool = True,
    hermes_home: Path,
    target: Path,
    available_tools: Optional[set[str]] = None,
    available_toolsets: Optional[set[str]] = None,
) -> int:
    """Copy each enabled hermes skill into ``<target>/<skill_name>/`` flat.

    Anthropic's skill convention is flat (one level under
    ``.claude/skills/``); hermes skills can nest. For each enabled skill,
    we slug its frontmatter name and copy the source directory. Returns
    the number of skills copied.

    ``include=False`` short-circuits the work entirely and returns ``0``,
    which keeps the sandbox manifest digest stable while letting callers
    skip the (potentially expensive) skills glob + copy.
    """
    if not include:
        return 0
    try:
        from agent.skill_utils import (
            iter_skill_index_files,
            parse_frontmatter,
            skill_matches_platform,
            get_disabled_skill_names,
        )
        from agent.prompt_builder import _skill_should_show  # noqa: WPS437
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("skill utilities unavailable: %s", exc)
        return 0

    skills_dir = hermes_home / "skills"
    if not skills_dir.exists():
        return 0

    disabled = get_disabled_skill_names()
    seen_names: Dict[str, Path] = {}
    copied = 0

    for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
        enabled = _skill_is_enabled(
            skill_file,
            disabled_names=disabled,
            available_tools=available_tools,
            available_toolsets=available_toolsets,
            skill_matches_platform=skill_matches_platform,
            parse_frontmatter=parse_frontmatter,
            skill_should_show=_skill_should_show,
        )
        if enabled is None:
            continue
        base_name, _frontmatter = enabled

        slug = _unique_slug(base_name, skill_file.parent, seen_names)
        dest = target / slug
        src = skill_file.parent
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)
        except (OSError, shutil.Error) as exc:
            logger.debug("Failed to copy skill %s → %s: %s", src, dest, exc)
            continue
        seen_names[slug] = src
        copied += 1

    return copied


def _extract_skill_conditions(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror ``prompt_builder.extract_skill_conditions`` shape."""
    try:
        from agent.prompt_builder import extract_skill_conditions  # type: ignore

        return extract_skill_conditions(frontmatter)
    except Exception:
        return {}


def _sanitize_skill_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name or "skill").strip("-")
    return slug or "skill"


# ---------------------------------------------------------------------------
# .mcp.json construction
# ---------------------------------------------------------------------------


def _build_mcp_config(
    *, session_id: str, hermes_home: Path, platform: Optional[str]
) -> Dict[str, Any]:
    base_env = {
        "HERMES_HOME": str(hermes_home),
        "HERMES_SESSION_ID": session_id,
    }
    if platform:
        base_env["HERMES_SESSION_PLATFORM"] = platform

    hermes_tools_env = {**base_env, "HERMES_MCP_AUTO_APPROVE": "1"}

    return {
        "mcpServers": {
            "hermes_tools": {
                "command": "hermes",
                "args": ["mcp", "tools-serve"],
                "env": hermes_tools_env,
            },
            "hermes_messaging": {
                "command": "hermes",
                "args": ["mcp", "serve"],
                "env": dict(base_env),
            },
        }
    }


# ---------------------------------------------------------------------------
# Manifest / caching
# ---------------------------------------------------------------------------


def _collect_manifest_inputs(
    *,
    hermes_home: Path,
    platform: Optional[str],
    model: Optional[str] = None,
    components: Optional[SandboxComponents] = None,
) -> Dict[str, Any]:
    """Return a dict whose hash captures everything that affects the sandbox."""
    items: Dict[str, Any] = {
        "hermes_home": str(hermes_home),
        "platform": (platform or os.environ.get("HERMES_SESSION_PLATFORM") or "").strip(),
        "python": sys.platform,
        "preamble_sha": hashlib.sha1(_PREAMBLE.encode("utf-8")).hexdigest(),
        "model": (model or "").strip(),
    }
    soul = hermes_home / "SOUL.md"
    if soul.exists():
        st = soul.stat()
        items["soul"] = {"mtime": st.st_mtime, "size": st.st_size}
    mem_dir = hermes_home / "memories"
    if mem_dir.exists():
        items["memories"] = sorted(
            (p.name, int(p.stat().st_mtime), p.stat().st_size)
            for p in mem_dir.glob("*.md")
        )
    skills_dir = hermes_home / "skills"
    if skills_dir.exists():
        items["skills"] = sorted(
            (
                str(p.relative_to(skills_dir)),
                int(p.stat().st_mtime),
                p.stat().st_size,
            )
            for p in skills_dir.rglob("SKILL.md")
        )
    config = hermes_home / "config.yaml"
    if config.exists():
        st = config.stat()
        items["config"] = {"mtime": st.st_mtime, "size": st.st_size}
    # When components is provided, fold it into the digest so changing the
    # preset invalidates the manifest. When omitted (the current default for
    # every call site in ``build_session_sandbox``), the digest is identical
    # to the pre-commit-1 shape — existing sandbox caches stay valid.
    if components is not None:
        items["components"] = dataclasses.asdict(components)
    return items


def _hash_inputs(inputs: Dict[str, Any]) -> str:
    blob = json.dumps(inputs, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


# ---------------------------------------------------------------------------
# Tool-name helpers used by CLAUDE.md composition
# ---------------------------------------------------------------------------


def _resolve_tool_names() -> List[str]:
    """Return the sorted list of hermes tool names the sandbox will surface."""
    try:
        from tools.registry import discover_builtin_tools, registry
        from hermes_mcp.tools_server import EXCLUDED_TOOLS

        discover_builtin_tools()
        return sorted(n for n in registry.get_all_tool_names() if n not in EXCLUDED_TOOLS)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("Could not resolve tool names for sandbox: %s", exc)
        return []
