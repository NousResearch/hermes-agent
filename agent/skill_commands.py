"""Shared slash command helpers for skills (CLI and gateway both invoke /skill-name)."""

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import display_hermes_home
from agent.prompt_cache_boundary import register_stable_prefix
from agent.skill_preprocessing import load_skills_config as _load_skills_config, preprocess_skill_content

logger = logging.getLogger(__name__)

_skill_commands: Dict[str, Dict[str, Any]] = {}
_skill_commands_platform: Optional[str] = None
_skill_commands_home: Optional[str] = None
# Guards the (map, platform-tag, home-tag) triple so publication and the
# freshness lookup always see a consistent snapshot. Scanning stays outside.
_publish_lock = threading.Lock()
_SKILL_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_SKILL_MULTI_HYPHEN = re.compile(r"-{2,}")

# Skill-scaffolding markers. A /skill (or /bundle) turn is expanded into a
# model-facing message embedding the full skill body; memory providers storing
# the raw user turn would capture the body instead of what the user asked, so
# ``extract_user_instruction_from_skill_message`` recovers just the instruction.
# The markers MUST stay byte-identical to the builders (``_build_skill_message``,
# ``_scaffold_header``).
_SKILL_INVOCATION_PREFIX = "[IMPORTANT: The user has invoked the "
_SINGLE_SKILL_MARKER = "The full skill content is loaded below.]"
_SINGLE_SKILL_INSTRUCTION = "The user has provided the following instruction alongside the skill invocation: "
_RUNTIME_NOTE = "\n\n[Runtime note:"
_BUNDLE_MARKER = " skill bundle,"
_BUNDLE_USER_INSTRUCTION = "\nUser instruction: "
_BUNDLE_FIRST_SKILL_BLOCK = "\n\n[Loaded as part of the "

# The skill name sits in the first quoted span of the activation note, for both
# the single-skill and the bundle header ("work" / "/clean /work").
_SKILL_NAME_RE = re.compile(re.escape(_SKILL_INVOCATION_PREFIX) + r'"([^"]*)"')

# SQL LIKE pattern for listing queries that recognize scaffolding before the row
# reaches Python (no LIKE wildcards in the prefix, so no ESCAPE clause needed).
SKILL_SCAFFOLD_SQL_LIKE = _SKILL_INVOCATION_PREFIX + "%"

# Marks where a preview query joined the head and tail of a long scaffolded
# message; ``describe_skill_invocation`` cuts there rather than show the body.
SKILL_EXCERPT_JOINT = "\x1e"


def slugify_skill_name(name: str) -> str:
    """Normalize a skill/bundle name to a ``/command`` slug (``Foo Bar`` -> ``foo-bar``);
    strips chars (``+``, ``/``) that would make invalid Telegram command names."""
    cmd = _SKILL_INVALID_CHARS.sub("", name.lower().replace(" ", "-").replace("_", "-"))
    return _SKILL_MULTI_HYPHEN.sub("-", cmd).strip("-")


def append_user_instruction(parts: list, instruction: str) -> str:
    """Append the instruction line to ``parts``; return the stable prefix, which
    ends exactly at the instruction marker so (registered with
    ``agent.prompt_cache_boundary``) the cache planner can break on the scaffold.
    Single construction site guarantees the prefix is a byte-prefix of the message.

    Shared by every builder that ends a static skill scaffold with the caller-supplied volatile instruction
    (single-skill invocations, cron job prompts). Keeping construction in one place guarantees the
    registered prefix stays a byte-prefix of the built message — the invariant the request-time split
    depends on. See #81867.
    """
    stable_prefix = "\n".join(parts) + "\n" + _SINGLE_SKILL_INSTRUCTION
    parts.append(f"{_SINGLE_SKILL_INSTRUCTION}{instruction}")
    return stable_prefix


def extract_user_instruction_from_skill_message(content: Any) -> Optional[str]:
    """Recover the user's instruction from a slash-skill-expanded turn: the
    string unchanged when it is NOT scaffolding, the extracted instruction when
    the scaffolding carried one, or ``None`` for a bare ``/skill`` invocation."""
    if not isinstance(content, str):
        return None
    if not content.startswith(_SKILL_INVOCATION_PREFIX):
        return content
    if _BUNDLE_MARKER in content:
        # Bundles put the instruction before the loaded skills: FIRST marker is the user's.
        return _cut_after(content, _BUNDLE_USER_INSTRUCTION, _BUNDLE_FIRST_SKILL_BLOCK, content.find)
    if _SINGLE_SKILL_MARKER in content:
        # The instruction follows the skill body (which may quote the marker): LAST marker is the user's.
        return _cut_after(content, _SINGLE_SKILL_INSTRUCTION, _RUNTIME_NOTE, content.rfind)
    return None


def describe_skill_invocation(content: Any, separator: str = " — ") -> Optional[str]:
    """Render a slash-skill-expanded turn the way the user typed it:
    ``"/work — fix the title leak"``, ``"/work"`` for a bare invocation, or
    ``None`` when *content* is not scaffolding. ``separator=" "`` gives the
    literal invocation as typed (chat transcripts)."""
    if not isinstance(content, str) or not content.startswith(_SKILL_INVOCATION_PREFIX):
        return None
    match = _SKILL_NAME_RE.match(content)
    name = (match.group(1) if match else "").strip()
    # Bundle headers already carry their typed "/a /b" keys; a single skill is a bare name.
    label = name if name.startswith("/") else f"/{name}"
    instruction = extract_user_instruction_from_skill_message(content)
    if instruction and instruction is not content:
        # An excerpt (head + tail joined by SKILL_EXCERPT_JOINT) can put the
        # joint inside the span — keep only the side the marker was found on.
        instruction = " ".join(instruction.split(SKILL_EXCERPT_JOINT)[0].split())
        if instruction:
            return f"{label}{separator}{instruction}" if name else instruction
    return label if name else None


def _cut_after(message: str, marker: str, stop_marker: str, find) -> Optional[str]:
    """Text between *marker* (located with ``find``) and *stop_marker*, stripped; None if absent/empty."""
    marker_idx = find(marker)
    if marker_idx < 0:
        return None
    return message[marker_idx + len(marker):].split(stop_marker, 1)[0].strip() or None


def _resolve_skill_commands_platform() -> Optional[str]:
    """Current platform scope for disabled-skill filtering, or None (CLI, RL,
    scripts). A change invalidates the scan cache so each platform sees its
    own ``skills.platform_disabled`` view.

    Used to detect when the active platform has shifted so :func:`get_skill_commands` can drop a stale cache
    that was populated for a different platform's ``skills.platform_disabled`` view (#14536).
    """
    try:
        from gateway.session_context import get_session_env
        resolved_platform = os.getenv("HERMES_PLATFORM") or get_session_env("HERMES_SESSION_PLATFORM")
    except Exception:
        resolved_platform = os.getenv("HERMES_PLATFORM")
    return resolved_platform or None


def _resolve_skill_commands_home() -> str:
    """Effective Hermes home the scan is scoped to (profiles carry their own
    ``skills.external_dirs``, so a profile switch must invalidate the cache).

    A gateway session can switch between profiles that each carry their own ``skills.external_dirs`` (via
    ``set_hermes_home_override``), but the module-level scan only tracked
    ``_resolve_skill_commands_platform()``. Switching profiles without a platform change left the previous
    profile's skill list cached, so ``get_skill_commands()`` reported a cache miss for skills that only
    exist under the new profile (#88023).
    """
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def _load_skill_payload(skill_identifier: str, task_id: str | None = None) -> tuple[dict[str, Any], Path | None, str] | None:
    """Load a skill by name/path and return (loaded_payload, skill_dir, display_name)."""
    raw_identifier = (skill_identifier or "").strip()
    if not raw_identifier:
        return None
    try:
        from tools.skills_tool import _skills_dir, skill_view
        from agent.skill_utils import normalize_skill_lookup_name
        normalized = normalize_skill_lookup_name(raw_identifier)
        loaded_skill = json.loads(skill_view(normalized, task_id=task_id, preprocess=False))
    except Exception:
        return None
    if not loaded_skill.get("success"):
        return None
    skill_path = str(loaded_skill.get("path") or "")
    skill_dir = None
    # Prefer the absolute skill_dir from skill_view() (correct for external
    # skills too); fall back to SKILLS_DIR-relative reconstruction for legacy responses.
    if loaded_skill.get("skill_dir"):
        skill_dir = Path(loaded_skill["skill_dir"])
    elif skill_path:
        try:
            skill_dir = _skills_dir() / Path(skill_path).parent
        except Exception:
            skill_dir = None
    return loaded_skill, skill_dir, str(loaded_skill.get("name") or normalized)


def _inject_skill_config(loaded_skill: dict[str, Any], parts: list[str]) -> None:
    """Append a ``[Skill config: ...]`` block with resolved ``metadata.hermes.config``
    values so the agent needn't read config.yaml. Any failure leaves the message without it."""
    try:
        from agent.skill_utils import extract_skill_config_vars, parse_frontmatter, resolve_skill_config_values
        raw_content = str(loaded_skill.get("raw_content") or loaded_skill.get("content") or "")
        frontmatter, _ = parse_frontmatter(raw_content)
        resolved = resolve_skill_config_values(extract_skill_config_vars(frontmatter))
        if not resolved:
            return
        parts.append("")
        parts.append(f"[Skill config (from {display_hermes_home()}/config.yaml):")
        parts.extend(f"  {key} = {str(value) if value else '(not set)'}" for key, value in resolved.items())
        parts.append("]")
    except Exception:
        pass


_SKILL_DIR_NOTE = (
    "Resolve any relative paths in this skill (e.g. `scripts/foo.js`, "
    "`templates/config.yaml`) against that directory, then run them "
    "with the terminal tool using the absolute path."
)
_SETUP_SKIPPED_NOTE = (
    "Required environment setup was skipped. Continue loading the skill "
    "and explain any reduced functionality if it matters."
)


def _setup_note(loaded_skill: dict[str, Any]) -> Optional[str]:
    if loaded_skill.get("setup_skipped"):
        return _SETUP_SKIPPED_NOTE
    return loaded_skill.get("gateway_setup_hint") or (
        loaded_skill.get("setup_note") if loaded_skill.get("setup_needed") else None
    ) or None


def _supporting_files(loaded_skill: dict[str, Any], skill_dir: Path | None) -> list[str]:
    """Skill-relative support file paths: from ``linked_files`` or a disk walk."""
    linked = (loaded_skill.get("linked_files") or {}).values()
    supporting = [entry for entries in linked if isinstance(entries, list) for entry in entries]
    if not supporting and skill_dir:
        for subdir in ("references", "templates", "scripts", "assets"):
            files = sorted((skill_dir / subdir).rglob("*"))
            supporting += [str(f.relative_to(skill_dir)) for f in files if f.is_file() and not f.is_symlink()]
    return supporting


def _build_skill_message(
    loaded_skill: dict[str, Any],
    skill_dir: Path | None,
    activation_note: str,
    user_instruction: str = "",
    runtime_note: str = "",
    session_id: str | None = None,
) -> str:
    """Format a loaded skill into a user/system message payload."""
    from tools.skills_tool import _skills_dir
    # Preprocess first so downstream blocks see the expanded content.
    content = preprocess_skill_content(
        str(loaded_skill.get("content") or ""), skill_dir, session_id, skills_cfg=_load_skills_config(),
    )
    parts = [activation_note, "", content.strip()]
    # Absolute skill dir lets the agent run bundled scripts without a skill_view() round-trip.
    if skill_dir:
        parts += ["", f"[Skill directory: {skill_dir}]", _SKILL_DIR_NOTE]
    _inject_skill_config(loaded_skill, parts)
    setup_note = _setup_note(loaded_skill)
    if setup_note:
        parts += ["", f"[Skill setup note: {setup_note}]"]
    supporting = _supporting_files(loaded_skill, skill_dir)
    if supporting and skill_dir:
        try:
            skill_view_target = str(skill_dir.relative_to(_skills_dir()))
        except ValueError:
            skill_view_target = skill_dir.name  # external dir — use the skill name
        parts += ["", "[This skill has supporting files (paths relative to the skill directory above):]"]
        parts += [f"- {sf}" for sf in supporting]
        parts.append(
            f'\nLoad any of these with skill_view(name="{skill_view_target}", '
            f'file_path="<path>"), or run scripts directly by absolute path '
            f"(e.g. `node {skill_dir}/scripts/foo.js`)."
        )
    stable_prefix = None
    if user_instruction:
        parts.append("")
        # Everything before the volatile instruction is a stable scaffold; the
        # registered boundary lets the cache planner break there (see append_user_instruction).
        # Everything before the caller-supplied instruction is a stable scaffold; declare the exact boundary
        # so the Anthropic cache planner can put a breakpoint on it instead of caching the whole message as
        # one atomic block (#81867). The static instruction prose stays on the stable side; the volatile
        # instruction (webhook payload, ticket IDs, timestamps) and any runtime note ride in the tail.
        stable_prefix = append_user_instruction(parts, user_instruction)
    if runtime_note:
        parts += ["", f"[Runtime note: {runtime_note}]"]
    message = "\n".join(parts)
    if stable_prefix is not None and message.startswith(stable_prefix) and len(message) > len(stable_prefix):
        register_stable_prefix(stable_prefix)
    return message


def _render_skill_block(
    loaded: tuple[dict[str, Any], Path | None, str], activation_note: str, task_id: str | None, **message_kwargs: str,
) -> str:
    """Bump Curator usage tracking (never fatal) and build the message block for one loaded skill."""
    loaded_skill, skill_dir, skill_name = loaded
    try:
        # Track active usage for Curator lifecycle management (#17782)
        # Track active usage for Curator lifecycle management (#17782)
        # Track active usage for Curator lifecycle management (#17782)
        from tools.skill_usage import bump_use
        bump_use(skill_name, task_id=task_id)
    except Exception:
        pass
    return _build_skill_message(loaded_skill, skill_dir, activation_note, session_id=task_id, **message_kwargs)


def _scaffold_header(
    subject: str, loaded_names: list[str], *, lead_lines: list[str] | None = None,
    missing: list[str] | None = None, disabled: list[str] | None = None,
    extra_instruction: str = "", user_instruction: str = "",
) -> str:
    """Header for multi-skill messages (bundles and stacked invocations).
    ``subject`` must end in " skill bundle" so the bundle-format extractor applies."""
    lines = [
        f"[IMPORTANT: The user has invoked the {subject}, "
        f"loading {len(loaded_names)} skills together. Treat every skill below "
        "as active guidance for this turn.]",
        "",
        *(lead_lines or []),
        f"Skills loaded: {', '.join(loaded_names)}",
    ]
    if missing:
        lines.append(f"Skills missing (skipped): {', '.join(missing)}")
    if disabled:
        lines.append(f"Skills disabled for this platform (skipped): {', '.join(disabled)}")
    if extra_instruction:
        lines += ["", f"Bundle instruction: {extra_instruction}"]
    if user_instruction:
        lines += ["", f"User instruction: {user_instruction}"]
    return "\n".join(lines)


_SCAN_SKIP_PARTS = {'.git', '.github', '.hub', '.archive'}


def _scan_skill_md(skill_md: Path, disabled: set, seen_names: set, commands: Dict[str, Dict[str, Any]], resolve_command) -> None:
    """Register one SKILL.md in *commands* (no-op when filtered or colliding)."""
    from tools.skills_tool import _parse_frontmatter, skill_matches_platform, skill_matches_environment
    if any(part in _SCAN_SKIP_PARTS for part in skill_md.parts):
        return
    frontmatter, body = _parse_frontmatter(skill_md.read_text(encoding='utf-8'))
    # OS gate is hard; environment gate (kanban/docker/s6) is offer-time only.
    if not skill_matches_platform(frontmatter) or not skill_matches_environment(frontmatter):
        return
    name = frontmatter.get('name', skill_md.parent.name)
    if name in seen_names or name in disabled:
        return
    description = frontmatter.get('description', '') or next(
        (line.strip()[:80] for line in body.strip().split('\n') if line.strip() and not line.strip().startswith('#')),
        '',
    )
    seen_names.add(name)
    cmd_name = slugify_skill_name(name)
    if not cmd_name:
        return
    # A collision with a core command (name or alias, via resolve_command) skips
    # auto-registration; the skill stays loadable via /skill <name>.
    if resolve_command(cmd_name) is not None:
        logger.warning("Skill %r generates slash command '/%s' which collides with a core Hermes command; "
                       "skipping auto-registration. Use '/skill %s' instead.", name, cmd_name, name)
        return
    # Dedup on the slug too: "git_helper" and "git-helper" normalize the same.
    # First-wins preserves project > local > external precedence.
    cmd_key = f"/{cmd_name}"
    if cmd_key in commands:
        logger.warning("Skill %r maps to slash command %s already claimed by %r; keeping the first and skipping this one.",
                       name, cmd_key, commands[cmd_key]["name"])
        return
    commands[cmd_key] = {"name": name, "description": description or f"Invoke the {name} skill",
                         "skill_md_path": str(skill_md), "skill_dir": str(skill_md.parent)}


def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Scan skill dirs and return {"/skill-name": {name, description, skill_md_path, skill_dir}}.
    Builds a local map and publishes once at the end: writing straight into the
    global exposed partial results to overlapping scans, which then logged
    bogus "already claimed" collisions against their own incumbents."""
    global _skill_commands, _skill_commands_platform, _skill_commands_home
    platform = _resolve_skill_commands_platform()
    home = _resolve_skill_commands_home()
    # Build into a local map and publish once, at the end. Writing straight into the global made a scan's
    # partial results visible to everything else in the process: a second, overlapping scan deduped against
    # its own (empty) ``seen_names`` but collided against the first scan's already- published slugs, logging
    # one bogus "already claimed" warning per skill — each naming the same skill as its own incumbent
    # (#74574).
    commands: Dict[str, Dict[str, Any]] = {}
    try:
        from tools.skills_tool import _skills_dir, _get_disabled_skill_names
        from agent.skill_utils import (
            get_external_skills_dirs, get_project_skills_dirs, iter_project_skill_files, iter_skill_index_files,
        )
        from hermes_cli.commands import resolve_command
        disabled = _get_disabled_skill_names()
        seen_names: set = set()
        # Precedence: project (through the quarantine chokepoint) > local > external.
        # Resolve the local dir at call time: import-time SKILLS_DIR is frozen to
        # the launch home, but a multiplexed profile scope may have changed it.
        # See #67277.
        skills_dir = _skills_dir()
        iters = [iter_project_skill_files(d) for d in get_project_skills_dirs()]
        local = [skills_dir] if skills_dir.exists() else []
        iters += [iter_skill_index_files(d, "SKILL.md") for d in local + get_external_skills_dirs()]
        for _iter in iters:
            for skill_md in _iter:
                try:
                    _scan_skill_md(skill_md, disabled, seen_names, commands, resolve_command)
                except Exception:
                    continue
    except Exception:
        pass
    # Publish map + tags as ONE step: a reader landing between bare assignments
    # could accept the new map under a stale platform tag and serve another
    # platform's disabled-skill view.
    with _publish_lock:
        # Bare assignments are not atomic together: a reader landing between them sees the NEW map still
        # carrying the OLD platform tag, and if that stale tag happens to match its own platform it accepts
        # the map without rescanning — serving another platform's disabled-skill view, exactly the leak
        # #14536 closed. Only the publish/lookup pair is locked; the scan above (file I/O, deferred imports)
        # stays outside it.
        _skill_commands = commands
        _skill_commands_platform = platform
        _skill_commands_home = home
    return commands


def get_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Return the current skill commands mapping (scan first if empty). Rescans
    when the platform scope (one gateway serving Telegram and Discord) or the
    active profile's home (Desktop profile switch) changes, so each sees its
    own ``platform_disabled`` / ``external_dirs`` view.

    See #14536, #88023.
    """
    current_platform = _resolve_skill_commands_platform()
    current_home = _resolve_skill_commands_home()
    with _publish_lock:
        commands = _skill_commands
        is_fresh = bool(commands) and (_skill_commands_platform, _skill_commands_home) == (current_platform, current_home)
    # Scan outside the lock — file I/O and deferred imports; concurrent scans
    # are safe since each builds its own map.
    return commands if is_fresh else scan_skill_commands()


def diff_command_snapshots(before: Dict[str, str], after: Dict[str, str]) -> Dict[str, Any]:
    """Diff two {name: description} snapshots into added/removed/unchanged/total.
    Removed entries carry the pre-rescan description (the file may be gone)."""
    return {
        "added": [{"name": n, "description": after[n]} for n in sorted(set(after) - set(before))],
        "removed": [{"name": n, "description": before[n]} for n in sorted(set(before) - set(after))],
        "unchanged": sorted(set(after) & set(before)),
        "total": len(after),
    }


def command_snapshot(cmds: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """``{"/slug": info}`` -> ``{"slug": description}`` for diff_command_snapshots."""
    return {key.lstrip("/"): (info or {}).get("description") or "" for key, info in cmds.items()}


def reload_skills() -> Dict[str, Any]:
    """Re-scan skill dirs and return a diff of the slash-command map (``added``
    / ``removed`` / ``unchanged`` / ``total`` / ``commands``; descriptions are the
    full frontmatter field). Does NOT invalidate the skills system-prompt cache:
    skills are called by name, so ``/reload-skills`` costs no cache reset."""
    before = command_snapshot(_skill_commands)
    new_commands = scan_skill_commands()
    result = diff_command_snapshots(before, command_snapshot(new_commands))
    result["commands"] = len(new_commands)
    return result


def resolve_skill_command_key(command: str) -> Optional[str]:
    """Resolve a user-typed /command to its canonical ``/slug`` key, or None.
    ``_`` ≡ ``-``: Telegram disallows hyphens, so ``/claude-code`` arrives as ``/claude_code``."""
    return resolve_slash_key(command, get_skill_commands())


def resolve_slash_key(command: str, table: Dict[str, Any]) -> Optional[str]:
    """``command`` -> ``"/slug"`` when present in *table* (``_`` normalized to ``-``), else None."""
    if not command:
        return None
    cmd_key = f"/{command.replace('_', '-')}"
    return cmd_key if cmd_key in table else None


def build_skill_invocation_message(
    cmd_key: str, user_instruction: str = "", task_id: str | None = None, runtime_note: str = "",
) -> Optional[str]:
    """Build the user message for a skill slash command, or None if not found."""
    skill_info = get_skill_commands().get(cmd_key)
    loaded = _load_skill_payload(skill_info["skill_dir"], task_id=task_id) if skill_info else None
    if not loaded:
        return None
    note = (f'[IMPORTANT: The user has invoked the "{loaded[2]}" skill, indicating they want '
            "you to follow its instructions. The full skill content is loaded below.]")
    return _render_skill_block(loaded, note, task_id, user_instruction=user_instruction, runtime_note=runtime_note)


# Stacked slash-skill invocations — `/skill-a /skill-b do XYZ` loads every
# leading skill (up to _MAX_STACKED_SKILLS). The message reuses the BUNDLE
# scaffolding markers so the memory extractor needs no new plumbing.
_MAX_STACKED_SKILLS = 5


def split_stacked_skill_commands(rest: str) -> tuple[list[str], str]:
    """Consume further leading ``/skill`` tokens from *rest* (text after the first
    matched command); stops at the first non-skill (or repeated) token, which
    starts the user instruction. Returns ``(extra_cmd_keys, remaining_instruction)``."""
    keys: list[str] = []
    remaining = rest or ""
    while len(keys) < _MAX_STACKED_SKILLS - 1:
        stripped = remaining.lstrip()
        if not stripped.startswith("/"):
            break
        token, tail = (stripped.split(None, 1) + [""])[:2]
        cmd_key = resolve_skill_command_key(token.lstrip("/"))
        if cmd_key is None or cmd_key in keys:
            break
        keys.append(cmd_key)
        remaining = tail
    return keys, remaining.strip()


def build_stacked_skill_invocation_message(
    cmd_keys: list[str], user_instruction: str = "", task_id: str | None = None,
) -> Optional[tuple[str, list[str], list[str]]]:
    """Build the user message for a stacked multi-skill slash invocation:
    ``(message, loaded_skill_names, missing_skill_names)``, or ``None`` when no skill loaded."""
    commands = get_skill_commands()
    keys = [k for k in cmd_keys if k]
    loaded_names, missing, _disabled, skill_blocks = _load_skill_blocks(
        keys,
        lambda cmd_key: _load_skill_payload(commands[cmd_key]["skill_dir"], task_id=task_id) if cmd_key in commands else None,
        lambda name: f'[Loaded as part of the stacked skill invocation "{name}".]',  # bundle block marker
        task_id, missing_label=lambda k: k.lstrip("/"),
    )
    if not skill_blocks:
        return None
    typed = " ".join(keys)
    header = _scaffold_header(f'"{typed}" stacked skill bundle', loaded_names, missing=missing, user_instruction=user_instruction)
    return ("\n\n".join([header, *skill_blocks]), loaded_names, missing)


def _disabled_skill_names(platform: str | None = None) -> set:
    """Operator-disabled skill names (empty set when config is unreadable)."""
    try:
        from agent.skill_utils import get_disabled_skill_names
        return get_disabled_skill_names(platform=platform)
    except Exception:
        return set()


def _load_skill_blocks(
    identifiers: list[str], load, activation_note, task_id: str | None, *,
    missing_label=lambda ident: ident, disabled_names: set | None = None, disabled_as_missing: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load each distinct identifier via *load* and render its block; returns
    ``(loaded_names, missing, disabled, blocks)``. With *disabled_names*, members
    whose canonical (LOADED — identifiers may be paths) name or identifier is
    disabled go to ``disabled`` (or ``missing`` when *disabled_as_missing*)."""
    loaded_names: list[str] = []
    missing: list[str] = []
    disabled: list[str] = []
    blocks: list[str] = []
    seen: set[str] = set()
    for identifier in identifiers:
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        loaded = load(identifier)
        if not loaded:
            missing.append(missing_label(identifier))
            continue
        skill_name = loaded[2]
        if disabled_names and (skill_name in disabled_names or identifier in disabled_names):
            if disabled_as_missing:
                missing.append(identifier)
            else:
                disabled.append(skill_name or identifier)
            continue
        blocks.append(_render_skill_block(loaded, activation_note(skill_name), task_id))
        loaded_names.append(skill_name)
    return loaded_names, missing, disabled, blocks


def build_preloaded_skills_prompt(skill_identifiers: list[str], task_id: str | None = None) -> tuple[str, list[str], list[str]]:
    """Load skills for session-wide CLI/TUI preloading; returns (prompt_text,
    loaded_skill_names, missing_identifiers). Disabled skills count as missing:
    this path bypasses the scan-time filter, and ``hermes -s <skill>`` must not
    force-load an operator-disabled skill.

    Disabled skills are treated the same as missing ones: this loads via a raw identifier straight into
    ``_load_skill_payload``, bypassing ``get_skill_commands()``'s scan-time disabled filter — mirrors the
    bundle-invocation gate (#59156).
    """
    loaded_names, missing, _disabled, prompt_parts = _load_skill_blocks(
        [(raw or "").strip() for raw in skill_identifiers],
        lambda identifier: _load_skill_payload(identifier, task_id=task_id),
        lambda name: (f'[IMPORTANT: The user launched this CLI session with the "{name}" skill '
                      "preloaded. Treat its instructions as active guidance for the duration of this "
                      "session unless the user overrides them.]"),
        task_id, disabled_names=_disabled_skill_names(), disabled_as_missing=True,
    )
    return "\n\n".join(prompt_parts), loaded_names, missing
