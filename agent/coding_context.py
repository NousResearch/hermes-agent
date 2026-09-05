"""Coding-context awareness: the single place that decides "are we coding?".

In a code workspace on an interactive surface Hermes adopts a **coding posture**: a
frozen :class:`RuntimeMode` built from a :class:`ContextProfile` (pure data). The
system prompt reads ``system_prompt_parts()``; the toolset collapses ONLY under opt-in
``focus`` (never strips a user-enabled toolset). ``agent.coding_context``: ``auto``
(default, prompt-only) / ``focus`` / ``on`` / ``off``. Resolved once, immutable; the
workspace snapshot is never re-probed per turn (cache safety).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli._subprocess_compat import bounded_git_probe

logger = logging.getLogger("hermes.coding_context")

CODING_TOOLSET = "coding"

# Surfaces where ``auto`` may adopt the posture; messaging platforms are deliberately absent.
INTERACTIVE_CODING_PLATFORMS = {"cli", "tui", "acp", "desktop", ""}
# Project-root signals (cheap filename checks) marking a code workspace even without git.
_PROJECT_MARKERS = (
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "package.json", "tsconfig.json", "deno.json",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "Gemfile", "composer.json", "mix.exs",
    "pubspec.yaml", "CMakeLists.txt", "Makefile", "Dockerfile", "AGENTS.md", "CLAUDE.md", ".cursorrules",
)
# Agent-instruction files surfaced separately from manifests in the snapshot.
_CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md", ".cursorrules")

# Extensions that make a manifest-less git repo a *code* workspace (a `git init` notes folder is not).
_CODE_EXTENSIONS = frozenset({
    ".py", ".pyi", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt", ".kts",
    ".scala", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".m", ".mm", ".dart", ".ex", ".exs",
    ".lua", ".sh", ".bash", ".zsh", ".sql", ".vue", ".svelte", ".r", ".jl", ".hs", ".clj", ".erl", ".pl",
})
_CODE_SCAN_SKIP_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", "target", ".next", ".turbo", "vendor",
})
# Bounded sweep: a code workspace reveals itself in the first handful of entries.
_CODE_SCAN_MAX_ENTRIES = 500
# Lockfile → package manager, checked in priority order.
_PY_LOCKFILES = (("uv.lock", "uv"), ("poetry.lock", "poetry"), ("Pipfile.lock", "pipenv"))
_JS_LOCKFILES = (
    ("pnpm-lock.yaml", "pnpm"), ("bun.lockb", "bun"), ("bun.lock", "bun"), ("yarn.lock", "yarn"), ("package-lock.json", "npm"),
)
# package.json scripts / Makefile targets worth surfacing as verify commands.
_VERIFY_TARGETS = ("test", "tests", "lint", "typecheck", "check", "build", "fmt", "format")
_MAX_VERIFY_COMMANDS = 8
_MAX_FACT_FILE_BYTES = 256 * 1024
_GIT_TIMEOUT = 2.5
# Per-model edit-format steering toward the `patch` mode each family was trained on (unknown
# families get nothing). GPT/Codex get V4A for ALL edits incl. single-file: codex-rs ships
# apply_patch as its ONLY editor. Anthropic and most open-weight coders were RL'd on str_replace
# editors. Substrings match the model id; aligned with TOOL_USE_ENFORCEMENT_MODELS.
_EDIT_FORMAT_GUIDANCE: dict[str, tuple[tuple[str, ...], str]] = {
    "patch": (
        ("gpt", "codex"),
        "- Edit format: author new files with `write_file`; for edits to "
        "existing code use `patch` with `mode='patch'` (V4A diff) — including "
        "single-file edits. It's the edit format you handle most reliably.",
    ),
    "replace": (
        ("claude", "sonnet", "opus", "haiku",
         "gemini", "gemma", "deepseek", "qwen", "kimi", "glm", "grok",
         "hermes", "llama", "mistral", "devstral", "minimax"),
        "- Edit format: author new files with `write_file`; for edits to "
        "existing code prefer `patch` in `mode='replace'` — match a unique "
        "snippet and swap it. Reach for `mode='patch'` (V4A) only when an edit "
        "genuinely spans several files at once.",
    ),
}

# Operating brief. Tool names referenced here are in the coding toolset and _HERMES_CORE_TOOLS.
CODING_AGENT_GUIDANCE = (
    "You are a coding agent pairing with the user inside their codebase. "
    "Operate like a careful senior engineer.\n"
    "\n"
    "Gather context first:\n"
    "- Read the relevant files with `read_file` and locate code with "
    "`search_files` before changing anything. Trace a symbol to its definition "
    "and usages rather than guessing its shape.\n"
    "- Batch independent lookups: when several reads/searches don't depend on "
    "each other, issue them together in one turn instead of one at a time.\n"
    "- Never invent files, symbols, APIs, or imports. If you haven't seen it in "
    "the repo, go look. Don't assume a library is available — check the project "
    "manifest (pyproject.toml / package.json / Cargo.toml / go.mod) and how "
    "neighbouring files import it.\n"
    "\n"
    "Make changes through the tools, not the chat:\n"
    "- Edit with `patch`/`write_file`. Do NOT print code blocks to the user as "
    "a substitute for editing — apply the change, then summarise it. Only show "
    "code when the user explicitly asks to see it.\n"
    "- Match the project's existing style and conventions; AGENTS.md / "
    "CLAUDE.md / .cursorrules already in context win over your defaults. Touch "
    "only what the task needs — no drive-by refactors, renames, or reformatting "
    "— and add any imports/dependencies your code requires.\n"
    "- If an edit fails to apply, re-read the file to get the current exact "
    "contents before retrying — don't repeat a stale patch. If the same region "
    "fails twice, rewrite the enclosing function or file with `write_file` "
    "instead of attempting a third patch.\n"
    "\n"
    "Verify, and know when to stop:\n"
    "- Use `terminal` for git, builds, tests, and inspection. Run the relevant "
    "tests/linter/build and confirm they pass before claiming the work is done.\n"
    "- Terminal state persists across calls: current directory and exported "
    "environment variables carry forward. Activate a virtualenv or export setup "
    "vars once, then reuse that state instead of re-sourcing it before every "
    "test command.\n"
    "- Fix root causes, not symptoms: when you find a bug, check sibling call "
    "paths for the same flaw and fix the class, not just the reported site.\n"
    "- When fixing linter/type errors on a file, stop after about three "
    "attempts on the same file and ask the user rather than looping.\n"
    "- Track multi-step work with `todo_list`. Reference code as `path:line` instead "
    "of pasting whole files.\n"
    "\n"
    "Respect the user's repo: don't commit, push, or rewrite history unless "
    "asked, and never read, print, or commit secrets — leave `.env` and "
    "credential files alone unless the user explicitly asks. The Workspace "
    "block below is a snapshot from session start — re-run `git status`/"
    "`git branch` before relying on it. Be concise: lead with the change or "
    "answer, not a preamble."
)
_TODO_SENTENCE = (
    "- Track multi-step work with `todo_list`. Reference code as "
    "`path:line` instead of pasting whole files."
)
_NO_TODO_SENTENCE = "- Reference code as `path:line` instead of pasting whole files."
# Clearly non-coding skill categories (deny-list; coding-adjacent and custom ones keep full entries).
_NON_CODING_SKILL_CATEGORIES = (
    "apple", "communication", "cooking", "creative", "email", "finance", "gaming", "gifs", "health", "media",
    "music", "note-taking", "productivity", "shopping", "smart-home", "social-media", "travel", "yuanbao",
)
_MODE_ALIASES = {
    **dict.fromkeys(("focus", "strict", "lean"), "focus"),
    **dict.fromkeys(("on", "true", "yes", "1", "always"), "on"),
    **dict.fromkeys(("off", "false", "no", "0", "never"), "off"),
}


@dataclass(frozen=True)
class ContextProfile:
    """A named operating posture (pure data). ``toolset``: collapse target under ``focus``
    (``None`` keeps the platform default). ``compact_skill_categories``: DEMOTED to
    names-only under ``focus`` — deny-list, never hidden, so recall keeps working."""

    name: str
    toolset: Optional[str] = None
    guidance: str = ""
    model_hint: Optional[str] = None
    compact_skill_categories: tuple[str, ...] = ()


GENERAL_PROFILE = ContextProfile(name="general")
CODING_PROFILE = ContextProfile(
    name="coding", toolset=CODING_TOOLSET, guidance=CODING_AGENT_GUIDANCE, model_hint="coding",
    compact_skill_categories=_NON_CODING_SKILL_CATEGORIES,
)


# ── Detection helpers ───────────────────────────────────────────────────────

def _model_family(model: Optional[str]) -> Optional[str]:
    """Edit-format family key for a model id, or ``None`` (neutral wording applies)."""
    lowered = (model or "").lower()
    for family, (needles, _line) in _EDIT_FORMAT_GUIDANCE.items():
        if lowered and any(n in lowered for n in needles):
            return family
    return None


def _agent_config_value(config: Optional[dict[str, Any]], key: str, default: Any, *, readonly: bool) -> Any:
    """``config["agent"][key]``, loading config when none was passed."""
    if config is None:
        try:
            from hermes_cli.config import load_config, load_config_readonly
            config = load_config_readonly() if readonly else load_config()
        except Exception:
            config = {}
    return ((config or {}).get("agent", {}) or {}).get(key, default)


def _coding_mode(config: Optional[dict[str, Any]]) -> str:
    """Normalized ``agent.coding_context`` mode (auto/focus/on/off)."""
    raw = _agent_config_value(config, "coding_context", "auto", readonly=True)
    return _MODE_ALIASES.get(str(raw).strip().lower(), "auto")


def _resolve_cwd(cwd: Optional[str | Path]) -> Path:
    if cwd:
        return Path(cwd).expanduser()
    try:
        from agent.runtime_cwd import resolve_agent_cwd
        return resolve_agent_cwd()
    except Exception:
        return Path(os.getcwd())


def _git_root(cwd: Path) -> Optional[Path]:
    current = cwd.resolve()
    return next((p for p in (current, *current.parents) if (p / ".git").exists()), None)


def _home() -> Optional[Path]:
    try:
        return Path.home().resolve()
    except (OSError, RuntimeError):
        return None


def _marker_root(cwd: Path) -> Optional[Path]:
    """Nearest ancestor (≤6 levels) that looks like a project root, or ``None``. ``$HOME``
    and the shared temp root are skipped: a Makefile/AGENTS.md in the home dir is global
    config, and a stray manifest in /tmp must not flip every session under it."""
    current = cwd.resolve()
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
    except Exception:
        temp_root = None
    skip = (_home(), temp_root)
    for parent in (current, *current.parents)[:7]:
        if parent not in skip and any((parent / marker).exists() for marker in _PROJECT_MARKERS):
            return parent
    return None


def _has_code_files(root: Path) -> bool:
    """Bounded check for source files in the root and its immediate subdirs."""
    seen = 0
    stack = [(root, True)]
    while stack:
        directory, is_root = stack.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                seen += 1
                if seen > _CODE_SCAN_MAX_ENTRIES:
                    return False
                try:
                    if entry.is_file():
                        if os.path.splitext(entry.name)[1].lower() in _CODE_EXTENSIONS:
                            return True
                    elif is_root and entry.is_dir() and entry.name not in _CODE_SCAN_SKIP_DIRS and not entry.name.startswith("."):
                        stack.append((Path(entry.path), False))
                except OSError:
                    continue
    return False


def _detect_profile(mode: str, platform: str, cwd: Path) -> ContextProfile:
    """``auto``/``focus``: coding when the surface is interactive AND the cwd is a code
    workspace (project root, or a git repo that actually holds code; a repo rooted at
    ``$HOME`` is NOT a signal). ``on``/``off`` force. Not memoized: one gateway serves many cwds."""
    if mode == "off":
        return GENERAL_PROFILE
    if mode == "on":
        return CODING_PROFILE
    if platform and platform.strip().lower() not in INTERACTIVE_CODING_PLATFORMS:
        return GENERAL_PROFILE
    if _marker_root(cwd) is not None:
        return CODING_PROFILE
    git_root = _git_root(cwd)
    if git_root is not None and git_root != _home() and _has_code_files(git_root):
        return CODING_PROFILE
    return GENERAL_PROFILE


def _enabled_mcp_servers(config: Optional[dict[str, Any]]) -> list[str]:
    """Names of MCP servers the user has enabled — kept in the coding posture."""
    try:
        from hermes_cli.config import read_raw_config
        from hermes_cli.tools_config import _parse_enabled_flag
        servers = read_raw_config().get("mcp_servers") or {}
        return [
            str(name) for name, cfg in servers.items()
            if isinstance(cfg, dict) and _parse_enabled_flag(cfg.get("enabled", True), default=True)
        ]
    except Exception:
        return []


# ── RuntimeMode (the seam) ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RuntimeMode:
    """The resolved operating posture for a session; immutable, built once via
    :func:`resolve_runtime_mode` and never re-resolved mid-session (prompt cache).
    ``config_mode``: normalized ``agent.coding_context`` (toolset collapse gated on
    ``focus``); ``model``: steers edit-format guidance only; ``instructions``:
    ``agent.coding_instructions``."""

    profile: ContextProfile
    surface: str
    cwd: Path
    config_mode: str = "auto"
    model: Optional[str] = None
    instructions: str = ""

    @property
    def kind(self) -> str:
        return self.profile.name

    @property
    def is_coding(self) -> bool:
        return self.profile.name == CODING_PROFILE.name

    def toolset_selection(self, config: Optional[dict[str, Any]] = None) -> Optional[list[str]]:
        """Toolset list (only under ``focus``), or ``None`` to keep the platform default. Callers
        apply it only when the user hasn't pinned a selection (``--toolsets``, ``HERMES_TUI_TOOLSETS``)."""
        if self.config_mode != "focus" or self.profile.toolset is None:
            return None
        return [self.profile.toolset, *_enabled_mcp_servers(config)]

    def system_prompt_parts(self, valid_tool_names=None) -> tuple[list[str], list[str], list[str]]:
        """Return (prefix, workspace, trailing) posture blocks in the historical flat order —
        brief, snapshot, operator instructions — so prompt assembly can put a cache boundary
        before the snapshot without changing persisted bytes. The brief carries the model-family
        edit-format nudge (one cached string); ``valid_tool_names`` drops the ``todo_list``
        sentence when that tool isn't loaded; operator instructions ride their own block so
        the brief stays byte-stable."""
        if not self.is_coding:
            return [], [], []
        prefix: list[str] = []
        if self.profile.guidance:
            brief = self.profile.guidance
            if valid_tool_names is not None and "todo_list" not in valid_tool_names:
                brief = brief.replace(_TODO_SENTENCE, _NO_TODO_SENTENCE)
            family = _model_family(self.model)
            if family is not None:
                brief = f"{brief}\n{_EDIT_FORMAT_GUIDANCE[family][1]}"
            prefix.append(brief)
        workspace = build_coding_workspace_block(self.cwd)
        trailing = [f"Operator instructions (from config):\n{self.instructions}"] if self.instructions else []
        return prefix, [workspace] if workspace else [], trailing

    def system_blocks(self) -> list[str]:
        """Posture blocks as one flat list in historical order."""
        prefix, workspace, trailing = self.system_prompt_parts()
        return [*prefix, *workspace, *trailing]

    def compact_skill_categories(self) -> frozenset[str]:
        """Skill categories to demote to names-only in the skill index. Gated on ``focus``
        like the toolset collapse (index changes under ``auto`` proved too surprising).
        Demoted, never hidden: pruning caused silent capability loss."""
        if not self.is_coding or self.config_mode != "focus":
            return frozenset()
        return frozenset(self.profile.compact_skill_categories)


def resolve_runtime_mode(
    *, platform: Optional[str] = None, cwd: Optional[str | Path] = None, config: Optional[dict[str, Any]] = None,
    model: Optional[str] = None,
) -> RuntimeMode:
    """Resolve the operating posture once (a handful of ``stat`` calls) — the single entry
    point every domain should call; the result is safe to hold for the session. ``model``
    only steers edit-format guidance; ``agent.coding_instructions`` (str or list) becomes
    the trailing block so a user can pin workflow rules without editing the shipped brief."""
    resolved_cwd = _resolve_cwd(cwd)
    mode = _coding_mode(config)
    raw = _agent_config_value(config, "coding_instructions", "", readonly=False)
    items = raw if isinstance(raw, (list, tuple)) else [raw or ""]
    instructions = "\n".join(str(item).strip() for item in items if str(item).strip())
    return RuntimeMode(
        profile=_detect_profile(mode, (platform or "").strip().lower(), resolved_cwd),
        surface=platform or "",
        cwd=resolved_cwd,
        config_mode=mode,
        model=model,
        instructions=instructions,
    )


# ── Functional API (thin wrappers over RuntimeMode) ──────────────────────────

def is_coding_context(*, platform: Optional[str] = None, cwd: Optional[str | Path] = None, config: Optional[dict[str, Any]] = None) -> bool:
    """Whether Hermes should operate in its coding posture right now."""
    return resolve_runtime_mode(platform=platform, cwd=cwd, config=config).is_coding


def coding_selection(*, platform: Optional[str] = None, cwd: Optional[str | Path] = None, config: Optional[dict[str, Any]] = None) -> Optional[list[str]]:
    """Toolset selection for the coding posture (``None`` unless ``focus`` and active)."""
    return resolve_runtime_mode(platform=platform, cwd=cwd, config=config).toolset_selection(config)


def coding_system_prompt_parts(
    *, platform: Optional[str] = None, cwd: Optional[str | Path] = None, config: Optional[dict[str, Any]] = None,
    model: Optional[str] = None, valid_tool_names=None,
) -> tuple[list[str], list[str], list[str]]:
    """Return coding prefix, workspace snapshot, and trailing guidance."""
    mode = resolve_runtime_mode(platform=platform, cwd=cwd, config=config, model=model)
    return mode.system_prompt_parts(valid_tool_names=valid_tool_names)


def coding_compact_skill_categories(*, platform: Optional[str] = None, cwd: Optional[str | Path] = None, config: Optional[dict[str, Any]] = None) -> frozenset[str]:
    """Skill categories the active posture demotes to names-only (empty outside ``focus``)."""
    return resolve_runtime_mode(platform=platform, cwd=cwd, config=config).compact_skill_categories()


# ── git/workspace probe ─────────────────────────────────────────────────────

def _git(cwd: Path, *args: str) -> str:
    """``git -C <cwd> <args>`` → stripped stdout, or ``""`` on any failure. bounded_git_probe
    bounds post-kill cleanup on Windows — plain ``subprocess.run(timeout=...)`` deadlocked
    when a killed git left a suspended descendant holding the pipe handles.

    See #66037.
    """
    return bounded_git_probe(["git", "-C", str(cwd), *args], timeout=_GIT_TIMEOUT)


def _parse_status(porcelain: str) -> tuple[dict[str, str], dict[str, int]]:
    """Parse ``git status --porcelain=2 --branch`` into branch + counts."""
    branch: dict[str, str] = {}
    counts = {"staged": 0, "modified": 0, "untracked": 0, "conflicts": 0}
    for line in porcelain.splitlines():
        if line.startswith(("# branch.head", "# branch.upstream")):
            parts = line.split(maxsplit=2)
            branch[parts[1].removeprefix("branch.")] = parts[-1]
        elif line.startswith("# branch.ab"):
            parts = line.split()
            branch["ahead"], branch["behind"] = parts[2].lstrip("+"), parts[3].lstrip("-")
        elif line.startswith(("1 ", "2 ")):
            xy = line.split(maxsplit=2)[1]
            counts["staged"] += xy[0] != "."
            counts["modified"] += xy[1] != "."
        elif line.startswith(("u ", "? ")):
            counts["conflicts" if line[0] == "u" else "untracked"] += 1
    return branch, counts


def _read_small(path: Path) -> str:
    """Read a small text file, or ``""`` — never raises, never reads huge files."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FACT_FILE_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


@dataclass(frozen=True)
class ProjectFacts:
    """Structured project facts — exposed so non-prompt consumers (desktop verify UI) don't re-detect."""

    manifests: list[str]
    package_managers: list[str]
    verify_commands: list[str]
    context_files: list[str]


def detect_project_facts(root: Path) -> ProjectFacts:
    """Detect manifests, package manager(s), verify commands, context files (single source of truth)."""
    verify: list[str] = []
    if (root / "scripts" / "run_tests.sh").is_file():
        verify.append("scripts/run_tests.sh")
    if (root / "package.json").is_file():
        try:
            scripts = json.loads(_read_small(root / "package.json") or "{}").get("scripts") or {}
        except (json.JSONDecodeError, AttributeError):
            scripts = {}
        js_pm = next((pm for lock, pm in _JS_LOCKFILES if (root / lock).is_file()), "npm")
        verify.extend(f"{js_pm} run {name}" for name in _VERIFY_TARGETS if name in scripts)
    if (root / "pytest.ini").is_file() or "[tool.pytest" in _read_small(root / "pyproject.toml"):
        verify.append("pytest")
    makefile = _read_small(root / "Makefile")
    verify.extend(
        f"make {name}" for name in _VERIFY_TARGETS
        if makefile and re.search(rf"^{re.escape(name)}\s*:", makefile, re.MULTILINE)
    )
    return ProjectFacts(
        manifests=[m for m in _PROJECT_MARKERS if m not in _CONTEXT_FILES and (root / m).is_file()],
        package_managers=list(dict.fromkeys(
            pm for lock, pm in (*_PY_LOCKFILES, *_JS_LOCKFILES) if (root / lock).is_file()
        )),
        verify_commands=list(dict.fromkeys(verify))[:_MAX_VERIFY_COMMANDS],
        context_files=[c for c in _CONTEXT_FILES if (root / c).is_file()],
    )


def _workspace_roots(cwd: Optional[str | Path]) -> tuple[Optional[Path], Optional[Path]]:
    """(git_root, workspace_root) for *cwd*; workspace root is git root else marker root."""
    resolved = _resolve_cwd(cwd)
    git_root = _git_root(resolved)
    return git_root, git_root or _marker_root(resolved)


def project_facts_for(cwd: Optional[str | Path] = None) -> Optional[dict[str, Any]]:
    """Structured project facts for ``cwd`` (desktop verify UI) — ``None`` outside a workspace."""
    _, root = _workspace_roots(cwd)
    if root is None:
        return None
    f = detect_project_facts(root)
    return {
        "root": str(root),
        "manifests": f.manifests,
        "packageManagers": f.package_managers,
        "verifyCommands": f.verify_commands,
        "contextFiles": f.context_files,
    }


def build_coding_workspace_block(cwd: Optional[str | Path] = None) -> str:
    """Workspace snapshot for the system prompt (empty outside a workspace): git state when
    in a repo, plus project facts — so marker-only (non-git) projects still get one."""
    git_root, root = _workspace_roots(cwd)
    if root is None:
        return ""
    lines = [
        "Workspace (snapshot at session start — re-check with `git` before acting on it):",
        f"- Root: {root}",
    ]
    if git_root is not None:
        branch, counts = _parse_status(_git(root, "status", "--porcelain=2", "--branch"))
        head = branch.get("head", "")
        if head == "(detached)":
            lines.append("- Branch: (detached HEAD)")
        elif head:
            ahead, behind = branch.get("ahead", "0"), branch.get("behind", "0")
            upstream = f" \u2192 {branch['upstream']}" if branch.get("upstream") else ""
            ab = f" (ahead {ahead}, behind {behind})" if upstream and (ahead, behind) != ("0", "0") else ""
            lines.append(f"- Branch: {head}{upstream}{ab}")

        # Linked worktree: say so (branches/stashes are shared state) but do NOT expose
        # the primary tree path — a second absolute path makes the model run commands
        # in the wrong directory.
        git_dir, common_dir = _git(root, "rev-parse", "--git-dir"), _git(root, "rev-parse", "--git-common-dir")
        if git_dir and common_dir and Path(git_dir).resolve() != Path(common_dir).resolve():
            lines.append("- Worktree: linked (git state shared with primary tree)")

        dirty = [f"{n} {label}" for label, n in counts.items() if n]
        lines.append(f"- Status: {', '.join(dirty) if dirty else 'clean'}")

        recent = _git(root, "log", "-3", "--pretty=%h %s")
        if recent:
            lines.extend(["- Recent commits:", *(f"    {c}" for c in recent.splitlines())])

    f = detect_project_facts(root)
    if f.manifests:
        managers = f" ({'/'.join(f.package_managers)})" if f.package_managers else ""
        lines.append(f"- Project: {', '.join(f.manifests[:6])}{managers}")
    if f.verify_commands:
        lines.append(f"- Verify: {'; '.join(f.verify_commands)}")
    if f.context_files:
        lines.append(f"- Context files: {', '.join(f.context_files)}")
    return "\n".join(lines)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def coding_system_blocks(
    *,
    platform: Optional[str] = None,
    cwd: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
    model: Optional[str] = None,
) -> list[str]:
    """Stable system-prompt blocks for the current posture (empty when general).

    ``model`` steers the brief's edit-format nudge toward the model's family.
    """
    return resolve_runtime_mode(
        platform=platform, cwd=cwd, config=config, model=model
    ).system_blocks()

_PROFILES: dict[str, ContextProfile] = {
    GENERAL_PROFILE.name: GENERAL_PROFILE,
    CODING_PROFILE.name: CODING_PROFILE,
}

def get_profile(name: str) -> ContextProfile:
    """Return a registered profile, falling back to ``general``."""
    return _PROFILES.get(name, GENERAL_PROFILE)
# ---- END PLUGIN-COMPAT ----
