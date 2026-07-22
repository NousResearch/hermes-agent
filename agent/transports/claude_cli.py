"""Claude CLI (`claude -p`) subprocess transport — Phase 2b multi-turn.

Speaks Claude Code's print-mode stream-json protocol: spawn
``claude -p --output-format stream-json ...``, feed a single user prompt,
consume JSONL events from stdout until a terminal ``result`` event (or
subprocess exit).

Phase 2a Hermes MCP tool bridge (unchanged):
  * ``--mcp-config`` pointing at ``hermes_tools_mcp_server`` (stdio)
  * ``--allowedTools mcp__hermes-tools__*`` so Claude may call Hermes tools
  * ``--disallowedTools`` for native Bash/Edit/Write/Read (Hermes tools
    only for fs/exec — no double-agent). Do **not** pass ``--tools ""``:
    that disables MCP tools too.
  * Permission model: ``--permission-mode bypassPermissions`` only after
    the allowlist/denylist has restricted the surface to Hermes MCP tools
    (replaces Phase 1's unrestricted ``--dangerously-skip-permissions``)

Phase 2b multi-turn (OpenClaw/codex-style session resume):
  * First turn in a Hermes conversation: ``--session-id <uuid>`` so Claude
    persists a named session under ``~/.claude/projects/<cwd-encoded>/``.
  * Subsequent turns: ``--resume <uuid>`` + only the new user message;
    Claude owns prior history + native compaction.
  * Never pass ``--no-session-persistence`` — resume requires disk files.
  * Stable ``cwd`` across turns so session files resolve.

Tool round-trip is **internal to** ``claude -p``: Claude spawns the MCP
server, executes Hermes tools via stdio MCP, and streams tool_use /
tool_result events on stdout. Hermes hosts the server + projects those
events into the UI; the runtime does **not** proxy tool calls.

Per-turn wire shape remains positional prompt + stdin=DEVNULL; multi-turn
context is carried by Claude's session, not by replaying Hermes history.

Clean-env + non-rotating setup token model is unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


# ---------------------------------------------------------------------------
# Clean env for Claude Code subprocess
# ---------------------------------------------------------------------------
#
# Confirmed (live step-0 test 2026-07-19): Claude Code injects pollution
# (ANTHROPIC_BASE_URL, ANTHROPIC_*, CLAUDE_CODE_*, CLAUDECODE, ...) into the
# parent shell when Hermes is itself driven by Claude. A subprocess that
# inherits that env hits the wrong endpoint / wrong auth. The rotating
# ~/.claude OAuth login also fails from a subprocess ("OAuth session expired
# and could not be refreshed"). The NON-rotating setup token (sk-ant-oat)
# injected as CLAUDE_CODE_OAUTH_TOKEN into a clean env bills against base Max.
#
# Mirrors OpenClaw's CLAUDE_CLI_CLEAR_ENV approach: start from an allowlist
# of identity/path vars, never copy the polluted parent env wholesale.

# Exact names that must never reach the child (even if present in allowlist).
CLAUDE_CLI_CLEAR_ENV_NAMES: frozenset[str] = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_TOKEN",
        "ANTHROPIC_API_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDECODE",
        "CLAUDE_CODE_OAUTH_TOKEN",  # re-injected from the resolved setup token
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SESSION",
        "CLAUDE_CODE_SSE_PORT",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_EFFORT",
        "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "CLAUDE_CODE_SIMPLE",
        "CLAUDE_CODE_SAFE_MODE",
    }
)

# Prefixes scrubbed from any env we start from (parent pollution set).
CLAUDE_CLI_CLEAR_ENV_PREFIXES: tuple[str, ...] = (
    "ANTHROPIC_",
    "CLAUDE_CODE_",
    "CLAUDE_PREVIEW_",
    "CLAUDE_AGENT_",
)

# Identity / path vars kept from the parent so the child can resolve home,
# binaries, and temp dirs. Everything else is dropped (env -i style).
CLAUDE_CLI_KEEP_ENV: frozenset[str] = frozenset(
    {
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SHELL",
        "XDG_RUNTIME_DIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        # macOS keychain / locale helpers Claude Code may need for process
        # startup (NOT for OAuth — we inject the setup token explicitly).
        "SSH_AUTH_SOCK",
    }
)


def _is_pollution_key(key: str) -> bool:
    if key in CLAUDE_CLI_CLEAR_ENV_NAMES:
        return True
    if key == "CLAUDECODE":
        return True
    if key.startswith("CLAUDE_EFFORT"):
        return True
    for prefix in CLAUDE_CLI_CLEAR_ENV_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


def build_claude_cli_clean_env(
    *,
    oauth_token: str,
    base_env: Optional[dict[str, str]] = None,
    extra: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Build a clean env for ``claude -p`` with the setup token injected.

    Starts from ``base_env`` (default: ``os.environ``) but only keeps the
    allowlisted identity/path vars. Strips every Claude-Code / Anthropic
    pollution key, then injects ``CLAUDE_CODE_OAUTH_TOKEN=<oauth_token>``.

    Never reads or forwards the rotating ``~/.claude`` OAuth login — that
    fails from a subprocess. The setup token (sk-ant-oat) is non-rotating
    and fork-safe.
    """
    if not oauth_token or not str(oauth_token).strip():
        raise ValueError(
            "claude_cli clean env requires a non-empty setup token "
            "(CLAUDE_CODE_OAUTH_TOKEN / resolve_anthropic_token)"
        )
    source = base_env if base_env is not None else os.environ
    clean: dict[str, str] = {}
    for key in CLAUDE_CLI_KEEP_ENV:
        val = source.get(key)
        if val is None or val == "":
            continue
        if _is_pollution_key(key):
            continue
        clean[key] = str(val)
    # Hard-strip any pollution that snuck into the keep set (defensive).
    for key in list(clean):
        if _is_pollution_key(key):
            clean.pop(key, None)
    clean["CLAUDE_CODE_OAUTH_TOKEN"] = str(oauth_token).strip()
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            if _is_pollution_key(k) and k != "CLAUDE_CODE_OAUTH_TOKEN":
                continue
            clean[str(k)] = str(v)
    return clean


def resolve_claude_bin(explicit: Optional[str] = None) -> str:
    """Locate the ``claude`` binary. Prefer explicit path, then PATH, then ~/.local/bin."""
    if explicit:
        return explicit
    found = shutil.which("claude")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/claude")
    if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
        return fallback
    return "claude"


# ---------------------------------------------------------------------------
# Hermes MCP tool bridge (Phase 2a)
# ---------------------------------------------------------------------------
#
# Mirrors hermes_cli.codex_runtime_plugin_migration._build_hermes_tools_mcp_entry
# for the Claude CLI ``--mcp-config`` JSON shape. Claude spawns the MCP server
# as a stdio child and owns the full tool round-trip; Hermes only hosts the
# server process and projects tool events from stream-json.

from agent.transports.hermes_tools_mcp_server import (  # noqa: E402
    AGENT_LOOP_TOOLS_EXCLUDED,
    CLAUDE_EXPOSED_TOOLS,
    MCP_SERVER_NAME,
    get_exposed_tools,
)

# Claude Code prefixes MCP tools as mcp__<server>__<tool>.
# Server name "hermes-tools" → mcp__hermes-tools__terminal, etc.
HERMES_MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"
HERMES_MCP_ALLOWED_TOOLS_GLOB = f"{HERMES_MCP_TOOL_PREFIX}*"

# Claude native filesystem / exec / search tools that MUST NOT freestyle
# the real filesystem. Hermes tools (via MCP) replace these.
CLAUDE_NATIVE_FS_EXEC_TOOLS: tuple[str, ...] = (
    "Bash",
    "Edit",
    "Write",
    "Read",
    "Glob",
    "Grep",
    "NotebookEdit",
    "MultiEdit",
    "WebSearch",
    "WebFetch",
    # Agent / task spawning that could bypass Hermes tooling.
    "Task",
    "TodoWrite",
    "TodoRead",
    "Skill",
)

# Default permission mode once the surface is restricted to Hermes MCP tools.
# bypassPermissions auto-allows the allowlisted MCP tools headlessly without
# re-opening the full native tool surface (those stay disallowed).
#
# IMPORTANT: do NOT pass ``--tools ""``. Claude Code treats that as "disable
# all tools" — including MCP tools — so Hermes tools never appear and the
# model only narrates. Rely on --allowedTools / --disallowedTools alone.
CLAUDE_CLI_PERMISSION_MODE = "bypassPermissions"

# Default max agentic turns for a tool-using print-mode session.
CLAUDE_CLI_DEFAULT_MAX_TURNS = 40


def _looks_like_test_tempdir(path: str) -> bool:
    """Heuristic: refuse to bake a pytest tempdir into MCP env."""
    if not path:
        return False
    needles = (
        "pytest-of-",
        "/pytest-",
        "/tmp/pytest",
        "/private/var/folders/",
    )
    normalized = path.lower()
    return any(needle in normalized for needle in needles)


def resolve_hermes_repo_root() -> str:
    """Repo root containing ``agent/transports/`` (this file's parents)."""
    return str(Path(__file__).resolve().parents[2])


def resolve_mcp_python_executable(explicit: Optional[str] = None) -> str:
    """Python that can import ``mcp`` + this worktree's Hermes packages.

    Claude spawns the MCP server with only the env we put in ``mcp-config``.
    Using a *resolved* base-interpreter path (``Path.resolve()`` on a venv
    symlink, or a bare Homebrew python) drops site-packages and the server
    dies with ``No module named 'mcp'``. Prefer ``sys.executable`` as-is so
    venv activation via argv[0] still works, and never realpath the venv.
    """
    if explicit:
        return explicit
    # sys.executable is already the active interpreter path Hermes is on.
    # Keep it verbatim — do not os.path.realpath / Path.resolve.
    return sys.executable or "python3"


def _venv_site_packages(python_executable: str) -> Optional[str]:
    """Best-effort site-packages dir for a venv python path."""
    try:
        exe = Path(python_executable)
        # .../.venv/bin/python → .../.venv/lib/pythonX.Y/site-packages
        venv_root = exe.parent.parent
        if not (venv_root / "pyvenv.cfg").is_file():
            # Active interpreter may be a venv even if command path differs.
            if getattr(sys, "prefix", None) and sys.prefix != getattr(
                sys, "base_prefix", sys.prefix
            ):
                venv_root = Path(sys.prefix)
            else:
                return None
        lib = venv_root / "lib"
        if not lib.is_dir():
            return None
        for child in sorted(lib.iterdir()):
            if child.name.startswith("python"):
                site = child / "site-packages"
                if site.is_dir():
                    return str(site)
    except Exception:
        return None
    return None


def build_hermes_tools_mcp_server_entry(
    *,
    profile: str = "claude",
    hermes_home: Optional[str] = None,
    python_executable: Optional[str] = None,
    pythonpath: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Build one Claude ``mcpServers.<name>`` stdio entry for hermes-tools.

    Mirrors codex's ``_build_hermes_tools_mcp_entry`` but returns the JSON
    shape Claude's ``--mcp-config`` expects (command/args/env).
    """
    env: dict[str, str] = {}

    hh = hermes_home if hermes_home is not None else (os.environ.get("HERMES_HOME") or "")
    if hh and _looks_like_test_tempdir(hh):
        hh = ""
    if hh:
        env["HERMES_HOME"] = hh

    py = resolve_mcp_python_executable(python_executable)

    # Ensure the MCP child can import this worktree's agent package AND the
    # active venv's site-packages (mcp SDK, hermes deps). Claude replaces the
    # child env with only this dict — belt-and-suspenders PYTHONPATH.
    pp_parts: list[str] = []
    if pythonpath is not None:
        if pythonpath:
            pp_parts.append(pythonpath)
    else:
        existing = os.environ.get("PYTHONPATH") or ""
        if existing:
            pp_parts.append(existing)
    repo_root = resolve_hermes_repo_root()
    if repo_root and repo_root not in pp_parts:
        # Prepend so worktree modules win over any installed package.
        pp_parts.insert(0, repo_root)
    site_pkgs = _venv_site_packages(py)
    if site_pkgs and site_pkgs not in pp_parts:
        # After repo root so local modules still win, but before system paths.
        insert_at = 1 if pp_parts and pp_parts[0] == repo_root else 0
        pp_parts.insert(insert_at, site_pkgs)
    if pp_parts:
        env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    # Preserve venv marker when Hermes itself is running inside one so child
    # tools that re-exec python inherit the same environment.
    if getattr(sys, "prefix", None) and sys.prefix != getattr(
        sys, "base_prefix", sys.prefix
    ):
        env.setdefault("VIRTUAL_ENV", sys.prefix)

    env["HERMES_QUIET"] = "1"
    env["HERMES_REDACT_SECRETS"] = "true"
    env["HERMES_TOOLS_MCP_PROFILE"] = (profile or "claude").strip().lower()
    # Fail-fast browser CDP probes during MCP server import (saves ~1s of
    # connection latency that races Claude's first tool_use).
    env.setdefault("BROWSER_CDP_URL", "")

    # Identity/path for terminal tool + child processes.
    for key in ("HOME", "PATH", "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP", "SHELL", "LANG", "LC_ALL"):
        val = os.environ.get(key)
        if val:
            env[key] = val

    if extra_env:
        for k, v in extra_env.items():
            if v is not None:
                env[str(k)] = str(v)

    return {
        "command": py,
        "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
        "env": env,
    }


def build_hermes_mcp_config(
    *,
    profile: str = "claude",
    hermes_home: Optional[str] = None,
    python_executable: Optional[str] = None,
    pythonpath: Optional[str] = None,
    extra_env: Optional[dict[str, str]] = None,
    server_name: str = MCP_SERVER_NAME,
) -> dict[str, Any]:
    """Full Claude ``--mcp-config`` JSON object (``mcpServers`` wrapper)."""
    entry = build_hermes_tools_mcp_server_entry(
        profile=profile,
        hermes_home=hermes_home,
        python_executable=python_executable,
        pythonpath=pythonpath,
        extra_env=extra_env,
    )
    return {"mcpServers": {server_name: entry}}


def write_hermes_mcp_config_file(
    config: Optional[dict[str, Any]] = None,
    *,
    path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Serialize MCP config JSON to a temp (or given) file; return path."""
    cfg = config if config is not None else build_hermes_mcp_config(**kwargs)
    if path is None:
        fd, path = tempfile.mkstemp(prefix="hermes-claude-mcp-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
    return path


def hermes_mcp_allowed_tools(
    *,
    server_name: str = MCP_SERVER_NAME,
    tool_names: Optional[tuple[str, ...]] = None,
    use_glob: bool = True,
) -> list[str]:
    """Claude ``--allowedTools`` values for the Hermes MCP surface."""
    if use_glob:
        return [f"mcp__{server_name}__*"]
    names = tool_names if tool_names is not None else get_exposed_tools("claude")
    return [f"mcp__{server_name}__{n}" for n in names]


def claude_native_disallowed_tools() -> list[str]:
    """Claude native tools blocked so the model must use Hermes MCP tools."""
    return list(CLAUDE_NATIVE_FS_EXEC_TOOLS)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCliError(RuntimeError):
    """Raised on claude_cli runtime failures (is_error result, timeout, exit).

    Message is shaped so Hermes' existing ``classify_api_error`` / fallback
    path can treat it like any other provider failure when the turn runner
    re-raises.
    """

    message: str
    is_error: bool = True
    exit_code: Optional[int] = None
    stderr_tail: str = ""
    result_event: Optional[dict[str, Any]] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = self.message
        if self.exit_code is not None:
            base = f"{base} (exit={self.exit_code})"
        if self.stderr_tail:
            base = f"{base}\nclaude stderr (tail):\n{self.stderr_tail}"
        return base


@dataclass
class ClaudeCliConcurrencyError(ClaudeCliError):
    """Host-wide ``claude -p`` concurrency cap saturated after wait timeout.

    Conversation loop treats this as classifiable saturation (rate_limit-
    flavoured) and activates the profile's fallback chain (grok/gpt) rather
    than hanging or spawning past the cap. See
    ``agent.transports.claude_cli_concurrency``.
    """

    max_concurrent: Optional[int] = None
    timeout_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Client: spawn one `claude -p` and stream JSONL
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCliSpawnConfig:
    """Args for a single ``claude -p`` invocation (Phase 2a MCP + 2b session)."""

    model: str
    prompt: str
    system_prompt: Optional[str] = None
    claude_bin: str = "claude"
    cwd: Optional[str] = None
    # Phase 2a: Hermes MCP bridge (default on). When False, reverts to a
    # pure-text skeleton without tools (legacy Phase 1 behaviour minus
    # dangerously-skip-permissions).
    enable_hermes_mcp: bool = True
    mcp_config_path: Optional[str] = None
    mcp_config: Optional[dict[str, Any]] = None
    allowed_tools: Optional[list[str]] = None
    disallowed_tools: Optional[list[str]] = None
    # None (default): do NOT pass --tools. Claude's ``--tools ""`` disables
    # *all* tools including MCP, which is what broke Phase 2a tool execution.
    # Pass a non-None string only for explicit built-in-tool allowlists in tests.
    tools_override: Optional[str] = None
    permission_mode: str = CLAUDE_CLI_PERMISSION_MODE
    # Phase 1 flag — kept for tests/back-compat but ignored when MCP is on.
    # When MCP is off, we still refuse to use dangerously-skip-permissions
    # by default (permission_mode handles headless allow).
    dangerously_skip_permissions: bool = False
    setting_sources: str = "user"
    strict_mcp_config: bool = True
    max_turns: int = CLAUDE_CLI_DEFAULT_MAX_TURNS
    timeout_seconds: float = 600.0
    extra_args: Optional[list[str]] = None
    hermes_mcp_profile: str = "claude"
    # Phase 2b multi-turn session (Claude Code 2.1.x):
    #   create: ``--session-id <uuid>`` (must be a valid UUID)
    #   resume: ``--resume <uuid>`` (mutually exclusive with session_id)
    # Never pass ``--no-session-persistence`` — resume needs disk files.
    session_id: Optional[str] = None
    resume: Optional[str] = None


class ClaudeCliClient:
    """Spawn ``claude -p`` once and yield stream-json events from stdout.

    Threading model matches codex_app_server: caller drives the turn
    synchronously; a stderr reader thread captures diagnostics; stdout is
    read on the caller thread as a line iterator.
    """

    def __init__(
        self,
        *,
        oauth_token: str,
        env: Optional[dict[str, str]] = None,
        claude_bin: Optional[str] = None,
    ) -> None:
        self._oauth_token = oauth_token
        self._claude_bin = resolve_claude_bin(claude_bin)
        self._env = env or build_claude_cli_clean_env(oauth_token=oauth_token)
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._system_prompt_file: Optional[str] = None
        self._mcp_config_file: Optional[str] = None
        self._owned_mcp_config_file = False
        self._closed = False

    # ---------- build argv ----------

    def build_argv(self, cfg: ClaudeCliSpawnConfig) -> list[str]:
        """Construct the ``claude -p`` argv for a Hermes-MCP (+ multi-turn) turn."""
        bin_path = resolve_claude_bin(cfg.claude_bin or self._claude_bin)
        argv = [
            bin_path,
            "-p",
            "--model",
            cfg.model,
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--setting-sources",
            cfg.setting_sources or "user",
        ]

        if cfg.max_turns and cfg.max_turns > 0:
            argv.extend(["--max-turns", str(int(cfg.max_turns))])

        # --- Phase 2b session create / resume (OpenClaw sessionMode: always) ---
        # Resume and session-id are mutually exclusive. Prefer resume when both
        # are set so a caller can't accidentally fork a new empty session.
        resume_id = (cfg.resume or "").strip() or None
        create_id = (cfg.session_id or "").strip() or None
        if resume_id:
            argv.extend(["--resume", resume_id])
        elif create_id:
            argv.extend(["--session-id", create_id])
        # Intentionally never pass --no-session-persistence.

        # --- Permission model (replaces --dangerously-skip-permissions) ---
        # With Hermes MCP on: restrict to MCP allowlist, block native fs/exec,
        # and use a headless permission mode that auto-allows the allowlisted
        # Hermes tools only.
        if cfg.enable_hermes_mcp:
            mcp_path = cfg.mcp_config_path
            if not mcp_path:
                # Materialize config now so argv is self-contained.
                mcp_path = write_hermes_mcp_config_file(
                    cfg.mcp_config,
                    profile=cfg.hermes_mcp_profile,
                )
                self._mcp_config_file = mcp_path
                self._owned_mcp_config_file = True
            else:
                self._mcp_config_file = mcp_path
                self._owned_mcp_config_file = False

            argv.extend(["--mcp-config", mcp_path])
            if cfg.strict_mcp_config:
                argv.append("--strict-mcp-config")

            allowed = cfg.allowed_tools
            if allowed is None:
                allowed = hermes_mcp_allowed_tools()
            if allowed:
                # Claude accepts space-separated tool specs after the flag.
                argv.append("--allowedTools")
                argv.extend(list(allowed))

            disallowed = cfg.disallowed_tools
            if disallowed is None:
                disallowed = claude_native_disallowed_tools()
            if disallowed:
                argv.append("--disallowedTools")
                argv.extend(list(disallowed))

            # Optional built-in tool allowlist. Default None — never pass
            # ``--tools ""`` (that drops MCP tools too; confirmed 2026-07-19).
            if cfg.tools_override is not None:
                argv.extend(["--tools", cfg.tools_override])

            if cfg.permission_mode:
                argv.extend(["--permission-mode", cfg.permission_mode])
        else:
            # Pure-text path: still avoid dangerously-skip-permissions unless
            # explicitly requested for a legacy test harness.
            if cfg.dangerously_skip_permissions:
                argv.append("--dangerously-skip-permissions")
            elif cfg.permission_mode:
                argv.extend(["--permission-mode", cfg.permission_mode])

        if cfg.system_prompt:
            # Prefer file form so large Hermes system prompts don't blow argv.
            fd, path = tempfile.mkstemp(prefix="hermes-claude-sys-", suffix=".txt")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(cfg.system_prompt)
                self._system_prompt_file = path
                argv.extend(["--append-system-prompt-file", path])
            except Exception:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                self._system_prompt_file = None
                # Fallback: inline (may truncate on very large prompts).
                argv.extend(["--append-system-prompt", cfg.system_prompt])
        if cfg.extra_args:
            argv.extend(cfg.extra_args)
        # Positional prompt last (claude -p "<prompt>").
        argv.append(cfg.prompt)
        return argv

    # ---------- lifecycle ----------

    def spawn(self, cfg: ClaudeCliSpawnConfig) -> subprocess.Popen:
        """Start the subprocess. Caller reads stdout via ``iter_stdout_lines``."""
        if self._proc is not None:
            raise RuntimeError("claude_cli client already has a live subprocess")
        argv = self.build_argv(cfg)
        cwd = cfg.cwd or tempfile.mkdtemp(prefix="hermes-claude-cli-")
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=self._env,
            bufsize=0,
        )
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stderr_reader.start()
        return self._proc

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw in iter(proc.stderr.readline, b""):
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    continue
                with self._stderr_lock:
                    self._stderr_lines.append(line)
                    # Cap buffer so a noisy child cannot unbounded-grow memory.
                    if len(self._stderr_lines) > 400:
                        self._stderr_lines = self._stderr_lines[-200:]
        except Exception:
            pass

    def stderr_tail(self, n: int = 20) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def iter_stdout_lines(self, *, timeout: float = 600.0) -> Iterator[str]:
        """Yield decoded stdout lines until EOF or timeout.

        On timeout the subprocess is kill()'d and a ClaudeCliError is raised.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("claude_cli client not spawned")
        deadline = time.monotonic() + timeout
        # Blocking readline on a pipe cannot be interrupted by a pure-Python
        # deadline without a reader thread. Use a small helper thread + queue
        # so we can kill the child when the watchdog fires.
        import queue as _queue

        q: _queue.Queue = _queue.Queue()
        sentinel = object()

        def _reader() -> None:
            try:
                for raw in iter(proc.stdout.readline, b""):
                    try:
                        line = raw.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    q.put(line)
            finally:
                q.put(sentinel)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.kill()
                tail = "\n".join(self.stderr_tail(20))
                raise ClaudeCliError(
                    message=(
                        f"claude_cli turn timed out after {timeout:.0f}s"
                    ),
                    exit_code=None,
                    stderr_tail=tail,
                )
            try:
                item = q.get(timeout=min(0.5, remaining))
            except _queue.Empty:
                # Child died without more output?
                if proc.poll() is not None and q.empty():
                    break
                continue
            if item is sentinel:
                break
            yield item  # type: ignore[misc]

    def wait(self, timeout: float = 5.0) -> int:
        proc = self._proc
        if proc is None:
            return 0
        try:
            return int(proc.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            self.kill()
            return int(proc.poll() or -1)

    def kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.kill()
        if self._system_prompt_file:
            try:
                os.unlink(self._system_prompt_file)
            except OSError:
                pass
            self._system_prompt_file = None
        if self._owned_mcp_config_file and self._mcp_config_file:
            try:
                os.unlink(self._mcp_config_file)
            except OSError:
                pass
            self._mcp_config_file = None
            self._owned_mcp_config_file = False

    def __enter__(self) -> "ClaudeCliClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Stream-json line decode helper (used by projector + tests)
# ---------------------------------------------------------------------------


def parse_stream_json_line(line: str) -> Optional[dict[str, Any]]:
    """Parse one stdout line as a stream-json event. Returns None for blanks/noise."""
    text = (line or "").strip()
    if not text:
        return None
    # Claude may occasionally emit non-JSON diagnostic lines when --verbose.
    if not text.startswith("{"):
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def strip_hermes_mcp_tool_prefix(name: str) -> str:
    """Map ``mcp__hermes-tools__terminal`` → ``terminal`` for UI display."""
    if not isinstance(name, str) or not name:
        return name or "unknown"
    prefix = HERMES_MCP_TOOL_PREFIX
    if name.startswith(prefix):
        return name[len(prefix):] or name
    # Generic mcp__server__tool
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3 and parts[2]:
            return parts[2]
    return name


__all__ = [
    "AGENT_LOOP_TOOLS_EXCLUDED",
    "CLAUDE_CLI_CLEAR_ENV_NAMES",
    "CLAUDE_CLI_CLEAR_ENV_PREFIXES",
    "CLAUDE_CLI_DEFAULT_MAX_TURNS",
    "CLAUDE_CLI_KEEP_ENV",
    "CLAUDE_CLI_PERMISSION_MODE",
    "CLAUDE_EXPOSED_TOOLS",
    "CLAUDE_NATIVE_FS_EXEC_TOOLS",
    "ClaudeCliClient",
    "ClaudeCliConcurrencyError",
    "ClaudeCliError",
    "ClaudeCliSpawnConfig",
    "HERMES_MCP_ALLOWED_TOOLS_GLOB",
    "HERMES_MCP_TOOL_PREFIX",
    "MCP_SERVER_NAME",
    "build_claude_cli_clean_env",
    "build_hermes_mcp_config",
    "build_hermes_tools_mcp_server_entry",
    "claude_native_disallowed_tools",
    "hermes_mcp_allowed_tools",
    "parse_stream_json_line",
    "resolve_claude_bin",
    "resolve_hermes_repo_root",
    "resolve_mcp_python_executable",
    "strip_hermes_mcp_tool_prefix",
    "write_hermes_mcp_config_file",
]
