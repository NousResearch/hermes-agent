# Contributing to Hermes Agent

Thank you for contributing to Hermes Agent! This guide covers everything you need: setting up your dev environment, understanding the architecture, deciding what to build, and getting your PR merged.

---

## Contribution Priorities

We value contributions in this order:

1. **Bug fixes** — crashes, incorrect behavior, data loss. Always top priority.
2. **Cross-platform compatibility** — macOS, different Linux distros, and WSL2 on Windows. We want Hermes to work everywhere.
3. **Security hardening** — shell injection, prompt injection, path traversal, privilege escalation. See [Security](#security-considerations).
4. **Performance and robustness** — retry logic, error handling, graceful degradation.
5. **New skills** — but only broadly useful ones. See [Should it be a Skill or a Tool?](#should-it-be-a-skill-or-a-tool)
6. **New tools** — rarely needed. Most capabilities should be skills. See below.
7. **Documentation** — fixes, clarifications, new examples.

---

## Should it be a Skill or a Tool?

This is the most common question for new contributors. The answer is almost always **skill**.

### Make it a Skill when:

- The capability can be expressed as instructions + shell commands + existing tools
- It wraps an external CLI or API that the agent can call via `terminal` or `web_extract`
- It doesn't need custom Python integration or API key management baked into the agent
- Examples: arXiv search, git workflows, Docker management, PDF processing, email via CLI tools

### Make it a Tool when:

- It requires end-to-end integration with API keys, auth flows, or multi-component configuration managed by the agent harness
- It needs custom processing logic that must execute precisely every time (not "best effort" from LLM interpretation)
- It handles binary data, streaming, or real-time events that can't go through the terminal
- Examples: browser automation (Browserbase session management), TTS (audio encoding + platform delivery), vision analysis (base64 image handling)

### Should the Skill be bundled?

Bundled skills (in `skills/`) ship with every Hermes install. They should be **broadly useful to most users**:

- Document handling, web research, common dev workflows, system administration
- Used regularly by a wide range of people

If your skill is official and useful but not universally needed (e.g., a paid service integration, a heavyweight dependency), put it in **`optional-skills/`** — it ships with the repo but isn't activated by default. Users can discover it via `hermes skills browse` (labeled "official") and install it with `hermes skills install` (no third-party warning, builtin trust).

If your skill is specialized, community-contributed, or niche, it's better suited for a **Skills Hub** — upload it to a skills registry and share it in the [Nous Research Discord](https://discord.gg/NousResearch). Users can install it with `hermes skills install`.

---

## Development Setup

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Git** | With `--recurse-submodules` support, and the `git-lfs` extension installed |
| **Python 3.11+** | uv will install it if missing |
| **uv** | Fast Python package manager ([install](https://docs.astral.sh/uv/)) |
| **Node.js 20+** | Optional — needed for browser tools and WhatsApp bridge (matches root `package.json` engines) |

### Clone and install

```bash
git clone --recurse-submodules https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create venv with Python 3.11
uv venv venv --python 3.11
export VIRTUAL_ENV="$(pwd)/venv"

# Install with all extras (messaging, cron, CLI menus, dev tools)
uv pip install -e ".[all,dev]"

# Optional: RL training submodule
# git submodule update --init tinker-atropos && uv pip install -e "./tinker-atropos"

# Optional: browser tools
npm install
```

### Configure for development

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env

# Add at minimum an LLM provider key:
echo "OPENROUTER_API_KEY=***" >> ~/.hermes/.env
```

### Run

```bash
# Symlink for global access
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes

# Verify
hermes doctor
hermes chat -q "Hello"
```

### Run tests

```bash
# Preferred — matches CI (hermetic env, 4 xdist workers); see AGENTS.md
scripts/run_tests.sh

# Alternative (activate the venv first). The wrapper is still recommended
# for parity with GitHub Actions before you open a PR:
pytest tests/ -v
```

---

## Project Structure

```
hermes-agent/
├── run_agent.py              # AIAgent class — core conversation loop, tool dispatch, session persistence
├── cli.py                    # HermesCLI class — interactive TUI, prompt_toolkit integration
├── model_tools.py            # Tool orchestration (thin layer over tools/registry.py)
├── toolsets.py               # Tool groupings and presets (hermes-cli, hermes-telegram, etc.)
├── hermes_state.py           # SQLite session database with FTS5 full-text search, session titles
├── batch_runner.py           # Parallel batch processing for trajectory generation
│
├── agent/                    # Agent internals (extracted modules)
│   ├── prompt_builder.py         # System prompt assembly (identity, skills, context files, memory)
│   ├── context_compressor.py     # Auto-summarization when approaching context limits
│   ├── auxiliary_client.py       # Resolves auxiliary OpenAI clients (summarization, vision)
│   ├── display.py                # KawaiiSpinner, tool progress formatting
│   ├── model_metadata.py         # Model context lengths, token estimation
│   └── trajectory.py             # Trajectory saving helpers
│
├── hermes_cli/               # CLI command implementations
│   ├── main.py                   # Entry point, argument parsing, command dispatch
│   ├── config.py                 # Config management, migration, env var definitions
│   ├── setup.py                  # Interactive setup wizard
│   ├── auth.py                   # Provider resolution, OAuth, Nous Portal
│   ├── models.py                 # OpenRouter model selection lists
│   ├── banner.py                 # Welcome banner, ASCII art
│   ├── commands.py               # Central slash command registry (CommandDef), autocomplete, gateway helpers
│   ├── callbacks.py              # Interactive callbacks (clarify, sudo, approval)
│   ├── doctor.py                 # Diagnostics
│   ├── skills_hub.py             # Skills Hub CLI + /skills slash command
│   └── skin_engine.py            # Skin/theme engine — data-driven CLI visual customization
│
├── tools/                    # Tool implementations (self-registering)
│   ├── registry.py               # Central tool registry (schemas, handlers, dispatch)
│   ├── approval.py               # Dangerous command detection + per-session approval
│   ├── terminal_tool.py          # Terminal orchestration (sudo, env lifecycle, backends)
│   ├── file_operations.py        # read_file, write_file, search, patch, etc.
│   ├── web_tools.py              # web_search, web_extract (Parallel/Firecrawl + Gemini summarization)
│   ├── vision_tools.py           # Image analysis via multimodal models
│   ├── delegate_tool.py          # Subagent spawning and parallel task execution
│   ├── code_execution_tool.py    # Sandboxed Python with RPC tool access
│   ├── session_search_tool.py    # Search past conversations with FTS5 + summarization
│   ├── cronjob_tools.py          # Scheduled task management
│   ├── skill_tools.py            # Skill search, load, manage
│   └── environments/             # Terminal execution backends
│       ├── base.py                   # BaseEnvironment ABC
│       ├── local.py, docker.py, ssh.py, singularity.py, modal.py, daytona.py
│
├── gateway/                  # Messaging gateway
│   ├── run.py                    # GatewayRunner — platform lifecycle, message routing, cron
│   ├── config.py                 # Platform configuration resolution
│   ├── session.py                # Session store, context prompts, reset policies
│   └── platforms/                # Platform adapters
│       ├── telegram.py, discord_adapter.py, slack.py, whatsapp.py
│
├── scripts/                  # Installer and bridge scripts
│   ├── install.sh                # Linux/macOS installer
│   ├── install.ps1               # Windows PowerShell installer
│   └── whatsapp-bridge/          # Node.js WhatsApp bridge (Baileys)
│
├── skills/                   # Bundled skills (copied to ~/.hermes/skills/ on install)
├── optional-skills/          # Official optional skills (discoverable via hub, not activated by default)
├── environments/             # RL training environments (Atropos integration)
├── tests/                    # Test suite
├── website/                  # Documentation site (hermes-agent.nousresearch.com)
│
├── cli-config.yaml.example   # Example configuration (copied to ~/.hermes/config.yaml)
└── AGENTS.md                 # Development guide for AI coding assistants
```

### User configuration (stored in `~/.hermes/`)

| Path | Purpose |
|------|---------|
| `~/.hermes/config.yaml` | Settings (model, terminal, toolsets, compression, etc.) |
| `~/.hermes/.env` | API keys and secrets |
| `~/.hermes/auth.json` | OAuth credentials (Nous Portal) |
| `~/.hermes/skills/` | All active skills (bundled + hub-installed + agent-created) |
| `~/.hermes/memories/` | Persistent memory (MEMORY.md, USER.md) |
| `~/.hermes/state.db` | SQLite session database |
| `~/.hermes/sessions/` | JSON session logs |
| `~/.hermes/cron/` | Scheduled job data |
| `~/.hermes/whatsapp/session/` | WhatsApp bridge credentials |

---

## Architecture Overview

### Core Loop

```
User message → AIAgent._run_agent_loop()
  ├── Build system prompt (prompt_builder.py)
  ├── Build API kwargs (model, messages, tools, reasoning config)
  ├── Call LLM (OpenAI-compatible API)
  ├── If tool_calls in response:
  │     ├── Execute each tool via registry dispatch
  │     ├── Add tool results to conversation
  │     └── Loop back to LLM call
  ├── If text response:
  │     ├── Persist session to DB
  │     └── Return final_response
  └── Context compression if approaching token limit
```

### Key Design Patterns

- **Self-registering tools**: Each tool file calls `registry.register()` at import time. `model_tools.py` triggers discovery by importing all tool modules.
- **Toolset grouping**: Tools are grouped into toolsets (`web`, `terminal`, `file`, `browser`, etc.) that can be enabled/disabled per platform.
- **Session persistence**: All conversations are stored in SQLite (`hermes_state.py`) with full-text search and unique session titles. JSON logs go to `~/.hermes/sessions/`.
- **Ephemeral injection**: System prompts and prefill messages are injected at API call time, never persisted to the database or logs.
- **Provider abstraction**: The agent works with any OpenAI-compatible API. Provider resolution happens at init time (Nous Portal OAuth, OpenRouter API key, or custom endpoint).
- **Provider routing**: When using OpenRouter, `provider_routing` in config.yaml controls provider selection (sort by throughput/latency/price, allow/ignore specific providers, data retention policies). These are injected as `extra_body.provider` in API requests.

---

## Code Style

- **PEP 8** with practical exceptions (we don't enforce strict line length)
- **Comments**: Only when explaining non-obvious intent, trade-offs, or API quirks. Don't narrate what the code does — `# increment counter` adds nothing
- **Error handling**: Catch specific exceptions. Log with `logger.warning()`/`logger.error()` — use `exc_info=True` for unexpected errors so stack traces appear in logs
- **Cross-platform**: Never assume Unix. See [Cross-Platform Compatibility](#cross-platform-compatibility)

---

## Adding a New Tool

Before writing a tool, ask: [should this be a skill instead?](#should-it-be-a-skill-or-a-tool)

Tools self-register with the central registry. Each tool file co-locates its schema, handler, and registration:

```python
"""my_tool — Brief description of what this tool does."""

import json
from tools.registry import registry


def my_tool(param1: str, param2: int = 10, **kwargs) -> str:
    """Handler. Returns a string result (often JSON)."""
    result = do_work(param1, param2)
    return json.dumps(result)


MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does and when the agent should use it.",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "What param1 is"},
                "param2": {"type": "integer", "description": "What param2 is", "default": 10},
            },
            "required": ["param1"],
        },
    },
}


def _check_requirements() -> bool:
    """Return True if this tool's dependencies are available."""
    return True


registry.register(
    name="my_tool",
    toolset="my_toolset",
    schema=MY_TOOL_SCHEMA,
    handler=lambda args, **kw: my_tool(**args, **kw),
    check_fn=_check_requirements,
)
```

**Wire into a toolset (required):** Built-in tools are auto-discovered: any
`tools/*.py` file that contains a top-level `registry.register(...)` call is
imported by `discover_builtin_tools()` in `tools/registry.py` when `model_tools`
loads. There is **no** manual import list in `model_tools.py` to maintain.

You must still add the tool name to the appropriate list in `toolsets.py`
(for example `_HERMES_CORE_TOOLS` or a dedicated toolset); otherwise the tool
registers but is never exposed to the agent. If you introduce a new toolset,
add it in `toolsets.py` and wire it into the relevant platform presets.

See `AGENTS.md` (section **Adding New Tools**) for profile-aware paths and
plugin vs core guidance.

---

## Adding a Skill

Bundled skills live in `skills/` organized by category. Official optional skills use the same structure in `optional-skills/`:

```
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # Required: main instructions
│       └── scripts/              # Optional: helper scripts
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

### SKILL.md format

```markdown
---
name: my-skill
description: Brief description (shown in skill search results)
version: 1.0.0
author: Your Name
license: MIT
platforms: [macos, linux]          # Optional — restrict to specific OS platforms
                                   #   Valid: macos, linux, windows
                                   #   Omit to load on all platforms (default)
required_environment_variables:    # Optional — secure setup-on-load metadata
  - name: MY_API_KEY
    prompt: API key
    help: Where to get it
    required_for: full functionality
prerequisites:                     # Optional legacy runtime requirements
  env_vars: [MY_API_KEY]           #   Backward-compatible alias for required env vars
  commands: [curl, jq]             #   Advisory only; does not hide the skill
metadata:
  hermes:
    tags: [Category, Subcategory, Keywords]
    related_skills: [other-skill-name]
    fallback_for_toolsets: [web]       # Optional — show only when toolset is unavailable
    requires_toolsets: [terminal]      # Optional — show only when toolset is available
---

# Skill Title

Brief intro.

## When to Use
Trigger conditions — when should the agent load this skill?

## Quick Reference
Table of common commands or API calls.

## Procedure
Step-by-step instructions the agent follows.

## Pitfalls
Known failure modes and how to handle them.

## Verification
How the agent confirms it worked.
```

### Platform-specific skills

Skills can declare which OS platforms they support via the `platforms` frontmatter field. Skills with this field are automatically hidden from the system prompt, `skills_list()`, and slash commands on incompatible platforms.

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders)
platforms: [macos, linux]     # macOS and Linux
platforms: [windows]          # Windows only
```

If the field is omitted or empty, the skill loads on all platforms (backward compatible). See `skills/apple/` for examples of macOS-only skills.

### Conditional skill activation

Skills can declare conditions that control when they appear in the system prompt, based on which tools and toolsets are available in the current session. This is primarily used for **fallback skills** — alternatives that should only be shown when a primary tool is unavailable.

Four fields are supported under `metadata.hermes`:

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # Show ONLY when these toolsets are unavailable
    requires_toolsets: [terminal]     # Show ONLY when these toolsets are available
    fallback_for_tools: [web_search]  # Show ONLY when these specific tools are unavailable
    requires_tools: [terminal]        # Show ONLY when these specific tools are available
```

**Semantics:**
- `fallback_for_*`: The skill is a backup. It is **hidden** when the listed tools/toolsets are available, and **shown** when they are unavailable. Use this for free alternatives to premium tools.
- `requires_*`: The skill needs certain tools to function. It is **hidden** when the listed tools/toolsets are unavailable. Use this for skills that depend on specific capabilities (e.g., a skill that only makes sense with terminal access).
- If both are specified, both conditions must be satisfied for the skill to appear.
- If neither is specified, the skill is always shown (backward compatible).

**Examples:**

```yaml
# DuckDuckGo search — shown when Firecrawl (web toolset) is unavailable
metadata:
  hermes:
    fallback_for_toolsets: [web]

# Smart home skill — only useful when terminal is available
metadata:
  hermes:
    requires_toolsets: [terminal]

# Local browser fallback — shown when Browserbase is unavailable
metadata:
  hermes:
    fallback_for_toolsets: [browser]
```

The filtering happens at prompt build time in `agent/prompt_builder.py`. The `build_skills_system_prompt()` function receives the set of available tools and toolsets from the agent and uses `_skill_should_show()` to evaluate each skill's conditions.

### Skill setup metadata

Skills can declare secure setup-on-load metadata via the `required_environment_variables` frontmatter field. Missing values do not hide the skill from discovery; they trigger a CLI-only secure prompt when the skill is actually loaded.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

The user may skip setup and keep loading the skill. Hermes only exposes metadata (`stored_as`, `skipped`, `validated`) to the model — never the secret value.

Legacy `prerequisites.env_vars` remains supported and is normalized into the new representation.

```yaml
prerequisites:
  env_vars: [TENOR_API_KEY]       # Legacy alias for required_environment_variables
  commands: [curl, jq]            # Advisory CLI checks
```

Gateway and messaging sessions never collect secrets in-band; they instruct the user to run `hermes setup` or update `~/.hermes/.env` locally.

**When to declare required environment variables:**
- The skill uses an API key or token that should be collected securely at load time
- The skill can still be useful if the user skips setup, but may degrade gracefully

**When to declare command prerequisites:**
- The skill relies on a CLI tool that may not be installed (e.g., `himalaya`, `openhue`, `ddgs`)
- Treat command checks as guidance, not discovery-time hiding

See `skills/gifs/gif-search/` and `skills/email/himalaya/` for examples.

### Skill guidelines

- **No external dependencies unless absolutely necessary.** Prefer stdlib Python, curl, and existing Hermes tools (`web_extract`, `terminal`, `read_file`).
- **Progressive disclosure.** Put the most common workflow first. Edge cases and advanced usage go at the bottom.
- **Include helper scripts** for XML/JSON parsing or complex logic — don't expect the LLM to write parsers inline every time.
- **Test it.** Run `hermes --toolsets skills -q "Use the X skill to do Y"` and verify the agent follows the instructions correctly.

---

## Adding a Skin / Theme

Hermes uses a data-driven skin system — no code changes needed to add a new skin.

**Option A: User skin (YAML file)**

Create `~/.hermes/skins/<name>.yaml`:

```yaml
name: mytheme
description: Short description of the theme

colors:
  banner_border: "#HEX"     # Panel border color
  banner_title: "#HEX"      # Panel title color
  banner_accent: "#HEX"     # Section header color
  banner_dim: "#HEX"        # Muted/dim text color
  banner_text: "#HEX"       # Body text color
  response_border: "#HEX"   # Response box border

spinner:
  waiting_faces: ["(⚔)", "(⛨)"]
  thinking_faces: ["(⚔)", "(⌁)"]
  thinking_verbs: ["forging", "plotting"]
  wings:                     # Optional left/right decorations
    - ["⟪⚔", "⚔⟫"]

branding:
  agent_name: "My Agent"
  welcome: "Welcome message"
  response_label: " ⚔ Agent "
  prompt_symbol: "⚔"

tool_prefix: "╎"             # Tool output line prefix
```

All fields are optional — missing values inherit from the default skin.

**Option B: Built-in skin**

Add to `_BUILTIN_SKINS` dict in `hermes_cli/skin_engine.py`. Use the same schema as above but as a Python dict. Built-in skins ship with the package and are always available.

**Activating:**
- CLI: `/skin mytheme` or set `display.skin: mytheme` in config.yaml
- Config: `display: { skin: mytheme }`

See `hermes_cli/skin_engine.py` for the full schema and existing skins as examples.

---

## Cross-Platform Compatibility

Hermes runs on Linux, macOS, and native Windows (plus WSL2). When writing code
that touches the OS, assume *any* platform can hit your code path.

> **Before you PR:** run `scripts/check-windows-footguns.py` to catch the
> common Windows-unsafe patterns in your diff. It's grep-based and cheap;
> CI runs it on every PR too.

### Critical rules

1. **Never call `os.kill(pid, 0)` for liveness checks.** `os.kill(pid, 0)`
   is a standard POSIX idiom to check "is this PID alive" — the signal 0
   is a no-op permission check. **On Windows it is NOT a no-op.** Python's
   Windows `os.kill` maps `sig=0` to `CTRL_C_EVENT` (they collide at the
   integer value 0) and routes it through `GenerateConsoleCtrlEvent(0, pid)`,
   which broadcasts Ctrl+C to the **entire console process group** containing
   the target PID. "Probe if alive" silently becomes "kill the target and
   often unrelated processes sharing its console." See [bpo-14484](https://bugs.python.org/issue14484)
   (open since 2012 — will never be fixed for compat reasons).

   **Preferred:** use `psutil` (a core dependency — always available):

   ```python
   import psutil
   if psutil.pid_exists(pid):
       # process is alive — safe on every platform
       ...
   ```

   If you specifically need the hermes wrapper (it has a stdlib fallback
   for scaffold-phase imports before pip install finishes), use
   `gateway.status._pid_exists(pid)`. It calls `psutil.pid_exists` first
   and falls back to a hand-rolled `OpenProcess + WaitForSingleObject`
   dance on Windows only when psutil is somehow missing.

   Audit grep for new callsites: `rg "os\.kill\([^,]+,\s*0\s*\)"`. Any hit
   in non-test code is presumptively a Windows silent-kill bug.

2. **Use `shutil.which()` before shelling out — don't assume Windows has
   tools Linux has.** `wmic` was removed in Windows 10 21H1 and later. `ps`,
   `kill`, `grep`, `awk`, `fuser`, `lsof`, `pgrep`, and most POSIX CLI tools
   simply don't exist on Windows. Test availability with
   `shutil.which("tool")` and fall back to a Windows-native equivalent —
   usually PowerShell via `subprocess.run(["powershell", "-NoProfile",
   "-Command", ...])`.

   For process enumeration: PowerShell's `Get-CimInstance Win32_Process` is
   the modern replacement for `wmic process`. See
   `hermes_cli/gateway.py::_scan_gateway_pids` for the pattern.

3. **`termios` and `fcntl` are Unix-only.** Always catch both `ImportError`
   and `NotImplementedError`:
   ```python
   try:
       from simple_term_menu import TerminalMenu
       menu = TerminalMenu(options)
       idx = menu.show()
   except (ImportError, NotImplementedError):
       # Fallback: numbered menu for Windows
       for i, opt in enumerate(options):
           print(f"  {i+1}. {opt}")
       idx = int(input("Choice: ")) - 1
   ```

4. **File encoding.** Windows may save `.env` files in `cp1252`. Always
   handle encoding errors:
   ```python
   try:
       load_dotenv(env_path)
   except UnicodeDecodeError:
       load_dotenv(env_path, encoding="latin-1")
   ```
   Config files (`config.yaml`) may be saved with a UTF-8 BOM by Notepad and
   similar editors — use `encoding="utf-8-sig"` when reading files that
   could have been touched by a Windows GUI editor.

5. **Process management.** `os.setsid()`, `os.killpg()`, `os.fork()`,
   `os.getuid()`, and POSIX signal handling differ on Windows. Guard with
   `platform.system()`, `sys.platform`, or `hasattr(os, "setsid")`:
   ```python
   if platform.system() != "Windows":
       kwargs["preexec_fn"] = os.setsid
   else:
       kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
   ```

   **Preferred:** for killing a process AND its children (what `os.killpg`
   does on POSIX), use `psutil` — it works on every platform:
   ```python
   import psutil
   try:
       parent = psutil.Process(pid)
       # Kill children first (leaf-up), then the parent.
       for child in parent.children(recursive=True):
           child.kill()
       parent.kill()
   except psutil.NoSuchProcess:
       pass
   ```

6. **Signals that don't exist on Windows: `SIGALRM`, `SIGCHLD`, `SIGHUP`,
   `SIGUSR1`, `SIGUSR2`, `SIGPIPE`, `SIGQUIT`, `SIGKILL`.** Python's
   `signal` module raises `AttributeError` at import time if you reference
   them on Windows. Use `getattr(signal, "SIGKILL", signal.SIGTERM)` or
   gate the whole block behind a platform check. `loop.add_signal_handler`
   raises `NotImplementedError` on Windows — always catch it.

7. **Path separators.** Use `pathlib.Path` instead of string concatenation
   with `/`. Forward slashes work almost everywhere on Windows, but
   `subprocess.run(["cmd.exe", "/c", ...])` and other shell contexts can
   require backslashes — convert with `str(path)` at the subprocess boundary,
   not inside Python logic.

8. **Symlinks need elevated privileges on Windows** (unless Developer Mode is
   on). Tests that create symlinks need `@pytest.mark.skipif(sys.platform ==
   "win32", reason="Symlinks require elevated privileges on Windows")`.

9. **POSIX file modes (0o600, 0o644, etc.) are NOT enforced on NTFS** by
   default. Tests that assert on `stat().st_mode & 0o777` must skip on
   Windows — the concept doesn't translate. Use ACLs (`icacls`, `pywin32`)
   for Windows secret-file protection if needed.

10. **Detached background daemons on Windows need `pythonw.exe`, NOT
    `python.exe`.** `python.exe` always allocates or attaches to a console,
    which makes it vulnerable to `CTRL_C_EVENT` broadcasts from any sibling
    process. `pythonw.exe` is the no-console variant. Combine with
    `CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
    CREATE_BREAKAWAY_FROM_JOB` in `subprocess.Popen(creationflags=...)`.
    See `hermes_cli/gateway_windows.py::_spawn_detached` for the reference
    implementation.

11. **`subprocess.Popen` with `.cmd` or `.bat` shims needs `shutil.which`
    to resolve.** Passing `"agent-browser"` to `Popen` on Windows finds
    the extensionless POSIX shebang shim in `node_modules/.bin/`, which
    `CreateProcessW` can't execute — you'll get `WinError 193 "not a valid
    Win32 application"`. Use `shutil.which("agent-browser", path=local_bin)`
    which honors PATHEXT and picks the `.CMD` variant on Windows.

12. **Don't use shell shebangs as a way to run Python.** `#!/usr/bin/env
    python` only works when the file is executed through a Unix shell.
    `subprocess.run(["./myscript.py"])` on Windows fails even if the file
    has a shebang line. Always invoke Python explicitly:
    `[sys.executable, "myscript.py"]`.

13. **Shell commands in installers.** If you change `scripts/install.sh`,
    make the equivalent change in `scripts/install.ps1`. The two scripts
    are the canonical example of "works on Linux does not mean works on
    Windows" and have drifted multiple times — keep them in lockstep.

14. **Known paths that are OneDrive-redirected on Windows:** Desktop,
    Documents, Pictures, Videos. The "real" path when OneDrive Backup is
    enabled is `%USERPROFILE%\OneDrive\Desktop` (etc.), NOT
    `%USERPROFILE%\Desktop` (which exists as an empty husk). Resolve the
    real location via `ctypes` + `SHGetKnownFolderPath` or by reading the
    `Shell Folders` registry key — never assume `~/Desktop`.

15. **CRLF vs LF in generated scripts.** Windows `cmd.exe` and `schtasks`
    parse line-by-line; mixed or LF-only line endings can break multi-line
    `.cmd` / `.bat` files. Use `open(path, "w", encoding="utf-8",
    newline="\r\n")` — or `open(path, "wb")` + explicit bytes — when
    generating scripts Windows will execute.

16. **Two different quoting schemes in one command line.** `subprocess.run
    (["schtasks", "/TR", some_cmd])` → schtasks itself parses `/TR`, AND
    the `some_cmd` string is re-parsed by `cmd.exe` when the task fires.
    Different parsers, different escape rules. Use two separate quoting
    helpers and never cross them. See `hermes_cli/gateway_windows.py::
    _quote_cmd_script_arg` and `_quote_schtasks_arg` for the reference
    pair.

### Testing cross-platform

Tests that use POSIX-only syscalls need a skip marker. Common ones:
- Symlinks → `@pytest.mark.skipif(sys.platform == "win32", ...)`
- `0o600` file modes → `@pytest.mark.skipif(sys.platform.startswith("win"), ...)`
- `signal.SIGALRM` → Unix-only (see `tests/conftest.py::_enforce_test_timeout`)
- `os.setsid` / `os.fork` → Unix-only
- Live Winsock / Windows-specific regression tests →
  `@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific regression")`

If you monkeypatch `sys.platform` for cross-platform tests, also patch
`platform.system()` / `platform.release()` / `platform.mac_ver()` — each
re-reads the real OS independently, so half-patched tests still route
through the wrong branch on a Windows runner.

---

## Security Considerations

Hermes has terminal access. Security matters.

### Existing protections

| Layer | Implementation |
|-------|---------------|
| **Sudo password piping** | Uses `shlex.quote()` to prevent shell injection |
| **Dangerous command detection** | Regex patterns in `tools/approval.py` with user approval flow |
| **Cron prompt injection** | Scanner in `tools/cronjob_tools.py` blocks instruction-override patterns |
| **Write deny list** | Protected paths (`~/.ssh/authorized_keys`, `/etc/shadow`) resolved via `os.path.realpath()` to prevent symlink bypass |
| **Skills guard** | Security scanner for hub-installed skills (`tools/skills_guard.py`) |
| **Code execution sandbox** | `execute_code` child process runs with API keys stripped from environment |
| **Container hardening** | Docker: all capabilities dropped, no privilege escalation, PID limits, size-limited tmpfs |

### When contributing security-sensitive code

- **Always use `shlex.quote()`** when interpolating user input into shell commands
- **Resolve symlinks** with `os.path.realpath()` before path-based access control checks
- **Don't log secrets.** API keys, tokens, and passwords should never appear in log output
- **Catch broad exceptions** around tool execution so a single failure doesn't crash the agent loop
- **Test on all platforms** if your change touches file paths, process management, or shell commands

If your PR affects security, note it explicitly in the description.

---

## Pull Request Process

### Branch naming

```
fix/description        # Bug fixes
feat/description       # New features
docs/description       # Documentation
test/description       # Tests
refactor/description   # Code restructuring
```

### Before submitting

1. **Run tests**: `scripts/run_tests.sh` (recommended; same as CI) or `pytest tests/ -v` with the project venv activated
2. **Test manually**: Run `hermes` and exercise the code path you changed
3. **Check cross-platform impact**: If you touch file I/O, process management, or terminal handling, consider macOS, Linux, and WSL2
4. **Keep PRs focused**: One logical change per PR. Don't mix a bug fix with a refactor with a new feature.

### PR description

Include:
- **What** changed and **why**
- **How to test** it (reproduction steps for bugs, usage examples for features)
- **What platforms** you tested on
- Reference any related issues

### Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

| Type | Use for |
|------|---------|
| `fix` | Bug fixes |
| `feat` | New features |
| `docs` | Documentation |
| `test` | Tests |
| `refactor` | Code restructuring (no behavior change) |
| `chore` | Build, CI, dependency updates |

Scopes: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`, etc.

Examples:
```
fix(cli): prevent crash in save_config_value when model is a string
feat(gateway): add WhatsApp multi-user session isolation
fix(security): prevent shell injection in sudo password piping
test(tools): add unit tests for file_operations
```

---

## Reporting Issues

- Use [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- Include: OS, Python version, Hermes version (`hermes version`), full error traceback
- Include steps to reproduce
- Check existing issues before creating duplicates
- For security vulnerabilities, please report privately

---

## Community

- **Discord**: [discord.gg/NousResearch](https://discord.gg/NousResearch) — for questions, showcasing projects, and sharing skills
- **GitHub Discussions**: For design proposals and architecture discussions
- **Skills Hub**: Upload specialized skills to a registry and share them with the community

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
