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

## Before You Start: Search First

A quick search before you build saves your time and keeps the PR queue clean — duplicates are common here, so it's worth a minute up front.

- **Search both open *and* merged PRs and issues** for your topic or error symptom — the duplicate-check in the PR template fires at review time, after you've already done the work:
  ```bash
  gh search issues --repo NousResearch/hermes-agent "<your terms>"
  gh search prs --repo NousResearch/hermes-agent --state all "<your terms>"
  ```
  Or use the web UI: [issues](https://github.com/NousResearch/hermes-agent/issues?q=) · [PRs (all states)](https://github.com/NousResearch/hermes-agent/pulls?q=is%3Apr).
- **The issue tracker can lag the code.** Many requested features are already implemented in-tree, so also search the source (`search_files`, or your editor's grep) for the capability before proposing it.
- **If an open PR already addresses it**, consider reviewing or improving that one instead of opening a competing duplicate.
- **For larger work**, comment on the issue to signal you're working on it, so others don't start the same thing.

Related: #38284 covers the agent-side analog — Hermes itself checking existing issues and PRs before deep self-troubleshooting. This section is the human-contributor complement.

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

If your skill is official and useful but not universally needed (e.g., a paid service integration, a heavyweight dependency), put it in **`optional-skills/`** — it ships with the repo but isn't activated by default. Users can discover it via `hermes skills browse` (labeled "official") and install it with `hermes skills install` (no third-party warning, built-in trust).

If your skill is specialized, community-contributed, or niche, it's better suited for a **Skills Hub** — upload it to a skills registry and share it in the [Nous Research Discord](https://discord.gg/NousResearch). Users can install it with `hermes skills install`.

---

## Memory Providers: Ship as a Standalone Plugin

**We are no longer accepting new memory providers into this repo.** The set of built-in providers under `plugins/memory/` (honcho, mem0, supermemory, byterover, hindsight, holographic, openviking, retaindb) is closed. If you want to add a new memory backend, publish it as a **standalone plugin repo** that users install into `~/.hermes/plugins/` (or via a pip entry point).

Standalone memory plugins:

- Implement the same `MemoryProvider` ABC (`agent/memory_provider.py`) — `sync_turn`, `prefetch`, `shutdown`, and optionally `post_setup(hermes_home, config)` for setup-wizard integration
- Use the same discovery system — `discover_memory_providers()` picks them up from user/project plugin directories and pip entry points
- Integrate with `hermes memory setup` via `post_setup()` — no need to touch core code
- Can register their own CLI subcommands via `register_cli(subparser)` in a `cli.py` file
- Get all the same lifecycle hooks and config plumbing as in-tree providers

PRs that add a new directory under `plugins/memory/` will be closed with a pointer to publish the provider as its own repo. Existing in-tree providers stay; bug fixes to them are welcome.

This isn't a quality bar — it's a coupling-and-maintenance decision. Memory providers are the most common plugin type and they shouldn't all live in this tree.

---

## Third-Party Product Integrations: Ship as a Standalone Plugin

The same rule extends to **any plugin that integrates someone else's product or project** — observability/metrics backends, vendor SaaS connectors, analytics dashboards, paid-service tie-ins, and similar third-party integrations. **These do not land in this repo.**

The reason is maintenance load, not quality. Every external product absorbed into the core tree becomes ours to keep working against a fast-moving codebase, for a backend we don't own and can't control. Hermes ships a lot and the core moves quickly; coupling third-party products into it creates an open-ended burden on the maintainers.

Publish these as a **standalone plugin repo** instead:

- Implement the relevant ABC and use the existing plugin discovery path (`~/.hermes/plugins/`, project `.hermes/plugins/`, or a pip entry point) — see [Build a Hermes Plugin](https://hermes-agent.nousresearch.com/docs/guides/build-a-hermes-plugin)
- Register lifecycle hooks (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`), tools (`ctx.register_tool`), and CLI subcommands (`ctx.register_cli_command`) through the surface we already expose — no core changes needed
- If your plugin needs a capability the framework doesn't expose, that's a feature request to **widen the generic plugin surface** (a new hook or `ctx` method) — never special-case your plugin in core
- Promote it in the [Nous Research Discord](https://discord.gg/NousResearch) `#plugins-skills-and-skins` channel so users can find and install it

A well-built third-party-product plugin can clear automated review and still be closed for this reason — it's a placement decision, not a verdict on the code. PRs that add such a directory under `plugins/` will be closed with a pointer to publish it as its own repo.

---

## Development Setup

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Git** | With the `git-lfs` extension installed |
| **Python 3.11–3.13** | uv will install it if missing |
| **uv** | Fast Python package manager ([install](https://docs.astral.sh/uv/)) |
| **Node.js 20+** | Optional — needed for browser tools and WhatsApp bridge (matches root `package.json` engines) |

### Install with the standard installer

For most contributors, the best development bootstrap is the same path users
take: run the standard installer, then work inside the repository it cloned.
The installer creates the Hermes venv, wires the `hermes` command, stamps the
install method for `hermes update`, and clones the full git project into
`$HERMES_HOME/hermes-agent` (usually `~/.hermes/hermes-agent`). That keeps your
development environment on the same layout the CLI, updater, lazy dependency
installer, gateway, and docs assume.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"

# Add dev/test extras on top of the standard install.
uv pip install -e ".[all,dev]"

# Optional: browser tools / docs site dependencies.
npm install
```

After that, create branches and run tests from that checkout:

```bash
git checkout -b fix/description
scripts/run_tests.sh
```

### Manual clone fallback

Use this only if you intentionally do not want Hermes' managed install layout
(for example, a throwaway clone inside a container or CI job). If you install
this way, make sure you run the `hermes` entrypoint from this venv; running the
system `python3 -m hermes_cli.main` can pick up unrelated system Python
packages.

Create the venv **outside** the cloned source tree. A venv that lives inside
the directory the agent operates from can be wiped by a relative-path command
the agent runs against its own checkout (`rm -rf venv`, `uv venv venv`, etc.),
which silently destroys the running runtime mid-session. Keeping it outside the
tree means no relative path from the workspace resolves to it.

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create venv with Python 3.11, OUTSIDE the source tree
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
export VIRTUAL_ENV="$HOME/.hermes/venvs/hermes-dev"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Install with all extras (messaging, cron, CLI menus, dev tools)
uv pip install -e ".[all,dev]"

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
# The standard installer already put `hermes` on PATH.
hermes doctor
hermes chat -q "Hello"
```

If you used the manual clone fallback, run `./hermes` from the checkout or
symlink this clone's venv explicitly:

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes
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
│   ├── session_search_tool.py    # Search past conversations with FTS5 + anchored windows
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
| `~/.hermes/sessions/` | Gateway routing index (`sessions.json`), request-dump breadcrumbs, gateway `*.jsonl` transcripts, and (optionally) per-session JSON snapshots when `sessions.write_json_snapshots: true` is set. The per-session snapshots are off by default; state.db is canonical. |
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
- **Session persistence**: All conversations are stored in SQLite (`hermes_state.py`) with full-text search and unique session titles. Per-session JSON snapshots in `~/.hermes/sessions/` were superseded by the SQLite store and are off by default; opt back in with `sessions.write_json_snapshots: true` if you have external tooling that consumes the JSON files directly.
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

## Prerequisites
Env vars, install steps, MCP setup, API key sourcing.

## How to Run
Canonical invocation through the `terminal` tool.

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

### Skill authoring standards (HARDLINE)

Every new or modernized skill — bundled, optional, or contributed — must meet these standards before merge. Reviewers reject PRs that violate them.

1. **`description` ≤ 60 characters, one sentence, ends with a period.** Long descriptions bloat the skill listing UI and dilute the model's attention when many skills are loaded. State the capability, not the implementation. No marketing words ("powerful", "comprehensive", "seamless", "advanced"). Don't repeat the skill name. Verify with:
   ```python
   import re, pathlib
   m = re.search(r'^description: (.*)$',
                 pathlib.Path('skills/<cat>/<name>/SKILL.md').read_text(),
                 re.MULTILINE)
   assert len(m.group(1)) <= 60, len(m.group(1))
   ```

   Good: `Search arXiv papers by keyword, author, category, or ID.`
   Bad: `A powerful and comprehensive skill that allows the agent to search arXiv for relevant academic papers using various criteria including keywords, authors, and categories.`

2. **Tools referenced in SKILL.md prose must be native Hermes tools or MCP servers the skill explicitly expects.** When the skill needs a capability, point at the proper tool by name in backticks: `` `terminal` ``, `` `web_extract` ``, `` `web_search` ``, `` `read_file` ``, `` `write_file` ``, `` `patch` ``, `` `search_files` ``, `` `vision_analyze` ``, `` `browser_navigate` ``, `` `delegate_task` ``, `` `image_generate` ``, `` `text_to_speech` ``, `` `cronjob` ``, `` `memory` ``, `` `skill_view` ``, `` `todo` ``, `` `execute_code` ``.

   Do NOT name shell utilities the agent already has wrapped:

   | Don't say | Say |
   |---|---|
   | `grep`, `rg` | `search_files` |
   | `cat`, `head`, `tail` | `read_file` |
   | `sed`, `awk` | `patch` |
   | `find`, `ls` | `search_files` (with `target='files'`) |
   | `curl` for content extraction | `web_extract` |
   | `echo > file`, `cat <<EOF` | `write_file` |

   If the skill depends on an MCP server, name the MCP server and document its setup in `## Prerequisites`. Third-party CLIs (e.g. `ffmpeg`, `gh`, a specific SDK) are fine to invoke from inside script files, but the prose should frame the interaction as "invoke through the `terminal` tool", not as a manual shell session.

3. **`platforms:` gating audited against actual script imports.** Skills that use POSIX-only primitives (`fcntl`, `termios`, `os.setsid`, `os.kill(pid, 0)` for liveness, `/proc`, hardcoded `/tmp` paths, `signal.SIGKILL`, bash heredocs, `osascript`, `apt`, `systemctl`) must declare their supported platforms via the `platforms:` frontmatter. Default posture is to fix it cross-platform first — `tempfile.gettempdir()`, `pathlib.Path`, `psutil.pid_exists()`, Python-level filtering instead of `grep`. Gate to a narrower set only when the dependency is genuinely platform-bound (e.g. `osascript` is macOS-only, `/proc` is Linux-only).

4. **`author` credits the human contributor first.** For external contributions, the contributor's real name + GitHub handle goes first (`Jane Doe (jane-doe)`); "Hermes Agent" is the secondary collaborator. If the contributor's commit shows "Hermes Agent" as author because they used Hermes to draft the skill, replace it with their actual name — credit the human, not the tool.

5. **SKILL.md body uses the modern section order.** `# <Skill> Skill` title, 2-3 sentence intro stating what it does and what it doesn't do, then:
   - `## When to Use` — trigger conditions
   - `## Prerequisites` — env vars, install steps, MCP setup, API key sourcing
   - `## How to Run` — canonical invocation through the `terminal` tool
   - `## Quick Reference` — flat command/API reference
   - `## Procedure` — numbered steps with copy-paste commands
   - `## Pitfalls` — known limits, rate limits, things that look broken but aren't
   - `## Verification` — single command that proves the skill works

   Target ~200 lines for a complex skill, ~100 lines for a simple one. Cut redundant intro fluff, marketing prose, and re-explanations of env vars already documented in `## Prerequisites`.

6. **Scripts go in `scripts/`, references in `references/`, templates in `templates/`.** Don't expect the model to inline-write parsers, XML walkers, or non-trivial logic every call — ship a helper script. Reference scripts from SKILL.md by path relative to the skill directory.

7. **Tests live at `tests/skills/test_<skill>_skill.py`** and use only stdlib + pytest + `unittest.mock`. No live network calls. Run via `scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q`. Must pass under the hermetic CI env (no API keys leaking through). Use `monkeypatch` and `tmp_path` for any env-var or filesystem dependencies.

8. **`.env.example` additions are isolated to a clearly delimited block.** Don't touch the surrounding file — contributor-supplied `.env.example` versions are usually stale, and edits outside the skill's own block will be dropped during salvage. Comment all values with `#` (it's documentation, not live config).

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

### Dependency pinning policy (supply chain hardening)

After the [litellm supply chain compromise](https://github.com/BerriAI/litellm/issues/24512) in March 2026 and the [Mini Shai-Hulud worm campaign](https://socket.dev/blog/tanstack-npm-packages-compromised-mini-shai-hulud-supply-chain-attack) in May 2026, all dependencies must follow these rules:

| Source type | Required treatment | Rationale |
|---|---|---|
| **PyPI package** | `>=floor,<next_major` | PyPI versions are immutable once published, but new versions can be pushed into your range. A `<next_major` ceiling stops a 1.x install from upgrading to a malicious 2.0.0. |
| **Git URL** (atroposlib, tinker, yc-bench, Baileys) | Full commit SHA | Branches and tags are mutable refs; SHA is content-addressed. |
| **GitHub Actions** | Full commit SHA + version comment | Action tags are mutable refs (e.g. tj-actions/changed-files March 2025). Pin as `uses: owner/action@<sha>  # vX.Y.Z` |
| **CI-only pip installs** | `==exact` | Hermetic CI builds; churn is acceptable. |

**Every new PyPI dependency in a PR must have a `<next_major` upper bound.** PRs adding unbounded `>=X.Y.Z` specs will be rejected by reviewers. The `supply-chain-audit.yml` CI workflow also flags dependency manifest changes for manual review.

**How to determine the ceiling:**
- If the package is at version `1.x.y`, use `<2`.
- If the package is at version `0.x.y` (pre-1.0), use `<0.(current_minor + 2)` — e.g. if current is `0.29.x`, use `<0.32`. This gives ~2 minor versions of headroom while keeping the window small enough that a hostile takeover version is unlikely to land inside it.
- Exception: packages with very stable APIs (e.g. `aiohttp-socks`) can use `<1` at reviewer discretion.

**Examples:**
```toml
# ✅ Correct — post-1.0
"openai>=2.21.0,<3"
"pydantic>=2.12.5,<3"

# ✅ Correct — pre-1.0 (tight minor window)
"asyncpg>=0.29,<0.32"
"aiosqlite>=0.20,<0.23"
"hindsight-client>=0.4.22,<0.5"

# ❌ Rejected — no upper bound
"some-package>=1.2.3"

# ❌ Rejected — too tight (blocks legitimate patches)
"some-package==1.2.3"

# ❌ Rejected — too loose for pre-1.0 (allows 80 minor versions)
"some-package>=0.20,<1"
```

**Reference PRs:** #2796 (litellm removal), #2810 (upper bounds pass), #9801 (SHA pinning + supply-chain-audit CI).

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

---

## Agent-Specific Contributor Guidance

The following sections were moved out of `AGENTS.md` so routine agent sessions do not automatically load contributor-only policy and deep development notes. `AGENTS.md` now keeps the compact runtime instructions; open this guide when doing PR review, subsystem work, or policy-sensitive contributions.

## Contribution Rubric — What We Want / What We Don't

This is the project's intent layer. Use it two ways:

1. **For humans and for your own work** — what gets merged and what gets
   rejected, so a contribution aims at the target.
2. **For automated review (the triage sweeper)** — guidance on when a PR is
   safe to close on the three allowed reasons (`implemented_on_main`,
   `cannot_reproduce`, `incoherent`) and, just as important, **when NOT to
   close** one. Taste-based "we don't want this / out of scope" closes are NOT
   an automated decision — those stay with a human maintainer. The sweeper's
   job here is to recognize design intent and *avoid wrongly closing a
   legitimate contribution*, not to make the won't-implement call itself.

Read the balance right: Hermes ships a **lot** — most merges are bug fixes to
real reported behavior, and the product surface (platforms, channels,
providers, models, desktop/TUI features) expands aggressively and on purpose.
The restraint below is aimed squarely at the **core agent + the model tool
schema**, the one place where every addition is paid for on every API call.
"Smallest footprint" governs *how a capability is wired into the core*, NOT
whether the product is allowed to grow. We are expansive at the edges and
conservative at the waist.

### What we want

- **Fix real bugs, well.** The bulk of what lands is `fix(...)` against an
  actual reported symptom. A good fix reproduces the symptom on current
  `main`, points to the exact line where it manifests, and fixes the whole bug
  class — sibling call paths included — not just the one site the reporter hit.
- **Expand reach at the edges.** New platform adapters, channels, providers,
  models, and desktop/TUI/dashboard features are welcome and land routinely,
  including large ones (a new messaging channel, a session-cap feature, a
  Windows PTY bridge). Breadth in the product is a goal, not a footprint
  concern — as long as it integrates with the existing setup/config UX
  (`hermes tools`, `hermes setup`, auto-install) rather than bolting on a raw
  env var.
- **Refactor god-files into clean modules.** Extracting a multi-thousand-line
  cluster out of `cli.py` / `run_agent.py` / `gateway/run.py` into a focused
  mixin or module is wanted work, even when the diff is huge and mechanical
  (large `+N/-N` refactors merge regularly). The "every line traces to the
  request" test applies to *feature* PRs; a declared refactor's request IS the
  extraction.
- **Keep the core narrow.** New *model tools* are the expensive exception —
  every tool ships on every API call. Prefer, in order: extend existing code →
  CLI command + skill → service-gated tool (`check_fn`) → plugin → MCP server
  in the catalog → new core tool (last resort). See "The Footprint Ladder."
- **Extend, don't duplicate.** Before adding a module/manager/hook, check
  whether existing infrastructure already covers the use case. When several PRs
  integrate the same *category*, design one shared interface instead of merging
  them one at a time (see the ABC + orchestrator note under the Footprint
  Ladder).
- **Behavior contracts over snapshots.** Tests should assert how two pieces of
  data must relate (invariants), not freeze a current value (model lists,
  config version literals, enumeration counts). See "Don't write
  change-detector tests."
- **E2E validation, not just green unit mocks.** For anything touching
  resolution chains, config propagation, security boundaries, remote
  backends, or file/network I/O, exercise the real path with real imports
  against a temp `HERMES_HOME`. Mocks hide integration bugs.
- **Cache-, alternation-, and invariant-safe.** Preserve prompt caching, strict
  message role alternation (never two same-role messages in a row; never a
  synthetic user message injected mid-loop), and a system prompt that is
  byte-stable for the life of a conversation.
- **Contributor credit preserved.** Salvage external work by cherry-picking
  (rebase-merge) so authorship survives in git history; don't reimplement from
  scratch when you can build on top.

### What we don't want (rejected even when well-built)

- **Speculative infrastructure.** Hooks, callbacks, or extension points with no
  concrete consumer. Adding a hook is easy; removing one after plugins depend
  on it is hard. A hook is NOT speculative if a contributor has a real, stated
  use case — even if the consumer ships separately.
- **New `HERMES_*` env vars for non-secret config.** `.env` is for secrets
  only (API keys, tokens, passwords). All behavioral settings — timeouts,
  thresholds, feature flags, display prefs — go in `config.yaml`. Bridge to an
  internal env var if the mechanism needs one, but user-facing docs point to
  `config.yaml`. Reject PRs that tell users to "set X in your .env" unless X
  is a credential.
- **A new core tool when terminal + file already do the job, or when a skill
  would.** If the only barrier is file visibility on a remote backend, fix the
  mount, not the toolset.
- **Lazy-reading escape hatches on instructional tools.** No `offset`/`limit`
  pagination on tools that load content the agent must read fully (skills,
  prompts, playbooks). Models will read page 1 and skip the rest.
- **"Fixes" that destroy the feature they secure.** A mitigation that kills the
  feature's purpose is the wrong mitigation. Read the original commit's intent
  (`git log -p -S`) before restricting behavior; find a fix that preserves the
  feature.
- **Outbound telemetry / usage attribution without opt-in gating.** No new
  analytics, third-party identifier tagging, or attribution tags until a
  generic user-facing opt-in (config gate + setup prompt + `hermes tools`
  toggle) exists. Park behind a label, do not merge.
- **Change-detector tests, cache-breaking mid-conversation, dead code wired in
  without E2E proof, and plugins that touch core files.** Plugins live in their
  own directory and work within the ABCs/hooks we provide; if a plugin needs
  more, widen the generic plugin surface, don't special-case it in core.
- **Third-party products / other people's projects integrated into the core
  tree.** Observability backends, vendor SaaS integrations, analytics dashboards,
  and similar "someone else's product" plugins do NOT land under `plugins/` in
  this repo. They place an ongoing maintenance burden on us to keep them working
  against a fast-moving core, for a backend we don't own. Ship them as a
  **standalone plugin repo** users install into `~/.hermes/plugins/` (or via a
  pip entry point), and promote them in the Nous Research Discord
  (`#plugins-skills-and-skins`). This is a coupling-and-maintenance decision, not
  a quality bar — the plugin can be excellent and still be a close. PRs that add
  such a directory to the tree are closed with a pointer to publish it as its own
  repo.

### Before you call it a bug — verify the premise (and when NOT to close)

The most common reason a well-written PR gets closed is not code quality — it
is that the change is built on a **wrong premise**, or it treats an
**intentional design as a gap**. These patterns cut both ways: they tell a
human reviewer what to scrutinize, and they tell the automated sweeper when a
PR is NOT safe to close as `implemented_on_main` / `cannot_reproduce` (when in
doubt, leave it open for a human). They are distilled from real closes.

- **"Intentional design, not a gap."** A limitation that looks like an
  oversight is often deliberate. Before "fixing" a missing link or a
  restriction, ask whether the isolation IS the design. Example: profiles are
  independent islands on purpose — a PR adding live config inheritance from the
  default profile was closed because coupling profiles together is exactly what
  the design prevents (the copy-at-creation `--clone` path already covers the
  legitimate "start from my default" case). Read the original commit's intent
  (`git log -p -S "<symbol>"`) before assuming something is unfinished.
- **"The premise doesn't hold against how X actually works."** A PR's
  justification frequently rests on a wrong mental model of an existing
  mechanism. Trace the real code/runtime before accepting the rationale. Two
  real closes: a rate-limit "re-probe during cooldown" PR (the breaker only
  trips on a *confirmed-empty* account bucket, so re-probing just hammers a
  bucket we've already proven empty); a usage-accumulation fix whose new branch
  **never executes at runtime** because an earlier guard already popped the
  state it depended on. If you can't point to the exact line where the bug
  manifests AND show the fix changes that line's behavior, you haven't verified
  the premise.
- **"This fix was wrong — the absence/omission was deliberate."** Adding the
  obvious-looking missing piece can break things the omission was protecting.
  Example: restoring "missing" `__init__.py` files made a test tree importable
  as a dotted package that shadowed the real plugin, deleting its `register()`
  at import time. The absence was load-bearing.
- **"Overreached / resurrected an approach we'd moved past."** Scope creep that
  supersedes an agreed-on base, or revives a direction the maintainers
  deliberately closed, gets rejected even when the code works. Keep the change
  to the narrow piece that was actually agreed; offer the rest as a focused
  follow-up.

The throughline: **verify the claim AND the intent against the codebase before
writing or merging a fix.** A confirmed reproduction on current `main` plus a
line-level account of where the fix acts beats a plausible-sounding rationale
every time. When in doubt about intent, it is cheaper to ask than to ship a
fix that fights the design.

### The Footprint Ladder (new capability decision)

Each rung adds more permanent surface than the one above. Choose the highest
(least-footprint) rung that correctly solves the problem:

1. **Extend existing code** — the capability is a variation of something that
   already exists. Zero new surface.
2. **CLI command + skill** — manages config/state/infra expressible as shell
   commands. The agent runs `hermes <subcommand>` guided by a skill. Zero
   model-tool footprint. Default choice for subscriptions, scheduled tasks,
   service setup. Examples: `hermes webhook`, `hermes cron`, `hermes tools`.
3. **Service-gated tool (`check_fn`)** — needs structured params/returns AND
   only appears when a prerequisite is configured. Zero footprint otherwise.
   Examples: Home Assistant tools (gated on token), memory-provider tools.
4. **Plugin** — third-party/niche/user-specific capability that doesn't ship in
   core. Lives in `~/.hermes/plugins/` or a pip package, discovered at runtime.
5. **MCP server (in the catalog)** — if the capability genuinely needs to be a
   tool (structured I/O the agent invokes) but isn't core-fundamental, prefer
   building it as an MCP server and adding it to the MCP catalog over growing
   the core toolset. The agent connects to it through the built-in MCP client;
   zero permanent core-schema footprint, and it's reusable by any MCP host.
6. **New core tool** — only when the capability is fundamental, broadly useful
   to nearly every user, and unreachable via terminal + file (or an MCP server).
   Examples of correct core tools: terminal, read_file, web_search,
   browser_navigate.

When 3+ open PRs try to integrate the same *category* of thing (memory
backends, providers, notifiers), don't merge them one at a time — design an
ABC + orchestrator, wrap the existing built-in as the first provider, and turn
the competing PRs into plugins against that interface.


## TUI Architecture (ui-tui + tui_gateway)

The TUI is a full replacement for the classic (prompt_toolkit) CLI, activated via `hermes --tui` or `HERMES_TUI=1`.

### Process Model

```
hermes --tui
  └─ Node (Ink)  ──stdio JSON-RPC──  Python (tui_gateway)
       │                                  └─ AIAgent + tools + sessions
       └─ renders transcript, composer, prompts, activity
```

TypeScript owns the screen. Python owns sessions, tools, model calls, and slash command logic.

### Transport

Newline-delimited JSON-RPC over stdio. Requests from Ink, events from Python. See `tui_gateway/server.py` for the full method/event catalog.

### Key Surfaces

| Surface | Ink component | Gateway method |
|---------|---------------|----------------|
| Chat streaming | `app.tsx` + `messageLine.tsx` | `prompt.submit` → `message.delta/complete` |
| Tool activity | `thinking.tsx` | `tool.start/progress/complete` |
| Approvals | `prompts.tsx` | `approval.respond` ← `approval.request` |
| Clarify/sudo/secret | `prompts.tsx`, `maskedPrompt.tsx` | `clarify/sudo/secret.respond` |
| Session picker | `sessionPicker.tsx` | `session.list/resume` |
| Slash commands | Local handler + fallthrough | `slash.exec` → `_SlashWorker`, `command.dispatch` |
| Completions | `useCompletion` hook | `complete.slash`, `complete.path` |
| Theming | `theme.ts` + `branding.tsx` | `gateway.ready` with skin data |

### Slash Command Flow

1. Built-in client commands (`/help`, `/quit`, `/clear`, `/resume`, `/copy`, `/paste`, etc.) handled locally in `app.tsx`
2. Everything else → `slash.exec` (runs in persistent `_SlashWorker` subprocess) → `command.dispatch` fallback

### Dev Commands

```bash
cd ui-tui
npm install       # first time
npm run dev       # watch mode (rebuilds hermes-ink + tsx --watch)
npm start         # production
npm run build     # full build (hermes-ink + tsc)
npm run typecheck # typecheck only (tsc --noEmit)
npm run lint      # eslint
npm run fmt       # prettier
npm test          # vitest
```

### TUI in the Dashboard (`hermes dashboard` → `/chat`)

The dashboard embeds the real `hermes --tui` — **not** a rewrite.  See `hermes_cli/pty_bridge.py` + the `@app.websocket("/api/pty")` endpoint in `hermes_cli/web_server.py`.

- Browser loads `web/src/pages/ChatPage.tsx`, which mounts xterm.js's `Terminal` with the WebGL renderer, `@xterm/addon-fit` for container-driven resize, and `@xterm/addon-unicode11` for modern wide-character widths.
- `/api/pty?token=…` upgrades to a WebSocket; auth uses the same ephemeral `_SESSION_TOKEN` as REST, via query param (browsers can't set `Authorization` on WS upgrade).
- The server spawns whatever `hermes --tui` would spawn, through `ptyprocess` (POSIX PTY — WSL works, native Windows does not).
- Frames: raw PTY bytes each direction; resize via `\x1b[RESIZE:<cols>;<rows>]` intercepted on the server and applied with `TIOCSWINSZ`.

**Do not re-implement the primary chat experience in React.** The main transcript, composer/input flow (including slash-command behavior), and PTY-backed terminal belong to the embedded `hermes --tui` — anything new you add to Ink shows up in the dashboard automatically. If you find yourself rebuilding the transcript or composer for the dashboard, stop and extend Ink instead.

**Structured React UI around the TUI is allowed when it is not a second chat surface.** Sidebar widgets, inspectors, summaries, status panels, and similar supporting views (e.g. `ChatSidebar`, `ModelPickerDialog`, `ToolCall`) are fine when they complement the embedded TUI rather than replacing the transcript / composer / terminal. Keep their state independent of the PTY child's session and surface their failures non-destructively so the terminal pane keeps working unimpaired.

### Electron Desktop Chat App (`apps/desktop/`)

A **separate** chat surface from both the classic CLI and the dashboard's embedded TUI. It is an Electron + React + nanostore renderer (`@assistant-ui/react`) that talks to a `tui_gateway` backend over JSON-RPC (`requestGateway(method, params)`). The WebSocket/JSON-RPC transport lives in the framework-agnostic `apps/shared` package (`@hermes/shared` — `JsonRpcGatewayClient` + WS URL helpers), which the web dashboard (`web/`) also consumes; **desktop has no build/runtime dependency on the dashboard frontend** — it spawns a headless `hermes serve` backend server (the same gateway `dashboard` serves, minus the browser UI). `dashboard` and `serve` share `cmd_dashboard`/`start_server` but are independent surfaces — neither launches the other. The one exception is a backward-compat *fallback*: `serve` is newer, so the desktop spawn (`electron/backend-command.cjs` + `backendSupportsServe()` in `main.cjs`) detects whether the resolved runtime registers `serve` and, only when it does not (an older managed install / PATH `hermes` the app hasn't updated yet), rewrites the argv to the legacy `dashboard --no-open`. Without that, a new app against an un-upgraded runtime would crash on an unknown subcommand and brick every mid-upgrade user. It does NOT embed `hermes --tui` — it has its own composer, transcript, and slash-command pipeline. Route desktop bugs to the `hermes-desktop-app-work` skill, not `hermes-dashboard-work`.

**Slash commands in the desktop app are curated client-side, then dispatched to the backend.** The pipeline:

- **Backend already provides everything.** `tui_gateway/server.py` `commands.catalog` (empty-query list) and `complete.slash` (typed-query completions) both include built-in commands, user `quick_commands`, AND skill-derived commands (`scan_skill_commands()` / `get_skill_commands()`). The desktop app does not need a new RPC to see skills.
- **The renderer curates via `apps/desktop/src/lib/desktop-slash-commands.ts`.** This is the load-bearing file. It holds `DESKTOP_COMMANDS` (the ~19 built-ins shown in the palette) plus block-lists for terminal-only / messaging-only / picker-owned / settings-owned / advanced commands that should NOT clutter the desktop popover.
  - `isDesktopSlashCommand(name)` — gates **execution**. Returns true for built-ins AND for any non-built-in (skill / quick command), so typed extension commands run.
  - `isDesktopSlashSuggestion(name)` — gates **discovery/completion**. Used by BOTH completion paths in `app/chat/composer/hooks/use-slash-completions.ts` (empty-query catalog filter + typed-query `complete.slash` filter) and by `filterDesktopCommandsCatalog`.
  - `isDesktopSlashExtensionCommand(name)` — true when the command is NOT a known Hermes built-in (i.e. a skill or user quick command). Both suggestion and catalog-filter paths allow extensions through so skill commands surface in the palette. (Added when fixing "skill commands missing from the desktop slash palette" — the curated allow-list was silently dropping every skill/quick command from completions even though they executed fine when typed.)
- **Dispatch** lives in `app/session/hooks/use-prompt-actions.ts` (`runSlash`): built-ins that the desktop owns (`/skin`, `/help`, `/new`, …) are handled locally or via `commands.catalog`; everything else goes to `slash.exec`, falling back to `command.dispatch` (which the gateway resolves into skill / alias / exec directives). A skill command resolves to `{type: "skill", message}` and is submitted as a normal prompt.

**Rule:** the desktop slash palette's curation is about hiding noise (terminal-only / messaging-only built-ins), NOT about hiding user-activated extensions. Skill commands and `quick_commands` are extensions the backend surfaces — they belong in completions. If you tighten `desktop-slash-commands.ts`, keep `isDesktopSlashExtensionCommand` flowing into both the suggestion and catalog-filter paths. Tests: `apps/desktop/src/lib/desktop-slash-commands.test.ts` (run via the repo-root `vitest`, since `apps/desktop` resolves deps from the root workspace install).

---


## Dependency Pinning Policy

All dependencies must have upper bounds to limit supply-chain attack surface.
This policy was established after the litellm compromise (PR #2796, #2810) and
reinforced after the Mini Shai-Hulud worm campaign (May 2026).

| Source type | Treatment | Example |
|---|---|---|
| PyPI package | `>=floor,<next_major` | `"httpx>=0.28.1,<1"` |
| Git URL | Commit SHA | `git+https://...@<40-char-sha>` |
| GitHub Actions | Commit SHA + comment | `uses: actions/checkout@<sha>  # v4` |
| CI-only pip | `==exact` | `pyyaml==6.0.2` |

**When adding a new dependency to `pyproject.toml`:**
1. Pin to `>=current_version,<next_major` for post-1.0 (e.g. `>=1.5.0,<2`).
2. For pre-1.0 packages, use `<0.(current_minor + 2)` (e.g. `>=0.29,<0.32`).
3. Never commit a bare `>=X.Y.Z` without a ceiling — CI and reviewers will reject it.
4. Run `uv lock` to regenerate `uv.lock` with hashes.

Reference: #2810 (bounds pass), #9801 (SHA pinning + audit CI).

---


## Skin/Theme System

The skin engine (`hermes_cli/skin_engine.py`) provides data-driven CLI visual customization. Skins are **pure data** — no code changes needed to add a new skin.

### Architecture

```
hermes_cli/skin_engine.py    # SkinConfig dataclass, built-in skins, YAML loader
~/.hermes/skins/*.yaml       # User-installed custom skins (drop-in)
```

- `init_skin_from_config()` — called at CLI startup, reads `display.skin` from config
- `get_active_skin()` — returns cached `SkinConfig` for the current skin
- `set_active_skin(name)` — switches skin at runtime (used by `/skin` command)
- `load_skin(name)` — loads from user skins first, then built-ins, then falls back to default
- Missing skin values inherit from the `default` skin automatically

### What skins customize

| Element | Skin Key | Used By |
|---------|----------|---------|
| Banner panel border | `colors.banner_border` | `banner.py` |
| Banner panel title | `colors.banner_title` | `banner.py` |
| Banner section headers | `colors.banner_accent` | `banner.py` |
| Banner dim text | `colors.banner_dim` | `banner.py` |
| Banner body text | `colors.banner_text` | `banner.py` |
| Response box border | `colors.response_border` | `cli.py` |
| Spinner faces (waiting) | `spinner.waiting_faces` | `display.py` |
| Spinner faces (thinking) | `spinner.thinking_faces` | `display.py` |
| Spinner verbs | `spinner.thinking_verbs` | `display.py` |
| Spinner wings (optional) | `spinner.wings` | `display.py` |
| Tool output prefix | `tool_prefix` | `display.py` |
| Per-tool emojis | `tool_emojis` | `display.py` → `get_tool_emoji()` |
| Agent name | `branding.agent_name` | `banner.py`, `cli.py` |
| Welcome message | `branding.welcome` | `cli.py` |
| Response box label | `branding.response_label` | `cli.py` |
| Prompt symbol | `branding.prompt_symbol` | `cli.py` |

### Built-in skins

- `default` — Classic Hermes gold/kawaii (the current look)
- `ares` — Crimson/bronze war-god theme with custom spinner wings
- `mono` — Clean grayscale monochrome
- `slate` — Cool blue developer-focused theme

### Adding a built-in skin

Add to `_BUILTIN_SKINS` dict in `hermes_cli/skin_engine.py`:

```python
"mytheme": {
    "name": "mytheme",
    "description": "Short description",
    "colors": { ... },
    "spinner": { ... },
    "branding": { ... },
    "tool_prefix": "┊",
},
```

### User skins (YAML)

Users create `~/.hermes/skins/<name>.yaml`:

```yaml
name: cyberpunk
description: Neon-soaked terminal theme

colors:
  banner_border: "#FF00FF"
  banner_title: "#00FFFF"
  banner_accent: "#FF1493"

spinner:
  thinking_verbs: ["jacking in", "decrypting", "uploading"]
  wings:
    - ["⟨⚡", "⚡⟩"]

branding:
  agent_name: "Cyber Agent"
  response_label: " ⚡ Cyber "

tool_prefix: "▏"
```

Activate with `/skin cyberpunk` or `display.skin: cyberpunk` in config.yaml.

---


### Skill authoring standards (HARDLINE)

Every new or modernized skill — bundled, optional, or contributed —
must meet these standards before merge. Reviewers reject PRs that
violate them.

1. **`description` ≤ 60 characters, one sentence, ends with a period.**
   Long descriptions bloat skill listings and dilute the model's
   attention when many skills are loaded. State the capability, not
   the implementation. No marketing words ("powerful",
   "comprehensive", "seamless", "advanced"). Don't repeat the skill
   name. Verify with:
   ```python
   import re, pathlib
   m = re.search(r'^description: (.*)$',
                 pathlib.Path('skills/<cat>/<name>/SKILL.md').read_text(),
                 re.MULTILINE)
   assert len(m.group(1)) <= 60, len(m.group(1))
   ```

2. **Tools referenced in SKILL.md prose must be native Hermes tools or
   MCP servers the skill explicitly expects.** When the skill needs a
   capability, point at the proper tool by name in backticks
   (`` `terminal` ``, `` `web_extract` ``, `` `read_file` ``,
   `` `patch` ``, `` `search_files` ``, `` `vision_analyze` ``,
   `` `browser_navigate` ``, `` `delegate_task` ``, etc.). Do NOT
   name shell utilities the agent already has wrapped — `grep` →
   `search_files`, `cat`/`head`/`tail` → `read_file`, `sed`/`awk` →
   `patch`, `find`/`ls` → `search_files target='files'`. If the skill
   depends on an MCP server, name the MCP server and document the
   expected setup in `## Prerequisites`. Anything else (third-party
   CLIs, shell pipelines, etc.) is fair game inside script files but
   should not be the headline interaction surface in the prose.

3. **`platforms:` gating audited against actual script imports.**
   Skills that use POSIX-only primitives (`fcntl`, `termios`,
   `os.setsid`, `os.kill(pid, 0)` for liveness, `/proc`, `/tmp`
   hardcoded, `signal.SIGKILL`, bash heredocs, `osascript`, `apt`,
   `systemctl`) must declare their supported platforms. Default
   posture: try to fix it cross-platform first — `tempfile.gettempdir`,
   `pathlib.Path`, `psutil.pid_exists`, Python-level filtering instead
   of `grep`. Gate to a narrower set only when the dependency is
   genuinely platform-bound.

4. **`author` credits the human contributor first.** For external
   contributions, the contributor's real name + GitHub handle goes
   first; "Hermes Agent" is the secondary collaborator. If the
   contributor's commit shows "Hermes Agent" as author (because they
   used Hermes to draft the skill), replace it with their actual name
   — credit the human, not the tool.

5. **SKILL.md body uses the modern section order.** `# <Skill> Skill`
   title, 2-3 sentence intro stating what it does and doesn't do,
   `## When to Use`, `## Prerequisites`, `## How to Run`,
   `## Quick Reference`, `## Procedure`, `## Pitfalls`,
   `## Verification`. Target ~200 lines for a complex skill,
   ~100 lines for a simple one. Cut redundant intro fluff, marketing
   prose, and re-explanations of env vars already in
   `## Prerequisites`.

6. **Scripts go in `scripts/`, references in `references/`,
   templates in `templates/`.** Don't expect the model to inline-write
   parsers, XML walkers, or non-trivial logic every call — ship a
   helper script. Reference it from SKILL.md by path relative to the
   skill directory.

7. **Tests live at `tests/skills/test_<skill>_skill.py`** and use only
   stdlib + pytest + `unittest.mock`. No live network calls. Run via
   `scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q`.

8. **`.env.example` additions are isolated to a clearly delimited
   block.** Don't touch the surrounding file — contributor-supplied
   `.env.example` versions are usually stale and edits outside the
   skill's own block must be dropped during salvage.

The full salvage / modernization checklist for external skill PRs
lives in the `hermes-agent-dev` skill at
`references/new-skill-pr-salvage.md` — load it before polishing
contributor skill PRs.

---


## Profiles: Multi-Instance Support

Hermes supports **profiles** — multiple fully isolated instances, each with its own
`HERMES_HOME` directory (config, API keys, memory, sessions, skills, gateway, etc.).

The core mechanism: `_apply_profile_override()` in `hermes_cli/main.py` sets
`HERMES_HOME` before any module imports. All `get_hermes_home()` references
automatically scope to the active profile.

### Rules for profile-safe code

1. **Use `get_hermes_home()` for all HERMES_HOME paths.** Import from `hermes_constants`.
   NEVER hardcode `~/.hermes` or `Path.home() / ".hermes"` in code that reads/writes state.
   ```python
   # GOOD
   from hermes_constants import get_hermes_home
   config_path = get_hermes_home() / "config.yaml"

   # BAD — breaks profiles
   config_path = Path.home() / ".hermes" / "config.yaml"
   ```

2. **Use `display_hermes_home()` for user-facing messages.** Import from `hermes_constants`.
   This returns `~/.hermes` for default or `~/.hermes/profiles/<name>` for profiles.
   ```python
   # GOOD
   from hermes_constants import display_hermes_home
   print(f"Config saved to {display_hermes_home()}/config.yaml")

   # BAD — shows wrong path for profiles
   print("Config saved to ~/.hermes/config.yaml")
   ```

3. **Module-level constants are fine** — they cache `get_hermes_home()` at import time,
   which is AFTER `_apply_profile_override()` sets the env var. Just use `get_hermes_home()`,
   not `Path.home() / ".hermes"`.

4. **Tests that mock `Path.home()` must also set `HERMES_HOME`** — since code now uses
   `get_hermes_home()` (reads env var), not `Path.home() / ".hermes"`:
   ```python
   with patch.object(Path, "home", return_value=tmp_path), \
        patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}):
       ...
   ```

5. **Gateway platform adapters should use token locks** — if the adapter connects with
   a unique credential (bot token, API key), call `acquire_scoped_lock()` from
   `gateway.status` in the `connect()`/`start()` method and `release_scoped_lock()` in
   `disconnect()`/`stop()`. This prevents two profiles from using the same credential.
   See `plugins/platforms/irc/adapter.py` for the canonical pattern.

6. **Profile operations are HOME-anchored, not HERMES_HOME-anchored** — `_get_profiles_root()`
   returns `Path.home() / ".hermes" / "profiles"`, NOT `get_hermes_home() / "profiles"`.
   This is intentional — it lets `hermes -p coder profile list` see all profiles regardless
   of which one is active.


## Testing

**ALWAYS use `scripts/run_tests.sh`** — do not call `pytest` directly. The script enforces
hermetic environment parity with CI (unset credential vars, TZ=UTC, LANG=C.UTF-8,
`-n auto` xdist workers, in-tree subprocess-isolation plugin). Direct `pytest`
on a 16+ core developer machine with API keys set diverges from CI in ways
that have caused multiple "works locally, fails in CI" incidents (and the reverse).

```bash
scripts/run_tests.sh                                  # full suite, CI-parity
scripts/run_tests.sh tests/gateway/                   # one directory
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # one test
scripts/run_tests.sh -v --tb=long                     # pass-through pytest flags
```

### Subprocess-per-test-file isolation

Every test file runs in a freshly-spawned Python subprocess via `run_tests_parallel.py`. This means module-level dicts/sets and
ContextVars from one test file cannot leak into the next.

### Why the wrapper

|                     | Without wrapper                             | With wrapper                              |
| ------------------- | ------------------------------------------- | ----------------------------------------- |
| Provider API keys   | Whatever is in your env (auto-detects pool) | All env vars except a specific few unset. |
| HOME / `~/.hermes/` | Your real config+auth.json                  | Temp dir per test                         |
| Timezone            | Local TZ (PDT etc.)                         | UTC                                       |
| Locale              | Whatever is set                             | C.UTF-8                                   |


### Don't write change-detector tests

A test is a **change-detector** if it fails whenever data that is **expected
to change** gets updated — model catalogs, config version numbers,
enumeration counts, hardcoded lists of provider models. These tests add no
behavioral coverage; they just guarantee that routine source updates break
CI and cost engineering time to "fix."

**Do not write:**

```python
# catalog snapshot — breaks every model release
assert "gemini-2.5-pro" in _PROVIDER_MODELS["gemini"]
assert "MiniMax-M2.7" in models

# config version literal — breaks every schema bump
assert DEFAULT_CONFIG["_config_version"] == 21

# enumeration count — breaks every time a skill/provider is added
assert len(_PROVIDER_MODELS["huggingface"]) == 8
```

**Do write:**

```python
# behavior: does the catalog plumbing work at all?
assert "gemini" in _PROVIDER_MODELS
assert len(_PROVIDER_MODELS["gemini"]) >= 1

# behavior: does migration bump the user's version to current latest?
assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]

# invariant: no plan-only model leaks into the legacy list
assert not (set(moonshot_models) & coding_plan_only_models)

# invariant: every model in the catalog has a context-length entry
for m in _PROVIDER_MODELS["huggingface"]:
    assert m.lower() in DEFAULT_CONTEXT_LENGTHS_LOWER
```

The rule: if the test reads like a snapshot of current data, delete it. If
it reads like a contract about how two pieces of data must relate, keep it.
When a PR adds a new provider/model and you want a test, make the test
assert the relationship (e.g. "catalog entries all have context lengths"),
not the specific names.

Reviewers should reject new change-detector tests; authors should convert
them into invariants before re-requesting review.
