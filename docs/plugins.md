# Hermes Plugin Author Guide

Plugins are Python packages that extend Hermes without touching core: new
tools the model can call, slash commands, lifecycle hooks, behavior-changing
middleware, gateway platforms, model/memory/media backends, skills, and CLI
subcommands. This guide documents the full authoring surface — everything a
plugin can register, where plugins live, how they load, and how to publish
one.

The source of truth is `hermes_cli/plugins.py` (the `PluginContext` facade
and `VALID_HOOKS`) and `hermes_cli/middleware.py` (`VALID_MIDDLEWARE`). When
this doc and the code disagree, the code wins — please open a docs PR.

## Table of contents

- [Quick start](#quick-start)
- [Where plugins live](#where-plugins-live)
- [The manifest: plugin.yaml](#the-manifest-pluginyaml)
- [Plugin kinds and how each loads](#plugin-kinds-and-how-each-loads)
- [Enabling and disabling](#enabling-and-disabling)
- [The register(ctx) entry point](#the-registerctx-entry-point)
- [Registering tools](#registering-tools)
- [Slash commands and CLI subcommands](#slash-commands-and-cli-subcommands)
- [Lifecycle hooks](#lifecycle-hooks)
  - [Hook catalog](#hook-catalog)
- [Middleware](#middleware)
- [Backend providers](#backend-providers)
- [Gateway platforms](#gateway-platforms)
- [Auxiliary LLM tasks](#auxiliary-llm-tasks)
- [Skills](#skills)
- [Host LLM access, message injection, tool dispatch](#host-llm-access-message-injection-tool-dispatch)
- [State and concurrency](#state-and-concurrency)
- [Distributing your plugin](#distributing-your-plugin)
- [Debugging](#debugging)
- [Converting a waiting PR into a plugin](#converting-a-waiting-pr-into-a-plugin)

## Quick start

A plugin is a directory with two files: a `plugin.yaml` manifest and an
`__init__.py` exposing `register(ctx)`.

```
~/.hermes/plugins/
└── my-first-plugin/
    ├── plugin.yaml
    └── __init__.py
```

`plugin.yaml`:

```yaml
name: my-first-plugin
version: 1.0.0
description: "Blocks rm -rf and adds a /coin slash command."
author: "@you"
provides_hooks:
  - pre_tool_call
```

`__init__.py`:

```python
import random


def _guard_rm(tool_name: str = "", args: dict | None = None, **_kwargs):
    """Block destructive shell commands before they reach approval."""
    if tool_name != "terminal":
        return None
    command = (args or {}).get("command", "")
    if "rm -rf" in command:
        return {"action": "block", "message": "Blocked by my-first-plugin: rm -rf"}
    return None


def _coin(raw_args: str) -> str:
    return f"🪙 {random.choice(['heads', 'tails'])}"


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _guard_rm)
    ctx.register_command("coin", _coin, description="Flip a coin")
```

Enable and verify:

```bash
hermes plugins enable my-first-plugin
hermes plugins list          # should show it enabled
HERMES_PLUGINS_DEBUG=1 hermes chat   # verbose discovery logs if it doesn't
```

That's the whole loop. Everything below is reference for the rest of the
surface.

## Where plugins live

Plugins are discovered from four sources, in order. Later sources override
earlier ones on name collision, so a user plugin with the same name as a
bundled plugin replaces it.

| # | Source | Location | Notes |
|---|--------|----------|-------|
| 1 | Bundled | `<repo>/plugins/<name>/` | Ships with hermes-agent. `memory/` and `context_engine/` subdirs have their own discovery paths. Override with `HERMES_BUNDLED_PLUGINS` (used by Nix/packaged installs). |
| 2 | User | `~/.hermes/plugins/<name>/` | Where `hermes plugins install` puts things. |
| 3 | Project | `./.hermes/plugins/<name>/` | Opt-in via `HERMES_ENABLE_PROJECT_PLUGINS` (untrusted-by-default: it's repo-controlled code). |
| 4 | Pip | any installed package | Exposes the `hermes_agent.plugins` entry-point group. |

Nested category plugins (e.g. `plugins/image_gen/openai/`) get a
path-derived registry key (`image_gen/openai`). Flat plugins use their
directory name (`disk-cleanup`). That key is what `plugins.enabled`,
`plugins.disabled`, and `hermes plugins list` match on.

## The manifest: plugin.yaml

```yaml
name: my-plugin              # required; defaults to the directory name
version: 1.2.0
description: "One-liner shown in hermes plugins list."
author: "@you"
kind: standalone             # standalone | backend | exclusive | platform | model-provider
manifest_version: 1          # optional; installer currently supports 1
requires_env:                # env vars the plugin needs (documentational)
  - MY_API_KEY
provides_tools:              # what the plugin registers (documentational)
  - my_tool
provides_hooks:
  - pre_tool_call
```

Unknown `kind` values fall back to `standalone` with a warning. Two
auto-coercions exist for user-installed plugins that omit `kind`: source
containing `register_memory_provider`/`MemoryProvider` is treated as
`exclusive` (memory provider), and `register_provider` + `ProviderProfile`
as `model-provider`.

## Plugin kinds and how each loads

Loading is kind- and source-aware (see `PluginManager.discover_and_load`):

| Kind | Bundled | User-installed |
|------|---------|----------------|
| `standalone` | opt-in via `plugins.enabled` | opt-in via `plugins.enabled` |
| `backend` | **auto-loads** (ships with Hermes, must just work) | opt-in via `plugins.enabled` |
| `platform` | **auto-registers lazily** — a deferred loader keyed on the platform name; the heavy SDK imports happen on first real use (gateway/cron/setup/send_message) | opt-in via `plugins.enabled` |
| `exclusive` | skipped by the general loader; the category's own discovery activates exactly one provider via `<category>.provider` config (e.g. memory) | same |
| `model-provider` | skipped by the general loader; `providers/__init__.py` lazily discovers on first `get_provider_profile()` call | same |

An explicit entry in `plugins.disabled` always wins, for every kind and
source.

## Enabling and disabling

Plugins are **opt-in by default**. The relevant `config.yaml` keys:

```yaml
plugins:
  enabled:                 # allow-list; only these load
    - my-first-plugin
    - image_gen/openai     # nested plugins use the path-derived key
  disabled:                # deny-list; always wins over enabled
    - some-plugin
  entries:                 # per-plugin trust grants
    my-first-plugin:
      allow_tool_override: true    # let this plugin replace built-in tools
      # llm: ...                   # ctx.llm override grants (see plugin_llm)
```

CLI shortcuts: `hermes plugins enable <key>`, `hermes plugins disable <key>`,
`hermes plugins list`, and an interactive `hermes plugins toggle`.

## The register(ctx) entry point

Every directory plugin's `__init__.py` must expose:

```python
def register(ctx) -> None:
    ...
```

`ctx` is a `PluginContext` — the complete facade a plugin gets. Everything
your plugin does starts from one of its methods:

| Surface | Method(s) |
|---------|-----------|
| Tools the model can call | `register_tool` |
| In-session slash commands | `register_command` |
| Terminal subcommands (`hermes foo`) | `register_cli_command` |
| Lifecycle hooks (observe/steer) | `register_hook` |
| Behavior-changing middleware | `register_middleware` |
| Gateway platform adapters | `register_platform` |
| Slack Block Kit actions | `register_slack_action_handler` |
| Backend providers | `register_context_engine`, `register_image_gen_provider`, `register_video_gen_provider`, `register_web_search_provider`, `register_browser_provider`, `register_tts_provider`, `register_transcription_provider`, `register_dashboard_auth_provider` |
| Auxiliary LLM tasks | `register_auxiliary_task` |
| Skills | `register_skill` |
| Host-owned LLM calls | `ctx.llm` |
| Conversation injection | `inject_message` |
| Calling tools from plugin code | `dispatch_tool` |
| Profile awareness | `ctx.profile_name` |

`register()` runs at discovery time in every Hermes process (CLI, gateway,
kanban workers, cron). Keep it cheap: register callbacks, don't open
connections. Defer expensive setup to first use (see
[State and concurrency](#state-and-concurrency)).

## Registering tools

```python
from tools.registry import tool_error, tool_result

MY_TOOL_SCHEMA = {
    "name": "my_tool",
    "description": "What the model reads when deciding to call it.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

def _handle_my_tool(args: dict, **kw) -> str:
    query = args.get("query", "")
    if not query:
        return tool_error("query is required")
    return tool_result({"answer": do_the_thing(query)})

def register(ctx) -> None:
    ctx.register_tool(
        name="my_tool",
        toolset="my-plugin",          # groups tools for /tools toggling
        schema=MY_TOOL_SCHEMA,        # flat {name, description, parameters} dict
        handler=_handle_my_tool,      # fn(args: dict, **kw) -> str (JSON string)
        check_fn=None,                # optional () -> bool availability gate
        requires_env=None,            # env vars this tool needs
        is_async=False,
        description="Shown in tool listings.",
        emoji="🔧",
        override=False,
    )
```

This delegates to `tools.registry.register()`, so plugin tools appear
alongside built-ins. Handlers return a JSON **string** — use the
`tool_result()` / `tool_error()` helpers from `tools.registry` — and should
accept `**kw` for forward-compatible dispatch kwargs. See
`plugins/spotify/tools.py` for a full worked example.

**Overriding built-in tools** is possible but gated. `override=True`
against a built-in name raises `PluginToolOverrideError` unless the
operator has opted in:

```yaml
plugins:
  entries:
    my-plugin:
      allow_tool_override: true
```

Bundled plugins are exempt (an override there is a maintainer decision).
The gate exists because an enabled plugin that silently replaces
`shell_exec` or `write_file` could exfiltrate everything the model routes
through it.

## Slash commands and CLI subcommands

**Slash commands** work in CLI *and* gateway sessions (Telegram, Discord,
Slack, …):

```python
def handler(raw_args: str) -> str | None:   # async also supported
    return f"you said: {raw_args}"

ctx.register_command(
    "mycmd", handler,
    description="Shown in command pickers",
    args_hint="<file>",    # lets Discord surface an argument field
)
```

Names are normalized (lowercased, `/` stripped, spaces → `-`). Conflicts
with built-in commands are rejected with a warning.

**CLI subcommands** add `hermes <name>` terminal commands:

```python
def setup(subparser):                 # receives an argparse subparser
    subparser.add_argument("--flag")

def run(args) -> None: ...

ctx.register_cli_command("mycli", help="...", setup_fn=setup, handler_fn=run)
```

## Lifecycle hooks

```python
ctx.register_hook("pre_tool_call", my_callback)
```

Semantics that apply to **every** hook:

- Callbacks are invoked with keyword arguments only. Always accept
  `**kwargs` so new fields never break you.
- All registered callbacks run, in registration order (plugin load order).
- Exceptions are caught and logged — a broken hook can never break the
  agent loop.
- Non-`None` return values are collected; each hook documents what (if
  anything) it does with them. Undocumented returns are ignored, so
  observer-style callbacks can return freely.
- Unknown hook names register with a warning instead of failing, so a
  plugin written against a newer Hermes still loads on an older one.
- Every payload carries `telemetry_schema_version` (`hermes.observer.v1`).

### Hook catalog

Canonical list: `VALID_HOOKS` in `hermes_cli/plugins.py`.

#### Tool lifecycle

| Hook | Fires | Return contract |
|------|-------|-----------------|
| `pre_tool_call` | before each tool executes (after request middleware) | `{"action": "block", "message": "..."}` blocks the call; the message is returned to the model. First valid block wins. Anything else = observer. Kwargs include `tool_name`, `args`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `middleware_trace`. |
| `post_tool_call` | after each tool executes | observer |
| `transform_tool_result` | after a tool returns, before the result goes to the model | return a **string** to replace the tool result; first valid string wins; non-strings ignored |
| `transform_terminal_output` | terminal tool output | return a string to replace the output |

#### LLM / API lifecycle

| Hook | Fires | Return contract |
|------|-------|-----------------|
| `pre_llm_call` | before each LLM call | `{"context": "..."}` (or a plain string) injects ephemeral context into the current turn's **user message** — never the system prompt, so the prompt-cache prefix survives. Injected context is never persisted to the session DB. |
| `post_llm_call` | after each LLM call | observer |
| `transform_llm_output` | before the response text reaches the user | return a string to replace the response; first non-None string wins |
| `pre_api_request` / `post_api_request` | around the raw provider API request | observer |
| `api_request_error` | when a provider API call raises | observer |
| `classify_api_error` *(pending [#58524](https://github.com/NousResearch/hermes-agent/pull/58524))* | at the top of `classify_api_error()`, **before** the built-in pipeline, once per failed API call | return `None` to pass, or `{"reason": "<FailoverReason name>", ...}` with optional `retryable` / `should_compress` / `should_rotate_credential` / `should_fallback` / `message` / `error_context` overrides. First valid result wins. Self-scope on the `provider` kwarg. Lets a model-provider plugin own its provider's error quirks. |

#### Verification

| Hook | Fires | Return contract |
|------|-------|-----------------|
| `pre_verify` | once per turn when the agent edited code and is about to verify/finish | `{"action": "continue", "message": "<follow-up>"}` keeps the agent going (run a check, tidy the diff). The Claude-Code Stop shape `{"decision": "block", "reason": "..."}` is accepted too. Bounded by `agent.max_verify_nudges`. |

#### Session lifecycle

`on_session_start`, `on_session_end`, `on_session_finalize`,
`on_session_reset`, `subagent_start`, `subagent_stop` — observers for
session and subagent boundaries. Use `on_session_end` for cleanup of
session-scoped resources.

#### Gateway

| Hook | Fires | Return contract |
|------|-------|-----------------|
| `pre_gateway_dispatch` | once per incoming `MessageEvent`, after the internal-event guard, **before** auth/pairing and agent dispatch | `{"action": "skip", "reason": "..."}` drops the message; `{"action": "rewrite", "text": "..."}` replaces `event.text`; `{"action": "allow"}` / `None` = normal dispatch. Kwargs: `event`, `gateway`, `session_store`. |

#### Approvals (observers only)

`pre_approval_request` / `post_approval_response` fire when a dangerous
command needs user approval — both for CLI-interactive prompts and
gateway/ACP approvals. Return values are **ignored**: plugins cannot veto
or pre-answer an approval here (use `pre_tool_call` to block a tool before
it reaches approval). Kwargs: `command`, `description`, `pattern_key`,
`pattern_keys`, `session_key`, `surface` (`"cli"` | `"gateway"`);
`post_approval_response` adds `choice`
(`"once" | "session" | "always" | "deny" | "timeout"`).

#### Kanban (observers only)

`kanban_task_claimed` / `kanban_task_completed` / `kanban_task_blocked`
fire after a task transition is committed to the board DB. Note which
process fires each: `claimed` fires in the **dispatcher** process;
`completed` and `blocked` fire in the **worker** subprocess. A plugin that
needs a central view hooks the dispatcher; per-task in-session context
hooks the worker. Common kwargs: `task_id`, `board`, `assignee`, `run_id`,
`profile_name`; plus `summary` (completed) / `reason` (blocked).

## Middleware

Hooks observe (with a few documented steering shapes). **Middleware
changes what happens**: request middleware rewrites the effective payload;
execution middleware wraps the actual call. Canonical list:
`VALID_MIDDLEWARE` in `hermes_cli/middleware.py`.

```python
ctx.register_middleware("tool_request", my_middleware)
```

| Kind | Contract |
|------|----------|
| `tool_request` | receives `tool_name`, `args`, `original_args`, context kwargs. Return `{"args": {...}}` to replace the effective tool arguments before hooks, guardrails, approvals, and execution see them. |
| `llm_request` | receives `request`, `original_request`, context kwargs. Return `{"request": {...}}` to replace the effective provider kwargs before Hermes sends them. |
| `tool_execution` | wraps tool execution. Receives the payload plus `next_call`; must call `next_call(payload)` exactly once to run the downstream chain (or not call it, to short-circuit) and return the result. |
| `llm_execution` | same shape, wrapping the provider call. |

Details that matter in practice:

- Request middleware chains: each callback sees the payload as rewritten
  by earlier callbacks; `original_*` always carries the pre-middleware
  copy. Payloads are deep-copied between callbacks — mutate freely.
- Optionally include `source` / `reason` / `name` strings in your returned
  dict; they land in the middleware trace for observability.
- `next_call` in execution middleware is **single-use**. Calling it twice
  raises (it would re-run the provider or tool).
- A middleware that raises is logged and skipped; the chain continues, and
  a downstream failure after your `next_call` propagates as itself. You
  cannot break the base runtime path from middleware.
- Payloads carry `middleware_schema_version` (`hermes.middleware.v1`).

## Backend providers

All backend registrations share one shape: subclass the abstract provider,
instantiate it, register it, and it becomes selectable by name in config.
Non-conforming providers are rejected with a warning (never an exception).

| Method | Base class | Config selector |
|--------|-----------|-----------------|
| `register_web_search_provider` | `agent.web_search_provider.WebSearchProvider` | `web.search_backend` / `web.extract_backend` / `web.backend` |
| `register_browser_provider` | `agent.browser_provider.BrowserProvider` | `browser.cloud_provider` |
| `register_image_gen_provider` | `agent.image_gen_provider` base | `image_gen.provider` |
| `register_video_gen_provider` | `agent.video_gen_provider` base | `video_gen.provider` |
| `register_tts_provider` | `agent.tts_provider.TTSProvider` | `tts.provider` |
| `register_transcription_provider` | `agent.transcription_provider.TranscriptionProvider` | `stt.provider` |
| `register_dashboard_auth_provider` | dashboard auth base | dashboard config |
| `register_context_engine` | `agent.context_engine.ContextEngine` | single slot — first registration wins; later attempts are rejected with a warning |

**A context engine owns the entire compression strategy.** The
`ContextEngine` ABC includes `should_compress()`, `compress()`,
`should_compress_preflight()`, `has_content_to_compress()`, session
lifecycle callbacks, and even engine-specific tool schemas — so
"custom compaction policy" features (when to compress, how to split,
how to summarize) can ship today as a context-engine plugin rather
than patches to the default engine.

Name-collision precedence for TTS/STT: built-in provider names always win,
and a `type: command` provider defined in config wins over a plugin
provider with the same name (config is more local than plugin install).

**Model providers** don't use `PluginContext` at all: a `model-provider`
plugin calls `register_provider()` from `providers/` with a
`ProviderProfile` and is discovered lazily by `providers/__init__.py`. See
`plugins/model-providers/README.md`. Note that a `ProviderProfile`
already owns its **model catalog**: `models_url` (explicit catalog
endpoint), an overridable `fetch_models()` (custom response shapes, or
`None` for providers with no REST catalog), and `fallback_models` (the
curated `/model`-picker list when the live fetch fails) — catalog
source and refresh behavior belong in the provider plugin, not core. **Memory providers** are `exclusive`
plugins activated via `memory.provider` config — see `plugins/memory/`.

## Gateway platforms

```python
ctx.register_platform(
    name="irc",
    label="IRC",
    adapter_factory=lambda cfg: IRCAdapter(cfg),   # PlatformConfig -> BasePlatformAdapter
    check_fn=lambda: True,                          # dependency check, called first
    validate_config=None,
    required_env=["IRC_SERVER"],
    install_hint="pip install irc",
    emoji="💬",
    setup_fn=irc_interactive_setup,                 # extra PlatformEntry kwargs OK
)
```

Bundled platform plugins register **deferred** loaders — the module (and
its heavy SDK imports) load on first real use, keeping `hermes chat`
startup fast. Your user-installed platform plugin loads normally once
enabled.

For Slack specifically, `register_slack_action_handler(action_id, callback)`
wires Block Kit interactivity into the Slack adapter's `AsyncApp`;
`action_id` accepts a string, compiled regex, or constraint dict, and the
async callback follows the slack_bolt `(ack, body, action)` convention.

## Auxiliary LLM tasks

Auxiliary tasks are LLM-backed side jobs (vision, compression, web extract,
smart approval, …) with their own `auxiliary.<key>` config block, so users
can pin a provider/model for them independent of the main chat model.
Plugins can declare their own:

```python
ctx.register_auxiliary_task(
    key="my_plugin_summarizer",           # snake_case; don't shadow built-ins
    display_name="My summarizer",
    description="What it does, one line",
    defaults={"provider": "auto", "timeout": 30},
)
```

After registration the task appears in the `hermes model` auxiliary picker,
gets `AUXILIARY_<KEY>_*` env bridging at gateway startup, and default
routing fields are merged into loaded configs. Reserved built-in keys:
`vision`, `compression`, `web_extract`, `approval`, `mcp`,
`title_generation`, `skills_hub`, `curator`. Duplicate keys across plugins
are rejected.

## Skills

```python
ctx.register_skill("my-skill", Path(__file__).parent / "skills/my-skill/SKILL.md",
                   description="...")
```

Plugin skills resolve as `<plugin_name>:<name>` via `skill_view()`. They do
**not** enter the flat `~/.hermes/skills/` tree and are **not** listed in
the system prompt's `<available_skills>` index — explicit loads only.

## Host LLM access, message injection, tool dispatch

- **`ctx.llm`** — a host-owned `PluginLlm` facade for chat/structured
  completions against the user's active model and auth, so plugins don't
  bring their own keys. Model/agent/auth overrides are fail-closed and
  gated via `plugins.entries.<plugin_id>.llm.*`. See `agent/plugin_llm.py`.
- **`ctx.inject_message(content, role="user")`** — queue a message into the
  active conversation (starts a turn if idle, interrupts if running).
  Interactive-CLI only; returns `False` in gateway mode.
- **`ctx.dispatch_tool(tool_name, args, **kwargs)`** — call any registered
  tool (e.g. `delegate_task`) from plugin code with parent-agent context
  resolved automatically. Returns the tool's JSON string.
- **`ctx.profile_name`** — the active profile (`"default"`, a profile id,
  or `"custom"`), derived from `HERMES_HOME`; works in CLI, gateway, and
  kanban workers alike.

## State and concurrency

Hermes sessions are multi-threaded (delegated tool calls, background
workers, the self-improvement fork), so the classic lazy-singleton
pattern —

```python
_client = None

def get_client():
    global _client
    if _client is None:          # two threads can both pass this check
        _client = ExpensiveClient()
    return _client
```

— is a race that leaks connections. Use the shared primitives in
`plugins/plugin_utils.py` instead:

```python
from plugins.plugin_utils import lazy_singleton, SingletonSlot

@lazy_singleton
def get_client():
    return ExpensiveClient(load_config())

get_client()          # built exactly once, safe across threads
get_client.reset()    # for tests/teardown
```

`SingletonSlot` covers accessors that take a config argument
(first-config-wins semantics). Both are stdlib-only imports.

## Distributing your plugin

This is the point of the plugin system: **your feature ships without
waiting for a core merge.** Publish a repo; users install it with:

```bash
hermes plugins install owner/repo                 # GitHub shorthand
hermes plugins install https://github.com/owner/repo.git
hermes plugins install git@github.com:owner/repo.git
hermes plugins install owner/repo/path/to/plugin  # subdirectory of a repo
hermes plugins install https://github.com/owner/repo/tree/main/path  # browser URL
```

Installs land in `~/.hermes/plugins/<name>/`. Related commands:
`hermes plugins update <name>`, `remove <name>`, `enable`, `disable`,
`list`, `toggle`.

Ship these in your repo:

- `plugin.yaml` — with `manifest_version: 1` (the currently supported
  installer version).
- `after-install.md` — shown to the user right after install; put env
  setup, config keys, and the `hermes plugins enable <name>` reminder here.
- `README.md` — what it does, config reference.

Remember: user-installed plugins are always opt-in via `plugins.enabled`,
and tool overrides additionally require the per-plugin
`allow_tool_override` grant. Design for that trust model — fail gracefully
when your env vars are missing (declare them in `requires_env`), and never
assume you're the only plugin registered on a hook.

## Debugging

- `HERMES_PLUGINS_DEBUG=1` — tees verbose plugin-discovery logs to stderr:
  which directories were scanned, which manifests parsed, which plugins
  were skipped (and why), what each `register(ctx)` registered, and full
  tracebacks on load failure. Everything also lands in
  `~/.hermes/logs/agent.log`.
- `hermes plugins list` — shows each discovered plugin's status, including
  the load error when one didn't activate (e.g. `not enabled in config`).
- Common gotchas:
  - Plugin doesn't load → it's opt-in; run `hermes plugins enable <key>`.
  - Nested plugin → enable by **key** (`image_gen/openai`), not name.
  - `PluginToolOverrideError` → the operator must set
    `plugins.entries.<key>.allow_tool_override: true`.
  - Hook never fires → check the name against `VALID_HOOKS` (typos load
    with only a warning) and accept `**kwargs` in your callback.
  - Heavy work in `register()` → slows every Hermes invocation; defer to
    first use with `lazy_singleton`.

## Converting a waiting PR into a plugin

If your PR has been in the queue for a while, check whether it fits an
existing seam — you can ship it **today** as an installable plugin and
convert the PR into a much smaller interface request (or close it):

| Your PR is... | The plugin surface |
|---------------|--------------------|
| a new tool / integration | `register_tool` + a toolset |
| a policy/security gate (block commands, protect paths, rate-limit) | `pre_tool_call` hook or `tool_request` middleware |
| rewriting tool args or LLM requests (inject headers, clamp params) | `tool_request` / `llm_request` middleware |
| provider error-classification quirks | `classify_api_error` hook (pending [#58524](https://github.com/NousResearch/hermes-agent/pull/58524)) |
| context injection / recall on every turn | `pre_llm_call` |
| output filtering/transformation | `transform_llm_output`, `transform_tool_result` |
| a messaging platform | `register_platform` (kind: `platform`) |
| a media/search/TTS/STT backend | the matching `register_*_provider` |
| a compression/compaction strategy | `register_context_engine` (the ABC owns `should_compress`/`compress`) |
| a model provider or model-catalog fix | `ProviderProfile` plugin (`fetch_models` / `models_url` / `fallback_models`) |
| a memory backend | `exclusive` memory-provider plugin |
| an LLM side-task with its own model routing | `register_auxiliary_task` |

If your change doesn't fit any seam, that's exactly the feedback the
maintainers asked for: say **which hook or registration point would have
made your PR a plugin** in the plugin-interface discussion, ideally with
the callback signature you wish existed.
