# PR Spec: Claude Code ACP as a First-Class Hermes Provider

**Date:** 2026-04-16
**Branch:** `docs/claude-code-acp-pr-spec`

> **Intent:** This is the PR-level spec for adding Claude Code as a first-class ACP-backed provider in Hermes, instead of limiting Claude ACP support to ad hoc `delegate_task(acp_command=...)` usage.

## TL;DR

Hermes should add a new explicit provider:

- **Provider slug:** `claude-code-acp`
- **Display name:** `Claude Code ACP`
- **Transport:** local or remote ACP server, surfaced to Hermes as `chat_completions`
- **Default launch command:** `claude --acp --stdio`
- **Default marker base URL:** `acp://claude-code`

**Strong opinion:** this should **not** be implemented as a one-off clone of the current `copilot-acp` path.
The right shape is:

1. **generalize** the current Copilot-specific ACP plumbing into a reusable external-process ACP substrate,
2. add `claude-code-acp` as the **second** first-class `external_process` provider,
3. keep `anthropic` as the existing **native Anthropic Messages API** path,
4. keep generic `delegate_task(acp_command=..., acp_args=...)` support for power users.

That gives Hermes three clean Claude-related options:

- `anthropic` → native API transport
- `claude-code-acp` → Claude Code CLI as the active backend transport
- `delegate_task(... acp_command="claude" ...)` → manual child-agent orchestration

Those are meaningfully different products and should stay distinct.

---

## Why this PR exists

Today Hermes already has enough ACP plumbing to prove Claude Code works, but not enough productization to make it feel native.

### What exists already

After inspecting the current Hermes codebase:

- `tools/delegate_tool.py` already supports generic ACP child delegation via `acp_command` + `acp_args`
- `hermes_cli/auth.py` already has an `external_process` provider type
- `copilot-acp` is the only productized first-class external-process provider today
- `agent/copilot_acp_client.py` already provides an OpenAI-compatible shim over ACP stdio
- `anthropic` already supports Claude Code credentials for the **native** Anthropic API path

### What is missing

Claude Code is **not** selectable as a first-class provider in the main setup/model flow.
The current code is hardcoded around Copilot in several places:

- `hermes_cli/auth.py`
- `hermes_cli/runtime_provider.py`
- `hermes_cli/main.py`
- `agent/auxiliary_client.py`
- `run_agent.py`
- `agent/copilot_acp_client.py`
- provider/model registries and tests

So Hermes can use Claude ACP today only through manual delegation, not as a normal top-level provider.

---

## Grounding from current reality

This spec is based on two important facts established during investigation.

### 1. Claude Code ACP works locally

On this machine, `claude --acp --stdio` works and Hermes successfully used it through the generic ACP delegation path.

### 2. Local Codex ACP does not currently work

On this machine, the installed Codex CLI does **not** accept `--acp --stdio`, and a Hermes ACP smoke test against Codex failed.

That matters because it means:

- this PR should focus on **Claude Code ACP**, not “generic Codex ACP” productization,
- Hermes should avoid implying that “ACP provider” means Codex compatibility today,
- any future Codex ACP provider should be a separate follow-up once the CLI actually supports ACP in practice.

---

## Product goal

Make Claude Code usable as a normal Hermes provider with the same level of polish users get from other first-class providers.

That means a user should be able to:

1. select **Claude Code ACP** in `hermes setup` / provider selection,
2. let Hermes detect `claude` locally or use a configured remote ACP endpoint,
3. pick a default Claude model hint,
4. persist the provider + model in `config.yaml`,
5. run Hermes normally in CLI, gateway, cron, and ACP-server contexts,
6. have Hermes launch Claude Code as the active backend automatically.

---

## Non-goals

This PR should **not** try to:

- replace the existing `anthropic` provider,
- merge `anthropic` and Claude Code ACP into one ambiguous “Claude” option,
- auto-detect and silently switch users from `anthropic` to `claude-code-acp`,
- invent a fully generic “any ACP server” provider UX in v1,
- expose Claude Code-specific slash commands through Hermes,
- guarantee model-level parity with Claude Code internals beyond passing Hermes’ requested model as a hint,
- productize Codex ACP before the local Codex CLI actually supports ACP.

---

## Recommendation

## Add a new explicit provider: `claude-code-acp`

I recommend the canonical provider ID be:

- **`claude-code-acp`**

with aliases such as:

- `claude-acp`
- `claude-code`

### Why this name is correct

`claude-code-acp` is a little verbose, but it avoids two major ambiguities:

1. **`anthropic` already exists** and is the native API path.
2. **Claude Code credentials already work with `anthropic`**, so `claude-code` alone would blur credential source and transport.

The slug should encode the transport choice, not just the model family.

### Why not just reuse `anthropic`

Because these are different transport layers:

- `anthropic` = Hermes talks directly to Anthropic’s API
- `claude-code-acp` = Hermes talks to the Claude Code CLI, which then acts as the model/tool agent backend

Conflating those would create confusing setup, debugging, and support behavior.

### Why not “delegation-only”

Because the generic delegation hook is good for power users, but not sufficient UX for a first-class provider:

- no provider selection UI
- no persisted status/config flow
- no top-level runtime resolution
- no coherent docs/tests/story for ongoing support

---

## Architecture recommendation

## Generalize the current Copilot ACP path into a reusable ACP client layer

The cleanest implementation is:

### Step 1: introduce a generic ACP chat shim

Refactor `agent/copilot_acp_client.py` into a generic implementation, for example:

- new file: `agent/acp_chat_client.py`
- new class: `ACPChatCompletionsClient`

Then either:

- keep `agent/copilot_acp_client.py` as a thin compatibility wrapper/re-export, or
- update imports in the same PR and remove the old name.

### Why this refactor is worth doing now

The current client is only nominally Copilot-specific. The actual mechanics are mostly generic:

- spawn a subprocess
- speak ACP JSON-RPC over stdio
- turn ACP responses into the minimal OpenAI-ish shape Hermes expects
- preserve tool-call blocks

What is Copilot-specific today is mostly naming and default env/base-url assumptions.

Without this refactor, adding Claude ACP would duplicate the same subprocess/ACP shim with a different command name. That is the wrong abstraction boundary.

---

## Provider metadata changes

### Extend `ProviderConfig` for external-process providers

Right now `ProviderConfig` is good for API-key and OAuth providers, but external-process providers are effectively hardcoded in helper functions.

Add first-class fields for subprocess-backed providers, e.g.:

- `default_command: str = ""`
- `command_env_vars: tuple[str, ...] = ()`
- `args_env_var: str = ""`
- `default_args: tuple[str, ...] = ()`
- `missing_command_hint: str = ""`
- `base_url_env_var: str = ""` (already exists)
- `inference_base_url: str = ""` (already exists)

Then define providers like:

### `copilot-acp`
- `default_command = "copilot"`
- `command_env_vars = ("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH")`
- `args_env_var = "HERMES_COPILOT_ACP_ARGS"`
- `default_args = ("--acp", "--stdio")`
- `inference_base_url = "acp://copilot"`

### `claude-code-acp`
- `default_command = "claude"`
- `command_env_vars = ("HERMES_CLAUDE_CODE_ACP_COMMAND", "CLAUDE_CODE_CLI_PATH")`
- `args_env_var = "HERMES_CLAUDE_CODE_ACP_ARGS"`
- `default_args = ("--acp", "--stdio")`
- `inference_base_url = "acp://claude-code"`
- `missing_command_hint = "Install Claude Code or set HERMES_CLAUDE_CODE_ACP_COMMAND/CLAUDE_CODE_CLI_PATH."`

**Strong opinion:** use explicit typed fields on `ProviderConfig`, not a bag of `extra[...]` strings. These values are part of core runtime behavior.

---

## Runtime behavior

### Resolved runtime shape

`resolve_runtime_provider(...)` should return the same general shape for `claude-code-acp` that it does for `copilot-acp`:

```python
{
  "provider": "claude-code-acp",
  "api_mode": "chat_completions",
  "base_url": "acp://claude-code",
  "api_key": "claude-code-acp",
  "command": "/path/to/claude",
  "args": ["--acp", "--stdio"],
  "source": "process",
}
```

### Why `api_mode` stays `chat_completions`

Internally Hermes is still talking to an OpenAI-compatible shim object. The ACP subprocess is the transport detail behind that shim.

So v1 should keep:

- `api_mode = "chat_completions"`

for all external-process ACP providers.

### Marker base URLs

Use marker URLs to preserve the current ergonomics:

- `acp://copilot`
- `acp://claude-code`

and continue allowing remote ACP endpoints with:

- `acp+tcp://host:port`

That keeps compatibility with the existing “no local command needed if remote ACP URL is configured” behavior.

---

## UX recommendation

## Claude Code ACP should be an explicit provider choice in setup/model flows

### Provider picker

Add `Claude Code ACP` anywhere provider choices are listed, including:

- `hermes setup`
- `/model` or equivalent provider switch flow
- canonical provider registries / labels / aliases

### Setup flow behavior

Add a dedicated `_model_flow_claude_code_acp(...)` in `hermes_cli/main.py`.

This flow should:

1. resolve the Claude Code command / args / base URL,
2. print a short explanation that Hermes will delegate requests to `claude --acp`,
3. show the resolved command or remote ACP URL,
4. offer a curated Claude model list,
5. allow a custom model name entry,
6. persist the provider selection in `config.yaml`.

### Model selection policy

Unlike `copilot-acp`, Hermes should **not** try to fetch a live model catalog here unless Claude ACP exposes a stable supported discovery path that Hermes can rely on.

For v1, use a curated list plus custom entry. Example list:

- `claude-opus-4.6`
- `claude-sonnet-4.6`
- `claude-sonnet-4.5`
- `claude-haiku-4.5`
- `Enter custom model name`

**Important:** Hermes should describe this clearly as a **model hint**, not a guaranteed CLI-enforced server-side model selection. That matches the current ACP shim behavior.

### Persisted config shape

```yaml
model:
  default: claude-sonnet-4.6
  provider: claude-code-acp
  base_url: acp://claude-code
  api_mode: chat_completions
```

---

## Auth and status behavior

### `get_external_process_provider_status(...)`

Refactor it to be provider-driven, not Copilot-driven.

It should:

1. look up provider metadata,
2. resolve command from provider-specific env vars,
3. resolve args from the provider-specific args env var,
4. resolve base URL from the provider’s base-url env var or default marker URL,
5. mark `configured=true` if either:
   - the local command resolves, or
   - the base URL is `acp+tcp://...`

For `claude-code-acp`, status output should surface:

- command
- resolved command
- args
- base_url
- configured / logged_in

### v1 status scope

I do **not** think this PR needs a deep `claude auth status` integration.

Why:

- it adds provider-specific subprocess probing logic,
- auth output/behavior may change across Claude Code versions,
- Hermes already treats `copilot-acp` as “command exists / remote endpoint exists” rather than “auth definitely valid.”

So in v1, “status” should mean **transport is configured**, not “Claude Code subscription is definitely healthy.”

Runtime errors from Claude Code should still bubble up clearly.

---

## Relationship to the existing `anthropic` provider

This is the most important product distinction in the PR.

### Keep `anthropic` unchanged

The existing `anthropic` provider should remain:

- native Anthropic API transport
- `api_mode = "anthropic_messages"`
- capable of using Anthropic API keys or Claude Code credential files / OAuth-derived tokens where Hermes already supports them

### Do not auto-upgrade or auto-route

Hermes should **not** automatically switch a user to `claude-code-acp` just because:

- `claude` exists on PATH
- Claude Code is logged in
- Claude Code credentials exist on disk

Why:

- transport changes affect latency, tooling behavior, debugging, and failure modes
- silent switching would be surprising
- many users will prefer the native Anthropic API path for stability and simplicity

### Recommendation for `auto`

In v1, `auto` should **not** include `claude-code-acp` in the automatic provider-resolution chain.

Make it explicit-selection only.

That avoids accidental provider flips and keeps the rollout safer.

---

## File-by-file implementation plan

## 1. Generic ACP client layer

### Create / refactor
- `agent/acp_chat_client.py` — new generic ACP subprocess-backed OpenAI-compatible shim

### Update or keep as compatibility wrapper
- `agent/copilot_acp_client.py`

### Required changes
- replace Copilot-specific defaults/messages with provider-supplied values
- allow arbitrary `acp://...` marker base URLs
- keep support for `command`, `args`, `acp_command`, `acp_args`, and `acp_cwd`

---

## 2. Provider registry and auth resolution

### Modify
- `hermes_cli/auth.py`

### Required changes
- add `DEFAULT_CLAUDE_CODE_ACP_BASE_URL = "acp://claude-code"`
- add `ProviderConfig("claude-code-acp", ...)`
- add aliases for `claude-acp` / `claude-code`
- generalize `get_external_process_provider_status(...)`
- generalize `resolve_external_process_provider_credentials(...)`
- remove Copilot-only error text from shared helpers

---

## 3. Runtime provider resolution

### Modify
- `hermes_cli/runtime_provider.py`

### Required changes
- treat `claude-code-acp` like `copilot-acp`
- ideally generalize the current `if provider == "copilot-acp"` branch into a provider-type-driven branch for `external_process`

---

## 4. Main model/provider flow

### Modify
- `hermes_cli/main.py`

### Required changes
- add provider selection branch for `claude-code-acp`
- add `_model_flow_claude_code_acp(...)`
- mirror the high-level shape of `_model_flow_copilot_acp(...)`
- use curated Claude model defaults instead of GitHub model-catalog fetching
- persist `provider`, `base_url`, and `api_mode`

---

## 5. Provider/model registries

### Modify
- `hermes_cli/models.py`
- `hermes_cli/model_normalize.py`
- `hermes_cli/providers.py`
- `agent/model_metadata.py`

### Required changes
- add `claude-code-acp` to canonical provider lists
- add labels + aliases
- add curated model list
- normalize `anthropic/claude-sonnet-4.6` → `claude-sonnet-4.6` for this provider like other strip-vendor providers
- add a Hermes provider overlay for `claude-code-acp`, matching the same ACP-transport conventions Hermes already uses for `copilot-acp`
- document the overlay choice clearly so future ACP providers do not reintroduce Copilot-specific assumptions

---

## 6. Main client construction path

### Modify
- `run_agent.py`
- `agent/auxiliary_client.py`

### Required changes
- stop hardcoding `copilot-acp` as the only ACP-backed direct client path
- instantiate the new generic ACP client for any supported external-process ACP provider
- do not special-case only `acp://copilot`
- support `acp://claude-code` and, ideally, provider-type-driven ACP detection

A good end-state is:

- `run_agent.py` decides “ACP client vs OpenAI client” by provider type / ACP scheme, not by Copilot slug
- `agent/auxiliary_client.py` does the same for auxiliary usage

---

## 7. Setup wizard integration

### Modify
- `hermes_cli/setup.py`

### Required changes
- include `claude-code-acp` in provider selection labels/defaults
- preserve current behavior where same-provider credential-pool prompts do **not** appear for external-process providers
- ensure vision-setup messaging still behaves sensibly for a provider that does not expose a Hermes-usable vision backend

---

## 8. Documentation

### Modify
- `docs/acp-setup.md`
- `docs/specs/claude-code-acp-pr-spec.md` (this file)

### Suggested additions
- short user-facing section: “Using Claude Code ACP as Hermes’ backend”
- clarify the distinction between:
  - Anthropic native provider
  - Claude Code credentials reused by Anthropic
  - Claude Code ACP transport

---

## Testing plan

Add or update tests in at least these files:

### CLI auth/provider tests
- `tests/hermes_cli/test_api_key_providers.py`
  - provider registry includes `claude-code-acp`
  - status resolution uses Claude env vars and defaults
  - credential resolution returns `acp://claude-code`
  - missing command error mentions Claude env vars

### provider persistence / setup tests
- `tests/hermes_cli/test_model_provider_persistence.py`
  - `_model_flow_claude_code_acp()` persists provider/base_url/model/api_mode together
- `tests/hermes_cli/test_setup_model_provider.py`
  - setup flow can save `claude-code-acp`
  - no same-provider fallback prompt for this external-process provider

### auxiliary client tests
- `tests/agent/test_auxiliary_client.py`
  - `resolve_provider_client("claude-code-acp")` instantiates the generic ACP client
  - requires configured or explicit model like `copilot-acp`

### run_agent tests
- `tests/run_agent/test_run_agent.py`
  - AIAgent chooses ACP client for `claude-code-acp`
- `tests/run_agent/test_run_agent_codex_responses.py`
  - if relevant, ensure no stale `copilot-acp` assumptions remain

### URL validation / metadata tests
- `tests/agent/test_proxy_and_url_validation.py`
  - `acp://claude-code` is accepted
- any provider metadata tests covering canonical provider lists / aliases

---

## Rollout and compatibility

### Backward compatibility

This PR should not break `copilot-acp`.

That means:

- Copilot env vars keep working
- `acp://copilot` keeps working
- tests for Copilot ACP remain green
- any compatibility wrapper around `CopilotACPClient` continues to work during the refactor

### Safe rollout strategy

1. Generalize the ACP substrate first.
2. Add `claude-code-acp` on top.
3. Keep `auto` unchanged.
4. Do not remove `copilot-acp` special cases until equivalent generic coverage exists in tests.

---

## Risks

### 1. Model selection is only a hint in v1

The current ACP shim passes Hermes’ selected model through the prompt transcript, not a provider-native guaranteed model switch.

That is acceptable for v1, but the UX copy should be honest about it.

### 2. Claude CLI auth/runtime failures may surface only at request time

Because v1 status is transport-oriented, a user may still have a broken Claude session when a real request runs.

That is acceptable if the resulting error is clear and actionable.

### 3. Overfitting shared code to Copilot assumptions

This is the main implementation trap.

If the refactor is shallow and keeps Copilot-specific names/messages/env assumptions in “shared” code, the Claude provider will work technically but remain brittle and confusing.

---

## Open questions

### Should Hermes expose Claude-specific auth diagnostics in v1?

My recommendation: **no**. Keep the first PR focused on transport productization.

### Should the generic ACP substrate become a public “custom ACP provider” feature?

My recommendation: **not in this PR**. First prove the abstraction with two providers:

- `copilot-acp`
- `claude-code-acp`

If that feels good, a later PR can consider a user-defined ACP provider surface.

---

## Concrete recommendation summary

If I were implementing this PR, I would make these exact decisions:

1. **Introduce `claude-code-acp` as a new first-class provider.**
2. **Generalize the current Copilot ACP path instead of duplicating it.**
3. **Keep `anthropic` separate and unchanged.**
4. **Keep `auto` unchanged; no silent Claude ACP auto-selection.**
5. **Use a curated Claude model list plus custom entry.**
6. **Treat the selected model as a hint in v1.**
7. **Preserve `copilot-acp` compatibility while refactoring.**

That is the smallest coherent product increment that makes Claude Code ACP feel truly native inside Hermes.

---

## Appendix: why this is better than the status quo

Today Hermes already has the raw pieces to spawn Claude Code through ACP, but the experience is fragmented:

- manual delegation works,
- main provider selection does not,
- provider resolution is Copilot-specific,
- direct-client codepaths are Copilot-specific,
- docs do not present Claude ACP as a supported top-level mode.

After this PR, Hermes would have a clean story:

- **Anthropic** if you want direct API access
- **Claude Code ACP** if you want Claude Code itself to be the backend agent transport
- **Generic ACP child delegation** if you want manual advanced orchestration

That is a much better product surface.