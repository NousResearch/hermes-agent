# TUI Refactor: Current to Ideal

Date: 2026-04-01

## Scope

- same repo refactor
- keep Python runtime
- replace PT-based interactive shell
- add Ink UI through a local gateway

## Current Environment

Interactive path is centered in `cli.py` with `prompt_toolkit` and `rich`.

Current technical shape:

- PT app shell and key handling in `cli.py`
  - `Application`, `KeyBindings`, `TextArea`, `patch_stdout`
- queue control in `cli.py`
  - `_pending_input`
  - `_interrupt_queue`
- approval and sudo callback globals in `tools/terminal_tool.py`
  - `_approval_callback`
  - `_sudo_password_callback`
- runtime entry in `run_agent.py`
  - `AIAgent.run_conversation()`
  - `AIAgent.chat()`

Current constraint:

- UI logic and runtime control are mixed, so UI replacement is expensive.

## Ideal Environment

Interactive path is split into three layers:

1. Ink UI (Node/TS)
2. local `tui_gateway` over stdio JSON-RPC
3. Python runtime (`AIAgent`, tools, sessions)

Rules for ideal state:

- no direct UI to `AIAgent` calls
- no PT dependency in gateway path
- keep current Hermes state/config contracts
  - `~/.hermes/.env`
  - `~/.hermes/config.yaml`
  - `~/.hermes/state.db`
  - profile behavior via `HERMES_HOME`

## Migration Path

## Cut 1: Headless Controller

Add:

- `tui_gateway/controller.py`
- `tui_gateway/session_state.py`
- `tui_gateway/events.py`

Change:

- `run_agent.py` callback wiring for controller events
- `cli.py` compatibility bridge into controller

Done:

- create/resume/prompt/interrupt/cancel work with no PT imports

## Cut 2: Local Gateway

Add:

- `tui_gateway/protocol.py`
- `tui_gateway/server.py`
- `tui_gateway/entry.py`

Methods:

- `session.create`
- `session.resume`
- `session.list`
- `session.interrupt`
- `session.cancel`
- `prompt.submit`
- `approval.respond`
- `sudo.respond`
- `clarify.respond`

Events:

- `message.delta`
- `tool.progress`
- `approval.requested`
- `sudo.requested`
- `clarify.requested`
- `error`

Done:

- simple client completes full prompt cycle through JSON-RPC

## Cut 3: Ink UI

Add:

- `ui-tui/src/main.tsx`
- `ui-tui/src/gatewayClient.ts`
- `ui-tui/src/state/store.ts`
- `ui-tui/src/components/Transcript.tsx`
- `ui-tui/src/components/Composer.tsx`
- `ui-tui/src/components/StatusBar.tsx`
- `ui-tui/src/components/ApprovalModal.tsx`
- `ui-tui/src/components/SudoPrompt.tsx`
- `ui-tui/src/components/ClarifyPrompt.tsx`

Change:

- `tools/terminal_tool.py` prompt adapters for gateway round-trip

Done:

- chat, tools, approval, sudo, clarify, interrupt all work through gateway

## Cut 4: Opt-In and Rollout

Entry points:

- `hermes --tui`
- `HERMES_EXPERIMENTAL_TUI=1`
- `display.experimental_tui: true`
- `/tui`, `/tui on`, `/tui off`, `/tui status`

Behavior:

- `/tui` starts gateway if needed and attaches
- failed attach falls back to PT mode with explicit error text
- `/tui off` disables auto-launch only

Rollout:

1. internal opt-in
2. external opt-in beta
3. default-on after checks pass
4. remove PT path later

## Acceptance Checks

- runtime: no PT import in controller/gateway path
- state: same config/profile/session continuity
- commands: slash command registry remains `hermes_cli/commands.py`
- permissions: approval/sudo/clarify protocol round-trip
- streaming: incremental assistant and tool updates
- opt-in: flag/env/config/slash command share one launch path

## Test Commands

- `python -m pytest tests/tui_gateway/test_controller.py -q`
- `python -m pytest tests/tui_gateway/test_protocol.py tests/tui_gateway/test_server_flow.py -q`
- `python -m pytest tests/tui_gateway/test_permissions_roundtrip.py -q`
- `python -m pytest tests/hermes_cli/test_tui_opt_in.py -q`
- `cd ui-tui && npm run build`
- `cd ui-tui && npm run test`
# Prompt Toolkit to Ink Migration Plan

Date: 2026-04-01

## Scope

This is a refactor in the same repo.

- no new repo
- no runtime rewrite
- no messaging gateway reuse for terminal UI

## Current Environment

Interactive Hermes today is `prompt_toolkit` plus `rich` inside `cli.py`.

Current structure:

- PT app shell and input handling in `cli.py`
  - `Application`
  - `KeyBindings`
  - `TextArea`
  - `patch_stdout`
- queue-based control flow in `cli.py`
  - `_pending_input`
  - `_interrupt_queue`
- approval and sudo callbacks in `tools/terminal_tool.py`
  - `_approval_callback`
  - `_sudo_password_callback`
- core runtime in `run_agent.py`
  - `AIAgent.run_conversation()`
  - `AIAgent.chat()`

Current issue:

- UI framework logic and runtime control flow are mixed in one path.
- Tool prompt routing depends on PT callback globals.
- Replacing UI without changing runtime is harder than it should be.

## Ideal Environment

Interactive Hermes is Ink UI plus a local TUI gateway.

Target model:

- Python runtime remains the source of truth.
- UI talks to `tui_gateway` over stdio JSON-RPC.
- `tui_gateway` talks to `AIAgent`.
- no direct UI to `AIAgent` coupling.

Target compatibility:

- same `~/.hermes/.env`
- same `~/.hermes/config.yaml`
- same `~/.hermes/state.db`
- same profile behavior through `HERMES_HOME`

## How To Get There

Use three delivery cuts and one switch cut.

## Cut 1: Headless Runtime Controller

Goal:

- separate runtime control from PT.

Add:

- `tui_gateway/controller.py`
- `tui_gateway/session_state.py`
- `tui_gateway/events.py`

Change:

- `run_agent.py` callback wiring needed by controller
- `cli.py` compatibility calls into controller

Done when:

- controller supports create/resume/prompt/interrupt/cancel
- controller path imports no PT modules
- tool progress and assistant deltas are typed events

## Cut 2: Local TUI Gateway

Goal:

- add stable protocol boundary for UI.

Add:

- `tui_gateway/protocol.py`
- `tui_gateway/server.py`
- `tui_gateway/entry.py`
- `tui_gateway/__init__.py`

Protocol methods:

- `session.create`
- `session.resume`
- `session.list`
- `session.interrupt`
- `session.cancel`
- `prompt.submit`
- `approval.respond`
- `sudo.respond`
- `clarify.respond`

Protocol events:

- `message.delta`
- `tool.progress`
- `approval.requested`
- `sudo.requested`
- `clarify.requested`
- `error`

Done when:

- a simple client can complete one full prompt cycle over stdio JSON-RPC

## Cut 3: Ink UI

Goal:

- usable clone flow through gateway.

Add:

- `ui-tui/package.json`
- `ui-tui/src/main.tsx`
- `ui-tui/src/gatewayClient.ts`
- `ui-tui/src/state/store.ts`
- `ui-tui/src/components/Transcript.tsx`
- `ui-tui/src/components/Composer.tsx`
- `ui-tui/src/components/StatusBar.tsx`
- `ui-tui/src/components/ApprovalModal.tsx`
- `ui-tui/src/components/SudoPrompt.tsx`
- `ui-tui/src/components/ClarifyPrompt.tsx`

Change:

- `tools/terminal_tool.py` adapters for gateway request/response prompt routing

Done when:

- user can chat, run tools, approve, deny, clarify, interrupt, and continue

## Cut 4: Opt-In Switch and Rollout

Goal:

- ship without forced cutover.

Entry points:

- `hermes --tui`
- `HERMES_EXPERIMENTAL_TUI=1`
- `display.experimental_tui: true`
- `/tui`, `/tui on`, `/tui off`, `/tui status`

Behavior:

- `/tui` starts gateway if needed, then attaches
- attach failure returns to PT mode with clear error text
- `/tui off` disables auto-launch only

Rollout sequence:

1. internal opt-in
2. external opt-in beta
3. default-on after checks pass
4. PT path removal later

## Acceptance Checks

- runtime
  - no PT import in controller or gateway path
  - deterministic interrupt/cancel
- state
  - same config, profile, and session continuity
- commands
  - slash command registry remains centralized in `hermes_cli/commands.py`
- permissions
  - approval, sudo, clarify round-trip through protocol
- streaming
  - incremental assistant and tool updates
- opt-in
  - flag, env, config, and slash command trigger the same launch path

## Test Commands

- `python -m pytest tests/tui_gateway/test_controller.py -q`
- `python -m pytest tests/tui_gateway/test_protocol.py tests/tui_gateway/test_server_flow.py -q`
- `python -m pytest tests/tui_gateway/test_permissions_roundtrip.py -q`
- `python -m pytest tests/hermes_cli/test_tui_opt_in.py -q`
- `cd ui-tui && npm run build`
- `cd ui-tui && npm run test`

## Non-Goals

- no ACP extraction work as prerequisite
- no new repository split
- no direct UI to `AIAgent` coupling
- no PT feature parity before gateway path is stable
# Prompt Toolkit to Ink Migration Plan

Date: 2026-04-01

## Scope

This is a refactor in the same repo.

- no new repo
- no runtime rewrite
- no messaging gateway reuse for terminal UI

## Current Environment (As-Is)

Interactive Hermes today is `prompt_toolkit` plus `rich` inside `cli.py`.

Facts from code:

- PT imports and app shell in `cli.py`
  - `Application`, `KeyBindings`, `TextArea`, `patch_stdout`
- PT queue control path in `cli.py`
  - `_pending_input` for normal input
  - `_interrupt_queue` for input while agent is running
- tool approval and sudo prompts use CLI callbacks in `tools/terminal_tool.py`
  - `_sudo_password_callback`
  - `_approval_callback`
- core agent runtime is Python in `run_agent.py`
  - `AIAgent.run_conversation()`
  - `AIAgent.chat()`

Current coupling problem:

- UI framework and runtime control flow are mixed in `cli.py`.
- Tool prompts depend on CLI callback globals.
- This blocks clean UI replacement.

## Ideal Environment (To-Be)

Interactive Hermes is Ink UI plus a local TUI gateway.

### Runtime

- `AIAgent` stays in Python.
- Tool execution stays in Python.
- Session storage and config remain unchanged.

### Boundary

- new `tui_gateway` process over stdio JSON-RPC
- UI talks only to gateway
- gateway talks to `AIAgent`

### UI

- Node/TypeScript Ink app
- transcript, composer, status, approvals, clarify, sudo, interrupt

### Compatibility

Use existing Hermes state and config:

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/state.db`
- profile behavior via `HERMES_HOME`

## How To Get There

Use three implementation cuts plus one switch cut.

## Cut 1: Headless Runtime Controller

Goal: separate runtime control flow from PT.

Add:

- `tui_gateway/controller.py`
- `tui_gateway/session_state.py`
- `tui_gateway/events.py`

Change:

- `run_agent.py` only for callback wiring needed by controller
- `cli.py` to call controller APIs in compatibility mode

Done when:

- controller can create/resume/prompt/interrupt/cancel without importing PT
- tool progress and assistant deltas are emitted as typed events

## Cut 2: Local TUI Gateway

Goal: protocol boundary between UI and runtime.

Add:

- `tui_gateway/protocol.py`
- `tui_gateway/server.py`
- `tui_gateway/entry.py`
- `tui_gateway/__init__.py`

Protocol methods:

- `session.create`
- `session.resume`
- `session.list`
- `session.interrupt`
- `session.cancel`
- `prompt.submit`
- `approval.respond`
- `sudo.respond`
- `clarify.respond`

Protocol events:

- `message.delta`
- `tool.progress`
- `approval.requested`
- `sudo.requested`
- `clarify.requested`
- `error`

Done when:

- a simple client can run one full prompt cycle over stdio JSON-RPC

## Cut 3: Ink UI

Goal: usable clone experience through gateway.

Add:

- `ui-tui/package.json`
- `ui-tui/src/main.tsx`
- `ui-tui/src/gatewayClient.ts`
- `ui-tui/src/state/store.ts`
- `ui-tui/src/components/Transcript.tsx`
- `ui-tui/src/components/Composer.tsx`
- `ui-tui/src/components/StatusBar.tsx`
- `ui-tui/src/components/ApprovalModal.tsx`
- `ui-tui/src/components/SudoPrompt.tsx`
- `ui-tui/src/components/ClarifyPrompt.tsx`

Change:

- `tools/terminal_tool.py` adapters so prompts round-trip through gateway path, not PT-only callbacks

Done when:

- user can chat, run tools, approve, deny, clarify, interrupt, and continue

## Cut 4: Opt-In Switch and Rollout

Goal: ship safely without forced cutover.

Entry points:

- `hermes --tui`
- `HERMES_EXPERIMENTAL_TUI=1`
- `display.experimental_tui: true`
- `/tui`, `/tui on`, `/tui off`, `/tui status` in legacy CLI

Behavior:

- `/tui` starts gateway if needed, then attaches
- attach failure returns to PT mode with clear error text
- `/tui off` disables auto-launch only

Rollout:

1. internal opt-in
2. external opt-in beta
3. default-on only after acceptance checks pass
4. PT path removal later

## Acceptance Checks

- runtime
  - no PT import in controller or gateway path
  - deterministic interrupt/cancel
- state
  - same config, profile, and session continuity
- commands
  - slash command registry stays centralized in `hermes_cli/commands.py`
- permissions
  - approval, sudo, clarify all round-trip through protocol
- streaming
  - incremental assistant and tool updates
- opt-in
  - flag, env, config, and slash command all trigger same launch path

## Test Commands

- `python -m pytest tests/tui_gateway/test_controller.py -q`
- `python -m pytest tests/tui_gateway/test_protocol.py tests/tui_gateway/test_server_flow.py -q`
- `python -m pytest tests/tui_gateway/test_permissions_roundtrip.py -q`
- `python -m pytest tests/hermes_cli/test_tui_opt_in.py -q`
- `cd ui-tui && npm run build`
- `cd ui-tui && npm run test`

## Non-Goals

- no ACP extraction work as prerequisite
- no new repository split
- no direct UI to `AIAgent` coupling
- no PT feature parity before gateway path is stable
# Ink Gateway TUI Migration Plan

Date: 2026-04-01

## Goal

Replace Hermes' interactive `prompt_toolkit` CLI with a React terminal UI built on `Ink`, while keeping the Python agent and tool runtime in place.

The new design should:

- remove `prompt_toolkit` from the interactive path entirely
- keep `AIAgent`, tool execution, and session logic in Python
- introduce a transport-neutral local UI gateway between backend and frontend
- use stock `Ink` first, not a Claude Code renderer transplant
- keep using the same Hermes config, profile, skills, memory, and session storage model

## Decision Summary

Hermes should not evolve the current `prompt_toolkit` shell. The replacement architecture is:

1. Python backend session server
2. local gateway transport over stdio JSON-RPC
3. Node/TypeScript `Ink` TUI frontend

This intentionally uses a dedicated local TUI gateway and keeps `acp_adapter` unchanged.

The new TUI is a new shell, not a new runtime.

## Compatibility Requirements

From the existing Hermes docs and setup flows, the new TUI should continue to use:

- the same `~/.hermes/.env` provider/auth configuration
- the same `~/.hermes/config.yaml` settings model
- the same `~/.hermes/state.db` session store
- the same `HERMES_HOME` profile layout and isolation rules
- the same skills, memories, and slash-command registry already shared across Hermes surfaces

The migration should not create:

- a separate TUI-only config file
- a separate TUI-only session database
- a separate prompt assembly path with drift from existing Hermes runtime behavior

## Why This Shape

The current interactive CLI is too coupled to `prompt_toolkit` to incrementally clean up in place:

- `cli.py` mixes input handling, rendering, approvals, clarify flows, voice, queues, and agent orchestration
- `tools/terminal_tool.py` assumes UI callbacks installed by the CLI
- the current event model is built around PT queues and threads, not a transport-neutral session API

At the same time, a full port to Claude Code's custom renderer is the wrong first move:

- Claude Code's TUI stack is not just `Ink`; it includes a product-coupled renderer fork and app bootstrap assumptions
- Hermes does not need that complexity to reach a good first-party TUI
- stock `Ink` is enough to validate the UI model and close the biggest UX gap first

Operationally, a Node/TypeScript frontend is acceptable here because Hermes already ships with a Node-aware install story and already supports Node-based surfaces in the wider product.

## Non-Goals

- reusing the messaging gateway as the TUI transport
- preserving `prompt_toolkit` compatibility
- matching Claude Code internals one-for-one
- rewriting Hermes' core agent or tool runtime in Node

## High-Level Architecture

The new interactive stack has three layers:

1. `python runtime`
   Owns `AIAgent`, tool execution, session state, approvals, interrupts, and filesystem/terminal tools.
2. `ui gateway`
   A local protocol server that exposes Hermes sessions as typed requests, responses, and events.
3. `ink tui`
   A React terminal app that renders transcript, composer, status, approvals, tool cards, and slash-command UX.

Suggested process model:

```text
hermes
  1. launch ink tui (node)
  2. spawn python ui gateway over stdio
  3. create/resume session
  4. exchange JSON-RPC requests + streaming events
```

## Why Not The Existing Messaging Gateway

The messaging gateway solves a different problem:

- multi-platform message routing
- user authorization and pairing
- per-platform delivery behavior
- long-running bot process management

That stack is useful as architecture background, but it is the wrong seam for a local terminal app.

The ACP adapter demonstrates the right boundary shape:

- backend runtime behind a protocol boundary
- callback/event bridging
- permission round-trips
- explicit session lifecycle

The new local UI gateway should target a Hermes TUI protocol directly, not an editor protocol.

## ACP Isolation Strategy

Do not use ACP extraction as a prerequisite.

Instead:

1. build `tui_gateway` directly around `AIAgent`
2. keep `acp_adapter/*` untouched during early migration
3. allow shared-runtime refactors later only if they reduce real maintenance cost

Reasons:

- ACP payloads are editor-shaped and add translation overhead
- ACP-first migration adds scope and time before the new TUI ships
- owner direction favors a fast clone path with gateway indirection, not transport unification work

## Proposed Backend Split

Extract the following concerns out of `cli.py`:

1. `session controller`
   A headless controller for create, resume, prompt, interrupt, cancel, and slash-command dispatch.
2. `event bridge`
   Converts agent callbacks and tool progress into structured UI events.
3. `permission bridge`
   Converts dangerous-command approval, sudo prompts, and clarify requests into request/response interactions.
4. `presentation adapters`
   Optional formatting helpers for transcript items and tool previews, without owning terminal rendering.
5. `gateway adapter`
   A thin request/event layer for `tui_gateway` over stdio JSON-RPC.

The backend must stop depending on a terminal UI framework for control flow.

## Shared Runtime Invariants

The backend remains the source of truth for:

- prompt assembly
- Honcho/memory synchronization
- tool dispatch
- approval policy
- slash-command execution
- session transitions

The frontend should render protocol state, not own core agent behavior.

In particular, the new TUI must not introduce UI-side blocking work into the turn path. Context, memory, Honcho prefetch, and similar backend concerns should stay behind the runtime boundary and preserve Hermes' existing caching and async-prefetch behavior.

## Proposed Transport

Start with stdio JSON-RPC.

Reasons:

- local CLI startup is simple
- process ownership is clear
- no port management

WebSocket can be added later if Hermes wants:

- remote terminal clients
- browser UI
- multiple concurrent viewers

But it should not be the first transport.

## Platform And Toolset Strategy

The new UI should run Hermes in a dedicated `tui` platform mode.

That mode should:

- share most behavioral semantics with the current interactive CLI
- reuse the canonical slash-command registry rather than fork it
- preserve session continuity with other Hermes surfaces where the shared state model already supports it
- avoid editor-specific payload conventions in the TUI protocol

Toolset strategy:

- start from current interactive CLI capabilities
- only introduce a dedicated `hermes-tui` toolset if the transport boundary proves it is needed
- keep transport constraints out of tool business logic as much as possible

## Protocol Shape

The TUI protocol should be explicit and event-driven.

Core requests:

- `session.create`
- `session.resume`
- `session.list`
- `session.interrupt`
- `session.cancel`
- `session.set_cwd`
- `prompt.submit`
- `command.run`
- `approval.respond`
- `sudo.respond`
- `clarify.respond`

Core events:

- `session.state`
- `message.start`
- `message.delta`
- `message.complete`
- `thinking.delta`
- `tool.started`
- `tool.progress`
- `tool.completed`
- `approval.requested`
- `sudo.requested`
- `clarify.requested`
- `error`

Design rule: every user-visible interactive state in the new TUI must come from protocol state, not local UI guesswork.

## Ink Frontend Scope

The first `Ink` frontend only needs a narrow set of surfaces:

- transcript view
- input composer
- status/footer bar
- slash-command picker/help
- approval modal
- sudo prompt
- clarify prompt
- tool activity cards

Do not start with:

- mouse-heavy interaction
- custom selection model
- custom renderer internals
- Claude-style terminal instrumentation

Those can come later if real gaps appear.

## Migration Phases

## Phase 1: Headless Runtime Extraction

Goal: make Hermes usable without `prompt_toolkit`.

Work:

- introduce a backend session/controller module
- move PT-specific queues and rendering concerns out of agent flow
- replace direct CLI callback assumptions with abstract request/response hooks
- isolate slash-command execution from the PT shell
- introduce a `platform="tui"` runtime path without forking core agent logic

Exit criteria:

- a non-PT backend can run a prompt, stream progress, request approval, and return a final response

## Phase 2: Local UI Gateway

Goal: expose the backend over stdio JSON-RPC.

Work:

- create a `ui_gateway` package or equivalent module group
- model session lifecycle and event streaming
- implement cancel/interrupt behavior
- adapt terminal approval and sudo flow into transport messages
- keep config, profile, and session storage identical to existing Hermes surfaces

Exit criteria:

- a minimal client can drive a full Hermes session over stdio without importing `cli.py`

## Phase 3: Ink MVP

Goal: ship a working Hermes TUI without `prompt_toolkit`.

Work:

- create a Node/TS package for the TUI
- connect to the Python gateway
- render transcript + composer + status
- support approvals, clarify prompts, and slash commands
- preserve interrupt-and-redirect behavior for active runs

Exit criteria:

- Hermes can be used end-to-end from the new TUI for normal chat and tool use

## Phase 4: Feature Parity

Goal: close the biggest regressions from the legacy CLI.

Work:

- port session picker/resume UX
- port tool previews and long-running command status
- port config-aware commands
- port voice or explicitly defer it behind a non-blocking boundary

Exit criteria:

- daily-driver workflows no longer require the PT CLI

## Phase 5: Cutover And Deletion

Goal: make the new TUI the default interactive path.

Work:

- switch `hermes` interactive startup to the Ink client
- keep legacy PT path only behind a temporary fallback flag if needed
- delete PT-specific code after a short stabilization window

Exit criteria:

- `prompt_toolkit` is no longer part of the main interactive CLI

## File-Level Refactor Targets

Initial hot spots:

- `cli.py`
- `tools/terminal_tool.py`
- `model_tools.py`
- `run_agent.py`
- `hermes_cli/commands.py`

Expected pattern:

- avoid importing PT code into the new backend path
- move any UI-specific formatting behind protocol events or thin adapters

## First Implementation Slices (PR Plan)

Keep early PRs narrow and mergeable. Do not start with a large branch that rewrites `cli.py` end-to-end.

1. `PR-1: headless session controller`
   - add a transport-neutral controller around `AIAgent` for create/resume/prompt/interrupt/cancel
   - no UI, no PT dependencies
2. `PR-2: local ui gateway (stdio json-rpc)`
   - add `ui_gateway` process entry
   - implement protocol requests/events for one full prompt cycle
3. `PR-3: ink shell bootstrap`
   - add Node/TS package with gateway client
   - render transcript + composer + status + streaming deltas
4. `PR-4: interactive controls parity`
   - approvals, sudo, clarify flows
   - interrupt-and-redirect and command routing
5. `PR-5: startup switch + fallback flag`
   - add explicit opt-in startup flag for Ink path (`HERMES_EXPERIMENTAL_TUI=1` or equivalent)
   - add CLI/config opt-in controls and `/tui` command entrypoint in legacy CLI
   - keep PT path behind a temporary env/flag gate during stabilization
6. `PR-6: parity hardening and PT deletion`
   - close remaining UX gaps from legacy CLI
   - remove PT path after stability window

## Concrete File Plan

Use fixed locations so contributors do not invent parallel structures.

`PR-1` files:

- add `tui_gateway/controller.py`
- add `tui_gateway/session_state.py`
- add `tui_gateway/events.py`
- update `run_agent.py` only where callback wiring is needed
- update `cli.py` only to call controller entry points in compatibility mode

`PR-2` files:

- add `tui_gateway/protocol.py`
- add `tui_gateway/server.py`
- add `tui_gateway/entry.py`
- add `tui_gateway/__init__.py`
- add `tests/tui_gateway/test_protocol.py`
- add `tests/tui_gateway/test_server_flow.py`

`PR-3` files:

- add `ui-tui/package.json`
- add `ui-tui/src/main.tsx`
- add `ui-tui/src/gatewayClient.ts`
- add `ui-tui/src/state/store.ts`
- add `ui-tui/src/components/Transcript.tsx`
- add `ui-tui/src/components/Composer.tsx`
- add `ui-tui/src/components/StatusBar.tsx`

`PR-4` files:

- add `ui-tui/src/components/ApprovalModal.tsx`
- add `ui-tui/src/components/SudoPrompt.tsx`
- add `ui-tui/src/components/ClarifyPrompt.tsx`
- update `tools/terminal_tool.py` to use gateway request/response adapters instead of PT-specific assumptions
- add `tests/tui_gateway/test_permissions_roundtrip.py`

`PR-5` files:

- update `hermes_cli/main.py` startup selection for `--tui` and env/config flags
- update `hermes_cli/commands.py` with `/tui` commands
- update `cli.py` command dispatch to launch/attach behavior
- add `tests/hermes_cli/test_tui_opt_in.py`

`PR-6` files:

- remove PT-only paths from `cli.py` once parity checks pass
- remove obsolete PT wiring helpers
- update docs and command help text

If path names change, keep one module per role and avoid duplicate gateway implementations.

## Protocol Envelope (v0)

Use one JSON-RPC envelope shape for all gateway traffic.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "req-123",
  "method": "prompt.submit",
  "params": {
    "session_id": "sess-1",
    "text": "hello"
  }
}
```

Event notification:

```json
{
  "jsonrpc": "2.0",
  "method": "event",
  "params": {
    "type": "message.delta",
    "session_id": "sess-1",
    "payload": {
      "text": "hi"
    }
  }
}
```

Error:

```json
{
  "jsonrpc": "2.0",
  "id": "req-123",
  "error": {
    "code": 4001,
    "message": "session not found"
  }
}
```

Protocol rules:

- all event ordering is per-session FIFO
- ids are opaque strings
- unknown event types are ignored by clients and logged
- protocol version is pinned in `tui_gateway/protocol.py`

## Acceptance Checks Per Phase

Each phase should ship with explicit checks:

- `runtime`
  - prompt executes end-to-end without importing `prompt_toolkit`
  - interrupt and cancel are deterministic
- `state continuity`
  - same `HERMES_HOME`, `config.yaml`, `state.db`, and profile behavior as existing Hermes surfaces
- `commands`
  - slash-command resolution uses shared registry (`hermes_cli/commands.py`)
- `permissions`
  - dangerous command approval, sudo prompt, and clarify prompt all round-trip through protocol events
- `streaming`
  - message/tool progress events stream incrementally; no UI-side polling loop for core turn output
- `opt-in controls`
  - `--tui`, env flag, config toggle, and `/tui` commands all resolve to the same launch behavior
  - failures fall back to PT mode with explicit error output

## Test Commands Per PR

`PR-1`:

- `python -m pytest tests/tui_gateway/test_controller.py -q`

`PR-2`:

- `python -m pytest tests/tui_gateway/test_protocol.py tests/tui_gateway/test_server_flow.py -q`

`PR-3`:

- `cd ui-tui && npm run build`
- `cd ui-tui && npm run test`

`PR-4`:

- `python -m pytest tests/tui_gateway/test_permissions_roundtrip.py -q`
- `cd ui-tui && npm run test`

`PR-5`:

- `python -m pytest tests/hermes_cli/test_tui_opt_in.py -q`

`PR-6`:

- `python -m pytest tests/ -q`

## Opt-In UX Surface

Expose TUI opt-in through user-facing TUI language, not transport language.

Entry points:

- startup flag: `hermes --tui`
- env flag: `HERMES_EXPERIMENTAL_TUI=1`
- config toggle: `display.experimental_tui: true`
- slash command in legacy CLI:
  - `/tui` (launch/attach)
  - `/tui on` (persist opt-in)
  - `/tui off` (disable auto-launch)
  - `/tui status` (show mode + process/attach state)

Behavior:

- if `/tui` is called and the local TUI gateway is not running, start it and attach
- if already running, attach/reuse session
- on startup/attach failure, print clear error and stay in PT mode
- `/tui off` disables future auto-launch; it does not terminate active sessions unless requested

## Rollout And Rollback

Rollout should be staged:

1. internal opt-in (`HERMES_EXPERIMENTAL_TUI=1` or equivalent)
2. external opt-in beta (still flag-gated, PT remains default)
3. default-on with PT fallback still available, only after acceptance checks are green
4. PT removal after a short stability window

Rollback path must remain simple until PT deletion:

- one switch to restore legacy interactive startup
- no data migration required between TUI and PT modes (shared state model)

## Main Risks

1. `cli.py` currently owns more state than it appears to. Extraction will uncover hidden coupling.
2. Approval and sudo flows are global/callback-driven today and need per-session protocol state.
3. Long-running tool output may currently assume terminal-local behavior that has to be normalized before transport.
4. Voice mode may carry PT assumptions and should be treated as optional during the first cut.
5. If the frontend demands behavior beyond stock `Ink`, the team may need to introduce custom terminal primitives later.

## Recommendation

Start with stock `Ink` and a direct `tui_gateway` over stdio JSON-RPC.

Do not:

- refactor `prompt_toolkit` forward
- route the terminal UI through the messaging gateway
- begin by vendoring Claude Code's renderer

The shortest path to a good Hermes TUI is:

1. extract headless backend control flow
2. expose it over stdio JSON-RPC
3. build the TUI in `Ink`
4. only customize deeper terminal behavior after real product pressure appears

## Success Criteria

This migration succeeds if Hermes can:

- start an interactive session without `prompt_toolkit`
- stream assistant and tool activity live into an `Ink` UI
- handle approvals, clarify requests, sudo prompts, and interrupts cleanly
- preserve the existing Python agent/tool runtime
- preserve existing Hermes config, profile, and session continuity expectations
- preserve shared slash-command semantics instead of inventing a second command surface
- avoid adding new blocking UI-driven work into the prompt path
- make the legacy PT shell deletable rather than permanent
