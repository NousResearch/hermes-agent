# Hermes AG-UI adapter

Run Hermes as an [AG-UI](https://docs.ag-ui.com/) HTTP/SSE server, so AG-UI
clients (e.g. CopilotKit) can drive it from a browser.

## Quick start

This checkout contains a local modernization prototype, not a merge-ready
release. The three launchers and explicit setup section are integrated, and
published-client browser handoff/approval probes pass, including CopilotKit's
runtime proxy and React hooks. Real-provider acceptance, final build/upstream
integration and remaining lifecycle review are still pending.

```bash
# 1. After preparing/activating a Hermes checkout, select the extra through PM
hermes pm install --extra agui

# 2. One-time: make sure Hermes has a provider + model configured
hermes model                        # pick provider/model, set credentials

# 3. Start the server (binds http://127.0.0.1:8000 by default)
hermes agui                         # also: hermes-agui / python -m agui_adapter
```

That's the whole thing on a local machine. `hermes-agui` **reuses your existing
`hermes model` provider and credentials** — the same ones the Hermes CLI uses —
so you don't set any API keys or endpoints yourself. Because it binds loopback,
**no auth token is required** (see [Security](#security)).

You should see:

```
Starting Hermes AG-UI adapter on 127.0.0.1:8000
```

Point any AG-UI client at **`http://127.0.0.1:8000/`** (the run endpoint is
`POST /`; there's also `GET /health`).

`hermes agui`, `hermes-agui`, and `python -m agui_adapter` share the same
`--host`, `--port`, `--token`, and `--check` options. `--check` verifies imports
without starting a server or making a model request. All launchers load the
selected Hermes home's `.env`; none requires an interactive TTY to serve.

## Docker

The image's default/no-argument command is the interactive Hermes experience,
not an AG-UI service. **Pass an explicit `agui` command** for headless hosting;
a restart policy cannot turn the interactive CLI into a server.

Build this checkout (the image's `all` selection includes the AG-UI extra):

```bash
docker build -t hermes-agent:agui .
```

Configure a dedicated persistent data volume. These one-time setup commands
are interactive; the long-running service below is not. In `setup agui`, choose
`0.0.0.0` inside the container and save a strong session token when prompted.
The provider credentials and token stay in that volume's `.env`.

```bash
docker run --rm -it --mount type=volume,source=hermes-agui-data,target=/opt/data hermes-agent:agui model
docker run --rm -it --mount type=volume,source=hermes-agui-data,target=/opt/data hermes-agent:agui setup agui
docker run --name hermes-agui --publish 127.0.0.1:8000:8000 --mount type=volume,source=hermes-agui-data,target=/opt/data hermes-agent:agui agui --host 0.0.0.0 --port 8000
```

Publishing on host `127.0.0.1` limits reachability to the host; the container
still requires a token because its own bind is non-loopback. `GET /health` is
open; JSON run requests require `X-Hermes-Session-Token`. Secrets may also be
injected through the documented credential environment variable. Do not
override the image entrypoint or mount your normal Hermes home for a smoke test.
Both normal s6/PID-1 startup and Docker's `--init` wrapper are supported.

## Connecting a client

Use AG-UI's standard `HttpAgent`. With **CopilotKit**, the usual shape is a Next.js
runtime route that proxies to this server:

```ts
// app/api/copilotkit/route.ts
import { HttpAgent } from "@ag-ui/client";
const AGENT_URL = process.env.AGENT_URL ?? "http://localhost:8000";
const agent = new HttpAgent({
  url: `${AGENT_URL.replace(/\/$/, "")}/`,
  // Required when the adapter has a token; keep it on the runtime server:
  headers: process.env.HERMES_AGUI_SESSION_TOKEN
    ? { "X-Hermes-Session-Token": process.env.HERMES_AGUI_SESSION_TOKEN }
    : {},
});
// ...register `agent` with CopilotRuntime as usual
```

The browser talks to your Next.js runtime; the runtime (server-side) talks to
`hermes-agui` on `:8000`. Because that hop is server-to-server on localhost, the
loopback/no-token setup is all you need. A remote/runtime-container hop needs
an appropriately reachable adapter URL and session token. No Hermes-specific
JavaScript package or Gateway API change is required.

If installing your JavaScript client fails with npm's `Cannot read properties
of null (reading 'edgesOut')`, check the npm version before changing Hermes.
This is an [upstream peer-resolver bug](https://github.com/npm/cli/pull/8448).
The CopilotKit 1.70.3 client/runtime was validated with Node 22.23.2 and npm
11.16.0 using normal peer resolution; npm 10.9.8 reproduced the crash even in
a standalone Vitest-only project. Use a project-local npm toolchain containing
the fix; disabling peer checks is not required for this validated setup.

## Configuration

Behavioral settings live in the `agui` section of `config.yaml`
(`~/.hermes/config.yaml`), alongside every other Hermes setting — nothing is
required for the local quick start above. The **only AG-UI-specific** environment
variable is the session token, because it's a secret (Hermes reserves `.env` for
secrets). Run `hermes setup agui` for explicit setup, or edit the section:

```yaml
# ~/.hermes/config.yaml
agui:
  host: 127.0.0.1          # bind interface; any non-loopback value requires a token
  port: 8000               # listen port
  toolsets: [hermes-acp]   # Hermes toolsets exposed to the agent
  # Optional model overrides — leave blank to defer to your `hermes model` config:
  provider: ""             # e.g. "custom" for an OpenAI-compatible endpoint
  model: ""                # e.g. "gpt-4o"
  api_mode: ""             # "chat_completions" or "codex_responses"
  base_url: ""             # set to bypass the resolver and call this endpoint directly
  frontend_tool_timeout: 60       # seconds to wait for a browser tool result
  max_conversations: 16           # active + idle agents; reject excess admission
  conversation_idle_timeout: 1800 # seconds before closing an idle conversation
```

| Setting | Where | Default | Purpose |
|---|---|---|---|
| `agui.host` | `config.yaml` | `127.0.0.1` | Bind interface. Non-loopback requires a token (see Security). |
| `agui.port` | `config.yaml` | `8000` | Listen port. |
| `agui.toolsets` | `config.yaml` | `[hermes-acp]` | Hermes toolsets enabled for the agent. |
| `agui.provider` / `model` / `api_mode` / `base_url` | `config.yaml` | *(your `hermes model` config)* | Optional per-adapter model overrides; blank defers to the resolver. |
| `agui.frontend_tool_timeout` | `config.yaml` | `60` | Positive finite wait in seconds; expiry cancels the pending execution. |
| `agui.max_conversations` | `config.yaml` | `16` | Positive integer cap, including idle agents; no early eviction. |
| `agui.conversation_idle_timeout` | `config.yaml` | `1800` | Positive finite idle lifetime in seconds. Active/parked turns are not idle. |
| `HERMES_AGUI_SESSION_TOKEN` | **`.env`** (secret) | — | Required off-loopback; optional defense-in-depth on loopback. |
| `--host` / `--port` / `--token` | `hermes agui` flags | — | Override the above per invocation. |

**Two ways it finds a provider:**

- **Default (local use):** `agui.base_url` blank → provider, credentials, and model
  are resolved when a conversation is created from your `hermes model` config, like the Hermes
  CLI. This is the zero-config path.
- **Explicit endpoint:** set `agui.base_url` → the hermes resolver is skipped and
  all calls go to that OpenAI-compatible URL (credentials from `OPENAI_API_KEY`).
  This is how the CopilotKit showcase points at a mock LLM, and how you'd target a
  self-hosted OpenAI-compatible server.

**Tool startup scope:** built-in and enabled general-plugin tools are supported.
This standalone launcher does not currently discover or start configured MCP
servers; selecting an MCP toolset does not establish that connection. The
existing Hermes MCP client is separate from the AG-UI frontend-tool relay.
Configured MCP lifecycle/approval integration needs its own validation before
it is advertised as supported here.

Example — point at a local OpenAI-compatible endpoint instead of your Hermes
config:

```yaml
# ~/.hermes/config.yaml
agui:
  base_url: http://localhost:4010/v1
  model: gpt-4o
```

```bash
OPENAI_API_KEY=sk-… python -m agui_adapter
```

## Security

The server dispatches terminal-capable agent work, so the bind interface
determines the auth posture (this mirrors the Hermes API server and dashboard):

- **Loopback (default) → no token.** `127.0.0.1` / `::1` / `localhost` trust the
  OS process boundary. A token is still honored if you set one.
- **Network bind → token required, fail-closed.** Binding `0.0.0.0`, `::`, or any
  specific non-loopback IP **refuses to start** without a usable
  `HERMES_AGUI_SESSION_TOKEN` (≥16 chars, not a placeholder) — an open bind to a
  terminal-capable agent is remote code execution. Clients send the token as the
  `X-Hermes-Session-Token` header (preferred) or a `?token=` query param (can
  leak into logs/history — prefer the header). `/health` is always open.
- **DNS-rebinding Host guard.** Every request's `Host` must match the bound
  interface (loopback names on a loopback bind; exact host on a specific-IP bind;
  any host on a `0.0.0.0`/`::` bind, where the token is the authorization). A
  mismatch is `400`.
- **CSRF defense.** `POST /` requires `Content-Type: application/json` (else
  `415`), forcing a browser CORS preflight so a hostile page can't silently drive
  even a loopback server.
- **Server tool names are reserved.** A frontend (`tools[]`) or state-writer
  declaration whose name matches a registered server tool — e.g. `terminal`,
  `write_file`, `execute_code` — **fails the run** with `RUN_ERROR` naming the
  offending declarations. Dispatch resolves purely by name, so accepting one
  would make the model's call execute that server tool instead of handing off to
  the client. A name also cannot be declared as both kinds, or switch between
  frontend and state-writer across runs in the same process (registration is
  process-global). Rename the client tool.
- **Dangerous-command approvals** ride AG-UI's native interrupt: the run finishes
  with `outcome:{type:"interrupt"}`, the client renders it (`useInterrupt`) and
  resumes with `resume:[{interruptId, status:"resolved", payload:{approved,
  scope}}]`. The Hermes turn parks until the decision or the approval timeout
  (default **60s** via `approvals.timeout` — raise it for a human-in-the-loop web
  UI). Silence/timeout → **deny**. Run a **single uvicorn worker**.

<details>
<summary><b>What the interrupt does NOT cover (read before exposing off-loopback)</b></summary>

The interrupt fires for **dangerous** `terminal()` commands (Hermes'
dangerous-command / Tirith checks; benign commands like `ls` run without a
prompt), including dangerous `terminal()` calls from inside an `execute_code`
script. It does **not** fire for:

- **`approvals.mode` bypasses** — `mode=off`/`--yolo` prompts for nothing;
  `mode=smart` may auto-approve via an auxiliary LLM. The interrupt/park flow is
  the default (`manual`) behavior. (The unconditional hardline floor —
  `rm -rf /`, `mkfs`, fork bombs — still blocks `terminal()` regardless of mode.)
- **`execute_code`'s direct code** — a script's own `subprocess`/`os.system`/
  `ctypes`/file calls never pass through `terminal()`, so they're neither
  interrupt-gated nor scanned by the hardline floor (`os.system("rm -rf /")` in a
  script is not caught). `execute_code`'s whole-script guard auto-approves in
  this interactive context (a Hermes-core limitation, #30882). `execute_code`
  is in the default `hermes-acp` toolset.
- **`delegate_task` sub-agents** — governed by `delegation.subagent_auto_approve`
  (default **false → auto-deny**, fail-closed), not by the interrupt. Also in
  `hermes-acp`.
- **Isolated-container backends** — for an isolated `env_type` (`docker` with no
  host bind-mount, `singularity`, `modal`, `daytona`) Hermes skips the command
  guards entirely (the sandbox is the boundary), *before* the hardline floor —
  so no interrupt and no hardline block fires for `terminal()` there. (`docker`
  *with* a host bind-mount goes through the normal flow.)
- **Already-approved / allowlisted commands** — a prior `session`/`always` grant
  or the permanent allowlist auto-approves with no interrupt; an `always` grant
  is global, so a command approved "always" on the CLI also runs here silently.
- **File-write tools (`write_file`, `patch`)** — the interrupt gates shell
  commands, **not** file edits, so `write_file`/`patch` run without a prompt (an
  agent can overwrite `.env`/SSH keys without an approval). This is a **known
  gap**: Hermes core exposes no file-edit approval hook this adapter can call, so
  gating it here would require a Hermes-core change — out of scope for this
  adapter. To omit dedicated file-edit tools, configure `config.yaml`:

```yaml
agui:
  toolsets: [terminal]
```

This retains `terminal` and process management while excluding `write_file`,
`patch`, `execute_code`, and `delegate_task`. Shell commands can still modify
files; the existing dangerous-command guards described above apply.
</details>

**Reverse proxy / embedding notes.** The Host guard rejects any `Host` that
doesn't match `agui.host`, so a proxy must either forward a matching `Host` or
you set `agui.host` to the public hostname (which then requires a token, since
it's non-loopback). If you call `create_app(bound_host=…)`
directly instead of using `hermes-agui`, pass the real serve interface — the
token/Host checks are enforced against that value.

## How it works

```
POST /  (RunAgentInput JSON)
  → acquire the conversation lease (one active turn per thread)
  → first turn: translate input and build the agent with stable tool schemas
  → later user turns: reuse that agent and its canonical returned messages
  → AIAgent.run_conversation(...) on a context-preserving worker thread
        text/reasoning → events.AGUIEventBridge → asyncio.Queue → SSE frames (live)
        frontend tool → publish real call id, park worker, finish HTTP run
        matching result on a new HTTP run → unblock the same worker
        state snapshot → StateSnapshotEvent after each state-writer tool call
  RUN_STARTED … RUN_FINISHED | RUN_ERROR
```

**No Hermes core changes** — the adapter builds on mechanisms Hermes already
exposes.

### Feature support

- **Streaming** — assistant text and provider reasoning stream as
  `TEXT_MESSAGE_*` / `REASONING_MESSAGE_*`; tool calls as `TOOL_CALL_*`.
- **Frontend (client-executed) tools** — tools in `RunAgentInput.tools` are
  advertised to the model but never run server-side. Each frontend tool name is
  registered with an adapter-local handler that parks its worker. The adapter
  emits the real call id (no result) and finishes that HTTP run. The client
  executes it and submits a new run with the matching tool result. No synthetic
  user message, replacement agent, or model-call replay is involved.
- **Mixed server + frontend tool calls in one turn** — a batch containing a
  frontend tool is never parallel-safe, so Hermes runs it sequentially; server
  actions execute once, in batch order. An action after the frontend call waits
  for its result rather than running ahead of the client.
- **Frontend context, forwarded props and inbound state** — initial data is
  rendered into leading context messages once. Changed data on later user turns
  is appended only to the new user content, including multimodal content. A
  changed continuation payload is appended as an envelope around its tool
  result. Past context is only rewritten by Hermes' normal compression;
  tool schemas remain fixed for the life of the conversation.
- **Multimodal** — image blocks pass through as OpenAI-style content parts;
  pure-text messages stay a plain string.
- **Resume** — frontend results reattach to a live worker. Approvals use their
  separate native `resume[]` interrupt flow. Neither path patches Hermes' turn
  setup or reconstructs an interrupted turn.
- **Compression** — the adapter uses Hermes' real compressor and keeps its
  returned canonical history on following turns. An instance-local prompt-build
  compatibility guard restores the admitted tool snapshot after Hermes'
  registry refresh at compaction; it does not replace the compressor or prompt
  builder. This preserves client/state-writer tools and their exact schemas.
  Start a new conversation to change the admitted tool contract.

### Conversation lifetime

Conversations and pending work are **process-local**, not durable. Use one
server worker (or route every request for a thread to the same process).
Reconnects can resume a live frontend/approval wait; restart, timeout, or idle
expiry cannot reconstruct it. Cold requests containing prior user turns are
rejected: start a new thread with a new user request. No automatic replay occurs.

The server keeps canonical model history, not the browser's transcript
projection. Accepted user history, user/run identities, frontend declarations,
toolsets and state-writer definitions are validated on subsequent turns. A
failed/cancelled execution retires its agent. Idle cleanup and shutdown close
agents under their owning profile context. Capacity rejects new threads rather
than evicting warm conversations and destroying their cached prefix early.
Pending approvals belong to the application. Session-scoped approval grants use
an internal conversation identity, not the client-provided thread id, and are
cleared when that conversation closes. Separate apps cannot share or consume
each other's pending decisions or session grants by reusing a thread id.

### Outbound shared state (state-writer tools)

CopilotKit shared-state demos need the agent to both **see** inbound state (via
the system-message injection above) and **emit** state updates when it mutates
state. Hermes has no first-class shared-state store, so the adapter provides one
per run:

1. **Seed** a run-scoped `session.RunState` from inbound `RunAgentInput.state`,
   so every emitted snapshot carries UI-set keys alongside agent-written keys.
2. **Declare** which server-executed tools write which state key via
   `forwarded_props["stateWriterTools"]` — a list or name→decl map of
   `{name, stateKey, arg?, mode?, description?, parameters?}`. `arg` picks which
   tool argument carries the value (omit → merge the whole args dict); `mode` is
   `"replace"` (default) or `"append"`.
   Frontend `tools[]` and `stateWriterTools` declarations must each have unique
   names, including explicit names in the state-writer mapping form.
3. **Write** — each declared tool gets a server-side handler that merges the
   call's args into the `RunState` and returns a confirmation (it does *not*
   interrupt), so the model reads the result and continues.
4. **Emit** — after each state-writer tool completes, emit a `StateSnapshotEvent`
   carrying the full merged state. These internal tools do not emit duplicate
   chat tool chips; snapshots can arrive before a later frontend handoff.

| demo | tool | declaration |
|---|---|---|
| `shared_state_read_write` | `set_notes(notes)` | `{stateKey:"notes", arg:"notes"}` |
| `gen_ui_agent` | `set_steps(steps)` | `{stateKey:"steps", arg:"steps"}` |
| `shared_state_streaming` | `write_document(document)` | `{stateKey:"document", arg:"document"}` |
| `subagents` | `research/writing/critique_agent` | `{stateKey:"delegations", mode:"append"}` (partial — see below) |

`StateDeltaEvent` (JSON-Patch) is intentionally not emitted — a full snapshot
per call is simpler, deterministic, and CopilotKit re-renders identically.

**Shared-state limitations.** `shared_state_streaming`'s per-token growth is not
replicated (the snapshot is emitted after the tool call completes; end state is
correct). `subagents` appends the raw call args, so it captures `task`/`sub_agent`
but not the sub-agent's computed `result` (that needs a code-level handler, not a
declarative mapping).

## Tests

```bash
# Use a PM-prepared development environment with the agui extra.
scripts/run_tests.sh -j 3 tests/agui_adapter --file-retries 0 --tb=short
```

- `test_translate.py`, `test_events.py` — pure unit tests.
- `test_auth.py`, `test_approvals.py`, `test_dispatch.py`,
  `test_tool_name_collisions.py` — security + lifecycle.
- `test_e2e_aimock.py` — end-to-end against a real
  [`@copilotkit/aimock`](https://www.npmjs.com/package/@copilotkit/aimock)
  fixture server. Requires Node.js plus a one-time install; these tests **skip**
  until it is present, so check for skips before trusting a green run:

  ```bash
  mkdir -p tests/agui_adapter/.aimock && cd tests/agui_adapter/.aimock
  echo '{"name":"aimock-fixture","private":true}' > package.json
  npm i @copilotkit/aimock@1.45.1
  ```

  Write that `package.json` first, and write it by hand — both details matter.
  Without a local `package.json`, npm walks up and installs into the
  **repo-root** `package.json` instead, leaving `.aimock/` empty and the tests
  still skipping. And `npm init -y` can't generate it here: it defaults the
  package name to the directory name, which npm rejects as invalid because
  `.aimock` starts with a dot. `.aimock/` is gitignored.

- `test_modernization_probe.py`, `test_parked_frontend.py`,
  `test_conversations.py`, `test_compression.py`, `test_approval_ownership.py` —
  real AIAgent, dispatch, approval, compression, HTTP/SSE and
  canonical-history contracts against a deterministic loopback model endpoint.
  These are not evidence of paid-provider or browser-client compatibility.
- `test_provider_startup.py` — actual subprocess launch paths load a test-owned
  provider profile, credential and general tool plugin from a temporary Hermes
  home, then dispatch the tool through a loopback model. No real provider calls.
