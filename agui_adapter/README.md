# Hermes AG-UI adapter

Run Hermes as an [AG-UI](https://docs.ag-ui.com/) HTTP/SSE server, so AG-UI
clients (e.g. CopilotKit) can drive it from a browser.

## Quick start

```bash
# 1. Install the adapter extra (adds the `hermes-agui` command + deps)
pip install -e '.[agui]'            # or: pip install 'hermes-agent[agui]'

# 2. One-time: make sure Hermes has a provider + model configured
hermes model                        # pick provider/model, set credentials

# 3. Start the server (binds http://127.0.0.1:8000 by default)
hermes-agui                         # equivalently: python -m agui_adapter
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

> There is no `hermes agui` subcommand — start it with `hermes-agui` or
> `python -m agui_adapter`.

## Connecting a client

Any AG-UI `HttpAgent` works. With **CopilotKit**, the usual shape is a Next.js
runtime route that proxies to this server:

```ts
// app/api/copilotkit/route.ts
import { HermesAgent } from "@ag-ui/hermes";
const AGENT_URL = process.env.AGENT_URL ?? "http://localhost:8000";
const agent = new HermesAgent({ url: `${AGENT_URL}/` });
// ...register `agent` with CopilotRuntime as usual
```

The browser talks to your Next.js runtime; the runtime (server-side) talks to
`hermes-agui` on `:8000`. Because that hop is server-to-server on localhost, the
loopback/no-token setup is all you need. (See `showcase/integrations/hermes/` in
the CopilotKit repo for a complete working example.)

## Configuration

All configuration is via environment variables — nothing is required for the
local quick start above.

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_AGUI_HOST` | `127.0.0.1` | Bind interface. Any non-loopback value requires a token (see Security). |
| `PORT` / `HERMES_AGUI_PORT` | `8000` | Listen port (`PORT` wins — PaaS convention). |
| `HERMES_AGUI_SESSION_TOKEN` | — | Required off-loopback; optional defense-in-depth on loopback. |
| `HERMES_AGUI_TOOLSETS` | `hermes-acp` | Comma-separated Hermes toolsets to enable for the agent. |
| `HERMES_AGUI_MODEL` | *(hermes config default)* | Override the model. |
| `HERMES_AGUI_PROVIDER` | *(hermes config)* | Override the provider. |
| `HERMES_AGUI_API_MODE` | *(auto)* | `chat_completions` or `responses`. |
| `HERMES_AGUI_BASE_URL` / `OPENAI_BASE_URL` | — | **Bypass** the hermes resolver and send every LLM call straight to this OpenAI-compatible endpoint. |
| `HERMES_AGUI_API_KEY` / `OPENAI_API_KEY` | — | Key for the explicit-endpoint path above. |

**Two ways it finds a provider:**

- **Default (local use):** no `*_BASE_URL` set → provider, credentials, and model
  are resolved per run from your `hermes model` config, exactly like the Hermes
  CLI. This is the zero-config path.
- **Explicit endpoint:** set `HERMES_AGUI_BASE_URL` (or `OPENAI_BASE_URL`) → the
  hermes resolver is skipped and all calls go to that URL. This is how the
  CopilotKit showcase points at a mock LLM, and how you'd target a self-hosted
  OpenAI-compatible server.

Example — point at a local OpenAI-compatible endpoint instead of your Hermes
config:

```bash
OPENAI_BASE_URL=http://localhost:4010/v1 OPENAI_API_KEY=sk-… \
  HERMES_AGUI_MODEL=gpt-4o hermes-agui
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
  adapter. To gate file edits, **drop `write_file`/`patch` from
  `HERMES_AGUI_TOOLSETS`**, or run on an isolated `env_type`.

To make the interrupt cover **all** local code execution: run the default
(`manual`) `approvals.mode` on a non-isolated `env_type`, and set
`HERMES_AGUI_TOOLSETS` to a toolset that **excludes `execute_code`,
`delegate_task`, `write_file`, and `patch`**. Those are kept in the default
toolset for capability parity; excluding them is the operator's hardening lever.
</details>

**Reverse proxy / embedding notes.** The Host guard rejects any `Host` that
doesn't match `HERMES_AGUI_HOST`, so a proxy must either forward a matching
`Host` or you set `HERMES_AGUI_HOST` to the public hostname (which then requires
a token, since it's non-loopback). If you call `create_app(bound_host=…)`
directly instead of using `hermes-agui`, pass the real serve interface — the
token/Host checks are enforced against that value.

## How it works

```
POST /  (RunAgentInput JSON)
  → translate.prepare_run(messages, context)   # AG-UI → Hermes history
  → session.build_run_agent(...)               # merge frontend + state-writer tool schemas; register handlers
  → AIAgent.run_conversation(...) on a worker thread
        text/reasoning → events.AGUIEventBridge → asyncio.Queue → SSE frames (live)
        tool events    → derived from the returned messages (real model ids)
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
  registered with a handler that calls `agent.interrupt()` and returns a
  placeholder; when the model calls it, the loop unwinds at its next
  top-of-loop interrupt check, the adapter emits the tool call (real id, no
  result), and the client executes it and starts a new run with the result.
- **Mixed server + frontend tool calls in one turn** — a batch containing a
  frontend tool is never parallel-safe, so Hermes runs it sequentially; server
  tools finish and append results before the frontend tool's handler interrupts.
- **Frontend context** — `RunAgentInput.context[]` is injected as a read-only
  system message (never merged into the user message, so fixture matching stays
  deterministic).
- **Forwarded props (agent config)** — `RunAgentInput.forwarded_props` (e.g.
  `{tone, expertise}` from `useAgentContext`) is rendered as its own read-only
  system message per run.
- **Inbound shared state** — `RunAgentInput.state` is injected as a
  `Current shared state: <json>` system message.
- **Multimodal** — image blocks pass through as OpenAI-style content parts;
  pure-text messages stay a plain string.
- **Resume** — when the history tail is a tool result with no new user turn,
  `resume_shim.py` (a narrow, flag-gated wrapper around `build_turn_context`)
  drops the synthetic trailing user turn so the run continues from history. It's
  a pure pass-through when the resume flag isn't set, keeping core untouched.

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
3. **Write** — each declared tool gets a server-side handler that merges the
   call's args into the `RunState` and returns a confirmation (it does *not*
   interrupt), so the model reads the result and continues.
4. **Emit** — after the run, for each state-writer tool call (in message order),
   the server emits the normal `TOOL_CALL_*` + `TOOL_CALL_RESULT` followed by a
   `StateSnapshotEvent` carrying the full merged state as of that call.

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
pip install -e '.[agui]' && pip install pytest pytest-asyncio
python -m pytest tests/agui_adapter/ -q
```

- `test_translate.py`, `test_events.py` — pure unit tests.
- `test_auth.py`, `test_approvals.py`, `test_resume_shim.py` — security + lifecycle.
- `test_e2e_aimock.py` — end-to-end against a real
  [`@copilotkit/aimock`](https://www.npmjs.com/package/@copilotkit/aimock)
  fixture server (auto-installed under `tests/agui_adapter/.aimock`, gitignored).
  Requires Node.js.
