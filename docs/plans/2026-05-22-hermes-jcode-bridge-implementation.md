# Hermes jcode bridge implementation

Date: 2026-05-22 PDT

This note continues the comparison in
`docs/plans/2026-05-22-hermes-jcode-comparison.md` by adding the first concrete
bridge inside Hermes. The design still keeps both upstreams intact: Hermes calls
jcode through jcode's documented wrapper CLI, and jcode does not need any local
source patch for this phase.

The intended end state is not "Hermes calls a slower external binary forever."
It is "Hermes keeps the gateway, autonomy, provider breadth, and tool
ecosystem; jcode contributes the Rust-speed local runtime, TUI/session feel,
browser profile workflow, and swarm coordination." This first bridge pins the
contract before replacing the one-shot subprocess path with a persistent
`jcode serve` client.

## Added files

```text
plugins/jcode_bridge/
|-- plugin.yaml
|-- README.md
|-- __init__.py
|-- contracts.py
|-- hermes_service.py
|-- safety.py
|-- tools.py
`-- webhook_dispatch.py

contracts/jcode_bridge/v1/
|-- README.md
|-- debug_command.schema.json
|-- debug_response.schema.json
|-- run_json.schema.json
|-- run_ndjson_event.schema.json
|-- run_ndjson_stream.schema.json
`-- upstream_sync_report.schema.json

contracts/hermes_service/v1/
|-- README.md
|-- service_request.schema.json
`-- service_response.schema.json

contracts/hermes_mcp/v1/
|-- README.md
|-- initialize_response.schema.json
|-- tools_list_response.schema.json
`-- tools_call_response.schema.json

tests/
|-- fixtures/jcode_bridge/
|   |-- debug_response_error.json
|   |-- debug_response_success.json
|   |-- run_json_success.json
|   `-- run_ndjson_success.ndjson
|-- fixtures/hermes_service/
|   |-- service_request_web_search.json
|   |-- service_response_error.json
|   `-- service_response_success.json
|-- fixtures/hermes_mcp/
|   |-- initialize_response.json
|   |-- tools_call_response_success.json
|   `-- tools_list_response.json
`-- plugins/test_jcode_bridge_plugin.py

scripts/hermes_service_bridge.py
scripts/jcode_bridge_compat.py
scripts/jcode_bridge_smoke.py
scripts/jcode_bridge_upstream_report.py
scripts/hermes_jcode_mother_repo.py

bridges/jcode-tool-hermes/
|-- Cargo.toml
|-- README.md
`-- src/main.rs

bridges/hermes-mcp-server/
|-- README.md
`-- hermes_mcp_server.py
```

The plugin registers a new `jcode` toolset with three tools:

- `jcode_run`: runs one prompt through `jcode run` and returns structured JSON
  to Hermes.
- `jcode_status`: runs lightweight status checks such as `version --json`,
  `auth status --json`, `provider current --json`, and `browser status`.
- `jcode_contract_check`: validates the `jcode-bridge.v1` fixtures/schemas and
  can run optional live version/run checks against a local jcode binary.

It also registers a `pre_gateway_dispatch` hook. When a Hermes webhook route is
configured with `dispatch: jcode`, the hook:

1. lets the webhook adapter authenticate, rate-limit, dedupe, render the
   prompt, and store delivery metadata as usual
2. skips the normal Hermes agent run for that event
3. runs the rendered prompt through `jcode_run`
4. sends jcode's result through the webhook adapter's existing `deliver` target

This gives Hermes webhook ingress to jcode without hardcoding jcode into
`gateway/platforms/webhook.py`.

The operator-facing guide lives at `plugins/jcode_bridge/README.md`. It is the
quick path for enabling the plugin, picking the fast sidecar mode, configuring
webhook dispatch, and running compatibility gates.

## Safety boundary

Hermes owns the policy boundary before a prompt is handed to jcode. The bridge
therefore blocks unattended routing for two high-risk prompt classes unless the
tool call or webhook route explicitly sets the relevant confirmation flag:

- `outbound_human_contact`: prompts that appear to send, reply, DM, post,
  text, call, or otherwise contact a person/account.
- `sensitive_person_data`: prompts that appear to find private personal
  contact or identity data such as phone numbers, home addresses, personal
  email, SSN, or date of birth.

The flags are:

```json
{
  "confirm_outbound_human_contact": true,
  "confirm_sensitive_person_data": true,
  "safety_override_reason": "Operator approved this route for a narrow workflow."
}
```

When confirmation is missing, `jcode_run` returns:

```json
{
  "success": false,
  "error": "jcode bridge safety confirmation required",
  "requires_confirmation": true,
  "risk_types": ["outbound_human_contact"],
  "confirmation_fields": ["confirm_outbound_human_contact"]
}
```

This is deliberately a Hermes-side guardrail. jcode can stay fast and small;
Hermes decides which automated gateway/webhook tasks are allowed to use that
speed without a visible approval step.

## CLI contract

The bridge relies on jcode's wrapper-oriented commands documented in
`.codex-research/jcode/docs/WRAPPERS.md`:

```bash
jcode --quiet --no-update --no-selfdev run --json "Reply with exactly OK"
jcode --quiet --no-update --no-selfdev run --ndjson "Reply with exactly OK"
jcode --quiet --no-update --no-selfdev debug --wait message "Reply with exactly OK"
jcode --quiet --no-update --no-selfdev debug list
jcode --quiet --no-update --no-selfdev version --json
jcode --quiet --no-update --no-selfdev auth status --json
jcode --quiet --no-update --no-selfdev provider current --json
```

Those flags are intentionally baked into the tool handlers:

- `--quiet` keeps wrapper output machine-oriented.
- `--no-update` avoids update work/noise during a Hermes tool call.
- `--no-selfdev` avoids repo auto-detection changing bridge behavior.

The bridge accepts `cwd`, `session`, `provider`, `model`, and
`provider_profile` and maps them to jcode global CLI flags before the
subcommand. It redacts the message body from the returned command audit list.
For server-backed paths, `ensure_server: true` asks jcode to start its
persistent Rust server via `jcode debug start` before falling back to slower
one-shot CLI execution.

## Compatibility contracts

The bridge now has a small explicit contract layer in
`plugins/jcode_bridge/contracts.py`. It validates:

- `jcode run --json`: object payload with a final text field, plus optional
  `session_id`, `provider`, `model`, and `usage`.
- `jcode run --ndjson`: event objects with string `type` fields and a final
  `done` event containing final text.
- direct debug socket requests: newline JSON objects shaped as
  `{"type": "debug_command", "id": ..., "command": ...}`.
- direct debug socket responses: `debug_response` or `error` envelopes.

The portable schema artifacts live in `contracts/jcode_bridge/v1/`. They cover
the same boundary plus the upstream-sync report shape. This lets a future
mother repo, Rust-side tests, or CI job consume the bridge contract without
importing Hermes Python modules.

The reverse service artifacts live in `contracts/hermes_service/v1/`. They
define the newline-JSON request/response envelope for a jcode client calling
selected Hermes services such as `web_search`, `web_extract`, `session_search`,
and `memory`. The stdlib implementation is
`plugins/jcode_bridge/hermes_service.py`, and the runnable wrapper is
`scripts/hermes_service_bridge.py`.

The first Rust-side caller lives in `bridges/jcode-tool-hermes/`. It is a
dependency-free binary that starts the Hermes service wrapper, sends one
`hermes-service.v1` request line, and prints the response line. It is not wired
into jcode's tool registry yet; it is the portable client to adapt there.

The first no-patch jcode integration lives in `bridges/hermes-mcp-server/`. It
implements the small stdio MCP surface that jcode's MCP manager uses and maps
MCP tool calls back onto the same `hermes-service.v1` envelope. In jcode, those
tools appear as `mcp__hermes__hermes_tool`,
`mcp__hermes__hermes_web_search`, `mcp__hermes__hermes_web_extract`,
`mcp__hermes__hermes_session_search`, and `mcp__hermes__hermes_memory`.
Its fixtures and schemas live in `contracts/hermes_mcp/v1/` and
`tests/fixtures/hermes_mcp/`.

The fixtures under `tests/fixtures/jcode_bridge/` are the update canaries. When
jcode changes its wrapper/debug protocol, update those fixtures first, then
adjust the JSON Schemas, `contracts.py`, and bridge runtime parsing. This keeps
upstream syncs concrete: a jcode update is compatible only if these boundary
fixtures, schemas, and the bridge tests still pass.

For environments without the full Hermes pytest setup, run the stdlib-only
compatibility gate:

```bash
scripts/jcode_bridge_compat.py
```

Run the reverse-service contract gate:

```bash
scripts/hermes_service_bridge.py check
```

Run the jcode-facing MCP contract gate:

```bash
python3 bridges/hermes-mcp-server/hermes_mcp_server.py --check --live
```

Probe local MCP bridge overhead without model or network calls:

```bash
scripts/jcode_bridge_latency_probe.py --iterations 50
```

Build the Rust caller:

```bash
cargo build --manifest-path bridges/jcode-tool-hermes/Cargo.toml
```

Optional live checks against a local jcode binary:

```bash
scripts/jcode_bridge_compat.py --live --jcode-bin /absolute/path/to/jcode
scripts/jcode_bridge_compat.py --live --live-run --jcode-bin /absolute/path/to/jcode
```

The same check is exposed inside Hermes as the `jcode_contract_check` tool, so
an agent or webhook preflight can verify compatibility before sending real work
to jcode.

For a no-pytest behavioral gate, run:

```bash
scripts/jcode_bridge_smoke.py
```

It covers the contract tool, safety confirmation block, bad jcode JSON
rejection, `ensure_server` server-backed execution, reverse-service contract
and safety behavior, the Rust jcode-side client, the stdio MCP wrapper and MCP
contract, mother-repo scaffold generation, and webhook preflight pass/block
behavior. The native jcode tool scaffold is source-validated by the scaffold
copy checks and is designed to compile inside a mother repo that has
`upstreams/jcode` checked out.

Run the native jcode tool gate when a jcode checkout is available:

```bash
scripts/jcode_native_tool_check.py --jcode /absolute/path/to/jcode
```

For upstream bumps, generate a combined sync report:

```bash
scripts/jcode_bridge_upstream_report.py --smoke --format markdown \
  --output docs/plans/2026-05-23-hermes-jcode-upstream-sync-report.md
```

That report records both repo SHAs, dirty-state samples, Graphify summaries,
Graphify artifact paths/sizes, bridge contract/schema results, reverse-service
and MCP transport status, latency-probe metrics, native jcode tool status, and
optional smoke results. Use it as the first gate before deciding whether a new
jcode/Hermes pair is compatible.

For a standalone mother-repo scaffold:

```bash
scripts/hermes_jcode_mother_repo.py scaffold --output /path/to/mother-agent
python3 /path/to/mother-agent/scripts/check_bridge_contract.py
```

The scaffold copies the bridge plugin, native jcode Hermes tool crate, portable
schemas, fixtures, reverse service wrapper, MCP wrapper, generated jcode MCP
config, latency probe, and plan docs into a separate workspace and records the
current Hermes/jcode pins in `hermes-jcode.manifest.json`. Its generated
contract check runs without importing Hermes gateway internals and validates
`jcode-bridge.v1`, `hermes-service.v1`, and `hermes-mcp.v1`, which proves the
compatibility boundary can move outside this checkout while the final product
surface moves into jcode's native Rust tool architecture.

## Enabling

The plugin is standalone, so Hermes should only load it when explicitly enabled
in `config.yaml`:

```yaml
plugins:
  enabled:
    - jcode_bridge
```

Set `JCODE_BIN` if `jcode` is not on `PATH`:

```bash
export JCODE_BIN=/absolute/path/to/jcode
```

For a source checkout, build jcode first and point `JCODE_BIN` at the produced
binary. The bridge does not compile or install jcode by itself.

## Tool examples

Example `jcode_status` call:

```json
{
  "checks": ["version", "auth_status", "provider_current", "server_list", "debug_sockets"]
}
```

Example `jcode_contract_check` call:

```json
{
  "live": true,
  "live_run": false,
  "jcode_bin": "/absolute/path/to/jcode"
}
```

Example `jcode_run` call:

```json
{
  "message": "Research the public API shape of this local repo and summarize it.",
  "cwd": "/Users/aayu/Workspace/developer/hermes",
  "output_mode": "json",
  "timeout_seconds": 600
}
```

Example server-backed run:

```json
{
  "message": "Use the existing jcode session to inspect the browser state.",
  "execution_mode": "server_debug",
  "ensure_server": true,
  "session": "fox",
  "socket": "/run/user/501/jcode.sock",
  "timeout_seconds": 600
}
```

Example direct debug-socket run, which skips the jcode CLI wrapper:

```json
{
  "message": "Use the existing jcode session to inspect the browser state.",
  "execution_mode": "debug_socket",
  "session": "fox",
  "debug_socket": "/var/folders/.../T/jcode-debug.sock",
  "timeout_seconds": 600
}
```

`execution_mode=auto` tries direct `debug_socket`, then `server_debug`, then
one-shot `jcode run --json` when the daemon paths are unavailable.

When `debug_socket` is omitted, the bridge looks in the same places jcode uses:

- explicit `socket` / `$JCODE_SOCKET`, converted to the sibling
  `*-debug.sock`
- `$JCODE_RUNTIME_DIR`
- `$XDG_RUNTIME_DIR`
- `$TMPDIR` on macOS
- the platform temp directory and `jcode-<uid>` fallback

It scans those runtime directories for `jcode*-debug.sock`, so a route can often
use `execution_mode: auto` without hardcoding the socket path.

Example resumed-session call:

```json
{
  "message": "Continue the previous browser workflow and report status.",
  "session": "fox",
  "output_mode": "ndjson"
}
```

## Webhook dispatch

Minimal route:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      routes:
        local-jcode:
          secret: "replace-me"
          dispatch: jcode
          prompt: "Handle this webhook payload:\n\n{__raw__}"
          deliver: telegram
          deliver_extra:
            chat_id: "12345"
          jcode:
            cwd: "/Users/aayu/Workspace/developer/hermes"
            execution_mode: auto
            ensure_server: true
            preflight_contract: true
            preflight_live: true
            debug_socket: "/var/folders/.../T/jcode-debug.sock"
            output_mode: json
            timeout_seconds: 600
            # Required before this route can perform account/person outreach.
            confirm_outbound_human_contact: false
```

Compact equivalent:

```yaml
dispatch:
  target: jcode
  cwd: "/Users/aayu/Workspace/developer/hermes"
  execution_mode: debug_socket
  ensure_server: true
  preflight_contract: true
  session: fox
  debug_socket: "/var/folders/.../T/jcode-debug.sock"
  output_mode: ndjson
  confirm_outbound_human_contact: true
  safety_override_reason: "Internal test route with human review upstream."
```

The plugin reads route config from the already-running webhook adapter. If the
route is not opted into `dispatch: jcode`, it returns `None` and Hermes
continues through its normal gateway path.

Webhook routes can preflight the bridge before handing work to jcode:

- `preflight_contract: true` validates the local `jcode-bridge.v1` fixtures.
- `preflight_live: true` additionally runs `jcode version --json`.
- `preflight_live_run: true` additionally runs one harmless
  `jcode run --json` prompt. Keep this off by default for high-volume webhook
  routes because it can spend model/API budget before the real task starts.

If preflight fails, Hermes delivers `jcode bridge contract preflight failed`
through the webhook adapter and skips the jcode task.

## Why this first

This is the lowest-risk bridge because it:

1. Uses a stable public boundary instead of importing Rust internals.
2. Keeps jcode updates portable: if `jcode run --json` remains compatible,
   upstream jcode can move freely.
3. Keeps Hermes updates portable: the bridge is a plugin, not a core loop patch.
4. Creates a place to route Hermes webhook events into jcode without first
   porting Hermes' gateway into jcode.

## Known limits

- `execution_mode=debug_socket` reaches a running Rust jcode server directly
  over the newline JSON debug socket when that socket path is known or
  discoverable. With `ensure_server: true`, the bridge can ask jcode to start
  that server before retrying the socket.
- `execution_mode=server_debug` reaches the same Rust server through `jcode
  debug`, which is more portable but still shells through the CLI transport.
- `execution_mode=cli` remains available as the portable fallback, but each
  dispatch can still pay jcode CLI startup cost.
- It does not yet start or discover `jcode serve`.
- It does not yet stream NDJSON back into Hermes' live tool-progress UI; it
  parses NDJSON after the subprocess exits.
- It does not add a shared browser provider contract yet.
- The reverse bridge can be reached through jcode's MCP manager, but it is not
  yet a native in-tree jcode `Tool`.
- The safety evaluator is conservative and lexical. It is intended to catch
  unattended webhook/account-action hazards early, not replace a full policy
  engine or final user approval UX.

## Next implementation steps

1. Keep expanding the direct socket client from debug command dispatch into a
   fuller persistent protocol client when the event contract is pinned.
2. Add contract fixtures and schemas for:
   - `debug --wait message`
   - `debug list`
   - `version --json`
   - `auth status --json`
   - `provider current --json`
3. Promote the safety evaluator into a shared approval contract for any future
   browser-provider bridge, so jcode can execute quickly while Hermes keeps the
   visible decision point for account/person actions.
