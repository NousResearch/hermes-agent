# jcode Bridge Plugin

Use this plugin as the bootstrap layer for a Rust-first Hermes/jcode supertool.
The final product should use jcode as the host runtime and expose Hermes
capabilities as native-feeling jcode tools.

This plugin is intentionally bridge-first. It does not fork Hermes or jcode,
and it does not import jcode internals. Hermes calls stable wrapper/debug
surfaces, validates their output, and keeps the gateway/safety decisions on the
Hermes side.

The broader mother-repo plan is documented in
`docs/plans/2026-05-23-hermes-jcode-mother-repo-blueprint.md` and
`docs/plans/2026-05-23-hermes-jcode-supertool-architecture.md`: keep upstream
Hermes and jcode pinned separately, make jcode the Rust host, and import Hermes
integrations through native jcode tool crates plus contracts.

## What This Gives You

- `jcode_run`: send one prompt to jcode through CLI, server debug, direct debug
  socket, or automatic fallback.
- `jcode_status`: check local jcode version/auth/provider/browser/server/debug
  socket state.
- `jcode_contract_check`: validate the `jcode-bridge.v1` compatibility
  fixtures/schemas and optionally run live checks against a local jcode binary.
- `dispatch: jcode`: opt a Hermes webhook route into jcode execution while
  keeping Hermes webhook auth, templating, delivery, and safety controls.
- `hermes-service.v1`: a reverse newline-JSON service contract so a local
  jcode client can call allowlisted Hermes services such as `web_search`,
  `web_extract`, `session_search`, and `memory`.
- `bridges/jcode-native-hermes-tool`: the intended supertool path, implementing
  jcode's native `Tool` trait for Hermes-backed capabilities.

The intended split:

- jcode owns the low-latency local runtime, primary UX, persistent server,
  browser/session feel, swarm-aware work, and model-facing tool registry.
- Hermes owns webhooks, messaging, policy, research/provider breadth, plugins,
  cron, memory-provider integrations, and the capability host behind native
  jcode tools.

## Enable

Add the plugin to Hermes config:

```yaml
plugins:
  enabled:
    - jcode_bridge
```

Point Hermes at jcode if it is not already on `PATH`:

```bash
export JCODE_BIN=/absolute/path/to/jcode
```

For a jcode source checkout, build jcode first and set `JCODE_BIN` to the
produced binary. This plugin does not compile or install jcode.

## Fast Sidecar Modes

`jcode_run` supports four execution modes:

| Mode | Behavior |
| --- | --- |
| `debug_socket` | Talk directly to a running jcode debug socket over newline JSON. Lowest bridge overhead when a server is already running. |
| `server_debug` | Shell through `jcode debug --wait message ...` to a running jcode server. More portable, still server-backed. |
| `auto` | Try direct debug socket, then `server_debug`, then one-shot `jcode run`. |
| `cli` | Run one `jcode run` process. Portable fallback with the most startup overhead. |

For server-backed modes, set `ensure_server: true` to ask jcode to start its
persistent server first:

```json
{
  "message": "Inspect this repo and summarize the local API surface.",
  "cwd": "/Users/aayu/Workspace/developer/hermes",
  "execution_mode": "auto",
  "ensure_server": true,
  "output_mode": "json"
}
```

When a debug socket path is omitted, the bridge checks:

- explicit `socket` or `$JCODE_SOCKET`, converted to the sibling
  `*-debug.sock`
- `$JCODE_RUNTIME_DIR`
- `$XDG_RUNTIME_DIR`
- `$TMPDIR` on macOS
- the platform temp directory and `jcode-<uid>` fallback
- any `jcode*-debug.sock` discovered in those runtime directories

## Webhook Dispatch

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
            output_mode: json
            timeout_seconds: 600
```

Production-style route with preflight:

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
            jcode_bin: "/absolute/path/to/jcode"
            cwd: "/Users/aayu/Workspace/developer/hermes"
            execution_mode: auto
            ensure_server: true
            preflight_contract: true
            preflight_live: true
            preflight_live_run: false
            output_mode: json
            timeout_seconds: 600
```

Preflight options:

| Option | Behavior |
| --- | --- |
| `preflight_contract` | Validate local `jcode-bridge.v1` fixtures before dispatch. |
| `preflight_live` | Also run `jcode version --json`. |
| `preflight_live_run` | Also run one harmless `jcode run --json` prompt. This can spend model/API budget. |
| `preflight_live_run_message` | Override the harmless live-run prompt. |

If preflight fails, Hermes delivers:

```text
jcode dispatch failed: jcode bridge contract preflight failed
```

and skips the jcode task.

## Safety Controls

Hermes blocks unattended routing for prompts that appear to involve:

- outbound human contact, such as DMs, posts, replies, texts, calls, or emails
- sensitive private-person data lookup, such as phone numbers, home addresses,
  personal emails, SSNs, or dates of birth

Tool calls or webhook routes must explicitly set:

```json
{
  "confirm_outbound_human_contact": true,
  "confirm_sensitive_person_data": true,
  "safety_override_reason": "Operator approved this narrow workflow."
}
```

These flags are audit markers for the bridge boundary. Higher-level approval
UX and policy should still live in Hermes.

## Compatibility Gates

Portable JSON Schema artifacts live in `contracts/jcode_bridge/v1/`:

```text
debug_command.schema.json
debug_response.schema.json
run_json.schema.json
run_ndjson_event.schema.json
run_ndjson_stream.schema.json
upstream_sync_report.schema.json
```

Those files are the cross-repo boundary a future mother repo can consume. The
Hermes plugin also keeps stdlib validators in `plugins/jcode_bridge/contracts.py`
so compatibility checks do not require installing a JSON Schema package.

Reverse service schemas live in `contracts/hermes_service/v1/`:

```text
service_request.schema.json
service_response.schema.json
```

The jcode-facing MCP transport has its own compatibility contract in
`contracts/hermes_mcp/v1/`:

```text
initialize_response.schema.json
tools_list_response.schema.json
tools_call_response.schema.json
```

That service is implemented by `plugins/jcode_bridge/hermes_service.py` and can
be run as newline JSON over stdio:

```bash
scripts/hermes_service_bridge.py check
scripts/hermes_service_bridge.py stdio
```

By default, it only allows `web_search`, `web_extract`, `session_search`, and
`memory`. Side-effect tools such as `send_message` must be explicitly
allowlisted and still require confirmation fields.

The first Rust client for this service lives at `bridges/jcode-tool-hermes/`:

```bash
cargo run --manifest-path bridges/jcode-tool-hermes/Cargo.toml -- \
  --service-command "python3 scripts/hermes_service_bridge.py stdio" \
  --tool web_search \
  --args-json '{"query":"Hermes jcode bridge","limit":3}'
```

That client is intentionally standalone. A later jcode patch can wrap the same
request/response logic in a native jcode `Tool`.

For integration without patching jcode, use the dependency-free MCP wrapper at
`bridges/hermes-mcp-server/`:

```bash
python3 bridges/hermes-mcp-server/hermes_mcp_server.py
```

Add it to jcode's `.jcode/mcp.json` or `~/.jcode/mcp.json`:

```json
{
  "servers": {
    "hermes": {
      "command": "python3",
      "args": [
        "/absolute/path/to/hermes/bridges/hermes-mcp-server/hermes_mcp_server.py"
      ],
      "env": {
        "JCODE_BRIDGE_ROOT": "/absolute/path/to/hermes"
      },
      "shared": true
    }
  }
}
```

jcode will expose the tools as `mcp__hermes__hermes_tool`,
`mcp__hermes__hermes_web_search`, `mcp__hermes__hermes_web_extract`,
`mcp__hermes__hermes_session_search`, and `mcp__hermes__hermes_memory`.

Validate that MCP wrapper boundary:

```bash
python3 bridges/hermes-mcp-server/hermes_mcp_server.py --check --live
```

Probe local bridge overhead without model or network calls:

```bash
scripts/jcode_bridge_latency_probe.py --iterations 50
```

Run the contract fixture gate:

```bash
scripts/jcode_bridge_compat.py
```

Run optional live checks:

```bash
scripts/jcode_bridge_compat.py --live --jcode-bin /absolute/path/to/jcode
scripts/jcode_bridge_compat.py --live --live-run --jcode-bin /absolute/path/to/jcode
```

Run the no-pytest behavioral smoke gate:

```bash
scripts/jcode_bridge_smoke.py
```

Generate an upstream-sync report:

```bash
scripts/jcode_bridge_upstream_report.py --smoke --format markdown \
  --output docs/plans/2026-05-23-hermes-jcode-upstream-sync-report.md
```

Use that report before bumping either upstream. It records both SHAs, Graphify
summaries, artifact paths/sizes, bridge contract/schema status, MCP transport
status, reverse Hermes service status, latency-probe metrics, and optional
smoke status.

Create a standalone mother-repo scaffold:

```bash
scripts/hermes_jcode_mother_repo.py scaffold --output /path/to/mother-agent
python3 /path/to/mother-agent/scripts/check_bridge_contract.py
```

The scaffold carries the bridge plugin, Rust service client, native jcode tool
crate, MCP server, schemas, fixtures, copied research docs, reverse-service
wrapper, generated jcode MCP config, latency probe, and a manifest with the
pinned Hermes/jcode state. Its contract check validates `jcode-bridge.v1`,
`hermes-service.v1`, and `hermes-mcp.v1`. The native tool crate is the practical
path for combining jcode's Rust hot path with Hermes integrations while still
being able to pull future upstream updates cleanly.

## Current Limits

- The bridge does not yet stream NDJSON into Hermes live tool progress.
- The bridge does not yet expose jcode's browser provider as a native Hermes
  browser provider.
- The bridge does not yet mirror jcode memory or swarm state into Hermes
  memory/kanban surfaces.
- `bridges/jcode-native-hermes-tool` is a native jcode `Tool` scaffold, but it
  still needs a small upstream jcode registration patch before it is available
  in jcode's default tool registry.
- `debug_socket` requires a running jcode server with debug socket enabled.
- `preflight_live_run` can spend model/API budget and should be reserved for
  deliberate compatibility checks.
