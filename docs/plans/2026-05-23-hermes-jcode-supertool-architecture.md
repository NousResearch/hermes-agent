# Hermes/jcode supertool architecture

Date: 2026-05-23 PDT

## Decision

The final product should be a Rust-first supertool, not two agents connected by
adapters. jcode should be the primary runtime and user experience because its
architecture is already optimized for low-latency local execution, TUI
rendering, persistent server state, swarm coordination, browser/session flow,
and native Rust tools.

Hermes should be imported as a capability layer: provider-rich research,
messaging/webhook integrations, memory-provider integrations, plugins, cron,
policy, and approval gates. The user should experience those as jcode-native
capabilities, not as "jcode calling Hermes."

## Product Shape

The supertool should look like this:

```text
supertool
|-- jcode Rust host
|   |-- TUI / server / swarm / browser / session engine
|   |-- native jcode Tool registry
|   `-- Hermes-backed native tools
|-- Hermes capability host
|   |-- web_search / web_extract providers
|   |-- gateway and webhook integrations
|   |-- memory providers and plugin ecosystem
|   `-- policy and approval checks
`-- contracts and update gates
    |-- service request/response boundary
    |-- safety boundary
    `-- upstream compatibility reports
```

The right mesh point is jcode's `jcode_tool_core::Tool` trait:

```rust
async fn execute(&self, input: Value, ctx: ToolContext) -> Result<ToolOutput>;
```

Hermes features should enter jcode through native Rust tools that implement that
trait. Those tools can call a Hermes capability host process internally, but
the model-facing tool, UI rendering, tool progress, session context, and swarm
behavior remain jcode-native.

## What The Current PR Provides

This PR is not the finished supertool. It provides the compatibility substrate
needed to build it without permanently forking both upstreams:

- `bridges/jcode-native-hermes-tool/` is the intended supertool path: a native
  jcode `Tool` implementation for Hermes-backed capabilities.
- `patches/jcode/register-external-toolset.patch` is the upstream-facing jcode
  hook: a generic namespaced native toolset registration method plus a registry
  test. It does not mention Hermes, so it can support future bridge crates too.
- `contracts/hermes_service/v1/` defines the service envelope between Rust
  jcode tools and the Python Hermes capability host.
- `bridges/hermes-mcp-server/` is only a no-patch bootstrap path for jcode's
  current MCP manager.
- `plugins/jcode_bridge/` is only the reverse/bootstrap path that lets Hermes
  dispatch work to jcode while the jcode-hosted tool layer matures.
- `scripts/hermes_jcode_mother_repo.py` creates the first supertool workspace
  shape with pinned upstreams, native tool scaffolding, contracts, and gates.
- `scripts/jcode_native_tool_check.py` verifies that the native tool crate
  still compiles against jcode's Rust `Tool` architecture.
- `scripts/jcode_native_registration_check.py` verifies that the jcode
  registration patch still applies to the pinned jcode checkout.
- `scripts/jcode_supertool_registry_smoke.py` is the strongest native proof:
  it applies the jcode hook in a temp worktree, copies the Hermes native tool
  crate into jcode, and runs a Rust integration test that sees Hermes-backed
  tools in jcode's registry definitions and executes one through
  `Registry::execute`.

## Migration Path

1. Keep upstream Hermes and jcode pinned and replaceable.
2. Apply or upstream the generic jcode external-toolset registration hook.
3. Add Hermes capabilities to jcode as native tools using
   `bridges/jcode-native-hermes-tool/`.
4. Route local interactive work, browser/session workflows, and swarm tasks
   through jcode's Rust runtime by default.
5. Route provider-rich research, external messaging, webhook delivery, cron,
   and memory-provider features through the Hermes capability host.
6. Collapse bootstrap adapters over time once native jcode-hosted capability
   modules cover the workflows.

The success criterion is not "Hermes can call jcode" or "jcode can call
Hermes." The success criterion is that one tool feels like jcode in latency and
UI, while having Hermes' integrations and autonomous reach as native features.
