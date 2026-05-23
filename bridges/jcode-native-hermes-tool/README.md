# jcode Native Hermes Tool

This crate is the intended product direction for the combined Hermes/jcode
supertool: jcode remains the Rust host, and selected Hermes capabilities become
native jcode tools.

Unlike the MCP wrapper, this is not meant to feel like "one agent calling
another agent." It implements jcode's `Tool` trait directly, so Hermes-backed
capabilities can participate in jcode's normal Rust agent loop, TUI rendering,
tool progress, session context, and swarm workflows.

The service boundary still exists because Hermes is Python and owns many
provider/plugin integrations. The boundary should be treated as a capability
module boundary inside the supertool, not as the product UX.

## Expected Mother-Repo Layout

```text
mother-agent/
|-- upstreams/
|   |-- jcode/
|   `-- hermes/
|-- bridges/
|   `-- jcode-native-hermes-tool/
`-- scripts/
    `-- hermes_service_bridge.py
```

The `Cargo.toml` path dependencies assume that layout.

## Integration Target

Wire `HermesNativeTool` into jcode's native tool registry with concrete tools
such as:

- `hermes_web_search`
- `hermes_web_extract`
- `hermes_session_search`
- `hermes_memory`

The MCP server remains useful as a no-patch bootstrap and compatibility test,
but this native crate is the mesh point for the final supertool.
