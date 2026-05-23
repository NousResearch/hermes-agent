# jcode Tool Hermes Bridge

This is the first jcode-side caller for the reverse bridge. It is a tiny
dependency-free Rust binary that speaks `hermes-service.v1` over newline JSON.

It does not import jcode or Hermes internals. It starts a local Hermes service
command, writes one request line, reads one response line, and prints that
response.

Example from a mother-repo scaffold:

```bash
cargo run --manifest-path bridges/jcode-tool-hermes/Cargo.toml -- \
  --service-command "python3 scripts/hermes_service_bridge.py stdio" \
  --tool web_search \
  --args-json '{"query":"Hermes jcode bridge","limit":3}'
```

Useful options:

- `--request-json`: pass a complete `hermes-service.v1` request object.
- `--tool` plus `--args-json`: build a request object locally.
- `--allow-tool`: forwarded to the service wrapper. Repeat for multiple tools.
- `--cwd`: run the service command from a mother-repo or Hermes checkout.

This crate is intentionally a bridge scaffold. The next step is to adapt the
same request/response logic into a native jcode `Tool` implementation.

For immediate jcode integration without patching upstream, use the sibling
`bridges/hermes-mcp-server/` wrapper. It exposes the same `hermes-service.v1`
boundary through jcode's existing MCP manager.
