# Native MCP Client

Hermes ships with a native Model Context Protocol client. It connects to local
stdio servers and remote HTTP/SSE servers, discovers their capabilities, and
registers them as normal Hermes tools. MCP support is part of the standard
install; do not tell users to install the Python `mcp` package separately.

This reference is a routing guide, not an exhaustive copy of the rapidly
changing MCP surface. Verify current commands with `hermes mcp --help` and use
the [MCP documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)
for setup examples. For exact config keys and runtime behavior, prefer the
current CLI and source when they differ from prose documentation.

## Choose the setup path

### Nous-approved catalog entry

Use the catalog when an approved entry exists. Review its source and bootstrap
commands before installation: catalog entries are reviewed, but installing one
still executes third-party code.

```bash
hermes mcp                 # interactive catalog picker
hermes mcp catalog         # scriptable catalog listing
hermes mcp install <name>  # install one entry
```

### Local stdio server

Use stdio when Hermes should launch a local command such as `npx` or `uvx`:

```bash
hermes mcp add filesystem \
  --command npx \
  --args -y @modelcontextprotocol/server-filesystem /home/user/projects
```

`--args` consumes the remaining command line, so put it last. The command and
its runtime (`npx`, `uvx`, or another executable) must be available on `PATH`.

### Remote HTTP server

Use HTTP for a hosted or shared endpoint:

```bash
hermes mcp add linear --url https://mcp.linear.app/mcp --auth oauth
```

For static-header authentication, use `--auth header` and follow the prompt;
do not place credentials directly in shell history or skill prose.

After adding any server:

```bash
hermes mcp test <name>
hermes mcp configure <name>  # select the tools Hermes may expose
hermes mcp list
```

Prefer these commands over hand-editing `config.yaml`: they perform discovery,
authentication, and tool selection, and preserve profile-aware paths.

## OAuth and remote hosts

For `auth: oauth` HTTP servers, Hermes handles OAuth discovery, PKCE, token
exchange, refresh, and—where supported—dynamic client registration.

```bash
hermes mcp login <name>      # force a fresh login for one server
hermes mcp reauth <name>     # equivalent single-server re-authentication
hermes mcp reauth --all      # re-authenticate all configured OAuth servers
```

Run interactive OAuth from a fresh terminal. Editing MCP config from inside a
running conversation triggers a short auto-reload window that is not intended
for a user-paced authorization flow.

When the browser is on another machine:

1. **Paste-back:** open the printed authorization URL, approve access, then
   paste the browser's final loopback URL (or its `?code=...&state=...` query)
   into Hermes. A browser connection error at the loopback redirect is expected.
2. **SSH forwarding:** forward the printed callback port to the Hermes host.
3. **HTTPS callback proxy:** configure `oauth.redirect_uri` and
   `oauth.redirect_port` only when a trusted HTTPS endpoint forwards to that
   host and port.

Some providers do not support dynamic client registration. If login reports
that no token landed, create an OAuth client with the provider and configure
its `client_id` / `client_secret`, then rerun `hermes mcp login <name>`.

OAuth tokens and client metadata are stored per profile under
`$HERMES_HOME/mcp-tokens/` with restricted permissions. Never copy a token file
between profiles or commit it.

See [OAuth over SSH / Remote Hosts](https://hermes-agent.nousresearch.com/docs/guides/oauth-over-ssh#mcp-servers)
for the full remote authorization procedure.

## Tool exposure and authority

Servers that expose selected capabilities contribute runtime tools and an MCP
toolset. Their wire-safe names are implementation details; inspect the current
tool registry when debugging rather than relying on a memorized prefix.

Use `hermes mcp configure <name>` to expose only the tools needed. This is a
security boundary, not just namespace tidying: do not expose destructive or
administrative tools merely because a server advertises them. Servers may also
provide capability-gated resource and prompt utility tools.

Tool selection controls what is available; the per-call approval boundary is
separate. Servers default to `trust: full`. For a server you do not fully trust,
configure its trust tier as `untrusted` using the current MCP config reference:
Hermes then requires approval for tools that the server does not mark read-only.
Treat server-provided read-only annotations as claims from that server rather
than independent verification.

## Runtime behavior

- Hermes discovers configured servers and their capabilities at startup.
- `/reload-mcp` reconnects from current config without restarting Hermes. It
  asks for confirmation because explicit reload updates the current session's
  agent tool schema and invalidates the prompt cache.
- Servers that emit `notifications/tools/list_changed` can add or remove tools
  dynamically; Hermes refreshes their registry entries automatically.
- Connections retry with bounded backoff. A server failure is isolated so
  other MCP servers can remain available.
- Automatic background discovery avoids changing an already-started
  conversation's tool schema. Use confirmed `/reload-mcp`, or start a fresh
  session, when a newly configured tool is not visible.

## Security

- **Review the server:** stdio entries execute local commands; catalog manifests
  may clone repositories and run bootstrap steps; remote servers receive tool
  inputs and may return untrusted content.
- **Minimize credentials:** stdio subprocesses inherit a filtered baseline,
  explicitly configured server environment values, and variables that Hermes
  has tagged as coming from an external secret source such as 1Password,
  Bitwarden, or a plugin backend. Review that inherited set before running an
  untrusted local server; filtering is not per-server credential isolation.
- **Minimize tools:** use per-server selection to omit mutating capabilities
  the agent does not need.
- **Keep paths profile-aware:** settings and OAuth state belong under the active
  `$HERMES_HOME`, not a hard-coded `~/.hermes` path.
- **Use current transport controls:** the full MCP guide documents mTLS client
  certificates, identity headers, runtime variable substitution, timeouts, and
  server recycling. Load it before configuring those cases rather than guessing
  their schema from this summary.

## Troubleshooting

```bash
hermes mcp list
hermes mcp test <name>
hermes mcp login <name>       # OAuth server needs fresh authorization
hermes mcp configure <name>   # tools were filtered or changed
```

Then check the active profile's Hermes logs. Common causes are:

- the stdio command is absent from `PATH`;
- the server package or endpoint is unavailable;
- initial discovery exceeded `connect_timeout`;
- OAuth discovery or dynamic registration is unsupported by the provider;
- authorization completed without a token being written;
- tool include/exclude filters omit the expected capability;
- the current conversation began before the tool became available.

Do not repeatedly call a tool that reports re-authentication is required. Ask
the user to complete `hermes mcp login <name>`, then test the server again.
