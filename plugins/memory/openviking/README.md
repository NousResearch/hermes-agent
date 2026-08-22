# OpenViking Memory Provider

Context database by Volcengine (ByteDance) with filesystem-style knowledge hierarchy, tiered retrieval, and automatic memory extraction.

## Requirements

- OpenViking installed with the `openviking-server` command available
- OpenViking server config initialized and validated (`openviking-server init`,
  then `openviking-server doctor`)
- OpenViking server running and reachable from Hermes

OpenViking 0.4.1 or newer is required for the MCP tool and actor-peer contract
used by Hermes.
Existing Hermes profiles can still run automatic memory lifecycle hooks
against some older servers, but explicit OpenViking tools will not be
available. New setup, or rerunning setup, requires OpenViking 0.4.1 or newer.

## Setup

Prepare OpenViking first:

```bash
openviking-server init
openviking-server doctor
openviking-server
```

Then configure Hermes:

```bash
hermes memory setup    # select "openviking"
```

Setup can import an existing `ovcli.conf` profile or create a new connection.
It then configures both parts of the integration:

- automatic recall, capture, and session lifecycle through OpenViking REST;
- explicit tools through Hermes' direct HTTP MCP client at `<endpoint>/mcp`.

No local MCP proxy or OpenViking Agent Plugin is required.

## Config

OpenViking's server config is separate from Hermes:

- `ov.conf` configures OpenViking storage, embedding/VLM models, auth, and
  server behavior. OpenViking reads it from `--config`,
  `OPENVIKING_CONFIG_FILE`, or `~/.openviking/ov.conf`.
- `ovcli.conf` stores client/CLI connection values such as `url`, `api_key`,
  `account`, and `user`. It is read from `OPENVIKING_CLI_CONFIG_FILE` or
  `~/.openviking/ovcli.conf`.

Hermes stores the non-secret endpoint and identity settings under
`memory.openviking` in the active profile's `config.yaml`. It stores only
`OPENVIKING_API_KEY` in the active profile's `.env`. The provider REST client
and MCP headers both resolve that same variable, so key rotation cannot update
one connection without the other.

When `OPENVIKING_API_KEY` is set, Hermes lets OpenViking derive account/user
identity from the key. In local or trusted deployments without an API key,
Hermes sends the configured account and user identity headers.

Run `hermes memory setup` again to change the endpoint or import a different
OpenViking CLI profile. Direct edits to `ovcli.conf` do not change an existing
Hermes profile after import.

## MCP Tools

Hermes discovers tools from the running OpenViking server. Their Hermes names
use the `mcp__openviking__<tool>` prefix, for example
`mcp__openviking__find`, `mcp__openviking__search`, and
`mcp__openviking__read`. New or changed OpenViking MCP tools become available
the next time Hermes connects; the memory plugin does not duplicate their
schemas or implementations.

The exact tool surface and argument contracts are owned by the installed
OpenViking version. This includes retrieval, browsing, memory and content
writes, editing, deletion, resource ingestion, and code-oriented search tools.

## Memory Writes

Hermes built-in `memory` tool additions are mirrored to OpenViking after the
local memory operation succeeds:

| Hermes action | OpenViking operation |
|---------------|----------------------|
| `add` | `content/write` with `mode=create` under the configured peer memory namespace |

Built-in `replace` and `remove` operations are not mirrored because Hermes
native memory entries do not yet carry stable OpenViking file URIs. Use the
OpenViking MCP tools when the user asks for an explicit OpenViking write,
change, or deletion.

## Upgrading Existing Hermes Profiles

The former `viking_*` native tool schemas are no longer exposed. After updating
Hermes, run `hermes memory setup`, select OpenViking, and import or enter the
same connection. Setup creates the direct MCP entry and preserves automatic
recall and capture. Existing linked `ovcli.conf` configurations remain readable
for lifecycle compatibility until they are imported this way.
