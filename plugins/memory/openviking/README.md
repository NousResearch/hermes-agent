# OpenViking Memory Provider

Context database by Volcengine (ByteDance) with filesystem-style knowledge hierarchy, tiered retrieval, and automatic memory extraction.

## Requirements

- OpenViking installed with the `openviking-server` command available
- OpenViking server config initialized and validated (`openviking-server init`,
  then `openviking-server doctor`)
- OpenViking server running and reachable from Hermes

OpenViking 0.2.10 or newer is recommended. For backward compatibility,
Hermes can identify older servers that expose the legacy status-only health
response, but only when anonymous OpenAPI metadata also identifies the service
as OpenViking. OpenViking 0.2.6 and earlier are deprecated for this integration;
upgrade them to receive the current health contract and compatibility fixes.
Selecting a peer with `X-OpenViking-Actor-Peer` requires OpenViking 0.4.0 or
newer; older 0.3.x servers retain their legacy agent-filtering limitation.

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

The setup can link an existing `~/.openviking/ovcli.conf` profile. Linking is
an explicit shared-file choice: the selected external file is read at runtime
and its values are not copied into `.env`. To isolate credentials, use
separate saved ovcli profiles or manual split profile configuration. Setup can
also create a minimal `ovcli.conf` when one does not exist.

Or manually:

```bash
hermes config set memory.provider openviking
```

If the server requires authentication, add only the API key to the active
profile's `.env` file. For the default profile that is `~/.hermes/.env`; for a
named profile use `~/.hermes/profiles/<profile>/.env`.

```text
OPENVIKING_API_KEY=...
```

## Config

OpenViking's server config is separate from Hermes:

- `ov.conf` configures OpenViking storage, embedding/VLM models, auth, and
  server behavior. OpenViking reads it from `--config`,
  `OPENVIKING_CONFIG_FILE`, or `~/.openviking/ov.conf`.
- `ovcli.conf` stores client/CLI connection values such as `url`, `api_key`,
  `account`, and `user`. It is read from `OPENVIKING_CLI_CONFIG_FILE` or
  `~/.openviking/ovcli.conf`.

For manual configuration, store the optional API key in the active profile's
`.env` and put the endpoint and local/trusted identity settings in that
profile's `config.yaml`:

```yaml
memory:
  provider: openviking
  openviking:
    endpoint: http://127.0.0.1:1933
    account: default
    user: default
    agent: hermes
```

Manual configuration keeps only `OPENVIKING_API_KEY` in the active profile's
`.env`; put `endpoint`, `account`, `user`, and `agent` under
`memory.openviking` in the active profile's `config.yaml`. The wizard's legacy
`Keep in Hermes only` option still stores selected connection values in `.env`;
users who want non-secrets out of `.env` should use the default `Mirror to
OpenViking store` option or manual split configuration. Legacy
`OPENVIKING_*` environment overrides remain supported. Bound
profiles resolve identity from their own secret scope; the process-level
`OPENVIKING_ENDPOINT` remains a non-secret fallback when the profile supplies
none.

When `OPENVIKING_API_KEY` is set, Hermes lets OpenViking derive account/user
identity from the key. In local or trusted deployments without an API key,
Hermes sends `OPENVIKING_ACCOUNT` and `OPENVIKING_USER` as identity headers.

## Tools

| Tool | Description |
|------|-------------|
| `viking_search` | Semantic search with fast/deep/auto modes |
| `viking_read` | Read content at a viking:// URI (abstract/overview/full) |
| `viking_browse` | Filesystem-style navigation (list/tree/stat) |
| `viking_remember` | Store a fact directly with OpenViking `content/write` |
| `viking_forget` | Delete one exact `viking://` memory file URI |
| `viking_add_resource` | Ingest URLs/docs into the knowledge base |

## Memory Writes And Deletes

`viking_remember` writes directly to OpenViking with `POST /api/v1/content/write`
and `mode=create`, using the current-user shorthand
`viking://user/memories/...`. Hermes sends `OPENVIKING_AGENT` as the peer
identity through the `X-OpenViking-Actor-Peer` header. OpenViking applies the
authenticated account's namespace policy and may return a canonical URI.
Peer selection requires OpenViking 0.4.0+; 0.3.x servers retain their legacy
agent-filtering limitation.
Explicit remembers do not depend on session commit extraction.

Hermes built-in `memory` tool additions are mirrored to OpenViking after the
local memory operation succeeds:

| Hermes action | OpenViking operation |
|---------------|----------------------|
| `add` | `content/write` with `mode=create` under the current-user shorthand namespace |

Built-in `replace` and `remove` operations are not mirrored because Hermes
native memory entries do not yet carry stable OpenViking file URIs. Use
`viking_forget` when the user explicitly asks to delete a specific OpenViking
memory URI.

`viking_forget` is intentionally narrow. It only accepts concrete user memory
file URIs, such as
`viking://user/memories/preferences/mem_abc123.md` or a canonical URI returned
by OpenViking. The tool rejects directories, resources, skills, sessions,
generated summary files, and URIs with query strings or fragments. Use
OpenViking's MCP, CLI, or admin APIs for broader resource and directory
cleanup.
