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

The setup can link to an existing `~/.openviking/ovcli.conf`, copy its current
connection values into Hermes, or create a minimal `ovcli.conf` when one does
not exist.

Or manually:

```bash
hermes config set memory.provider openviking
```

Add the connection settings to the active profile's `.env` file. For the
default profile that is `~/.hermes/.env`; for a named profile use
`~/.hermes/profiles/<profile>/.env`.

```text
OPENVIKING_ENDPOINT=http://127.0.0.1:1933
# OPENVIKING_API_KEY=...
# OPENVIKING_ACCOUNT=default
# OPENVIKING_USER=default
# OPENVIKING_AGENT=hermes
```

## Config

OpenViking's server config is separate from Hermes:

- `ov.conf` configures OpenViking storage, embedding/VLM models, auth, and
  server behavior. OpenViking reads it from `--config`,
  `OPENVIKING_CONFIG_FILE`, or `~/.openviking/ov.conf`.
- `ovcli.conf` stores client/CLI connection values such as `url`, `api_key`,
  `account`, and `user`. It is read from `OPENVIKING_CLI_CONFIG_FILE` or
  `~/.openviking/ovcli.conf`.

Hermes-side provider config is read from environment variables in the active
profile's `.env`:

| Env Var | Default | Description |
|---------|---------|-------------|
| `OPENVIKING_ENDPOINT` | `http://127.0.0.1:1933` | Server URL |
| `OPENVIKING_API_KEY` | (none) | User/admin API key for authenticated servers |
| `OPENVIKING_ACCOUNT` | `default` | Tenant account for local/trusted mode |
| `OPENVIKING_USER` | `default` | Tenant user for local/trusted mode |
| `OPENVIKING_AGENT` | `hermes` | Hermes peer ID in OpenViking, used for peer-scoped memories |

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
and `mode=create`. It creates peer-scoped memory files under
`viking://user/peers/${OPENVIKING_AGENT}/memories/...`; OpenViking may return a
canonical user-scoped form such as
`viking://user/default/peers/${OPENVIKING_AGENT}/memories/...` in API-key mode.
Explicit remembers do not depend on session commit extraction.

Successful Hermes built-in `memory` mutations are mirrored to OpenViking in
FIFO order. Hermes persists the exact OpenViking URI for each native memory
entry in the active profile at
`$HERMES_HOME/openviking/memory_mirror_registry.json`, so later mutations never
need to guess a target by semantic similarity:

| Hermes action | OpenViking operation |
|---------------|----------------------|
| `add` | `content/write` with `mode=create`; store the exact returned URI in the mirror registry |
| `replace` | resolve one registry entry from `target` + `old_text`, then replace the same URI and wait for semantic/vector refresh |
| `remove` | resolve one registry entry from `target` + `old_text`, then delete that exact URI and wait for semantic cleanup |

The mirror registry owns only memories created through this built-in-memory
bridge. Session-extracted memories and explicit `viking_remember` writes are not
registered or modified by it. Existing OpenViking memories created before the
registry was introduced also have no safe automatic mapping: a later built-in
`replace` or `remove` for such an entry fails closed with a warning and leaves
OpenViking unchanged rather than guessing which memory to mutate. Use
`viking_forget` with an exact URI for manual cleanup of those entries.

`viking_forget` is intentionally narrow. It only accepts concrete user memory
file URIs, such as
`viking://user/peers/hermes/memories/preferences/mem_abc123.md` or the canonical
`viking://user/default/peers/hermes/memories/preferences/mem_abc123.md`. Files
directly under `memories/`, such as `viking://user/default/memories/profile.md`,
are also allowed because OpenViking supports them. The tool rejects directories,
resources, skills, sessions, generated summary files, and URIs with query
strings or fragments. Use OpenViking's MCP, CLI, or admin APIs for broader
resource and directory cleanup.
