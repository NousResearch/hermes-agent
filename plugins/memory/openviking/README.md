# OpenViking Memory Provider

Context database by Volcengine (ByteDance) with filesystem-style knowledge hierarchy, tiered retrieval, and automatic memory extraction.

## Setup

```bash
hermes memory setup    # select "openviking"
```

Setup offers three paths:

1. **OpenViking Service (VolcEngine Cloud)** — a managed cloud service that
   requires an API key and preserves identity validation.
2. **Quick local setup** — configures a local server, installs OpenViking when
   needed, ensures Ollama and `qwen3-embedding:0.6b` are available, creates a
   Hermes-scoped local workspace, and reuses Hermes' effective static API-key LLM
   configuration. Generated files live under the active Hermes home in
   `openviking/`.
3. **Connect to an existing server** — uses the user's own local or remote
   endpoint by linking a detected `ovcli.conf` profile or accepting another
   server URL. That server remains managed separately.

The quick-local server starts automatically when Hermes needs it. On a clean
Hermes shutdown, tracked workflows and queued semantic/vector work can finish
briefly in the background; the plugin then stops only the exact child process
started by that provider instance. Configuration and stored data are preserved,
and Ollama is left running because other applications may share it. If work
completion or process ownership cannot be verified, the server is left running
rather than risk interrupting work. Existing local and remote servers are never
controlled at runtime. If an existing local endpoint is down during setup,
Hermes can start it once only when a separate OpenViking `ov.conf` already
exists; otherwise setup offers the local-server setup path or another endpoint.

Hermes OAuth, cloud-native, and external-process credentials are not copied to
OpenViking because they may expire or require Hermes-specific refresh logic.
Use a static API-key LLM for quick local setup, or configure and connect an
OpenViking server separately.

Setup links one OpenViking-owned profile and records only that profile path in
the active Hermes profile. New service/custom connections are stored as
`ovcli.conf.<name>` profiles; Quick Local uses its Hermes-scoped `ovcli.conf`.
Editing the linked profile therefore updates the next Hermes run without
duplicating credentials into Hermes configuration.

For compatibility, a manually configured Hermes profile can still use
environment variables:

```bash
hermes config set memory.provider openviking
```

Add the connection settings to the active profile's `.env` file. For the
default profile that is `~/.hermes/.env`; for a named profile use
`~/.hermes/profiles/<profile>/.env`.

```text
OPENVIKING_URL=http://127.0.0.1:1933
# OPENVIKING_API_KEY=...
# OPENVIKING_ACCOUNT=default
# OPENVIKING_USER=default
# OPENVIKING_ACTOR_PEER_ID=hermes
```

## Config

OpenViking's server config is separate from Hermes:

- `ov.conf` configures OpenViking storage, embedding/VLM models, auth, and
  server behavior. OpenViking reads it from `--config`,
  `OPENVIKING_CONFIG_FILE`, or `~/.openviking/ov.conf`.
- `ovcli.conf` and named `ovcli.conf.<name>` profiles store client connection
  values such as `url`, `api_key`, `account`, `user`, and `actor_peer_id`.
  Hermes setup links one exact profile; `OPENVIKING_CLI_CONFIG_FILE` is used
  only when no explicit profile path is linked.

Hermes can link an OpenViking CLI profile or read connection values from
environment variables in the active profile's `.env`:

| Env Var | Default | Description |
|---------|---------|-------------|
| `OPENVIKING_URL` | `http://127.0.0.1:1933` | Server URL |
| `OPENVIKING_API_KEY` | (none) | User/admin API key for authenticated servers |
| `OPENVIKING_ACCOUNT` | `default` | Tenant account for local/trusted mode |
| `OPENVIKING_USER` | `default` | Tenant user for local/trusted mode |
| `OPENVIKING_ACTOR_PEER_ID` | `hermes` | Agent ID in OpenViking, used for peer-scoped memories |

`OPENVIKING_ENDPOINT` and `OPENVIKING_AGENT` remain supported as legacy
fallbacks. Canonical variables take precedence when both forms are present.

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
`viking://user/peers/${OPENVIKING_ACTOR_PEER_ID}/memories/...`; OpenViking may
return a canonical user-scoped form such as
`viking://user/default/peers/${OPENVIKING_ACTOR_PEER_ID}/memories/...` in
API-key mode.
Explicit remembers do not depend on session commit extraction.

Hermes built-in `memory` tool additions are mirrored to OpenViking after the
local memory operation succeeds:

| Hermes action | OpenViking operation |
|---------------|----------------------|
| `add` | `content/write` with `mode=create` under the configured peer memory namespace |

Built-in `replace` and `remove` operations are not mirrored because Hermes
native memory entries do not yet carry stable OpenViking file URIs. Use
`viking_forget` when the user explicitly asks to delete a specific OpenViking
memory URI.

`viking_forget` is intentionally narrow. It only accepts concrete user memory
file URIs, such as
`viking://user/peers/hermes/memories/preferences/mem_abc123.md` or the canonical
`viking://user/default/peers/hermes/memories/preferences/mem_abc123.md`. Files
directly under `memories/`, such as `viking://user/default/memories/profile.md`,
are also allowed because OpenViking supports them. The tool rejects directories,
resources, skills, sessions, generated summary files, and URIs with query
strings or fragments. Use OpenViking's MCP, CLI, or admin APIs for broader
resource and directory cleanup.
