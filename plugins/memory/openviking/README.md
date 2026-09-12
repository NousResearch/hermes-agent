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
| `OPENVIKING_AGENT` | (none) | Optional peer ID for separate assistant context |

When `OPENVIKING_API_KEY` is set, Hermes lets OpenViking derive account/user
identity from the key. In local or trusted deployments without an API key,
Hermes sends `OPENVIKING_ACCOUNT` and `OPENVIKING_USER` as identity headers.
Hermes also sends `User-Agent: openviking-memory-hermes/<version>` on
OpenViking requests. This standard harness identifier contains the Hermes
version, but no per-user identifier, and does not add a separate request.

### Optional peer identity

New connections use the OpenViking user's memory directory by default. Setup
does not ask for a peer ID. Without a configured peer, Hermes sends neither
`X-OpenViking-Actor-Peer` nor assistant-message `peer_id`.

For separate assistant context, set the existing `agent` field in the active
profile's `config.yaml`:

```yaml
memory:
  openviking:
    agent: work-assistant
```

Existing non-empty `OPENVIKING_AGENT`, YAML `agent`, and linked OpenViking
`actor_peer_id` or legacy `agent_id` values retain their behavior. Resolution
order remains environment, linked OpenViking config, then Hermes YAML. To use
no peer, remove the peer value from each configured source and start a new
Hermes session.

Upgrades do not move or delete existing memories. Installations that relied
on the old implicit `hermes` peer now use user memory for new writes. Without
a peer ID, default OpenViking search covers user memory and existing peer
memories under the same OpenViking user. Old peer memories stay at their
existing paths and remain searchable. Ranking and result limits determine
which memories are returned. Keep a peer ID if you need the narrower view.

Set `agent: hermes` to restore peer-scoped writes. Memories written at user
scope before this change stay there and remain searchable. This setting
changes future writes, not the location of existing memories.

## Tools

| Tool | Description |
|------|-------------|
| `viking_search` | Semantic search with fast/deep/auto modes |
| `viking_read` | Read content at a viking:// URI (abstract/overview/full) |
| `viking_browse` | Filesystem-style navigation (list/tree/stat) |
| `viking_remember` | Submit a fact through OpenViking session memory extraction |
| `viking_forget` | Delete one exact `viking://` memory file URI |
| `viking_add_resource` | Ingest URLs/docs into the knowledge base |

## Memory Writes And Deletes

`viking_remember` creates a one-shot `hermes-remember-<random>` OpenViking
session, adds the fact as one message, and commits the session with no retained
tail. The session remains available in OpenViking for audit. OpenViking then
classifies the source and can add, merge, or skip a memory through its normal
extraction pipeline. The tool returns the one-shot session ID and the
extraction task ID when the server provides one. Extraction continues
asynchronously after the tool returns.

The tool returns `status: submitted` because extraction can add a memory, merge
the fact into an existing memory, or produce no memory operation. It does not
promise that OpenViking created a distinct memory file. The fact is submitted
as an unchanged `user` message so OpenViking owns the final classification.
The legacy `category` argument is still accepted from existing callers but is
not advertised or used. The one-shot session is separate from the live Hermes
conversation, so an explicit remember does not commit or rotate the active
conversation session.

If the message request or commit fails, the error includes the canonical
session URI, the failed stage, the observed message status, and an `ov session
commit <session-id>` recovery command. Inspect the session first. An archive
means the commit completed. A non-empty live `messages.jsonl` with no archive
means the message was accepted but still needs a commit. An empty live file
without an archive is ambiguous and must not trigger an automatic resubmission.
Use the same OpenViking profile and credentials as Hermes for manual recovery.
OpenViking server auto-commit is disabled by default, so an accepted message
whose explicit commit fails normally remains live and unextracted until it is
manually committed.

Successful Hermes built-in `memory` mutations are mirrored to OpenViking in
FIFO order. Hermes persists the exact OpenViking URI for each native memory
entry in the active profile at
`$HERMES_HOME/openviking/memory_mirror_registry.json`, so later mutations never
need to guess a target by semantic similarity:

| Hermes action | OpenViking operation |
|---------------|----------------------|
| `add` | `content/write` with `mode=create` under user memory, or the configured peer memory directory; store the exact returned URI in the mirror registry |
| `replace` | resolve one registry entry from `target` + `old_text`, then replace the same URI and wait for semantic/vector refresh |
| `remove` | resolve one registry entry from `target` + `old_text`, then delete that exact URI and wait for semantic cleanup |

URI construction uses the same server-asserted explicit user identity and
connection snapshot as the current OpenViking provider. The registry stores a
fingerprint of that connection with each mapping, without storing the raw API
key. It also stores the full current text of each mirrored native memory so it
can resolve the native tool's later `old_text` reference. This prevents a later
connection from reusing mappings that belong to another OpenViking endpoint,
identity, or peer.

Changing the endpoint, credentials, identity, or peer starts a new registry
scope. Mappings from the earlier connection then fail closed instead of being
applied to the new connection.

The mirror registry owns only memories created through this built-in-memory
bridge. Session-extracted memories and explicit `viking_remember` writes are not
registered or modified by it. Existing OpenViking memories created before the
registry was introduced also have no safe automatic mapping: a later built-in
`replace` or `remove` for such an entry fails closed with a warning and leaves
OpenViking unchanged rather than guessing which memory to mutate. Use
`viking_forget` with an exact URI for manual cleanup of those entries.

`replace` and `remove` resolve the same `old_text` supplied by the native memory
tool against the registry's pre-mutation content. If that text no longer maps
to exactly one registry entry, the OpenViking mutation fails closed rather than
selecting a URI heuristically.

The mirror is asynchronous and intentionally not a distributed transaction.
The local Hermes memory mutation commits before the OpenViking request. If that
remote request then fails (for example because of a transient network error),
the failure is logged at WARNING and the mirror does not durably replay the
event. Hermes and OpenViking can therefore drift until the entry is repaired
manually or later reconciliation/durable-outbox work handles it. Drift can also
occur if OpenViking accepts a mutation but the following local registry save
fails. In that case, later mutations fail closed because Hermes cannot prove
the exact remote mapping.

Mirror events are ordered within each provider. Registry updates are also
serialized across provider instances that use the same profile in one Hermes
process. The registry does not provide cross-process serialization.

Registry writes request file mode `0600` on POSIX systems. That mode is not an
equivalent Windows ACL guarantee, so Windows deployments should rely on the
normal account/filesystem access controls protecting `HERMES_HOME`.

An unreadable registry, an invalid registry structure, or an unsupported
registry version blocks all mirror operations. This prevents Hermes from
creating OpenViking files that it cannot track and prevents an older Hermes
version from overwriting mappings written by a newer version. The warning
identifies the registry path and distinguishes an unsupported version from an
invalid format.

Stop Hermes before repairing the registry. For an unsupported version, run a
Hermes version that supports the reported registry version; do not delete or
replace the file. For an unreadable or invalid registry, restore a valid backup.
If no backup exists, rename the invalid file and restart Hermes only if you
accept that its existing OpenViking files will no longer have stable mappings.
Those files then require manual cleanup by exact URI.

`viking_forget` is intentionally narrow. It only accepts concrete user memory
file URIs, such as
`viking://user/default/peers/hermes/memories/preferences/mem_abc123.md` (any
explicit user id works; `viking://~/...` input is passed through untouched for
deployments where the server expands the home alias). Files
directly under `memories/`, such as `viking://user/default/memories/profile.md`,
are also allowed because OpenViking supports them. The tool rejects directories,
resources, skills, sessions, generated summary files, and URIs with query
strings or fragments. Use OpenViking's MCP, CLI, or admin APIs for broader
resource and directory cleanup.
