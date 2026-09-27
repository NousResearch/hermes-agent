# litco-agent: the matter host

`litco-agent` is a fork of [Hermes Agent](https://github.com/NousResearch/hermes-agent) that runs as the agent for one litigation matter on its own machine. A firm's LitKit instance drives it over HTTP through the **turn server** described here. Everything LitCo adds lives in the top-level `litco/` package and the `plugins/platforms/litco_turn/` platform plugin, so the upstream tree stays syncable (`git fetch upstream && git merge upstream/main`).

## How it runs

The turn server is a Hermes gateway platform named `litco_turn`. It is a bundled platform plugin, so the gateway discovers it with no change to core code, and it runs in the same process as cron, memory, skills, and the other channels (Slack, Telegram). The gateway enables it whenever `LITCO_HOST_SECRET` is set, or when `gateway.platforms.litco_turn.enabled` is true in `config.yaml`.

Each turn runs as a Hermes `AIAgent` built the same way the API-server adapter builds one: the provider and model come from the profile's config, the toolsets come from `platform_toolsets.litco_turn`, and the transcript is the profile's SessionDB. The agent's streaming and tool callbacks are translated into LitKit's agent2 events (see `litco/hermes_runner.py`).

The server can also run on its own, as a sidecar started by the same systemd unit: `python -m litco.turn_server --host 127.0.0.1 --port 8765`. The gateway platform is the intended mode.

## Environment

| Variable | Required | Meaning |
|---|---|---|
| `LITCO_HOST_SECRET` | yes | Shared secret LitKit sends in `X-Host-Secret`. It is also the HMAC key for user assertions. With no secret, the server refuses every request. |
| `LITCO_MATTER_ID` | yes | The one LitKit matter this host serves. A turn for any other matter gets 403. |
| `LITCO_MATTER_HOME` | no | Root of the per-thread working directories. Default `~/matter`. |
| `LITCO_AGENT_TOKEN` | for LitKit tools | The matter-pinned LitKit bearer token (`lkm_` plus 43 base64url characters). The server sends it when fetching attachments; the `litkit` toolset sends it on every call. |
| `LITCO_INSTANCE_URL` | for LitKit tools | Base URL of the firm's LitKit instance, for example `https://<firm>.litco.ai`. |
| `LITCO_TURN_HOST` | no | Bind address. Default `127.0.0.1`. |
| `LITCO_TURN_PORT` | no | Port. Default `8765`. |

## Authentication

Every endpoint except `GET /health` requires `X-Host-Secret`, compared in constant time with `LITCO_HOST_SECRET`.

A request may also name the human it acts for. It then sends both `X-LitKit-Acting-User: <userId>` and `X-LitKit-User-Assertion: <assertion>`. The assertion uses the wire format LitKit already mints in `src/lib/agent-daemon/user-assertion.ts`:

```
v1.<userId>.<matterId>.<issuedAtMs>.<ttlMs>.<base64url HMAC-SHA256>
```

The MAC covers `v1.<userId>.<matterId>.<issuedAtMs>.<ttlMs>` and is keyed on the host secret. Both times are milliseconds, as in the TypeScript code. The server rejects the request with 401 when the MAC fails, the assertion has expired (60 seconds of clock skew allowed), it names another matter, the header user differs from the asserted user, or only one of the two headers is present. When an assertion verifies, the body's `userId` must equal the asserted user (403 otherwise). A request with neither header is a service request, such as cron work.

## Endpoints

### `POST /turn`

Body:

```json
{
  "matterId": "…", "userId": "…", "sessionId": "…", "text": "…",
  "attachments": [{"fileId": "…", "mime": "application/pdf", "filename": "complaint.pdf", "url": "https://…"}],
  "channel": "slack" | "web" | "telegram",
  "kind": "channel" | "dm",
  "budgetMs": 600000
}
```

`matterId`, `sessionId`, and `text` are required. `channel` defaults to `web` and `kind` to `channel`. A `dm` turn requires `userId`. `budgetMs` is optional; without it, a turn has no time or token ceiling.

The response is `text/event-stream`. The header `X-Turn-Id` carries the turn id. Each frame is `event: <type>` followed by one `data:` line of JSON, and every payload carries `type`, `turnId`, `stepId` (1, 2, 3, … within the turn), and `ts` (milliseconds since the epoch). A `: keepalive` comment is sent every 15 seconds of silence. The frames, in the order they can occur:

| Event | Fields | When |
|---|---|---|
| `goal_accepted` | `sessionId` | Always first. |
| `assistant_delta` | `delta` | Streamed answer text. |
| `assistant_reset` | `reason` | The text streamed so far was interim commentary (for example, before a tool call). Discard it; later deltas start fresh. |
| `tool_started` | `call{toolCallId,name}`, `args` | A tool call begins. |
| `tool_progress` | `call`, `message` | Progress from a long tool, such as a delegated subagent. |
| `tool_complete` | `call`, `result{status,summary,durationMs}` | A tool call ends. `status` is `ok` or `error`, and follows Hermes's own verdict: a terminal command with a non-zero exit, a result with an `error` field or `success: false`, and anything Hermes logs as "returned error" arrive as `error`. The summary never carries the tool's output, only a tool-supplied summary, an error message, an exit code (`command failed with exit code 1`), or the size of the result. |
| `error_classified` | `category`, `message`, `recovery` | The turn failed. |
| `loop_halted` | `reason`, `explanation` | The turn stopped early: `interrupted`, `budget_exhausted`, or `shutdown`. |
| `final` | `text`, `citations`, `usage{inputTokens,outputTokens,cacheReadTokens?,cacheWriteTokens?}`, `durationMs`, `modelUsed?`, `deliverables?` | Always last. |

Hermes has no plan events, so `plan_drafted` and `plan_step_*` are never sent.

`final.deliverables` lists the files the turn hands to the thread, as `{fileId, filename, mime, path, deliverableClass?}`:

1. every file the agent registered with the `litco_deliver_local` tool, in registration order, whatever its type (a file registered from outside `deliverables/`, or under another `name`, is first copied into the thread's `deliverables/` folder); then
2. every other file created or changed under `deliverables/` whose extension is `.docx`, `.xlsx`, `.pptx`, `.pdf`, `.md`, `.txt`, `.csv`, `.png`, or `.jpg`/`.jpeg`, unless it is scratch: a dotfile or a file in a dot-folder, `*.spec.json`, `*.tmp`, or an Office lock file (`~$*`).

Anything else under `deliverables/` (JSON, scripts, zips) stays on the host unless registered. `deliverableClass` appears when the agent gave one on registration. `path` is relative to `LITCO_MATTER_HOME`. `fileId` encodes that path and can be fetched from `GET /deliverables/{fileId}`.

If the client disconnects, the turn keeps running and its transcript lands in the session. `POST /interrupt/{turnId}` still stops it.

### `POST /interrupt/{turnId}`

Stops the turn. Returns 202 while the turn winds down, 200 if it had already finished, and 404 for an unknown id. The turn's stream ends with `loop_halted{reason:"interrupted"}` and `final`.

### `GET /health`

Returns `{ok, version, hermesVersion, uptimeSeconds, activeTurns, matterId}`. No secret is required. `hermesVersion` comes from Hermes's own identity resolver (install stamp, then git), for example `0.21.5+3720.g3754997` for a fork commit past the `0.21.5` release; the package metadata's `0.0.0` placeholder is never reported.

### `GET /deliverables/{fileId}`

Returns the bytes of a file listed in `final.deliverables`. Ids that resolve outside a `deliverables/` folder get 404.

## Sessions and concurrency

One LitKit thread is one `sessionId`, and one `sessionId` is one Hermes session. A second turn on the same `sessionId` continues the conversation. Hermes may rotate its internal session id when it compacts a long conversation; the mapping from `sessionId` to the current Hermes id is kept in `$HERMES_HOME/litco_sessions.json`, so the thread follows the rotation.

Turns on the same `sessionId` run one at a time, in arrival order. Turns on different `sessionId`s run concurrently. There is no global queue.

## Working directories

The server creates these under `LITCO_MATTER_HOME` on first use:

```
shared/                   channel threads (the whole case team)
  deliverables/
  memories/               the matter's shared Hermes memory (MEMORY.md, USER.md)
users/<userId>/           dm threads (one lawyer's private work)
  deliverables/
  memories/               that lawyer's private Hermes memory
inbox/<turnId>/           attachments fetched for one turn
```

A turn's working directory is `shared/` for a `channel` thread and `users/<userId>/` for a `dm` thread. The terminal and file tools start there, so a relative path such as `deliverables/memo.docx` lands in the thread's own folder.

### Memory

Hermes's built-in memory (`MEMORY.md` for notes, `USER.md` for the user profile, both read into the system prompt) is scoped by thread, not by profile. A `dm` turn reads and writes `users/<userId>/memories/`; a `channel` turn reads and writes `shared/memories/`. The profile-wide `$HERMES_HOME/memories/` is never read or written by a turn, so a note from one lawyer's private thread does not reach any other lawyer's turn.

No Hermes core file changes for this. `MemoryStore` resolves its files through its overridable `_path_for` on every load and write, and every consumer in a turn (the `memory` tool, the system-prompt snapshot, compaction reloads, the background memory review, which shares the parent agent's store object) reaches the files through `agent._memory_store`. The runner replaces that store, after building the agent and before the first prompt is assembled, with a `ScopedMemoryStore` bound to the thread's folder (`litco/memory_scope.py`). The binding is on the object, so a background review that finishes after the turn still writes to the right folder.

Limits: this covers turns that arrive through the turn server. Turns from native Hermes adapters (a Slack or Telegram bot configured directly in the profile) and cron jobs still use the profile-wide memory, and so do memory writes staged for approval (`write_approval` for memory, off by default), which are applied later to the profile-wide files; keep that gate off on a matter host. An external memory provider (`memory.provider`) keeps its own keying. A session created before this change keeps the system prompt it stored on its first turn, which may carry profile-wide memory.

Attachments that carry a `url` are downloaded into `inbox/<turnId>/` with `Authorization: Bearer $LITCO_AGENT_TOKEN`, and the agent is told where each one landed. An attachment without a `url`, or one that fails to download, is named in the prompt with its LitKit `fileId`.

## LitKit toolset

The `litkit` toolset is the host's access to the firm's LitKit instance. It is a bundled backend plugin (`plugins/litkit/`), so it loads with no `plugins.enabled` entry, and its tools stay hidden until `LITCO_INSTANCE_URL` and `LITCO_AGENT_TOKEN` are set. The code is in `litco/litkit/`: `client.py` (HTTP), `tools.py` (the tools), `files.py` (where tools write), `context.py` (the turn identity).

### Env contract

| Variable | Needed for | Use |
|---|---|---|
| `LITCO_INSTANCE_URL` | every call | Base URL; paths are appended to it. |
| `LITCO_AGENT_TOKEN` | every call | `Authorization: Bearer lkm_…`. Pinned to one matter; any other matter answers `403 agent_token_matter_mismatch`. |
| `LITCO_HOST_SECRET` | acting for a lawyer | HMAC key for the per-call user assertion. Without it, a turn with an acting user cannot call LitKit. |
| `LITCO_MATTER_ID` | optional | The matter. When unset, the client reads `GET /api/matters`, which returns exactly the token's one matter. |
| `LITCO_MATTER_HOME` | optional | Root of the working directories (default `~/matter`). |

Credentials are read through `agent.secret_scope.get_secret`, so a profile's `.env` works as well as the process environment.

### Who a call acts for

The turn server verifies the turn's user assertion and records the lawyer as `acting_user`. The runner binds a `TurnIdentity` (turn id, matter, acting user, working directory) for the length of the turn, and Hermes copies it into the threads that run tool calls and delegated subagents. On every LitKit request the client sends:

```
Authorization: Bearer lkm_…
X-LitKit-Acting-User: <userId>
X-LitKit-User-Assertion: v1.<userId>.<matterId>.<issuedAtMs>.300000.<mac>
```

The assertion is minted fresh for each request and each retry (MAC = base64url, no padding, of HMAC-SHA256 keyed on the host secret over `v1.<userId>.<matterId>.<issuedAtMs>.<ttlMs>`). A turn whose assertion did not verify, and all work outside a turn (cron), sends neither acting-user header, so LitKit applies the Matter Agent user's own viewer role. No `Origin` header is sent.

### Errors and retries

- 401 and 403 raise `LitKitPermissionError` and come back to the model as `{"error": "not permitted for this user on this matter (…)", "status": 403, "permission_denied": true}`. They are never retried.
- Reads (`GET`, and the read-only POSTs: bulk text export, quote checks, LitLex cite checks) retry on 429, 500, 502, 503 and 504 with exponential backoff (four retries, `Retry-After` honored). Writes retry only on 429, 503, or a connection that never opened, so a deliverable or a proposal is never committed twice.
- 404 reads as "not found, or not visible to this user"; LitKit hides walled documents as missing.
- The token and the host secret never appear in logs, reprs, or error text.

### Tools

| Tool | LitKit route(s) | Result |
|---|---|---|
| `litkit_matter` | `GET /api/matters/{id}`, `…/docs?countOnly=1`, `…/search/facets` | name, document count, custodians, productions, Bates prefixes, acting user |
| `litkit_search` | `GET …/search` | compact hits; notes the 5 s budget, the 500-hit cap, ranked fallback |
| `litkit_docs` | `GET …/docs?cursor=` | one page plus `nextCursor`; with `saveAs`, every page into `census/<name>.jsonl` |
| `litkit_document` | `GET /api/documents/{id}` (Bates via `…/bates-resolve`) | metadata, tags, productions |
| `litkit_text` | `GET /api/documents/{id}/text` | `texts/<bates>.txt` with a self-citing header, preview |
| `litkit_pdf` | `GET /api/documents/{id}/pdf` or `/native` | `pdfs/<bates>.pdf` or `natives/…`, sha256 |
| `litkit_export_text` | `POST …/export/text` (NDJSON, 500 ids per call, looped) | `texts/<bates>.txt` per document, `texts/index.json`, resumable |
| `litkit_memos` | `GET /api/matter-files?kind=memo`, `GET /api/matter-files/{id}` | memo list; `memos/<title>.md` |
| `litkit_files` | LitSpace list, search, content, upload | list, hits, `files/<name>`, upload result |
| `litkit_deliver` | `POST …/deliverables` (multipart) | `blocked`, gate summary, version; full response in `qa/` |
| `litkit_quote_check` | `POST …/quote-check/file` or `…/quote-check` | verified and unverified quotations; full response in `qa/` |
| `litkit_review` | review-jobs list, status, records, resume, cancel, pause | pass-through |
| `litkit_ingest` | productions, progress, exceptions, ingests, ingest jobs; resume, cancel, reingest, retry | pass-through |
| `litkit_proposals` | `POST`/`GET …/proposals` | proposal id and status |
| `litkit_tags` | `/api/tags`, `/api/documents/{id}/tags`, `…/bulk-tag` | list, create, apply, remove |
| `litkit_work_sets` | `/api/work-sets` | list, get, create, close, reopen |
| `litkit_litlex` | LitLex search, opinion (saved to `litlex/`), citator, authorities, statute, cite-resolve, cite-check, brief-check | pass-through |
| `litkit_notify` | `POST /api/notifications/emit` | to the acting lawyer (default), a named member, or the matter |
| `litkit_remember`, `litkit_recall` | `POST /api/agent/actions` (`remember`, `recall`) | `scope:"user"` keeps a note private to the acting lawyer |
| `litkit_actions` | `POST /api/agent/actions` | `term_frequency`, `find_redacted`, `hot_documents`, `refresh_dossier`, `diagnose_issue`, `diagnose_ingest`, `litlex_format_cite` |
| `litkit_attachment` | `GET …/chat/attachments/{fileId}` | `inbox/<fileId>/<filename>` |
| `litco_deliver_local` | none (local) | registers a file for this turn's `final.deliverables`, with optional `name` and `deliverableClass`; offered on any matter host, LitKit configured or not |

A tool given a path that does not exist (`litkit_deliver`, `litkit_quote_check`, `litkit_files upload`, `litkit_litlex brief_check`, `litco_deliver_local`) returns `file_missing: true` and an error that says to write the file first, confirm it exists, and call again. Nothing is sent to LitKit.

A blocked deliverable (`422`) returns `blocked: true`, the gate findings, and an instruction to report them to the user rather than resubmit; nothing was committed.

### Where tools write

Everything a LitKit tool writes lands under the turn's working directory (`shared/` or `users/<id>/`): `texts/`, `texts_x/`, `pdfs/`, `natives/`, `census/`, `memos/`, `files/`, `litlex/`, `inbox/`, `qa/`, and `litkit/results/` for spilled results. Names that come from LitKit (Bates numbers, filenames) are reduced to one safe path segment, `dir` arguments cannot climb out, and files handed to `litkit_deliver`, `litkit_quote_check`, `litkit_files upload` and `litkit_litlex brief_check` must sit inside the working directory or the matter home. Gate findings go to `qa/`, never to `deliverables/`, so they are not posted back to the thread.

Results larger than 12,000 characters follow Hermes's spill convention: the full JSON is saved under `litkit/results/` and the model gets a `<persisted-output>` block with the path and a preview.

### Host configuration

- Hermes defers plugin tools behind its tool-search bridge by default. On a matter host the LitKit tools are the main surface, so the profile should set `tools.tool_search.enabled: "off"` (or accept the bridge, which lists the tools and calls them through `tool_call`).
- The legal skills live in `litco/skills/legal/`. Point the profile at them with `skills.external_dirs: [<checkout>/litco/skills]`.

## Legal skills

`litco/skills/legal/` holds the firm's procedures rewritten for the matter host: `litkit-corpus-pull` (census, bulk text, hot-document ranking), `litkit-deliverable` (build, quote-check, commit, report the gates), `deposition-prep-package`, `discovery-letter-brief`, `docket-document-retrieval`, `expanded-legal-letter-redlines`, `legal-cite-check`, and `production-data-analysis` (with `references/litkit-access-procedure.md`). They call the `litkit` tools instead of a cookie-jar session and keep the verification discipline: quotations checked against extracted text before delivery, certified transcripts only, as-filed ECF copies. They carry no client names or matter facts.

## Known upstream noise

- `pm/shell.py:13` raises a `SyntaxWarning: invalid escape sequence` on import (a Windows path in a docstring). It is upstream code and harmless; it is left alone so the tree stays syncable.
- The gateway's loop-tick watchdog binds an `AF_UNIX` socket at `$HERMES_HOME/state/gateway.loop-tick.<pid>.sock`. With a deep `HERMES_HOME` (typical of a macOS dev checkout under a scratch folder) the path passes the 104-byte limit and the watchdog logs `AF_UNIX path too long`; liveness then falls back to the heartbeat file. Upstream hardcodes the location (both the gateway and the `hermes gateway` probe compute it), so it cannot be moved without a core change. The host image's `HERMES_HOME` is short, so this appears only in local runs; use a short `HERMES_HOME` there.

## Code map

| Path | Role |
|---|---|
| `litco/turn_server.py` | aiohttp app: auth, parsing, SSE framing, per-session locks, budgets, interrupts, deliverables. |
| `litco/hermes_runner.py` | Builds the `AIAgent` for a turn and maps Hermes callbacks to events. |
| `litco/assertion.py` | Host-secret comparison and the user-assertion MAC. |
| `litco/homes.py` | Working-directory layout, deliverable ids, the deliverable rule, and the `litco_deliver_local` registry. |
| `litco/memory_scope.py` | Per-thread scoping of Hermes's built-in memory. |
| `litco/litkit/` | LitKit client, the `litkit` tools, the turn identity, the write boundary. |
| `plugins/litkit/` | Registers the `litkit` toolset (bundled backend plugin). |
| `litco/skills/legal/` | Legal skills for the matter host. |
| `plugins/platforms/litco_turn/` | Registers the `litco_turn` gateway platform. |
| `tests/litco/` | Contract tests with a fake runner, runner tests with a fake agent, and LitKit client and toolset tests against a fake LitKit server. |

## Testing

```
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -e ".[messaging]" --group dev
.venv/bin/python -m pytest tests/litco -q
```
