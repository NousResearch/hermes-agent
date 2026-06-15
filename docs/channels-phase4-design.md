# Sessions as Channels — Phase 4: Mesh & Org Scale (meshboard-cloud)

Status: **design — decisions made** (2026-06-10). The six product/security calls (D1–D6, final section) are now **decided** (owner delegated; defaults chosen for the smallest blast radius and the most reversible path, each noting where a later phase revisits). Slice 4.0 is ready to implement. Owner: operator + desktop lane.
Related: [`channels-design.md`](channels-design.md) (Phases 1–3, shipped), [`session-presence.md`](session-presence.md), meshboard-cloud (`~/Workspaces/Projects/meshboard-cloud`), mesh feature `F-003-multi-participant-channels`.

> **Naming note.** "Phase 4" here is the *Hermes channels* roadmap (Phases 1–3 already shipped per `channels-design.md`). It is unrelated to the meshboard-cloud README's internal "Phase 4 — Teams," which is *already deployed* and is the substrate this design builds on.

## 0. Scope & non-goals

Phase 4 extends channels from "one user's own devices" (Phases 1–3, peer-to-peer over local/tailnet WS) to:

- **(a) Cross-mesh** — one org/mesh (a meshboard-cloud account) shares a specific channel with another account, so members of both co-view and post.
- **(b) Cross-org** — the same primitive scoped across organizational tenancy boundaries with strict data isolation.

The cloud is a **relay + durable log + membership authority for channels that opt in**, never a replacement for the local path. The hosting gateway remains the ordering authority for the live agent turn (the Phase-3 invariant). meshboard-cloud is purely additive — it is the NAT-crossing transport and the offline backlog store, plus the identity/membership plane that local P2P cannot provide.

**Non-goals for Phase 4:** running the agent in the cloud for a channel (the agent stays on the hosting gateway; the cloud lane in `cloud_lane.ts` is a separate product); CRDT/offline-edit merge of message bodies; end-to-end encryption as a *default* (called out as an open decision); replacing Syncthing presence propagation on meshes that have it.

## 1. Data model (D1)

New tables in a `migrations/0009_channels.sql`. Conventions match the existing schema exactly: `TEXT` ULID primary keys (`ulid()` from `util.ts`), `account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE` as the tenancy anchor on every row, `CURRENT_TIMESTAMP` text timestamps (UTC, space-separated, no `Z` — readers normalize exactly like `account_do.ts` `deriveLiveness` and `events.ts` `TS_NORM`), and `IF NOT EXISTS` for re-runnable migrations.

The column names deliberately mirror the **local** Hermes shapes so a row round-trips with no field remapping:
- local `messages(role, content, sender_device, timestamp, tool_calls, tool_name, reasoning, finish_reason, token_count, ...)` → `channel_messages`
- local `sessions(title, model, cwd, device_name, source, started_at, archived, ...)` → `channels`
- local presence record (`session_key`, `instance_id`, `host`, `client`, `endpoint`, `updated_at`/`expires_at`) → `channel_participants`

```sql
-- migrations/0009_channels.sql
-- Phase 4 (Hermes channels): cloud-backed channels = the durable, relayable
-- projection of a Hermes session that opted into the cloud. Mirrors the local
-- state.db sessions/messages shapes (hermes_state.py) and the presence record
-- (hermes_cli/session_presence.py) so rows round-trip without remapping.

-- channels: one row per session promoted to a cloud channel. origin_* identify
-- the local session + the gateway that owns its turn ordering (the Phase-3
-- "hosting gateway is the ordering authority" invariant survives: the cloud row
-- is a projection, never a fork).
CREATE TABLE IF NOT EXISTS channels (
  id                 TEXT PRIMARY KEY,                       -- ulid (cloud channel id)
  owner_account_id   TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  origin_session_key TEXT NOT NULL,                          -- local sessions.id / presence.session_key
  origin_device_id   TEXT REFERENCES devices(id),            -- gateway/device that hosts the turn
  title              TEXT,
  model              TEXT,
  source             TEXT,                                   -- 'tui_gateway' | 'cli' | ...
  visibility         TEXT NOT NULL DEFAULT 'owner-private',  -- owner-private | team | shared
  status             TEXT NOT NULL DEFAULT 'active',         -- active | archived
  last_seq           INTEGER NOT NULL DEFAULT 0,             -- monotonic high-water mark (see §2)
  created_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  archived_at        TEXT,
  UNIQUE(owner_account_id, origin_session_key)               -- idempotent promote
);
CREATE INDEX IF NOT EXISTS idx_channels_owner  ON channels(owner_account_id, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_channels_origin ON channels(origin_session_key);

-- channel_messages: the relayed/persisted message log. seq is cloud-assigned
-- (canonical read order); origin_message_id is the local messages.id for
-- dedupe + idempotency. sender_* mirrors the Phase-1 attribution unit
-- (messages.sender_device) plus the additive sender_account_id.
CREATE TABLE IF NOT EXISTS channel_messages (
  id                 TEXT PRIMARY KEY,                       -- ulid (cloud msg id)
  channel_id         TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
  account_id         TEXT NOT NULL,                          -- = channels.owner_account_id (denormalized for tenancy filter)
  seq                INTEGER NOT NULL,                       -- cloud-assigned, dense per channel
  origin_message_id  TEXT,                                   -- local messages.id (stringified) — dedupe key
  origin_device_id   TEXT,                                   -- which gateway emitted it
  role               TEXT NOT NULL,                          -- user | assistant | tool | system
  content            TEXT,                                   -- message body (see §6: server-visible vs e2e)
  sender_device      TEXT,                                   -- mirrors messages.sender_device (label)
  sender_account_id  TEXT REFERENCES accounts(id),           -- additive: which account posted (cross-mesh attribution)
  tool_name          TEXT,
  tool_calls         TEXT,                                   -- JSON
  finish_reason      TEXT,
  token_count        INTEGER,
  origin_ts          REAL,                                   -- local messages.timestamp (epoch float) for display
  created_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,-- cloud receive time
  UNIQUE(channel_id, seq),
  UNIQUE(channel_id, origin_device_id, origin_message_id)    -- idempotent push (replay-safe)
);
CREATE INDEX IF NOT EXISTS idx_chmsg_channel_seq ON channel_messages(channel_id, seq);
CREATE INDEX IF NOT EXISTS idx_chmsg_account     ON channel_messages(account_id, created_at);

-- channel_participants: presence/roster — the cloud projection of the Phase-3
-- session.participants roster, fed by AccountDO/ChannelDO socket attach/detach
-- + heartbeats. Mirrors the presence record; liveness derived like account_do.ts.
CREATE TABLE IF NOT EXISTS channel_participants (
  id              TEXT PRIMARY KEY,                          -- ulid
  channel_id      TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
  account_id      TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE, -- viewer's account
  device_id       TEXT REFERENCES devices(id),
  instance_id     TEXT,                                      -- mirrors presence.instance_id ("host-pid")
  display_name    TEXT,
  client          TEXT,                                      -- 'desktop' | 'tui' | 'cli' | 'phone'
  host            TEXT,
  role            TEXT NOT NULL DEFAULT 'viewer',            -- viewer | poster (channel-level)
  last_seen_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(channel_id, account_id, instance_id)
);
CREATE INDEX IF NOT EXISTS idx_chpart_channel ON channel_participants(channel_id, last_seen_at);

-- channel_members: authoritative membership / sharing grants. Subject is an
-- account (per-user) — see open decision D3. history_floor_seq gates backfill
-- visibility (D2): members read messages with seq >= this (0 = full history).
CREATE TABLE IF NOT EXISTS channel_members (
  channel_id        TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
  account_id        TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
  permission        TEXT NOT NULL DEFAULT 'read',            -- read | post | admin
  granted_via       TEXT,                                    -- 'owner' | team_id | peer_mesh_id (provenance)
  granted_by        TEXT NOT NULL,                           -- account_id of granter
  history_floor_seq INTEGER NOT NULL DEFAULT 0,
  joined_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  revoked_at        TEXT,
  PRIMARY KEY (channel_id, account_id)
);
CREATE INDEX IF NOT EXISTS idx_chmember_account ON channel_members(account_id, revoked_at);

-- channel_invites: mirrors team_invites (0005) — short-lived single-use token,
-- redeemed by an authed account. Cross-mesh + cross-org invites by email.
CREATE TABLE IF NOT EXISTS channel_invites (
  token       TEXT PRIMARY KEY,
  channel_id  TEXT NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
  email       TEXT NOT NULL,
  permission  TEXT NOT NULL DEFAULT 'read',
  invited_by  TEXT NOT NULL,                                 -- account_id
  created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  expires_at  TEXT NOT NULL,
  accepted_at TEXT,
  accepted_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_chinvite_channel ON channel_invites(channel_id, accepted_at);
CREATE INDEX IF NOT EXISTS idx_chinvite_email   ON channel_invites(email);

-- peer_meshes: a directed trust edge — account A trusts account B's Ed25519
-- signing key, enabling B to be invited into A's channels. Reuses the Ed25519
-- precedent from publisher_keys (0006) / crypto_verify.ts. Cross-ORG sharing is
-- the same edge where the two accounts belong to different organizations (§4).
CREATE TABLE IF NOT EXISTS peer_meshes (
  id                  TEXT PRIMARY KEY,
  account_id          TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE, -- the trusting side (A)
  peer_account_id     TEXT REFERENCES accounts(id),          -- the trusted side (B), once known
  peer_label          TEXT,
  peer_public_key_b64 TEXT NOT NULL,                         -- raw 32-byte Ed25519 pubkey, base64
  trust_state         TEXT NOT NULL DEFAULT 'pending',       -- pending | active | revoked
  created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  activated_at        TEXT,
  revoked_at          TEXT,
  UNIQUE(account_id, peer_public_key_b64)
);
CREATE INDEX IF NOT EXISTS idx_peer_meshes_account ON peer_meshes(account_id, trust_state);
```

**Why these keys/indexes.** `channel_messages` carries both a cloud `seq` (`UNIQUE(channel_id, seq)`, dense, the canonical read order) and a `(channel_id, origin_device_id, origin_message_id)` unique constraint so a re-push of the same local message — the common case after a reconnect — is a no-op `INSERT OR IGNORE`. The `account_id` denormalization on `channel_messages` mirrors what `dispatches`/`dispatch_events` already do (so every tenancy filter is a single-table index hit, never a join — the pattern `events.ts` relies on).

## 2. Mesh ↔ cloud sync

### 2.1 Promotion (opt-in, additive)
A local session becomes a cloud channel only when a user explicitly **promotes** it ("Share to cloud" in the desktop, or a CLI flag). Promotion is the single moment the zero-dep core touches the cloud:

`POST /v1/channels {origin_session_key, title, model, source, origin_device_id, visibility}` → creates the `channels` row (idempotent on `UNIQUE(owner_account_id, origin_session_key)`), returns `{channel_id, last_seq}`. The gateway stores `channel_id` in the local session's `state_meta` (key `cloud_channel_id`) so the binding survives restart.

### 2.2 Push (gateway → cloud)
The **hosting gateway** (the one that owns the turn, per Phase 3) is the only writer. On each persisted local message (the same hook that today calls `_publish_session_presence`), if the session has a `cloud_channel_id`, the gateway pushes a batch to `POST /v1/channels/{id}/messages`. The cloud assigns `seq = last_seq + 1 …` in a single D1 batch, bumps `channels.last_seq`, and `INSERT OR IGNORE`s on the dedupe constraint; the gateway advances a local `cloud_pushed_seq` watermark and re-pushes everything after it on reconnect (idempotent). After the D1 write, the handler fires `ctx.waitUntil()` to push to the channel's DO socket subscribers — the same `dispatches.ts → AccountDO /push` pattern in production.

### 2.3 Pull (cloud → gateway/client)
- **Backfill:** `GET /v1/channels/{id}/messages?since_seq=&limit=` — watermark cursor identical in shape to `GET /v1/events?since=`. A client opening a channel it isn't hosting pulls history from its `history_floor_seq` forward.
- **Live tail:** the channel WebSocket (§3.4), plus an SSE fallback `GET /v1/channels/{id}/stream?since_seq=` modeled on `events.ts` `streamEvents`.

### 2.4 Conflict / ordering model
**No CRDTs — single-writer per channel, exactly as Phase 3 mandates.** The hosting gateway serializes turns locally; the cloud `seq` is assigned server-side in arrival order from that single writer, so there is no merge.
- **Cross-mesh posters** do not write the cloud log directly. Their post is relayed *to the hosting gateway* (§3.4), enters its turn queue as a normal `session.prompt` carrying `sender_device` + `sender_account_id`, and comes back out through the §2.2 push. The host stays the sole writer; ordering never forks.
- **Host offline:** the cloud log is read-only (members see history + presence; the agent can't take a turn). New posts queue client-side and flush when the host returns — the same liveness story as local cross-device attach.

### 2.5 What stays local-only
`state.db` itself; `model_config`, `system_prompt`, `billing_*`, `cwd`, cost/token columns, `handoff_*`, compression locks, FTS; any session **not** promoted (promotion is per-session and revocable via `DELETE /v1/channels/{id}`); tool *execution* and file contents (only the message log is relayed, subject to §6).

## 3. Cross-mesh sharing (one mesh invites another)

A "mesh" = a meshboard-cloud `accounts` row. Cross-mesh sharing = granting a non-owner account a `channel_members` row.

### 3.1 Identity
- **Account-level:** existing `accounts(id, email)` + bearer `sessions`. Humans authenticate via magic-link (`auth_magic.ts`) or team SSO (`sso.ts`).
- **Device/gateway:** the existing paired-client bearer (`sessions.kind='device'`) — the credential the agent already uses. The hosting gateway connects to the channel WS with its device bearer.
- **Mesh-to-mesh trust:** `peer_meshes` with an Ed25519 key, reusing `crypto_verify.ts` + the `publisher_keys` precedent.

### 3.2 Invite / membership flow (mirrors `teams.ts`)
1. **Owner invites** (`admin`): `POST /v1/channels/{id}/invites {email, permission}` → mints a single-use TTL token, logs `channel.invited`.
2. **Invitee signs in** (magic-link/SSO → bearer) and **accepts**: `POST /v1/channels/invites/accept?token=` → atomic `UPDATE … RETURNING` guard, inserts a `channel_members` row with `history_floor_seq = channels.last_seq` (default-gated backfill — D2), logs `channel.joined`.
3. **Discovery:** the invitee's desktop lists the channel via `GET /v1/channels` (owned + member), pulls history, connects the WS.

### 3.3 Permissions (channel-level, three tiers, `atLeast`/`roleOf` pattern)
`read` (pull history + live tail + roster), `post` (also relay a prompt to the host), `admin` (also invite/revoke/rename/archive). Owner is implicitly `admin`. A `requireChannel(env, accountId, channelId, min)` helper returns the member row or `403`, called at the top of every channel handler.

### 3.4 Channel relay (`ChannelDO`)
A per-channel Durable Object (bound `CHANNEL_DO`), modeled on **DispatchDO** (device-bound relay) crossed with **AccountDO** (multi-socket fan-out): viewers join a fan-out set on `WS /v1/channels/{id}/socket`; the host joins `?role=host`; a `post`-permission socket's message is forwarded to the host socket as a `prompt` op (carrying `sender_device` + `sender_account_id`); new messages the host pushes and roster changes fan out to all viewers; socket attach/detach upserts `channel_participants` and broadcasts a `participants` op (the cloud projection of the Phase-3 `session.participants` event).

### 3.5 Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| `POST` | `/v1/channels` | bearer (owner) | promote; idempotent |
| `GET` | `/v1/channels` | bearer | owned + member |
| `GET`/`PATCH`/`DELETE` | `/v1/channels/{id}` | `requireChannel(read/admin/admin)` | get / rename+visibility / teardown |
| `POST` | `/v1/channels/{id}/messages` | device bearer, host, `post` | push batch; cloud assigns `seq` |
| `GET` | `/v1/channels/{id}/messages` | `requireChannel(read)` | backfill from `history_floor_seq` |
| `GET` | `/v1/channels/{id}/stream` | `requireChannel(read)` | SSE (replay + live) |
| `WS` | `/v1/channels/{id}/socket` | `requireChannel(read)` | live tail + poster→host relay + roster |
| `GET` | `/v1/channels/{id}/participants` | `requireChannel(read)` | roster + liveness |
| `POST` | `/v1/channels/{id}/invites` | `requireChannel(admin)` | mint invite |
| `POST` | `/v1/channels/invites/accept` | bearer | redeem |
| `GET`/`PATCH`/`DELETE` | `/v1/channels/{id}/members[/{accountId}]` | `requireChannel(read/admin/admin)` | list / change perm / revoke |
| `POST`/`GET`/`DELETE` | `/v1/peer-meshes[/{id}]` | bearer (owner/admin) | trust edges |
| `POST` | `/v1/peer-meshes/{id}/activate` | bearer + Ed25519 sig | activate handshake |

All routes register in `index.ts`'s `route()` (literal-before-`{id}` discipline), join `maybeAliasPath`'s `cockpitRoots`, and inherit `withSecurityHeaders` + the http→https 308.

## 4. Cross-org sharing (tenancy & isolation)

- **Tenancy model.** Reuse `account_id` as the isolation boundary: a channel + all its messages/participants belong to exactly one `owner_account_id`. Cross-org sharing does **not** copy rows across tenants; it grants read/post access via `channel_members` to an account in another org.
- **Shared vs private.** Shared: the channel's title/model, messages at `seq >= member.history_floor_seq`, the live roster. Private (never crosses the tenancy line): the owning account's other channels, `devices`, `tasks`, `dispatches`, `billing_*`, `audit_log` — all excluded by the `account_id` filter. A member sees exactly the channels they hold a `channel_members` row for.
- **Scoping.** A `channel_messages` row's tenancy is `account_id` (= owner); its author org is derivable from `sender_account_id`, enabling cross-org attribution without leaking either org's private data. Cross-org invites are **gated by an `active` `peer_meshes` edge** (D5) — otherwise `403 cross-org trust required` — making org↔org sharing an explicit, signed, revocable handshake.
- **Read-path enforcement.** Every handler resolves access **only** through `channel_members`/`channels.owner_account_id` via `requireChannel()` — never an ad-hoc `account_id ==` check. Message queries are `WHERE channel_id = ? AND seq >= ?` with `channel_id` already proven accessible, so a member can't widen to another channel by guessing IDs.

## 5. Zero-dependency (cloud strictly additive)

The hard constraint from `channels-design.md` is preserved verbatim: **no layer may REQUIRE mesh/cloud/tailnet.** A user with no meshboard-cloud account, no Tailscale, no Syncthing is completely unaffected — sessions, single-device co-viewing (gateway `FanoutTransport`), and sender attribution all run with zero network identity; cross-device attach still works via the Phase-2 explicit `host:port` endpoint on LAN. **No cloud client is instantiated** unless a session is promoted, and promotion is guarded by "is a cloud account configured?" If the cloud is configured but unreachable, promoted channels degrade to local-only (the gateway buffers the push watermark and flushes on reconnect; the live turn never depends on the cloud).

### Degradation matrix

| Capability | No cloud/tailnet/Syncthing | + Syncthing | + Tailscale | + meshboard-cloud |
|---|---|---|---|---|
| Single-device channel (co-view, attribution) | ✅ gateway fanout | ✅ | ✅ | ✅ |
| Cross-device attach (own devices) | ✅ explicit `host:port` | ✅ + presence-folder discovery | ✅ + MagicDNS | ✅ + cloud presence |
| Presence/session-list propagation | ✅ local presence dir | ✅ synced folder | ✅ | ✅ cloud roster |
| NAT-crossing real-time fan-out | ❌ (LAN/tailnet) | ❌ | ✅ if both on tailnet | ✅ ChannelDO relay |
| Offline backlog for absent members | ❌ | ⚠️ file-sync (not real-time) | ❌ | ✅ `channel_messages` log |
| Cross-mesh / cross-org sharing | ❌ (single-user core) | ❌ | ❌ | ✅ members + invites |
| Identity for sharing | device name | device name + label | tailnet hostname | account + email/SSO |

The bottom three rows are the only net-new Phase-4 capabilities, and every one is `❌` (gracefully absent, not broken) without the cloud. Nothing in columns 1–3 regresses.

## 6. Security

- **Authz.** Every channel route: `requireAuth` (unchanged) → `requireChannel(min)`. Device-only ops (host push, host WS role) require `sessions.kind='device'`. Governance actions write `audit_log`.
- **Cross-org leak prevention.** Single-source-of-truth `requireChannel()` + the `channel_id`-scoped message query (§4) confine a member to their channels; cross-org invites require an `active` `peer_meshes` edge; revoked members' sockets are force-closed and excluded by `revoked_at IS NOT NULL`.
- **Server-visible vs E2E.** *Decision D1.* As designed, `content` is server-visible (relay + previews + SSE/WS fan-out assume plaintext). E2E would store ciphertext and relay only — incompatible with cloud search/preview/agent. See Open Decisions.
- **Tokens/secrets.** Reuse existing posture: opaque revocable bearer rows; sha256-hashed `mbk_…` API keys; single-use TTL'd invite tokens; `peer_meshes` stores only a public key (private key never leaves the peer). No new secret class; `MASTER_SECRET` untouched.
- **Rate limits.** KV-backed IP + account limits on promote/invite (mirroring magic-link); the high-volume host-push endpoint is limited **by channel** at the `ChannelDO` (not per-request) to avoid D1 hot-row contention; over-limit → `429`, gateway backs off via its watermark buffer.
- **Poster→host relay abuse.** A `post` member injects prompts into the host's agent — already untrusted input (same turn queue as a local user); owner can downgrade to `read` or revoke instantly. @-mention gating becomes the throttle when multiple humans collide.

## 7. Phased rollout

- **4.0 — own-mesh cloud channels (smallest shippable).** Migration `0009` (`channels` + `channel_messages` only) + `POST /v1/channels`, `POST/GET /v1/channels/{id}/messages`, `GET …/stream` (SSE, copy `events.ts`). No DO, no membership. Value: a promoted session is durably logged + readable across the *owner's own* devices even when the host is offline. Gateway change: the promote action + the push hook on the existing presence-publish path.
- **4.1 — live relay + roster.** `ChannelDO`, `WS …/socket`, `channel_participants`, `GET …/participants`. Multi-client co-viewing crosses NAT in real time. Wire the desktop participant chips to the cloud roster (already renders the Phase-3 `session.participants` shape).
- **4.2 — cross-mesh sharing.** `channel_members` + `channel_invites`, invite/accept/members endpoints (copy `teams.ts`), `requireChannel()`, the poster→host relay op. A second account can be invited to `read`/`post`. Audit on.
- **4.3 — cross-org trust.** `peer_meshes` + the Ed25519 activation handshake; gate cross-org invites on an `active` edge. Ship `history_floor_seq` + `visibility` controls.
- **4.4 — hardening.** Per-channel DO rate limits, force-close-on-revoke, `peer_meshes` cascade revocation, and (pending D1) any E2E work; a "Shared channels" admin view.

Each slice is independently deployable, extends `test/smoke.sh`, keeps `openapi.yaml` byte-synced (the CI check), and adds no required dependency to the Hermes core.

---

## Decisions (made 2026-06-10)

These were product/security calls; the owner delegated them. Each is now **decided** — the principle was: pick the smallest-blast-radius, most-reversible default that keeps the zero-dependency core intact, and design the schema so the heavier option stays an *additive* later migration. Each notes the phase that revisits it.

- **D1 — Server-visible vs E2E → DECIDED: server-visible plaintext for 4.0–4.2.** `channel_messages.content` is plaintext (enables cloud backfill, SSE/WS relay, preview, and a future cloud-hosted agent — and matches the rest of the platform). The schema reserves room so a `ciphertext`/`enc_alg` column is a purely additive migration. **Revisit at 4.3**: offer the hybrid (plaintext within an org, E2E across `peer_meshes` boundaries) *only if* a cross-org customer requires it — not by default.
- **D2 — Pre-join history → DECIDED: owner-chooses per invite, default join-forward.** The invite carries an optional admin-only `include_history` flag; absent it, a new member's `history_floor_seq` is pinned to `channels.last_seq` at join (they see only messages from when they joined). Join-forward is the privacy-safe default so an outside org never silently inherits prior internal discussion; the owner can opt a specific invite into full history.
- **D3 — Sharing identity → DECIDED: per-account for 4.2.** Grants are `channel_members.account_id` rows (finest-grained, maps cleanly onto the existing bearer/account model and the `teams.ts` invite ergonomics, and keeps revocation per-account + auditable). **Add at 4.3**: a per-team convenience grant that *expands* to per-account member rows at grant time (provenance recorded in `granted_via = team_id`) — so the team layer is sugar over per-account, not a separate coarse path.
- **D4 — Agent on host vs cloud lane → DECIDED: always the owner's hosting gateway.** The cloud is pure relay; a relayed post enters the host's turn queue as a normal `session.prompt`. This preserves the single-writer ordering invariant, the user's chosen model/tools/keys, and the zero-dependency story. **Host offline ⇒ channel is read-only** until it returns (honest, simple). A cloud-lane fallback is explicitly *not* built — only ever reconsidered far later as an opt-in per-channel setting.
- **D5 — Cross-org sharing → DECIDED: require a signed `peer_meshes` handshake.** Any invite whose invitee account belongs to a *different* organization than the channel owner requires an `active` Ed25519 `peer_meshes` trust edge between the two accounts; otherwise `403 cross-org trust required`. Cross-org data flow is an org-admin-level, signed, revocable, auditable decision — never a side effect of one channel admin knowing an email. **Within a single org, plain email invites suffice** (no `peer_meshes` needed).
- **D6 — Retention/deletion → DECIDED: retain until the channel is deleted, with an explicit hard-erase, for 4.0.** `DELETE /v1/channels/{id}` hard-deletes all `channel_messages` (GDPR-style erase); the local session is untouched. **Add before any paid org tier**: owner-configurable retention (TTL/cap) with a hard maximum, and a legal/compliance review of cross-org retention before cross-org GA.

---

## Files referenced (for implementers)

- Existing design: `docs/channels-design.md`, `docs/session-presence.md`
- Local schema to mirror: `hermes_state.py` (`sessions`, `messages` incl. `sender_device`)
- Local presence: `hermes_cli/session_presence.py` (`write_session_presence`/`list_session_presence` record shape)
- Gateway hooks: `tui_gateway/server.py` (`_publish_session_presence`, `session.presence_list`, `_sanitize_sender_device`, `session.participants` roster), `tui_gateway/transport.py` (`FanoutTransport.attach/detach`)
- Cloud schema precedents: meshboard-cloud `migrations/0001_initial.sql` (accounts/sessions/devices/organizations), `0005_teams.sql` (teams/invites/audit — the invite template), `0006_ecosystem.sql` (Ed25519 `publisher_keys` — the `peer_meshes` precedent)
- Cloud patterns to copy: meshboard-cloud `src/teams.ts` (invite/accept/role enforcement), `src/auth.ts` (`requireAuth`), `src/events.ts` (watermark pagination + SSE), `src/do/account_do.ts` (multi-socket fan-out + `/push` + liveness), `src/do/dispatch_do.ts` (device-bound relay), `src/index.ts` (route registration + `maybeAliasPath`)
