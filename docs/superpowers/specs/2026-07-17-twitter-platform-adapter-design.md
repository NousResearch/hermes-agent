# Twitter/X Platform Adapter Design

**Status:** Approved design, pending implementation plan

**Date:** 2026-07-17

**Target:** `plugins/platforms/twitter/` on current `main`

**Source work:** [PR #12352](https://github.com/NousResearch/hermes-agent/pull/12352), salvaged and redesigned rather than merged wholesale

## Summary

Add a first-class Twitter/X gateway platform plugin that lets automated accounts such as `@heywebcmd` participate safely in public, multi-user conversation branches and private DMs.

The adapter will:

- register through `ctx.register_platform()`;
- implement the current `BasePlatformAdapter.send(chat_id, content, reply_to, metadata)` contract;
- ingest real mentions and DM events using X API v2 polling;
- route public replies by conversation branch and DMs by real DM conversation ID;
- use OAuth 2.0 Authorization Code with S256 PKCE only;
- store OAuth tokens and adapter state beneath profile-aware `get_hermes_home()` paths;
- enforce `allowed_users` / `allow_all_users` before enrichment or dispatch;
- enrich public triggers with bounded conversation trees, participant profiles, metrics, and media;
- support outbound image media;
- expose bookmarks and post metrics through service-gated plugin tools;
- serialize API work through bounded, rate-aware queues; and
- include integration-oriented tests against a temporary `HERMES_HOME`, with no real credentials or live X API calls.

The implementation will not claim or accept app-only bearer-token authentication or OAuth 1.0a.

## Context and Maintainer Feedback

PR #12352 implemented a large Twitter adapter against an older core gateway shape. The maintainer review identified a salvageable feature set but several architectural and correctness problems:

- it did not register through the platform-plugin registry;
- its `send()` signature did not match the current base contract;
- OAuth PKCE produced a hex digest instead of an unpadded base64url S256 challenge;
- token storage hard-coded `~/.hermes` instead of using the active profile home;
- mention ingestion and DM routing were incomplete or incorrect;
- setup, authorization, cron delivery, and credential locking were not integrated with current platform hooks.

The port therefore takes endpoint intent and test ideas from the old PR while implementing a new adapter against current platform APIs.

The current Discord adapter is the primary operational reference. Its history shows the importance of:

- supervising receiver tasks after startup;
- deduplicating replayed events after reconnect;
- fetching context from the exact routed conversation;
- treating context from unauthorized participants as untrusted background;
- keeping cron delivery independent from stale live HTTP sessions;
- bounding attachment work;
- avoiding blocking work on the event loop; and
- measuring queue wait and network latency separately.

## Goals

1. Make an authenticated X account behave like a careful human participant in public conversations:
   - answer direct mentions;
   - answer mentions made in replies to third-party posts;
   - continue nested branches after the bot replies;
   - allow additional authorized users to join that branch;
   - observe relevant sibling replies as context without replying to every post in the conversation.
2. Support reliable one-to-one and group DM intake and replies using actual DM conversation IDs.
3. Preserve profile isolation and gateway security boundaries.
4. Degrade gracefully when optional enrichment is unavailable or rate-limited.
5. Keep the adapter entirely under `plugins/platforms/twitter/` and avoid Twitter-specific core changes.

## Non-Goals

- OAuth 1.0a support.
- App-only bearer-token support.
- Filtered-stream or Account Activity webhook deployment in the first version.
- Full-archive conversation search.
- Automatically replying to every post in a conversation merely because the bot previously participated.
- Automatically inserting bookmarks into prompts.
- Automatically polling engagement metrics after every post.
- Video or animated GIF upload in the first version.
- Automatic splitting of oversized output into a new multi-post thread.
- Durable replay of queued outbound posts after process restart.
- Exact-once delivery after an ambiguous network failure; the X create-post API provides no suitable idempotency key.

## Plugin Layout

```text
plugins/platforms/twitter/
├── __init__.py
├── adapter.py       # BasePlatformAdapter lifecycle, ingestion, routing, send
├── client.py        # OAuth-authenticated X API v2 client and response parsing
├── oauth.py         # PKCE, callback exchange, token refresh, token persistence
├── queue.py         # Bounded endpoint-aware request scheduling
├── state.py         # Profile-safe cursors, deduplication, and branch mappings
├── tools.py         # Bookmarks and metrics plugin tools
└── plugin.yaml
```

Small data models may remain colocated until extraction materially improves clarity. The adapter must not modify core files to special-case Twitter.

## Registration and Platform Hooks

`register(ctx)` will call `ctx.register_platform()` with:

- `name="twitter"` and label `Twitter / X`;
- an adapter factory accepting the current `PlatformConfig`;
- an `is_connected` callback that verifies a usable profile-scoped OAuth token file;
- an interactive setup function;
- a YAML-to-internal-environment bridge for current platform authorization and cron hooks;
- `allowed_users_env="TWITTER_ALLOWED_USERS"`;
- `allow_all_env="TWITTER_ALLOW_ALL_USERS"`;
- `cron_deliver_env_var="TWITTER_HOME_CHANNEL"`;
- a standalone sender that creates a fresh profile-scoped client; and
- the X post length as the platform message-length boundary.

Non-secret user settings are authored in `config.yaml`. Internal environment variables are only a compatibility bridge for existing gateway hooks. OAuth access and refresh tokens are never written to `.env`.

## Configuration

The proposed user-facing configuration shape is:

```yaml
twitter:
  client_id: "public-oauth-client-id"
  redirect_uri: "http://127.0.0.1:8765/callback"
  allowed_users: []
  allow_all_users: false
  home_channel: timeline
  poll_interval_seconds: 30
  initial_backfill: 0
  conversation:
    max_depth: 8
    max_posts: 40
    siblings_per_parent: 5
  media:
    max_download_bytes: 10485760
    max_upload_bytes: 5242880
  queue:
    max_pending: 100
    max_wait_seconds: 900
```

Defaults will be conservative and documented. Invalid limits fail setup or adapter construction with actionable errors rather than being silently coerced to unsafe values.

`allowed_users` values are canonical X user IDs. Setup may accept `@handles` for convenience, but it resolves and stores the numeric IDs so authorization remains stable across handle changes. All X IDs remain strings end-to-end.

## OAuth 2.0 PKCE

Only OAuth 2.0 Authorization Code with PKCE is supported.

The setup flow will:

1. Generate a high-entropy `code_verifier` using a cryptographically secure source.
2. Compute `SHA256(code_verifier)`.
3. Encode that digest as URL-safe base64 with trailing `=` padding removed.
4. Send `code_challenge_method=S256`.
5. Generate and validate a cryptographically random `state` value.
6. Bind a loopback callback server to the configured loopback redirect URI.
7. Exchange the code only after matching the callback state.
8. Request only scopes used by the implemented endpoints:
   - `tweet.read`
   - `tweet.write`
   - `users.read`
   - `offline.access`
   - `dm.read`
   - `dm.write`
   - `bookmark.read`
   - `bookmark.write`
   - `media.write`
9. Resolve `/2/users/me` and persist the authenticated account ID and username with the token record.

The callback server has a bounded timeout, validates the request path, does not log authorization codes, and shuts down on success, rejection, timeout, or cancellation.

Token refresh is serialized so concurrent pollers and sends do not race to rotate the same refresh token. A failed refresh marks the adapter disconnected and raises a retryable fatal error where appropriate; it never silently falls back to a different profile or authentication method.

## Profile-Safe State and Credential Storage

All paths derive from `get_hermes_home()` at operation time, not module import time:

```text
<get_hermes_home()>/twitter/oauth2.json
<get_hermes_home()>/twitter/state.json
```

`oauth2.json` contains access token, refresh token, expiry, granted scopes, client ID binding, and authenticated account identity. It is written with `atomic_json_write(..., mode=0o600)`.

`state.json` contains only non-secret operational state:

- newest processed mention ID;
- newest processed DM event ID and pagination state needed for overlap-safe polling;
- a bounded deduplication set;
- bounded `bot_post_id -> participation_anchor_id` mappings; and
- any schema version required for safe migration.

State writes are atomic. Corrupt state is quarantined or ignored with a warning; token files are never discarded automatically. Adapter startup uses the existing platform credential lock keyed by the token/account identity so two active gateway instances cannot consume and reply as the same X account concurrently.

Tests change `HERMES_HOME` between cases, so no path or loaded state may be cached globally across profiles.

## Public Conversation Model

Every post in an X reply tree shares the root post's `conversation_id`. That identifier alone is too broad for an automated account because a popular root can contain many unrelated parallel branches.

Public chat IDs therefore have this form:

```text
tweet:<conversation_id>:<participation_anchor_id>
```

The participation anchor is the first inbound post that summons the bot in a branch when no ancestor is already associated with an existing branch.

### Branch Mapping

When the bot sends a reply, the adapter persists:

```text
<new bot post ID> -> <participation anchor ID>
```

For each later inbound trigger, the adapter walks the trigger's direct ancestor chain:

- if it reaches a mapped bot-authored post, the event joins that participation branch;
- otherwise, the trigger post ID becomes a new participation anchor.

This makes a branch shared by its participants rather than owned by the original tagger. Authorization is still evaluated independently for every triggering author.

### Trigger Rules

An authorized public post triggers Hermes when at least one is true:

1. its structured mention entities include the authenticated bot account;
2. `in_reply_to_user_id` is the authenticated bot account; or
3. it quotes a bot-authored post in a way X treats as explicitly summoning the account.

Text substring matching is not sufficient. The adapter uses structured post fields and authenticated account identity.

Posts that are merely siblings or nearby descendants do not trigger the agent. They may appear in bounded conversation context when a genuine trigger is processed.

This rule matches X's automated-reply restriction: a self-serve API client can reply only when the target post's author explicitly summoned the replying account by mention or quote.

### Approved Scenario

Given:

```text
T0 Alice: third-party root post
└── T1 Bob: reply that tags @heywebcmd
    ├── T2 @heywebcmd: bot reply
    │   └── T3 Carol: direct reply to @heywebcmd
    └── T2b Carol: sibling reply to Bob
```

- T1 triggers a new participation branch anchored at T1.
- T2 is sent with `reply_to=T1` and maps back to anchor T1.
- T3 joins the T1 branch and triggers if Carol is independently authorized.
- T2b is context-only unless it explicitly mentions or quotes `@heywebcmd`.
- If T2b summons the bot, it starts a separate branch anchored at T2b because its ancestors do not pass through T2.

## Mention Ingestion

The mention poller calls `GET /2/users/{authenticated_user_id}/mentions` with:

- `since_id` from profile state;
- expansions for author, mention entities, referenced posts, and attachment media;
- post fields for author, timestamps, conversation/reply structure, edit history, metrics, sensitivity, and attachments;
- user fields needed for bounded profile enrichment; and
- media fields needed for safe inbound handling.

Responses are paginated until the saved cursor boundary or configured per-cycle cap. Events are sorted oldest-first before dispatch. The cursor advances only after each event has been classified and either safely dispatched, deliberately ignored, or recorded as a permanent non-retryable failure. Retryable processing failures leave the cursor before the failed event.

On first setup, `initial_backfill: 0` establishes the latest cursor without dispatching historical mentions. A positive configured value dispatches at most that many newest eligible events in chronological order. Once a cursor exists, restart processing resumes missed events normally.

The adapter filters its own posts, reposts that are not supported triggers, duplicates, and structurally invalid events before authorization.

## Direct Message Ingestion and Routing

The DM poller uses the X API v2 DM lookup endpoint as a paginated REST API. It does not treat `/2/dm_events` as a streaming response.

Each inbound DM event is normalized with:

- the actual DM event ID as `message_id`;
- `dm:<dm_conversation_id>` as `chat_id`;
- the sender's X user ID as `user_id`;
- the sender's username and display name when available;
- media metadata and bounded downloaded media; and
- the original DM event ID as gateway message metadata for deduplication and traceability.

The adapter ignores outbound events authored by the authenticated account. DMs are authorized using the same fail-closed policy as public mentions.

`send()` routes `dm:<conversation_id>` to:

```text
POST /2/dm_conversations/<conversation_id>/messages
```

with the documented `{text, attachments}` body. It never passes a DM conversation ID to a participant-ID endpoint.

The base `reply_to` argument is accepted for contract compatibility but is not sent for DMs because the X DM create endpoint does not expose a message-reply reference. The destination remains the originating DM conversation, and the result records that no server-side reply threading was applied.

## Authorization and Trust Boundaries

The plugin registers `TWITTER_ALLOWED_USERS` and `TWITTER_ALLOW_ALL_USERS` with the common gateway authorization layer.

The adapter also performs an early authorization check after minimal event parsing and before:

- profile lookup beyond included response data;
- conversation search;
- ancestor lookup;
- media download; or
- event dispatch.

Default behavior is fail-closed. No allowlist plus false `allow_all_users` means no inbound user is authorized.

For a multi-user public branch:

- each triggering author is checked independently;
- authorization of the original tagger does not authorize later participants;
- unauthorized posts never become user turns; and
- posts from other participants included in an authorized trigger's tree snapshot are labeled unverified, untrusted background.

Profiles, biographies, post text, alt text, quoted content, usernames, and conversation context are all untrusted user-controlled content. They are placed in `channel_context` or equivalent user-context fields, never in `channel_prompt` or a mutable system prompt.

## Conversation Enrichment

After authorization, a public trigger receives best-effort enrichment under a strict deadline.

### Tree Reconstruction

The adapter queries recent search with:

```text
conversation_id:<root conversation ID>
```

and reconstructs parent-child relationships from `referenced_tweets` reply references. The resulting bounded neighborhood contains:

1. the direct ancestor spine from root to the trigger, up to `max_depth`;
2. bot-authored posts mapped to the active participation branch;
3. direct replies to those bot posts;
4. up to `siblings_per_parent` nearby replies around each relevant parent; and
5. no more than `max_posts` total posts.

Selection is deterministic. Ancestors and the exact trigger branch take precedence over siblings. Within a tie, recent posts are preferred and final output is rendered chronologically.

Only the triggering post becomes the event's user message. The tree snapshot is contextual material and must not synthesize additional user turns.

If recent search is unavailable, forbidden by the account's API access, older than its time window, or rate-limited beyond the enrichment deadline, the adapter falls back to direct parent lookup. If parent lookup also fails, the trigger still dispatches without tree enrichment.

### Profile Enrichment

For the trigger author, and for included participants when already available through expansions, context may include:

- display name and username;
- account description;
- account creation time;
- location when public;
- verification state; and
- public follower/following/post/listed metrics.

No follower graph or recent user timeline is fetched automatically. Profile data is clearly labeled as self-authored, untrusted metadata.

### Metrics

The trigger post includes its available public metrics. Private/non-public metrics are requested only for posts authored by the authenticated account and only through the explicit metrics tool.

The adapter does not automatically poll engagement after sending because that would add continuous cost and background activity unrelated to an inbound user turn.

### Inbound Media

Included media metadata carries type, dimensions, alt text, preview URL, and public metrics when available.

Supported inbound images are downloaded only after authorization, through the existing bounded profile-aware media cache. Downloads enforce:

- HTTPS and existing SSRF-safe fetch rules;
- declared and streamed byte caps;
- supported image MIME types;
- limited attachment counts; and
- request deadlines.

Unsupported video/GIF media remains descriptive metadata and a safe URL/preview reference; it does not block the text trigger.

## Adapter Lifecycle

`connect()` performs these steps:

1. Load and validate profile-scoped OAuth credentials.
2. Acquire the platform credential lock.
3. Refresh the token if necessary.
4. Resolve and verify `/2/users/me` against the stored account identity.
5. Load cursors, deduplication state, and branch mappings.
6. Establish first-run baselines when required.
7. Start supervised mention and DM polling tasks.
8. Mark the adapter connected only after identity verification, state loading, and successful initial poll/baseline operations.

Each receiver task has a done callback. An unexpected post-startup exit becomes a retryable fatal adapter error so the gateway supervisor can remove and reconnect the adapter. One poller cannot silently die while the adapter remains reported as healthy.

`disconnect()`:

- marks the shutdown intentional;
- cancels and awaits all pollers and queue workers;
- closes the HTTP client;
- flushes bounded non-secret state;
- releases the credential lock; and
- leaves no stale task callback capable of marking a newer connection fatal.

All HTTP requests, sleeps, callback waits, enrichment work, and task shutdowns are bounded and cancellation-safe.

## Send Contract and Routing

The adapter implements exactly:

```python
async def send(
    self,
    chat_id: str,
    content: str,
    reply_to: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SendResult:
    ...
```

Routing is explicit:

- `tweet:<conversation_id>:<anchor_id>` creates a public post; when `reply_to` is present, it sets `reply.in_reply_to_tweet_id` to that exact post ID.
- `dm:<conversation_id>` sends to the existing DM conversation.
- `timeline` creates a new top-level public post for cron and explicit delivery.

Bare numeric IDs are rejected rather than guessed as post, user, or DM conversation IDs.

Public replies preserve the active branch mapping after a successful response. The adapter validates that `reply_to` is numeric text but never converts it through floating point.

The adapter returns success only after X confirms creation and supplies an ID. A queued request is not reported as delivered.

Content must fit the supported X post limit after platform validation. The adapter does not silently split one Hermes response into an unsolicited multi-post thread.

## Outbound Media

`metadata` may provide existing gateway media paths. The adapter accepts supported local image files only in the first version.

Before upload it validates:

- the path is an allowed local file path;
- the file exists and is regular;
- MIME and extension match a supported image format;
- file size is within the configured cap; and
- the number of images is within X's per-post limit.

It uploads through the OAuth 2.0 media endpoint using `media.write`, receives string media IDs, and attaches them to the post or DM body. Partial upload failure prevents post creation and returns an actionable error; it does not create a text-only post unexpectedly.

## Rate-Limit Queue

The client owns bounded FIFO queues per endpoint/rate-limit bucket so a limited read endpoint does not block an unrelated write endpoint.

Queue behavior:

- reject new work with a clear backpressure error when `max_pending` is reached;
- record enqueue time, start time, completion time, endpoint class, and attempt count without logging content or tokens;
- honor `Retry-After` and `x-rate-limit-reset` using a monotonic deadline plus bounded jitter;
- wake cleanly on cancellation or disconnect;
- keep public replies and DMs ordered within their applicable write bucket;
- give required delivery work precedence over optional enrichment work without starving either indefinitely; and
- enforce `max_wait_seconds`.

For `send()`, the caller awaits the real result. If the maximum wait expires before an HTTP attempt begins, the item is removed and returns failure. It is not delivered later after the gateway may have initiated fallback behavior.

Known-safe failures such as a 429 before processing may be retried after reset. Ambiguous failures after a create-post or send-DM request begins—connection loss, timeout after write, or an indeterminate server failure—are not automatically retried because that can duplicate public posts or DMs. The result identifies the outcome as uncertain and logs the correlation metadata needed for diagnosis.

The queue is intentionally process-local and non-durable. Restart does not replay pending outbound content.

## Standalone Cron Delivery

The registered standalone sender supports out-of-process cron delivery when no live gateway adapter exists.

It:

- resolves credentials and configuration inside the requested profile scope;
- creates a fresh HTTP client and queue;
- accepts `timeline` or an explicit typed route;
- sends and closes the client in the same call; and
- never reuses a session owned by a disconnected live adapter.

`TWITTER_HOME_CHANNEL` is the internal bridge for `twitter.home_channel`, defaulting to `timeline` when configured for public cron posts. DM cron delivery requires an explicit `dm:<conversation_id>` destination.

## Plugin Tools

Tools are registered by the Twitter plugin and gated on the profile having usable OAuth credentials. They do not become core tools.

### `twitter_bookmarks`

Operations:

- `list` with bounded pagination;
- `add` with a string post ID; and
- `remove` with a string post ID.

The list operation calls `/2/users/<authenticated_user_id>/bookmarks`; it never uses `/2/users/me/bookmarks` and does not send unsupported `since_id` parameters.

### `twitter_post_metrics`

Accepts one or more string post IDs within a small bound and returns:

- public metrics for visible posts; and
- non-public metrics only for recent posts authored by the authenticated account when the API returns them.

Tools use the same OAuth refresh, rate-limit, ID-validation, error-sanitization, and profile-isolation paths as the adapter.

## Error Handling and Observability

Logs use structured, sanitized messages and never contain:

- access or refresh tokens;
- authorization codes or PKCE verifiers;
- full DM bodies;
- full public post bodies; or
- unredacted API error payloads that may echo credentials.

Useful operational fields include:

- route type, not raw private conversation content;
- event/post IDs as strings;
- conversation and branch IDs where safe;
- queue wait duration;
- HTTP duration measured immediately around the request;
- rate-limit reset duration;
- enrichment duration and fallback reason; and
- cursor advancement or deduplication decisions.

Authentication errors are distinguished from authorization failures, rate limits, transient transport failures, invalid routes, API reply restrictions, and ambiguous delivery outcomes.

## Testing Strategy

No test uses real X credentials or a live X endpoint. HTTP behavior is supplied by a local fake server or request transport stub while exercising real plugin imports and adapter/client code.

### Registration and Contract Tests

- Import `plugins/platforms/twitter` through the plugin loader.
- Assert registration occurs through `ctx.register_platform()`.
- Exercise the shared plugin-platform interface tests.
- Assert the adapter's `send()` signature matches the base contract.
- Assert declared authentication support is OAuth 2.0 PKCE only.

### OAuth Tests

- S256 challenge matches an RFC-compatible known vector.
- Challenge is URL-safe base64 without padding, never a hex digest.
- State mismatch, callback timeout, provider denial, and malformed callbacks fail safely.
- Token exchange and refresh form bodies are correct.
- Scope validation reports missing required scopes.
- Tokens are written atomically with mode `0600` beneath the active temporary `HERMES_HOME`.
- Changing profiles produces isolated token files and identities.

### Ingestion Tests

- First-run baseline with zero backfill dispatches nothing.
- Positive initial backfill is bounded and chronological.
- Mention pagination, `since_id`, cursor persistence, overlap, and deduplication work across adapter restart.
- Direct mention, mention in a third-party reply, nested reply to a bot post, and a new participant continuing the branch all route correctly.
- Same-level sibling posts are context-only unless they summon the bot.
- A new summoned sibling creates a separate participation branch.
- Bot-authored posts and duplicate events do not dispatch.
- DM events route by real conversation ID and replies use the conversation endpoint.
- Receiver task failure notifies the gateway supervisor; intentional disconnect does not.

### Authorization Tests

- Empty allowlist plus false allow-all denies mentions and DMs.
- Allowed IDs pass; non-allowed IDs do not.
- Later branch participants are checked independently.
- Authorization occurs before conversation search, profile lookup, or media download.
- Context from non-triggering or unauthorized participants is labeled unverified and never becomes a synthetic user turn.

### Enrichment Tests

- Conversation search results reconstruct the correct parent-child tree.
- Ancestor precedence, branch mapping, sibling caps, depth caps, and total caps are deterministic.
- Parent-channel/sibling leakage outside the selected conversation is impossible.
- Search rate-limit or access failure falls back to ancestor lookup and still dispatches the trigger.
- Profile fields, metrics, alt text, and media are placed only in untrusted user context.
- Media size, count, MIME, SSRF, timeout, and mixed-media behavior are bounded.

### Send and Queue Tests

- Public reply, top-level post, DM conversation, and invalid/bare route behavior.
- Exact `reply_to` propagation and string-safe IDs above `2**53`.
- Successful image upload and attachment for public posts and DMs.
- Partial media failure does not create a post.
- FIFO ordering per bucket and independence between read and write buckets.
- 429 handling honors fake reset headers without real sleeping.
- Queue overflow and max-wait cancellation do not later deliver the item.
- Ambiguous POST failure is not automatically retried.
- `SendResult.success` is true only after a confirmed API response.

### Temporary `HERMES_HOME` Integration Tests

At least one integration-oriented suite will:

1. create a temporary `HERMES_HOME`;
2. load the real Twitter plugin registration;
3. write fake OAuth credentials using the production persistence path;
4. connect the real adapter to a fake HTTP transport;
5. ingest mention and DM fixtures through its actual pollers;
6. observe normalized gateway events and branch mappings;
7. send replies through the registered adapter;
8. restart under the same temporary home and verify cursor/dedup state;
9. switch to a second temporary profile and prove isolation; and
10. invoke the standalone cron sender with a fresh client.

## Documentation

User documentation will cover:

- creating an X developer app and configuring its loopback redirect URI;
- the exact OAuth 2.0 scopes requested;
- setup through Hermes rather than hand-editing token files;
- `allowed_users` / `allow_all_users` fail-closed behavior;
- public branch semantics and when the bot responds;
- X's automated-reply restrictions;
- DM routing and cron destinations;
- media limitations;
- first-run backfill behavior;
- rate-limit and ambiguous-delivery behavior; and
- the explicit absence of bearer-token and OAuth 1.0a support.

## Acceptance Criteria

The work is complete when:

1. The adapter exists entirely under `plugins/platforms/twitter/` and registers through the current plugin surface.
2. Shared platform interface tests recognize it and its `send()` contract is exact.
3. OAuth 2.0 S256 PKCE, refresh, scope validation, profile-safe storage, and credential locking work without any alternate auth claims.
4. Real mention and DM polling paths normalize and dispatch authorized events correctly.
5. The approved multi-user nested-thread scenario produces branch-safe sessions and correct reply targets.
6. Same-level sibling activity enriches context but does not trigger an unsolicited response.
7. Conversation, profile, metrics, and media enrichment is bounded, untrusted, and best-effort.
8. Bookmarks and metrics are available as gated plugin tools.
9. Outbound image media and typed public/DM routes work.
10. The rate-aware queue preserves ordering, applies backpressure, and avoids duplicate retries after ambiguous writes.
11. Live and standalone cron delivery use safe current-profile clients.
12. Integration tests prove token/state isolation under temporary `HERMES_HOME` values.
13. No test or setup step makes a live X API call or requires real X credentials.
14. The relevant focused test suites pass.
