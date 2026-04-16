# Hermes msteams Parity Checklist vs OpenClaw

Updated after implementation round 3.

## Scope benchmark
Reference sources:
- `/Users/moatable/.nvm/versions/node/v24.12.0/lib/node_modules/openclaw/docs/channels/msteams.md`
- `/Users/moatable/.nvm/versions/node/v24.12.0/lib/node_modules/openclaw/dist/plugin-sdk/src/config/types.msteams.d.ts`

## Round 1 implemented
- [x] `Platform.MSTEAMS` added to Hermes platform/config registry
- [x] `hermes-msteams` toolset added
- [x] CLI platform registry entry added
- [x] Gateway setup surface includes Microsoft Teams credentials
- [x] Gateway status surface includes Microsoft Teams
- [x] Base native adapter file created: `gateway/platforms/msteams.py`
- [x] Bot Framework webhook ingress skeleton (`/api/messages`, configurable host/port/path)
- [x] Basic inbound message normalization for text activities
- [x] DM/group/channel chat type classification (basic)
- [x] In-memory conversation reference registry
- [x] Outbound text send path via Azure bot credentials / Bot Framework connector token
- [x] Basic platform prompt hint added
- [x] Initial tests for config wiring + adapter normalization/send fallback
- [x] Group mention gating default for non-DM conversations
- [x] Basic reply style support (`thread` default, `top-level` override)

## Not yet at OpenClaw parity
### Inbound
- [x] full mention gating semantics per team/channel override (global + team/channel requireMention implemented; implicit reply-to-bot supported)
- [x] DM policy (`pairing`, `allowlist`, `open`, `disabled`)
- [x] group policy (`allowlist`, `open`, `disabled`)
- [x] allowlist/team/channel policy semantics (ID-first with optional dangerous name matching; route allowlist MVP)
- [x] thread/top-level reply style semantics (global + team/channel overrides, stable reply anchor)
- [x] duplicate suppression / idempotent activity delivery handling (Hermes TTL dedupe by conversation+activity ID)
- [x] per-conversation queueing / serialization (Hermes lock-per-conversation worker model)
- [x] Teams inbound message-per-activity behavior matches current OpenClaw runtime (no Teams-specific burst debounce found in OpenClaw audit)

### Outbound
- [x] real mention entity construction for users
- [x] multi-user mention support
- [x] adaptive cards
- [x] poll sending
- [x] typing indicator behavior
- [x] rich chunking policy parity (`chunkMode`, `textChunkLimit` config)

### Attachments / media
- [x] native media send baseline (inline image send + inline document fallback)
- [x] DM file consent flow (`application/vnd.microsoft.teams.card.file.consent` + pending upload tracking + `fileConsent/invoke` accept/decline handling)
- [x] group/channel SharePoint upload flow (`share_point_site_id` / `MSTEAMS_SHAREPOINT_SITE_ID` + Graph upload + Teams file info card send)
- [x] inbound attachment caching to local files for agent context (`media_urls` / `media_types`)

### Graph-backed enrichments
- [x] history fetch baseline (channel reply thread context injection)
- [x] member-info action parity foundation
- [x] richer user lookup / out-of-conversation mention resolution foundation
- [x] richer inbound Graph fallback media recovery (hosted contents + reference attachments via Graph message lookup)
- [x] channel-list / channel-info action parity foundation
- [x] read action parity foundation
- [x] edit baseline
- [x] delete / pins / search / reactions parity

### Config / runtime
- [x] Teams-specific env vars integrated for policy/routing surfaces (`require_mention`, `reply_style`, `dm_policy`, `group_policy`, allowlists, dangerous name matching, teams JSON)
- [x] additional Teams runtime env vars wired (`text_chunk_limit`, `max_body_bytes`, `idempotency_ttl_seconds`, `auth_cache_ttl_seconds`, `pending_upload_ttl_seconds`, `state_path`, `share_point_site_id`, `media_allow_hosts`, `media_auth_allow_hosts`)
- [x] inbound media allowlist + auth-allowlist enforcement for attachment downloads (`media_allow_hosts`, `media_auth_allow_hosts`)
- [x] cron/send_message platform delivery wiring
- [x] real Bot Framework JWT validation scaffold (OpenID config + JWKS cache + audience/issuer checks + serviceUrl claim verification)
- [x] durable conversation reference persistence for direct-send/home-channel continuity
- [x] explicit Teams conversation-id parsing in `send_message` (`msteams:19:...`)
- [x] local env secret blocklist updates
- [x] user/developer docs pages
- [x] full adapter tests + fixture coverage

## Assessment
Current Hermes implementation now has a **strong policy/routing/auth foundation**, **durable direct-send state**, **per-conversation delivery serialization**, **OpenClaw-aligned chunking controls**, a **usable Graph/media/history baseline**, **real outbound file flows for both DM FileConsentCard and group/channel SharePoint uploads**, **richer inbound Graph fallback media recovery**, and the **core OpenClaw Teams action surface** (`edit`, `delete`, `pins`, `reactions`, `search`, `member-info`, `channel-list`, `channel-info`) implemented in Hermes-native code, and now covers the documented OpenClaw Teams capability surface in Hermes-native code with matching tests and docs. Remaining work is mostly PR cleanup / upstreaming, not missing Teams capability parity.
