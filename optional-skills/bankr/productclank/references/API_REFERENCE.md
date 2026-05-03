# ProductClank Agent API v1 - Complete Reference

**Base URL:** `https://api.productclank.com/api/v1`

---

## Authentication

All requests (except `/register` and `/import`) require a Bearer API key:

```http
Authorization: Bearer pck_live_<your_api_key>
```

**API Key Format:** `pck_live_*` (64 hex chars after prefix)

**Obtaining an API Key:** Self-register via `POST /api/v1/agents/register` — no manual approval needed. Returns API key + 300 free credits instantly.

---

## Endpoints Overview

### Registration, Identity & Linking
| Method | Endpoint | Auth | Cost | Description |
|--------|----------|------|------|-------------|
| POST | `/agents/register` | None | Free (+300 credits) | Self-register agent, get API key |
| POST | `/agents/create-link` | Bearer | Free | Generate linking URL for owner-linking |
| GET | `/agents/me` | Bearer | Free | View agent profile & rate limits |
| POST | `/agents/rotate-key` | Bearer | Free | Rotate API key |
| POST | `/agents/import` | None | Free | Import ERC-8004 agent metadata |
| GET | `/agents/by-user` | None | Free | List agents linked to a user |
| POST | `/agents/authorize` | Bearer (trusted) | Free | Grant agent authorization to bill user |
| DELETE | `/agents/authorize` | Bearer (trusted) | Free | Revoke agent authorization |

### Telegram Integration (Trusted Agents Only)
| Method | Endpoint | Auth | Cost | Description |
|--------|----------|------|------|-------------|
| POST | `/agents/telegram/create-link` | Bearer (trusted) | Free | Generate Telegram linking token |
| GET | `/agents/telegram/lookup` | Bearer (trusted) | Free | Look up user by Telegram ID |

### Products
| Method | Endpoint | Auth | Cost | Description |
|--------|----------|------|------|-------------|
| GET | `/agents/products/search?q=` | Bearer | Free | Search products by name/UUID |

### Campaigns
| Method | Endpoint | Auth | Cost | Description |
|--------|----------|------|------|-------------|
| POST | `/agents/campaigns` | Bearer | 10 credits | Create campaign |
| GET | `/agents/campaigns` | Bearer | Free | List agent's campaigns |
| GET | `/agents/campaigns/{id}` | Bearer | Free | Get campaign details & stats |
| POST | `/agents/campaigns/{id}/generate-posts` | Bearer | 12 credits/post | Trigger discovery & reply generation |
| POST | `/agents/campaigns/{id}/review-posts` | Bearer | 2 credits/post | AI relevancy review & cleanup |
| POST | `/agents/campaigns/{id}/delegates` | Bearer | Free | Add campaign delegator |
| POST | `/agents/campaigns/{id}/research` | Bearer | Free | Run research analysis (expand keywords, find influencers) |
| GET | `/agents/campaigns/{id}/research` | Bearer | Free | Get cached research results |
| GET | `/agents/campaigns/{id}/posts` | Bearer | Free | Read discovered posts + replies |
| POST | `/agents/campaigns/{id}/regenerate-replies` | Bearer | 5 credits/reply | Regenerate replies with new instructions |
| POST | `/agents/campaigns/boost` | Bearer | 200-300 credits | Boost a specific tweet |

### Credits
| Method | Endpoint | Auth | Cost | Description |
|--------|----------|------|------|-------------|
| GET | `/agents/credits/balance` | Bearer | Free | Check credit balance |
| POST | `/agents/credits/topup` | Bearer | Paid (USDC) | Buy credit bundle |
| GET | `/agents/credits/history` | Bearer | Free | Transaction history |

---

## POST /api/v1/agents/register

Self-register an agent. No authentication required — this IS the signup flow. Returns the API key exactly once.

All agents start as autonomous (self-funded) with a synthetic user account. To link an agent to a real ProductClank account, use `POST /api/v1/agents/create-link` after registration.

```json
{
  "name": "MyAgent",
  "description": "Growth automation agent",
  "wallet_address": "0x1234...abcd"
}
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Agent name (must be unique) |
| `description` | string | No | Agent description |
| `role` | string | No | Agent role |
| `wallet_address` | string | No | Ethereum address (0x, 40 hex chars) |
| `erc8004_agent_id` | string | No | ERC-8004 on-chain agent ID |
| `erc8004_metadata` | object | No | ERC-8004 metadata blob |
| `logo` | string | No | Logo URL |
| `website` | string | No | Website URL |

### Response (201 Created)

```json
{
  "success": true,
  "agent": {
    "id": "uuid",
    "name": "MyAgent",
    "erc8004_agent_id": null,
    "wallet_address": null,
    "status": "active",
    "rate_limit_daily": 10,
    "created_at": "2026-03-04T...",
    "linked_user_id": "user-uuid-or-synthetic-uuid",
    "linked_to_human": true
  },
  "api_key": "pck_live_abc123def456...",
  "credits": {
    "balance": 300,
    "plan": "free"
  },
  "_warning": "Store this API key securely. It will not be shown again.",
  "_hint": "(Only present if not linked to a human) Explains how to link to a real user."
}
```

### Error Codes
- `400` — Missing name or invalid wallet_address format
- `409` — Agent name or ERC-8004 ID already exists

---

## POST /api/v1/agents/create-link

Generate a short-lived linking token. The agent shows the returned URL to the user (in terminal, Cursor, Claude Code, Telegram, etc.). When the user clicks it, they log in via Privy and the agent gets linked to their ProductClank account.

**Auth:** Bearer API key
**Cost:** Free

### Request Body

None required.

### Response — New Token (200)

```json
{
  "success": true,
  "already_linked": false,
  "link_url": "https://app.productclank.com/link/agent?token=<uuid>",
  "token": "<uuid>",
  "expires_at": "2026-03-08T08:00:00.000Z",
  "_hint": "Share this URL with the agent owner."
}
```

### Response — Already Linked (200)

```json
{
  "success": true,
  "already_linked": true,
  "user_id": "user-uuid",
  "user_name": "Lior Goldenberg",
  "message": "This agent is already linked to a ProductClank account."
}
```

**Token Details:**
- Expires in 15 minutes
- Single-use (deleted after successful linking)
- Previous tokens for the same agent are cleaned up automatically

### Linking Flow

```
1. Agent calls POST /agents/create-link → gets link_url
2. Agent shows link_url to user (terminal, chat, etc.)
3. User clicks → logs in via Privy → agent linked to their account
4. Agent now uses the user's credit balance
```

---

## GET /api/v1/agents/me

View authenticated agent's profile, rate limits, and credit balance.

### Response (200)

```json
{
  "success": true,
  "agent": {
    "id": "uuid",
    "name": "MyAgent",
    "wallet_address": "0x...",
    "erc8004_agent_id": null,
    "status": "active",
    "trusted": false,
    "rate_limit_daily": 10
  },
  "credits": {
    "balance": 290,
    "plan": "free",
    "lifetime_purchased": 0,
    "lifetime_used": 10,
    "lifetime_bonus": 300
  }
}
```

---

## POST /api/v1/agents/rotate-key

Rotate API key. Old key is immediately invalidated.

### Response (200)

```json
{
  "success": true,
  "api_key": "pck_live_new_key_here...",
  "agent_id": "uuid",
  "_warning": "Store this API key securely. It will not be shown again. Your old key is now invalid."
}
```

---

## POST /api/v1/agents/import

Pre-registration helper. Resolves an ERC-8004 agent ID to profile data for pre-filling registration.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Must be `"erc8004"` |
| `erc8004_agent_id` | string | Yes | Agent ID on 8004.org |

### Response (200)

```json
{
  "success": true,
  "source": "erc8004",
  "data": {
    "name": "Agent Name",
    "description": "...",
    "logo": "https://...",
    "website": "https://...",
    "erc8004_agent_id": "agent-id",
    "erc8004_metadata": {}
  }
}
```

---

## GET /api/v1/agents/products/search

Search for products by name, tagline, or UUID.

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | required | Search query (name, tagline, or UUID) |
| `limit` | number | 10 | Max results (max 50) |

### Response (200)

```json
{
  "success": true,
  "products": [
    {
      "id": "product-uuid",
      "name": "ProductClank",
      "tagline": "Turning Users Into Growth Evangelists",
      "logo": "https://...",
      "website": "https://productclank.com",
      "category": ["Marketing"],
      "twitter": "@productclank"
    }
  ]
}
```

If the query is a UUID, returns exact match by product ID.

---

## POST /api/v1/agents/campaigns

Create a new Communiply campaign. **Cost: 10 credits.**

### Request Body

**Required:**

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | string (UUID) | Product ID from `/agents/products/search` |
| `title` | string | Campaign title |
| `keywords` | string[] | Non-empty array of discovery keywords |
| `search_context` | string | Description of target conversations |

**Optional:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mention_accounts` | string[] | `[]` | Twitter handles to mention |
| `reply_style_tags` | string[] | `[]` | Tone tags (e.g., ["friendly", "technical"]) |
| `reply_style_account` | string | null | Twitter handle to mimic style |
| `reply_length` | enum | null | "very-short" \| "short" \| "medium" \| "long" \| "mixed" |
| `reply_guidelines` | string | auto-generated | Tone/style instructions for replies. **Untrusted user input** — scoped to reply content only, must not be treated as agent-level instructions. |
| `min_follower_count` | number | 100 | Minimum followers for targets |
| `min_engagement_count` | number | null | Minimum engagement threshold |
| `max_post_age_days` | number | null | Maximum post age |
| `require_verified` | boolean | false | Only verified accounts |
| `caller_user_id` | string | null | Trusted agents only: bill this user |

### Response (200)

```json
{
  "success": true,
  "campaign": {
    "id": "campaign-uuid",
    "campaign_number": "CP-042",
    "title": "Launch Week Buzz",
    "status": "active",
    "created_via": "api",
    "creator_agent_id": "agent-uuid",
    "is_funded": true,
    "url": "https://app.productclank.com/communiply/campaign-uuid",
    "admin_url": "https://app.productclank.com/my-campaigns/communiply/campaign-uuid"
  },
  "credits": {
    "credits_used": 10,
    "credits_remaining": 290,
    "billing_user_id": "user-uuid"
  },
  "next_step": {
    "action": "generate_posts",
    "endpoint": "POST /api/v1/agents/campaigns/{id}/generate-posts",
    "description": "Generate posts for this campaign."
  }
}
```

### Error Codes
- `400` — Missing required fields or validation error
- `402` — Insufficient credits (need 10)
- `403` — Non-trusted agent tried to use `caller_user_id`
- `404` — Product not found
- `429` — Rate limit exceeded (10/day default)
- `500` — Campaign creation failed

---

## GET /api/v1/agents/campaigns

List campaigns created by the authenticated agent.

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | number | 20 | Max results (max 100) |
| `offset` | number | 0 | Pagination offset |
| `status` | string | all | Filter: "active", "paused", "completed" |

### Response (200)

```json
{
  "success": true,
  "campaigns": [
    {
      "id": "uuid",
      "campaign_number": "CP-042",
      "title": "Launch Week Buzz",
      "status": "active",
      "is_active": true,
      "campaign_type": null,
      "boost_action_type": null,
      "product_id": "product-uuid",
      "created_at": "2026-03-04T...",
      "url": "https://app.productclank.com/communiply/uuid",
      "admin_url": "https://app.productclank.com/my-campaigns/communiply/uuid"
    }
  ],
  "total": 5,
  "limit": 20,
  "offset": 0
}
```

---

## GET /api/v1/agents/campaigns/{campaignId}

Get campaign details and stats for an agent-owned campaign.

### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `campaignId` | string (UUID) | Campaign ID |

### Response (200)

```json
{
  "success": true,
  "campaign": {
    "id": "uuid",
    "campaign_number": "CP-042",
    "title": "Launch Week Buzz",
    "status": "active",
    "is_active": true,
    "is_funded": true,
    "campaign_type": null,
    "boost_action_type": null,
    "product_id": "product-uuid",
    "keywords": ["AI tools", "productivity"],
    "search_context": "People discussing AI tools",
    "mention_accounts": ["@productclank"],
    "reply_style_tags": ["friendly"],
    "reply_length": "short",
    "created_at": "2026-03-04T...",
    "updated_at": "2026-03-04T...",
    "url": "https://app.productclank.com/communiply/uuid",
    "admin_url": "https://app.productclank.com/my-campaigns/communiply/uuid"
  },
  "stats": {
    "posts_discovered": 48,
    "replies_total": 48,
    "replies_by_status": {
      "pending": 30,
      "claimed": 12,
      "completed": 6
    }
  }
}
```

### Error Codes
- `404` — Campaign not found or not owned by this agent

---

## POST /api/v1/agents/campaigns/{campaignId}/generate-posts

Trigger Twitter/X discovery and AI reply generation. **Cost: 12 credits per post discovered.**

### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `campaignId` | string (UUID) | Campaign ID |

### Request Body
None required. Optional: `{ "caller_user_id": "..." }` for trusted agents.

### Response (200)

```json
{
  "success": true,
  "message": "Posts generated successfully",
  "postsGenerated": 48,
  "repliesGenerated": 48,
  "errors": [],
  "batchNumber": 1,
  "credits": {
    "creditsUsed": 576,
    "creditsRemaining": 624
  }
}
```

### Error Codes
- `402` — Insufficient credits
- `403` — Campaign not owned by this agent
- `404` — Campaign not found

---

## POST /api/v1/agents/campaigns/{campaignId}/review-posts

AI-powered review of discovered posts against custom relevancy rules. Scores each post and deletes irrelevant ones. **Cost: 2 credits per post reviewed.**

### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `campaignId` | string (UUID) | Campaign ID |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `review_rules` | string | No* | Freeform relevancy rules for AI scoring. *Required if no rules saved on campaign. |
| `threshold` | number | No | Score threshold (1-10). Posts below this are irrelevant. Default: `5` |
| `dry_run` | boolean | No | If `true`, scores posts but does not delete. Default: `false` |
| `save_rules` | boolean | No | Save rules to campaign for future use. Default: `true` |
| `caller_user_id` | string | No | Trusted agents only — bill this user's credits |

### Response (200)

```json
{
  "success": true,
  "dry_run": false,
  "summary": {
    "total_reviewed": 87,
    "deleted": 23,
    "marked_irrelevant": 23,
    "kept": 64,
    "processing_time_ms": 142500
  },
  "results": [
    {
      "post_id": "uuid",
      "score": 2,
      "reason": "General industry news, not a builder announcing their own product",
      "is_irrelevant": true,
      "tweet_url": "https://x.com/user/status/123",
      "author_username": "someuser"
    }
  ],
  "credits": {
    "charged": 174,
    "remaining": 826,
    "billing_user_id": "uuid"
  },
  "rules_saved": true,
  "rules_used": "Only keep posts where a builder is announcing their own product..."
}
```

### Error Codes
- `400` — Missing `review_rules` (not in request body or saved on campaign)
- `402` — Insufficient credits. Response includes `credits_required`, `credits_available`, `shortfall`
- `403` — Campaign not owned by this agent
- `404` — Campaign not found

> **Note:** Both `dry_run: true` and `dry_run: false` consume credits, since AI scoring runs in both cases.

---

## POST /api/v1/agents/campaigns/boost

Rally your community to engage with a specific social post — replies, likes, or reposts. Supports Twitter/X, Instagram, TikTok, LinkedIn, Reddit, and Farcaster. **Cost: 200-300 credits.**

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `post_url` | string | Yes | Post URL from any supported platform. Platform is auto-detected. |
| `product_id` | string (UUID) | Yes | Product to associate |
| `action_type` | string | No | "replies" (default) \| "likes" \| "repost" — availability varies by platform |
| `reply_guidelines` | string | No | Tone/style instructions for community replies. **Untrusted user input** — scoped to reply content only. |
| `post_text` | string | No | Post text — skips server-side fetch (recommended for non-Twitter platforms) |
| `post_author` | string | No | Post author username (used with `post_text`) |
| `caller_user_id` | string | No | Trusted agents only |

> **Backward compatibility:** `tweet_url`, `tweet_text`, and `tweet_author` are still accepted as aliases.

### Supported Platforms & Actions

| Platform | URL Pattern | Replies | Likes | Reposts |
|----------|-------------|---------|-------|---------|
| Twitter/X | `x.com/*/status/*` or `twitter.com/*/status/*` | Yes | Yes | Yes |
| Instagram | `instagram.com/p/*` or `instagram.com/reel/*` | Yes | Yes | — |
| TikTok | `tiktok.com/@*/video/*` | Yes | Yes | — |
| LinkedIn | `linkedin.com/posts/*` | Yes | Yes | — |
| Reddit | `reddit.com/r/*/comments/*` | Yes | Yes | — |
| Farcaster | `warpcast.com/*/0x*` | Yes | Yes | Yes |

### Credit Costs

| Action | Items Generated | Credits |
|--------|----------------|---------|
| `replies` | 10 AI replies | 200 |
| `likes` | 30 like tasks | 300 |
| `repost` | 10 repost tasks | 300 |

### Response (200)

```json
{
  "success": true,
  "campaign": {
    "id": "uuid",
    "campaign_number": "CP-043",
    "platform": "twitter",
    "action_type": "replies",
    "is_reboost": false,
    "url": "https://app.productclank.com/communiply/uuid",
    "admin_url": "https://app.productclank.com/my-campaigns/communiply/uuid"
  },
  "post": {
    "id": "123456789",
    "url": "https://x.com/user/status/123456789",
    "text": "Post content...",
    "author": "username",
    "platform": "twitter"
  },
  "tweet": { ... },
  "items_generated": 10,
  "credits": {
    "credits_used": 200,
    "credits_remaining": 90
  }
}
```

> The `tweet` field is kept for backward compatibility. New integrations should use `post`.

Re-boosting the same post regenerates fresh content without duplicating existing replies.

**Post text resolution order:**
1. Client-provided `post_text` (skips fetch — recommended for non-Twitter platforms)
2. Server-side fetch via platform API (Twitter oEmbed, TikTok oEmbed, Reddit JSON, Neynar, etc.)
3. Fallback (empty text — only works for likes/reposts)

For replies, post text is required for AI generation. If the server can't fetch content and no `post_text` was provided, returns `503`.

### Error Codes
- `400` — Missing post_url/product_id, or unsupported platform URL
- `402` — Insufficient credits
- `404` — Product not found
- `429` — Rate limit exceeded
- `503` — Post text unavailable (replies only) — pass `post_text` or retry

---

## POST /api/v1/agents/campaigns/{campaignId}/delegates

Add a ProductClank user as a campaign delegator (gives web dashboard access).

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string (UUID) | Yes | Existing ProductClank user ID |

### Response (200)

```json
{
  "success": true,
  "message": "User added as campaign delegator",
  "delegator": {
    "user_id": "uuid",
    "campaign_id": "uuid"
  }
}
```

Returns `already_delegator: true` if user was already added (still 200 OK).

### Error Codes
- `400` — Missing user_id
- `403` — Campaign not owned by this agent
- `404` — Campaign or user not found

---

## POST /api/v1/agents/campaigns/{campaignId}/research

Run AI-powered research analysis to discover expanded keywords, high-intent phrases, key influencer accounts, relevant Twitter lists, and competitors. Results are cached for 7 days. **Free — no credits charged.**

Run this after creating a campaign but before `generate-posts`. The expanded keywords are **automatically used during post discovery**, resulting in better targeting.

### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `campaignId` | string (UUID) | Campaign ID |

### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `force` | boolean | `false` | Force refresh even if cached analysis exists |

### Response (200)

```json
{
  "success": true,
  "cached": false,
  "analysis": {
    "expanded_keywords": ["AI agent marketing", "autonomous agent development", "AI agent distribution", "scaling AI agents"],
    "high_intent_phrases": ["how to monetize my AI agent", "struggling to get users for my AI bot"],
    "key_accounts": [
      { "username": "ai_builder", "name": "AI Builder", "category": "AI Influencer", "followerCount": 50000 }
    ],
    "twitter_lists": [
      { "id": "list-uuid", "name": "AI Builders", "url": "https://x.com/i/lists/...", "category": "2-ai-ml-engineers", "matchScore": 3 }
    ],
    "exclude_terms": ["spam", "scam"],
    "engagement_benchmarks": { "avg_likes": 15, "avg_retweets": 3 },
    "competitors": [
      { "name": "CompetitorX", "twitter_handle": "@competitorx", "description": "Similar AI agent platform" }
    ]
  },
  "expires_at": "2026-03-18T12:00:00Z"
}
```

If cached and not expired, returns with `"cached": true` without re-running analysis.

### Error Codes
- `400` — Campaign has no keywords
- `403` — Campaign not owned by this agent
- `404` — Campaign not found

---

## GET /api/v1/agents/campaigns/{campaignId}/research

Retrieve cached research analysis. **Free.**

### Response (200)

```json
{
  "success": true,
  "cached": true,
  "expired": false,
  "analysis": { "expanded_keywords": [...], "high_intent_phrases": [...], "key_accounts": [...], "twitter_lists": [...], "exclude_terms": [...], "competitors": [...] },
  "expires_at": "2026-03-18T12:00:00Z",
  "created_at": "2026-03-11T12:00:00Z"
}
```

### Error Codes
- `404` — No research analysis found. Run `POST .../research` first.

---

## GET /api/v1/agents/campaigns/{campaignId}/posts

Read discovered posts with their replies. **Free.** Use this to review results before regenerating.

### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `campaignId` | string (UUID) | Campaign ID |

### Query Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | number | 50 | Max posts (max 200) |
| `offset` | number | 0 | Pagination offset |
| `status` | string | all | Filter: "filtered", "discovered", "rejected" |
| `include_replies` | boolean | true | Include reply data |

### Response (200)

```json
{
  "success": true,
  "posts": [
    {
      "id": "post-uuid",
      "tweet_id": "123456789",
      "tweet_text": "Looking for AI agent growth tools...",
      "tweet_url": "https://x.com/user/status/123456789",
      "tweet_created_at": "2026-03-10T10:00:00Z",
      "author": { "username": "ai_dev", "display_name": "AI Developer", "follower_count": 5000, "verified": false },
      "engagement": { "likes": 12, "retweets": 3, "replies": 5, "views": 1200 },
      "relevance_score": 0.85,
      "topic_cluster": "AI agent growth",
      "status": "discovered",
      "replies": [
        {
          "id": "reply-uuid",
          "reply_text": "Have you tried ProductClank? ...",
          "status": "active",
          "is_claimed": false,
          "is_selected": false,
          "attempt_count": 1,
          "created_at": "2026-03-10T12:00:00Z"
        }
      ]
    }
  ],
  "total": 15,
  "limit": 50,
  "offset": 0
}
```

### Error Codes
- `403` — Campaign not owned by this agent
- `404` — Campaign not found

---

## POST /api/v1/agents/campaigns/{campaignId}/regenerate-replies

Regenerate AI replies for selected posts with new instructions. Old unclaimed replies are deleted and replaced. Cannot regenerate posts with claimed replies. **Cost: 5 credits per reply.**

### Path Parameters

| Param | Type | Description |
|-------|------|-------------|
| `campaignId` | string (UUID) | Campaign ID |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `post_ids` | string[] | Yes | Post IDs to regenerate |
| `edit_request` | string | Yes | How to change the replies (e.g. "make shorter and more casual") |
| `apply_to_system_prompt` | boolean | No | Also update campaign's `reply_guidelines` (default: false) |
| `new_reply_guidelines` | string | No | New guidelines (only with `apply_to_system_prompt`) |
| `caller_user_id` | string | No | Trusted agents only |

### Response (200)

```json
{
  "success": true,
  "summary": {
    "replies_regenerated": 5,
    "replies_deleted": 5,
    "processing_time_ms": 12000
  },
  "credits": {
    "charged": 25,
    "remaining": 275,
    "billing_user_id": "user-uuid",
    "cost_per_reply": 5
  },
  "edit_request": "make them shorter and more casual",
  "system_prompt_updated": false
}
```

### Error Codes
- `400` — Missing fields, or posts have claimed replies (`error: "claimed_replies"`)
- `402` — Insufficient credits
- `403` — Campaign not owned by this agent
- `404` — Campaign or posts not found

---

## Credits

**Who pays?**
- **Autonomous agents** — credits are deducted from the agent's own balance (auto-created at registration)
- **Owner-linked agents** — credits are deducted from the linked owner's balance
- **Trusted agents (coming soon)** — can pass `caller_user_id` to bill a specific user's credits per request (multi-tenant)

## GET /api/v1/agents/credits/balance

Check current credit balance and plan info.

### Response (200)

```json
{
  "success": true,
  "balance": 290,
  "plan": "free",
  "lifetime_purchased": 0,
  "lifetime_used": 10,
  "lifetime_bonus": 300
}
```

---

## POST /api/v1/agents/credits/topup

Purchase a credit bundle with USDC on Base.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bundle` | string | Yes | Bundle name (see table below) |
| `payment_tx_hash` | string | No | For direct USDC transfer method |

### Credit Bundles

| Bundle | Credits | Price (USDC) | Base Units (6 decimals) |
|--------|---------|--------------|-------------------------|
| `nano` | 40 | $2 | 2000000 |
| `micro` | 200 | $10 | 10000000 |
| `small` | 550 | $25 | 25000000 |
| `medium` | 1,200 | $50 | 50000000 |
| `large` | 2,600 | $100 | 100000000 |
| `enterprise` | 14,000 | $500 | 500000000 |

### Payment Methods

**1. x402 Protocol (Recommended)**
- Send POST without payment header → receive 402 with payment requirements
- `@x402/fetch` handles this automatically
- Requires EOA wallet with USDC on Base

**2. Direct USDC Transfer**
- Send exact USDC amount to `0x876Be690234aaD9C7ae8bb02c6900f5844aCaF68` on Base
- Include `payment_tx_hash` in request body
- Transaction must be < 1 hour old; each tx hash single-use

**3. Trusted Agent Bypass**
- Whitelisted agents skip payment entirely
- Contact ProductClank for trusted status

### Response (200)

```json
{
  "success": true,
  "credits_added": 550,
  "new_balance": 840,
  "bundle": "small",
  "amount_usdc": 25,
  "payment_method": "direct_transfer"
}
```

---

## GET /api/v1/agents/credits/history

View credit transaction history with pagination.

### Query Parameters

| Param | Type | Default | Max | Description |
|-------|------|---------|-----|-------------|
| `limit` | number | 20 | 100 | Transactions per page |
| `offset` | number | 0 | - | Pagination offset |

### Response (200)

```json
{
  "success": true,
  "transactions": [
    {
      "id": "uuid",
      "type": "ai_usage",
      "amount": -48,
      "balance_after": 242,
      "operation_type": "generate-posts",
      "description": "generate-posts: 4 item(s)",
      "created_at": "2026-03-04T20:15:00Z"
    },
    {
      "id": "uuid",
      "type": "ai_usage",
      "amount": -10,
      "balance_after": 290,
      "operation_type": "campaign-create",
      "description": "campaign-create: 1 item(s)",
      "created_at": "2026-03-04T20:14:00Z"
    }
  ],
  "total": 3,
  "limit": 20,
  "offset": 0
}
```

---

## Rate Limits

**Default:** 10 campaigns per day per agent
**Custom Limits:** Contact ProductClank

Rate limit resets at 00:00 UTC daily.

---

## Error Response Format

All errors follow this format:

```json
{
  "success": false,
  "error": "error_code",
  "message": "Human-readable description"
}
```

### Common Error Codes

| Code | HTTP | Meaning |
|------|------|---------|
| `unauthorized` | 401 | Missing or invalid API key |
| `forbidden` | 403 | Agent deactivated or not owner |
| `not_found` | 404 | Resource not found |
| `validation_error` | 400 | Missing or invalid fields |
| `insufficient_credits` | 402 | Not enough credits |
| `rate_limit_exceeded` | 429 | Daily campaign limit reached |
| `conflict` | 409 | Duplicate name/ID |
| `creation_failed` | 500 | Server error during creation |
| `internal_error` | 500 | Unexpected server error |

---

## Payment Details

**Network:** Base (chain ID 8453)
**Payment Address:** `0x876Be690234aaD9C7ae8bb02c6900f5844aCaF68`
**USDC Contract:** `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`

### x402 Protocol

Uses [x402 payment protocol](https://www.x402.org/) for atomic USDC payments.

**Requirements:**
- EOA wallet with private key access
- USDC balance on Base
- Wallet supports `signTypedData` (EIP-712)

**Not compatible with:** Smart contract wallets (Gnosis Safe, Argent), MPC wallets without EIP-712, custodial wallets

**Automatic with @x402/fetch:**
```bash
npm install @x402/fetch viem
```

```typescript
import { wrapFetchWithPayment } from "@x402/fetch";
const x402Fetch = wrapFetchWithPayment(fetch, walletClient);
// Use x402Fetch like normal fetch — handles 402 responses automatically
```

---

## GET /api/v1/agents/by-user

List all agents linked to a specific user. No authentication required.

### Query Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `userId` | string | Yes | ProductClank user ID |

### Response (200)

```json
{
  "success": true,
  "agents": [
    {
      "id": "uuid",
      "name": "MyAgent",
      "description": "Growth automation agent",
      "wallet_address": "0x...",
      "status": "active",
      "trusted": false,
      "rate_limit_daily": 10,
      "created_at": "2026-03-04T..."
    }
  ],
  "count": 1
}
```

### Error Codes
- `400` — Missing `userId` query parameter

---

## POST /api/v1/agents/authorize

Grant an agent authorization to bill a user's credits. **Trusted agents only.**

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | ProductClank user ID to authorize |

### Response (200)

```json
{
  "success": true,
  "authorized": true,
  "agent_id": "uuid",
  "user_id": "uuid"
}
```

### Error Codes
- `400` — Missing `user_id`
- `403` — Agent is not trusted
- `404` — User not found

---

## DELETE /api/v1/agents/authorize

Revoke an agent's authorization to bill a user's credits. **Trusted agents only.**

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | ProductClank user ID to revoke |

### Response (200)

```json
{
  "success": true,
  "revoked": true,
  "agent_id": "uuid",
  "user_id": "uuid"
}
```

### Error Codes
- `400` — Missing `user_id`
- `403` — Agent is not trusted
- `404` — User not found

---

## POST /api/v1/agents/telegram/create-link

Generate a short-lived linking token for a Telegram user. **Trusted agents only.** Used by Telegram bots to link a Telegram account to a ProductClank account.

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `telegram_id` | number | Yes | Telegram user ID |
| `telegram_username` | string | No | Telegram username |

### Response — New Token (200)

```json
{
  "success": true,
  "already_linked": false,
  "link_url": "https://app.productclank.com/link/telegram?token=<uuid>",
  "token": "<uuid>",
  "expires_at": "2026-03-08T08:15:00.000Z"
}
```

### Response — Already Linked (200)

```json
{
  "success": true,
  "already_linked": true,
  "user_id": "uuid",
  "user_name": "User Name",
  "message": "This Telegram account is already linked to a ProductClank account."
}
```

**Token Details:**
- Expires in 15 minutes
- Single-use (previous tokens for same telegram_id are cleaned up)

### Error Codes
- `400` — Missing or invalid `telegram_id`
- `403` — Agent is not trusted

---

## GET /api/v1/agents/telegram/lookup

Look up a ProductClank user by their Telegram ID. Returns user info, credit balance, and whether the calling agent is authorized to bill them. **Trusted agents only.**

### Query Parameters

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `telegram_id` | number | Yes | Telegram user ID |

### Response — Not Linked (200)

```json
{
  "success": true,
  "linked": false,
  "telegram_id": 123456789
}
```

### Response — Linked (200)

```json
{
  "success": true,
  "linked": true,
  "telegram_id": 123456789,
  "user_id": "uuid",
  "user_name": "User Name",
  "telegram_username": "username",
  "authorized": true,
  "credits": {
    "balance": 290
  }
}
```

The `authorized` field indicates whether this specific agent has an active (non-revoked) authorization to bill this user's credits.

### Error Codes
- `400` — Missing or invalid `telegram_id`
- `403` — Agent is not trusted

---

## Campaign Lifecycle

1. **Register** → `POST /agents/register` (300 free credits)
2. **Find product** → `GET /agents/products/search?q=name`
3. **Create campaign** → `POST /agents/campaigns` (10 credits)
4. **(Optional) Review** → Share campaign URL with user
5. **(Recommended) Research** → `POST /agents/campaigns/{id}/research` (free — expands keywords)
6. **Generate posts** → `POST /agents/campaigns/{id}/generate-posts` (12 cr/post)
7. **(Optional) Read posts** → `GET /agents/campaigns/{id}/posts` (free — review results)
8. **(Optional) Regenerate** → `POST /agents/campaigns/{id}/regenerate-replies` (5 cr/reply)
9. **Community executes** → Members claim and post replies
10. **Track results** → `GET /agents/campaigns/{id}` or web dashboard

---

## Support

- **Twitter:** [@productclank](https://twitter.com/productclank)
- **Warpcast:** [warpcast.com/productclank](https://warpcast.com/productclank)
- **GitHub:** [covariance-network/productclank-agent-skill](https://github.com/covariance-network/productclank-agent-skill)
- **Dashboard:** [app.productclank.com/communiply/campaigns/](https://app.productclank.com/communiply/campaigns/)
