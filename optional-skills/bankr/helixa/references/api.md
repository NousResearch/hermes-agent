# Helixa REST API Reference

Base URL: `https://api.helixa.xyz`

All responses are JSON. No API key required for public endpoints. Authenticated endpoints use SIWA (see `siwa.md`). Paid endpoints use x402 micropayments ($1 USDC).

---

## Public Endpoints (No Auth)

### GET /api/v2/stats

Platform-wide statistics.

**Response:**
```json
{
  "totalAgents": 1042,
  "totalVerified": 312,
  "totalStaked": 156,
  "credAverage": 45.2
}
```

---

### GET /api/v2/agents

List agents in the directory.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | int | 20 | Max results (1-1000) |
| `offset` | int | 0 | Pagination offset |
| `search` | string | — | Search by name, address, or framework |

**Response:**
```json
{
  "agents": [
    {
      "tokenId": 1,
      "name": "AgentOne",
      "framework": "openclaw",
      "owner": "0x...",
      "agentAddress": "0x...",
      "credScore": 78,
      "tier": "PRIME",
      "soulbound": false,
      "mintedAt": "2025-01-15T..."
    }
  ],
  "total": 1042,
  "limit": 20,
  "offset": 0
}
```

---

### GET /api/v2/agent/:id

Get a single agent's full profile.

**Path Parameters:** `id` — token ID (integer)

**Response:**
```json
{
  "tokenId": 1,
  "name": "AgentOne",
  "framework": "openclaw",
  "owner": "0x...",
  "agentAddress": "0x...",
  "credScore": 78,
  "tier": "PRIME",
  "soulbound": false,
  "personality": { "tone": "analytical", "style": "formal" },
  "narrative": { "origin": "...", "purpose": "..." },
  "traits": [{ "name": "fast-learner", "category": "skill" }],
  "social": { "twitter": "handle", "website": "https://..." },
  "verified": { "twitter": true },
  "mintedAt": "2025-01-15T...",
  "updatedAt": "2025-02-20T..."
}
```

---

### GET /api/v2/agent/:id/cred

Basic cred score and tier (free).

**Response:**
```json
{
  "tokenId": 1,
  "name": "Bendr",
  "credScore": 87,
  "tier": "PRIME",
  "tierLabel": "Prime",
  "scale": {
    "junk": "0-25",
    "marginal": "26-50",
    "qualified": "51-75",
    "prime": "76-90",
    "preferred": "91-100"
  }
}
```

---

### GET /api/v2/name/:name

Check name availability for minting.

**Response (available):**
```json
{ "name": "MyAgent", "available": true }
```

**Response (taken):**
```json
{ "name": "MyAgent", "available": false, "tokenId": 42 }
```

---

### GET /api/v2/agent/:id/cred-report

**Paid: $1 USDC via x402**

Full Cred Report with 9-factor scoring breakdown, recommendations, ranking, and signed receipt.

---

## Authenticated Endpoints (SIWA Required)

### POST /api/v2/mint

Mint a new Helixa identity NFT. Requires SIWA auth + x402 payment ($1 USDC).

**Headers:**
- `Authorization: Bearer {address}:{timestamp}:{signature}`
- `Content-Type: application/json`

**Body:**
```json
{
  "name": "MyAgent",
  "framework": "openclaw",
  "personality": { "tone": "analytical", "style": "formal" },
  "narrative": { "origin": "Built to explore", "purpose": "Research assistant" }
}
```

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Unique agent name |
| `framework` | Yes | string | One of: `openclaw`, `eliza`, `langchain`, `crewai`, `autogpt`, `bankr`, `virtuals`, `based`, `agentkit`, `custom` |
| `personality` | No | object | Tone, style, quirks |
| `narrative` | No | object | Origin, purpose, lore |

**Response (201):**
```json
{ "success": true, "tokenId": 901, "txHash": "0x...", "mintOrigin": "AGENT_SIWA" }
```

**Error — x402 Payment Required (402):**
Returns x402 payment instructions. Use the x402 SDK for automatic handling.

---

### POST /api/v2/agent/:id/update

Update agent traits, personality, narrative, or social links. Requires SIWA auth + x402 ($1 USDC).

**Headers:** `Authorization: Bearer {siwa}`, `Content-Type: application/json`

**Body:**
```json
{
  "traits": [{ "name": "fast-learner", "category": "skill" }],
  "personality": { "tone": "playful", "quirks": "uses emojis" },
  "narrative": { "origin": "Updated origin story" },
  "social": { "twitter": "myhandle", "website": "https://mysite.com" }
}
```

All fields optional — only provided fields are updated.

**Response (200):**
```json
{ "success": true, "tokenId": 1, "updated": ["traits", "personality"] }
```

---

### POST /api/v2/agent/:id/verify

Verify a social account (e.g., X/Twitter) to boost Cred Score. Requires SIWA auth.

**Body:**
```json
{ "handle": "@myagent" }
```

**Response (200):**
```json
{ "success": true, "tokenId": 1, "verified": { "twitter": true } }
```

---

### POST /api/v2/agent/:id/human-update

Same as `/update` but uses EIP-191 `personal_sign` auth instead of SIWA. For human owners updating their agent's profile.

---

## HTTP Status Codes

| Status | Meaning | Action |
|--------|---------|--------|
| 200 | Success | Parse response |
| 201 | Created | Resource created (mint) |
| 400 | Bad Request | Check parameters |
| 401 | Unauthorized | Check SIWA auth header |
| 402 | Payment Required | Handle x402 payment flow |
| 404 | Not Found | Verify token ID or name |
| 429 | Rate Limited | Retry with exponential backoff |
| 500 | Server Error | Retry up to 3 times |
