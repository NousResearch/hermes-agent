# Discovering & Browsing x402 Services

Two complementary ways to find services: the **agentic.market REST API** (structured, filterable, no CLI needed) and the **awal bazaar CLI** (vector search, richer schemas). Use whichever fits the context, or combine them.

---

## agentic.market REST API

No authentication or wallet required.

### List all services

```bash
curl https://api.agentic.market/v1/services
```

Returns a complete directory of registered x402 services with endpoint URLs, methods, prices, and categories.

### Search by name or keyword

```bash
curl 'https://api.agentic.market/v1/services/search?q={query}'
```

Examples:

```bash
curl 'https://api.agentic.market/v1/services/search?q=flight'
curl 'https://api.agentic.market/v1/services/search?q=crypto+price'
curl 'https://api.agentic.market/v1/services/search?q=web+scraping'
```

---

## awal Bazaar CLI

Vector search across the CDP bazaar — finds services by semantic meaning, not just keyword match. Also the only way to inspect per-endpoint payment requirements without making a request.

### Search the bazaar

```bash
npx awal@latest x402 bazaar search <query> [-k <n>] [--network <network>] [--scheme <scheme>] [--max-price <price>] [--json]
```

| Option                | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `-k, --top <n>`       | Number of results, 1–20 (default: 20)                                |
| `--network <name>`    | Filter by chain (base, base-sepolia, polygon, solana, solana-devnet) |
| `--scheme <scheme>`   | Filter by payment scheme: `exact` or `upto`                          |
| `--max-price <price>` | Maximum price in USD (e.g. `0.01`)                                   |
| `--asset <address>`   | Filter by payment asset address                                      |
| `--pay-to <address>`  | Filter by recipient wallet address                                   |
| `--extensions <type>` | Filter by extension type (e.g. `outputSchema`, `bazaar`)             |
| `--json`              | Output as JSON                                                       |

### List all bazaar resources

```bash
npx awal@latest x402 bazaar list [--network <network>] [--full] [--refresh] [--json]
```

| Option             | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `--network <name>` | Filter by chain (base, base-sepolia, polygon, solana, solana-devnet) |
| `--full`           | Show complete details including schemas                              |
| `--refresh`        | Re-fetch resource index from CDP API                                 |
| `--json`           | Output as JSON                                                       |

### Inspect payment requirements

```bash
npx awal@latest x402 details <url> [--json]
```

Auto-detects the correct HTTP method by trying each until it gets a 402 response, then displays price, accepted payment schemes, network, and input/output schemas.

---

## When to use which

| Situation | Use |
|-----------|-----|
| Need to enumerate or filter the full service list | agentic.market REST API |
| Know a category / keyword, want ranked results | Either (try REST first; fall back to bazaar for semantic search) |
| Want to verify an endpoint exists and see its price before paying | `awal x402 details <url>` |
| Need input/output schema for a specific endpoint | `awal x402 details <url>` |
| Searching for something niche (no exact keyword match) | `awal x402 bazaar search` (semantic) |

---

## Known Services Catalog

Use this as a first lookup — if the endpoint you need is listed here, skip discovery and go straight to `pay.md`. All prices are USDC per request. Endpoints marked ✓ have been live-tested.

### Quick-Pick by Use Case

| I need…                         | Provider            | Price         |
| ------------------------------- | ------------------- | ------------- |
| Web search                      | Exa                 | $0.001–$0.007 |
| Web search with LLM answer      | Perplexity          | $0.01–$0.10   |
| Computational / math            | Wolfram Alpha       | $0.01–$0.02   |
| Web scraping                    | Firecrawl           | $0.01         |
| Browser automation              | Browserbase         | $0.002        |
| LLM inference (cheap)           | Venice.ai           | $0.001        |
| Audio transcription             | Deepgram            | $0.01–$1.00   |
| Flight / hotel search + booking | Amadeus             | $0.002–$0.09  |
| Live flight tracking            | FlightAware         | $0.04–$0.12   |
| Restaurant / attraction data    | Tripadvisor         | $0.01         |
| Email / messaging               | AgentMail           | Free          |
| Image generation                | Run402              | $0.03         |
| Blockchain RPC                  | Alchemy / QuickNode | Free–$10      |
| Subgraph queries                | The Graph           | Free          |
| Crypto spot prices (BTC, ETH…)  | CoinMarketCap ✓     | $0.01         |
| Crypto spot prices (cheap)      | Zapper / CoinStats  | $0.001        |
| Onchain token price by address  | CoinGecko           | $0.01         |
| DeFi portfolio / wallet balance | Zapper              | $0.001–$0.002 |
| Wallet analytics / SQL          | Allium              | $0.01–$0.03   |
| Historical price on exchange    | CoinStats           | $0.001        |
| Deep crypto research / AI       | Messari             | $0.10–$0.55   |

### Search & Web

**Exa** — AI-native web search with content retrieval.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://api.exa.ai/search` | POST | $0.007 |
| `https://api.exa.ai/contents` | POST | $0.001 |

```bash
npx awal@latest x402 pay 'https://api.exa.ai/search' -X POST \
  --max-amount 7000 \
  --data '{"query":"latest AI news","numResults":5}'
```

**Perplexity** — Web search with LLM-synthesized answers (Sonar models).

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://pplx.x402.paysponge.com/search` | POST | $0.01 | Basic search |
| `https://pplx.x402.paysponge.com/v1/sonar` | POST | $0.10 | Sonar model |
| `https://pplx.x402.paysponge.com/v1/agent` | POST | $0.01 | Agentic search |

**Wolfram Alpha** — Computational intelligence: math, science, unit conversion, data lookup.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://wolframalpha.x402.paysponge.com/v1/result` | GET | $0.01 | Simple text answer |
| `https://wolframalpha.x402.paysponge.com/v2/query` | GET | $0.02 | Full structured result |

**Firecrawl** (via Heurist) — Web scraping and crawling.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://mesh.heurist.xyz/x402/agents/.../firecrawl_web_search` | POST | $0.01 |
| `https://mesh.heurist.xyz/x402/agents/.../firecrawl_scrape_url` | POST | $0.01 |

Run `npx awal@latest x402 bazaar search "firecrawl"` to resolve the full endpoint URL.

**Browserbase** — Cloud browser sessions for dynamic page rendering.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://x402.browserbase.com/browser/session/create` | POST | $0.002 |

### LLM Inference

**Venice.ai** (Anthropic / OpenAI / DeepSeek) — On-demand LLM calls without managing API keys.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://api.venice.ai/api/v1/chat/completions` | POST | $0.001 | Chat (OpenAI-compatible) |
| `https://api.venice.ai/api/v1/responses` | POST | $0.001 | Responses API |
| `https://api.venice.ai/api/v1/audio/speech` | POST | $0.001 | TTS |
| `https://api.venice.ai/api/v1/audio/transcriptions` | POST | $0.001 | STT |
| `https://api.venice.ai/api/v1/images/generations` | POST | $0.001 | Image gen |

**Parallel** — Search and long-running agentic tasks.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://parallelmpp.dev/api/search` | POST | $0.01 |
| `https://parallelmpp.dev/api/task` | POST | $0.30 |

### Media & Audio

**Deepgram** — Audio transcription, TTS, and text intelligence.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://deepgram.x402.paysponge.com/v1/listen` | POST | $1.00 | Speech-to-text |
| `https://deepgram.x402.paysponge.com/v1/speak` | POST | $0.01 | Text-to-speech |
| `https://deepgram.x402.paysponge.com/v1/read` | POST | $0.01 | Text intelligence |

### Travel

**Amadeus** — Flight and hotel search and booking.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://stabletravel.dev/api/flights/search` | GET/POST | $0.05 | Search flights |
| `https://stabletravel.dev/api/flights/book` | POST | $0.09 | Book a flight |
| `https://stabletravel.dev/api/hotels/search` | GET | $0.03 | Search hotels |
| `https://stabletravel.dev/api/hotels/book` | POST | $0.002 | Book a hotel |

**FlightAware** — Live flight tracking and historical flight data.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://stabletravel.dev/api/flightaware/flights/search` | GET | $0.10 |
| `https://stabletravel.dev/api/flightaware/airports/{id}/flights` | GET | $0.04 |
| `https://stabletravel.dev/api/flightaware/history/flights/{id}/track` | GET | $0.12 |

### Location & Places

**Tripadvisor** — Location details and search for restaurants, hotels, and attractions.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://tripadvisor.x402.paysponge.com/api/v1/location/:locationId/details` | GET | $0.01 |
| `https://tripadvisor.x402.paysponge.com/api/v1/location/search` | GET | $0.01 |

### Social & Communication

**AgentMail** — 90+ email and messaging operations, all free.

```bash
npx awal@latest x402 bazaar search "agentmail"
```

### Compute & Infra

**Run402** — Code execution tiers and image generation.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://api.run402.com/tiers/v1/prototype` | POST | $0.10 |
| `https://api.run402.com/tiers/v1/hobby` | POST | $5.00 |
| `https://api.run402.com/generate-image/v1` | POST | $0.03 |

**Bankr** — Wallet transfers and agent task prompting, all free.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://llm.bankr.bot/v1/chat/completions` | POST | Free |
| `https://api.bankr.bot/wallet/transfer` | POST | Free |
| `https://api.bankr.bot/agent/prompt` | POST | Free |
| `https://api.bankr.bot/token-launches/deploy` | POST | Free |

### Blockchain & Onchain

**Alchemy** — NFT data, token balances, and EVM RPC, all free.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://x402.alchemy.com/{chainNetwork}/v2` | POST | Free |
| `https://x402.alchemy.com/{chainNetwork}/nft/v3/getNFTsForOwner` | GET | Free |
| `https://x402.alchemy.com/data/v1/assets/tokens/by-address` | POST | Free |

**The Graph** — Subgraph and deployment queries, free.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://gateway.thegraph.com/api/x402/subgraphs/id/{id}` | POST | Free |
| `https://gateway.thegraph.com/api/x402/deployments/id/{id}` | POST | Free |

**QuickNode** — 80+ blockchain RPC proxies, $0.00–$10.00 depending on network and call type.

```bash
npx awal@latest x402 bazaar search "quicknode"
```

### Crypto Market Data

**CoinMarketCap ✓** — Crypto quotes, DEX data, and listings.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://pro-api.coinmarketcap.com/x402/v3/cryptocurrency/quotes/latest` | GET | $0.01 | Query: `symbol=BTC,ETH`, `convert=USD` |
| `https://pro-api.coinmarketcap.com/x402/v1/dex/search` | GET | $0.01 | DEX token/pair search |

```bash
npx awal@latest x402 pay 'https://pro-api.coinmarketcap.com/x402/v3/cryptocurrency/quotes/latest' \
  --max-amount 10000 \
  --query '{"symbol":"BTC,ETH","convert":"USD"}'
```

**CoinGecko** — Spot prices, onchain token prices, trending pools.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://pro-api.coingecko.com/api/v3/x402/simple/price` | GET | $0.01 | Query: `ids=bitcoin,ethereum`, `vs_currencies=usd` |
| `https://pro-api.coingecko.com/api/v3/x402/onchain/networks/{network_id}/tokens/{contract_address}` | GET | $0.01 | Replace path params |

**CoinStats** — Historical price at a specific exchange and timestamp.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://x402.coinstats.app/coins/price/exchange` | GET | $0.001 | Query: `exchange`, `from`, `to`, `timestamp` (unix) |

```bash
npx awal@latest x402 pay 'https://x402.coinstats.app/coins/price/exchange' \
  --max-amount 1000 \
  --query '{"exchange":"binance","from":"bitcoin","to":"usd","timestamp":"1700000000"}'
```

**Zapper** — Token prices, portfolio totals, DeFi balances.

| Endpoint | Method | Price |
|----------|--------|-------|
| `https://public.zapper.xyz/x402/token-price` | GET | $0.001125 |
| `https://public.zapper.xyz/x402/portfolio-totals` | GET | $0.001875 |
| `https://public.zapper.xyz/x402/defi-balances` | GET | $0.001125 |

**Allium** — Onchain token prices and wallet analytics with SQL-level access.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://agents.allium.so/api/v1/developer/prices` | POST | $0.02 | Token price lookup |
| `https://agents.allium.so/api/v1/developer/wallet/balances` | POST | $0.03 | Wallet balance |
| `https://agents.allium.so/api/v1/explorer/queries/run-async` | POST | $0.01 | Custom SQL query |

**Messari** — Crypto research, AI chat, asset metrics, and signal intelligence.

| Endpoint | Method | Price | Notes |
|----------|--------|-------|-------|
| `https://api.messari.io/ai/v2/chat/completions` | POST | $0.25 | LLM with crypto context |
| `https://api.messari.io/metrics/v2/assets/details` | GET | $0.10 | Asset fundamentals |
| `https://api.messari.io/signal/v1/assets` | GET | $0.55 | Signal intelligence |

---

## Error Handling

- "CDP API returned 429" — Rate limited on the bazaar CLI; cached data will be used if available
- "No X402 payment requirements found" — URL is not an x402 endpoint; verify via `curl -i <url>`
- Empty results from agentic.market search — try a broader keyword or use `bazaar search` for semantic matching
