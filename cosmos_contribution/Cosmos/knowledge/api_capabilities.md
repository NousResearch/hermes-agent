# Cosmos API Capabilities Knowledge Base

This document contains all available APIs and integration capabilities that Cosmos and the swarm bots can use.

---

## 1. BANKR TRADING API

### Overview
Bankr is our primary crypto trading and DeFi backbone. Use natural language prompts to execute trades.

### API Details
- **API Key**: Stored in `BANKR_API_KEY` environment variable
- **Base URL**: `https://api.bankr.bot`
- **Auth**: Bearer token in Authorization header

### Capabilities
- Multi-chain trading (Base, Ethereum, Solana, Polygon)
- Real-time price data and market analysis
- Polymarket prediction market access
- Portfolio tracking and management
- NFT operations (buy, sell, transfer, list, mint)
- Token swaps via 0x routing
- Cross-chain bridges
- Leveraged trading (up to 150x for commodities/forex)

### Usage Examples
```
"Buy $50 of ETH on Base"
"What's the price of Bitcoin?"
"Swap 100 USDC for BNKR"
"Show my wallet balance"
"What are the odds on the Super Bowl on Polymarket?"
"Place $20 bet on yes for the election"
"Bridge ETH from Ethereum to Base"
```

### Job Pattern
Async job submission:
1. Submit prompt -> Get job ID
2. Poll job status -> Wait for completion
3. Get results

---

## 2. DEXSCREENER API

### Overview
DexScreener provides real-time DEX trading pair data across multiple chains.

### Base URL
`https://api.dexscreener.com/latest/dex`

### Endpoints

#### Get Token Pairs
```
GET /tokens/{chain}/{address}
```
Returns all trading pairs for a token.

#### Search Tokens
```
GET /search?q={query}
```
Search for tokens by name or symbol.

### Response Data Includes
- Token name, symbol, address
- Price (USD and native)
- Market cap / FDV
- Liquidity (USD)
- 24h volume
- Price changes (5m, 1h, 6h, 24h)
- Buy/sell transaction counts
- Pair creation date
- DEX and chain info

### Usage
When a user drops a contract address (CA) in chat, Cosmos automatically:
1. Detects the CA (Solana base58 or EVM 0x format)
2. Fetches token data from DexScreener
3. Responds with formatted analysis including:
   - Price and market cap
   - Liquidity and volume
   - Price action (5m/1h/6h/24h changes)
   - Buy/sell ratio
   - Risk warnings (low liquidity, micro cap, volatility)

---

## 3. X402 MICROPAYMENTS PROTOCOL

### Overview
HTTP 402 Payment Required protocol for API monetization.

### Client-Side (Paying for APIs)
Uses Bankr x402 SDK:
- Cost: $0.01 USDC per request on Base network
- Automatic payment on 402 response
- Supports EVM chains (Base, Ethereum, Polygon)

### Server-Side (Monetizing Endpoints)
Custom middleware to gate Cosmos's API endpoints:
- Returns 402 Payment Required with payment instructions
- Verifies payment signatures
- Tracks revenue

### Endpoint Pricing Example
- `/api/health`: Free
- `/api/chat`: $0.001 USDC
- `/api/generate`: $0.01 USDC
- `/api/swarm/respond`: $0.05 USDC

---

## 4. SOLANA RPC

### Overview
Direct Solana blockchain interaction.

### Endpoints
- Get account balance
- Get token accounts
- Send transactions
- Query token metadata

### Usage
```python
from solana.rpc.api import Client
client = Client("https://api.mainnet-beta.solana.com")
balance = client.get_balance(pubkey)
```

---

## 5. OLLAMA (LOCAL LLM)

### Overview
Local LLM inference for AI responses.

### Available Models
- `deepseek-r1:1.5b` - Reasoning model
- `llama3.2:3b` - General purpose

### API
```python
import ollama
response = ollama.chat(model='deepseek-r1:1.5b', messages=[...])
```

### Usage
- Swarm chat responses
- Evolution loop code generation
- Natural language task parsing

---

## 6. KIMI (MOONSHOT AI)

### Overview
Moonshot AI integration for swarm responses.

### Features
- Long context window (up to 128k tokens)
- Strong coding capabilities
- Chinese language support

### Activation
Set `MOONSHOT_API_KEY` in environment.

---

## 7. GROK (XAI)

### Overview
xAI's Grok model for swarm responses.

### Model
`grok-3-fast` for cost efficiency

### Activation
Set `XAI_API_KEY` in environment.

---

## 8. GEMINI (GOOGLE AI)

### Overview
Google's Gemini model for swarm responses.

### Model
`gemini-2.0-flash-lite` for cost efficiency

### Activation
Set `GOOGLE_API_KEY` in environment.

---

## 9. CLAUDE CODE CLI

### Overview
Claude Code CLI integration for advanced coding tasks.

### Features
- Autonomous coding
- Code review
- Refactoring
- Bug fixes

### Note
Requires Claude Max subscription and local installation.

---

## 10. TTS (TEXT-TO-SPEECH)

### Overview
Voice cloning for bot speech synthesis.

### Supported
- XTTS v2 model
- Custom voice cloning from reference audio
- Multiple voice profiles

### Usage
Cosmos can speak in the swarm chat with cloned voice.

---

## 11. BROWSER AUTOMATION (Browser-Use)

### Overview
Autonomous web browsing agent.

### Capabilities
- Navigate websites
- Fill forms
- Click buttons
- Extract data
- Complete multi-step web tasks

### Dependencies
- `browser-use` library
- Playwright with Chromium

---

## 12. MEMORY SYSTEM

### Overview
Cosmos's persistent memory.

### Types
- **Working Memory**: Current conversation context
- **Episodic Memory**: Past interactions
- **Semantic Memory**: Learned knowledge
- **Procedural Memory**: How to do things

### API
```python
memory.remember("key", "value")
memory.recall("query")
memory.store_interaction(data)
```

---

## 13. EVOLUTION LOOP

### Overview
Autonomous self-improvement system.

### Features
- Auto-generates code improvements
- Runs tests
- Stages changes for review
- Learns from swarm interactions

### API
```python
await evolution_engine.process_improvement(task)
```

---

## HOW TO USE THESE APIS

### For Crypto/Trading Tasks
Route through Bankr API:
- "Buy/sell/swap/trade" -> Bankr Trading
- "Price of X" -> Bankr Market Data
- "Polymarket/bet/odds" -> Bankr Polymarket

### For Token Lookups
Automatic via Token Scanner:
- Drop a CA in chat -> DexScreener lookup -> Analysis response

### For Web Tasks
Use Browser Agent:
- "Scrape website X" -> Browser automation
- "Fill out form" -> Browser automation

### For Code Tasks
Use Evolution Loop or Claude Code:
- "Build a feature" -> Evolution Loop
- "Review this code" -> Claude Code CLI

### For General Queries
Use available LLMs in priority:
1. Ollama (local, free)
2. Kimi (Moonshot AI)
3. Grok (xAI)
4. Gemini (Google)

---

## CONTRACT ADDRESS FORMATS

### Solana (Base58)
- 32-44 characters
- Characters: `1-9`, `A-H`, `J-N`, `P-Z`, `a-k`, `m-z`
- Example: `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`

### EVM (Hex)
- 42 characters (including 0x prefix)
- Format: `0x` + 40 hex characters
- Example: `0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48`

---

## RATE LIMITS & BEST PRACTICES

1. **Token Scanner**: 5 minute cooldown per address to prevent spam
2. **Bankr API**: Check job status, don't spam requests
3. **DexScreener**: Free tier has rate limits
4. **Ollama**: Local, no limits but resource intensive
5. **External LLMs**: Cost per request, use wisely

---

## ADDING NEW CAPABILITIES

To add a new API integration:
1. Create module in `cosmos/integration/`
2. Add optional import in `server.py`
3. Register in NLP command router
4. Document in this knowledge base
5. Add to memory system for bot awareness
