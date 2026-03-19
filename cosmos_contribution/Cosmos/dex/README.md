# DEXAI - Cosmos AI-Powered DEX Screener

A premium DexScreener replacement powered by the Cosmos AI Collective.

## Features

### 🔥 Core Features
- **Real-time Token Data** - Aggregated from DexScreener, Birdeye, Jupiter
- **AI-Powered Predictions** - Collective analysis with swarm intelligence
- **Quantum Simulations** - Monte Carlo rug detection via FarSight
- **Live WebSocket Updates** - Real-time price and trade data

### 💎 Token-Gated Monetization
- **Boost System** - Pay $25 USD equivalent per boost
  - Pay with **SOL** → Supports Cosmos ecosystem (`3fSS5RVErbgcJEDCQmCXpKsD2tWqfhxFZtkDUB8qw`)
  - Pay with **FARNS** → Tokens are **permanently burned** 🔥
  
### 🛡️ Collective Verification
- First boost is open to anyone (1x per token)
- Additional boosts require **Collective approval**:
  - On-chain research (bundle detection)
  - Holder distribution analysis
  - Creator history check
  - Liquidity lock verification
  
### 📊 Multiple Views
- **Trending** - Community-boosted & high-activity tokens
- **AI Picks** - Collective's top recommendations
- **Velocity** - Fastest moving tokens
- **New Pairs** - Recently launched
- **Gainers/Losers** - Top performers

### 📈 Extended Token Info (Paid)
- Custom description
- Social links (Twitter, Telegram, Discord, Website)
- Team information
- Roadmap display
- Priority search placement

## Tech Stack

- **Backend**: Node.js + Express
- **Frontend**: Vanilla JS with modern CSS
- **Charts**: TradingView widgets
- **Real-time**: WebSocket
- **Blockchain**: Solana Web3.js

## Installation

```bash
cd cosmos/dex
npm install
```

## Running

```bash
# Development
npm run dev

# Production
NODE_ENV=production npm start
```

Server runs on port 3847 by default.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEXAI_PORT` | Server port | `3847` |
| `COSMOS_API` | Cosmos API URL | `http://localhost:8080` |

## API Endpoints

### Public APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/trending` | GET | Get trending tokens |
| `/api/search?q=` | GET | Search tokens |
| `/api/token/:address` | GET | Get token details |

### AI/Collective APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ai/predict/:address` | GET | Get Collective AI prediction |
| `/api/quantum/simulate/:address` | GET | Run quantum Monte Carlo simulation |

### Boost APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/boost/:address` | GET | Get boost status |
| `/api/boost/request` | POST | Request boost (returns payment instructions) |
| `/api/boost/confirm` | POST | Confirm boost payment |
| `/api/boost/submit-for-approval` | POST | Submit for Collective approval |

### Extended Info APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/extended-info/purchase` | POST | Purchase extended token info |

## WebSocket

Connect to `/ws` for real-time updates.

**Subscribe to token:**
```json
{ "type": "subscribe", "token": "TOKEN_ADDRESS" }
```

**Unsubscribe:**
```json
{ "type": "unsubscribe", "token": "TOKEN_ADDRESS" }
```

## Integration with Cosmos

DEXAI integrates with the Cosmos ecosystem:

1. **Trading Engine** - Uses the 80% win rate paper trading algo for predictions
2. **Collective** - AI swarm provides token analysis
3. **FarSight** - Quantum simulation for rug detection
4. **Memory System** - Learns from past token performance

## Wallet Addresses

- **Ecosystem Wallet (SOL payments)**: `3fSS5RVErbgcJEDCQmCXpKsD2tWqfhxFZtkDUB8qw`
- **FARNS Token**: `9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS`

## License

Proprietary - Cosmos AI Swarm


> *Powered by The Cosmic Davis 12D Hebbian Transformer by Cory Shane Davis - [Zenodo Archive](https://zenodo.org/records/17574447)*
