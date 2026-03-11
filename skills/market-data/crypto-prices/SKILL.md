---
name: crypto-prices
description: Fetch real-time crypto prices (price, 24h change, market cap, volume) for top N coins or specific tickers from CoinGecko. Renders a colour-coded table in the terminal. No API key needed.
triggers:
  - "crypto prices"
  - "bitcoin price"
  - "eth price"
  - "top 10 coins"
  - "coin market cap"
  - "BTC ETH SOL price"
  - "real-time crypto"
  - "coingecko"
dependencies:
  - python3 (stdlib only — urllib, json, argparse)
---

## Overview

Uses the CoinGecko free public API (no key required) to fetch live market
data and render it as a clean aligned table with ANSI colour coding:
  - Green / red for positive / negative 24h change
  - Up/down arrows next to % change
  - Auto-scaled price format ($1.37T, $242B, $1.0004, etc.)

## Usage

Run the script directly from its saved path:

  python3 ~/.hermes/skills/market-data/crypto-prices/scripts/crypto_prices.py

With options:
  --top N          Show top N coins by market cap (default: 10)
  --coins TICKERS  Specific tickers, comma-separated (e.g. btc,eth,sol,doge)

## Steps

1. Load the script path: ~/.hermes/skills/market-data/crypto-prices/scripts/crypto_prices.py

2. To show the top 10 coins:
   python3 <script_path> --top 10

3. To show specific coins:
   python3 <script_path> --coins btc,eth,sol

4. To show a wider list:
   python3 <script_path> --top 25

5. To run it inline from the agent, use terminal():
   terminal(command="python3 ~/.hermes/skills/market-data/crypto-prices/scripts/crypto_prices.py --top 10")

## Supported Tickers (--coins flag)

btc, eth, sol, bnb, xrp, usdt, usdc, ada, avax, doge,
dot, matic, link, ltc, uni, atom, xlm, near

For any other coin, pass the CoinGecko coin ID directly
(e.g. --coins pepe,bonk,wif).

## Sending the Image to Telegram

Do NOT use send_message() for images — it only sends text and will print
the file path literally. Use the Telegram Bot API sendPhoto endpoint with
requests and a multipart file upload. Read the bot token and chat_id from
the TELEGRAM_BOT_TOKEN and TELEGRAM_HOME_CHANNEL env vars. See the
scripts/telegram_send_photo.py helper in this skill.

## API Details

Endpoint: https://api.coingecko.com/api/v3/coins/markets
  vs_currency=usd, order=market_cap_desc, price_change_percentage=24h

Rate limit: ~30 req/min on the free tier. If rate-limited (HTTP 429),
wait 60 seconds before retrying.

## Output Columns

  # | Name | Symbol | Price (USD) | 24h Chg | Market Cap | 24h Volume

## Pitfalls

- Python 3.10+ type union syntax `list[str] | None` breaks on 3.9. Use
  plain `= None` for optional args (already done in the script).
- CoinGecko free API occasionally returns 429; back off 60s.
- Stablecoins appear in the top 10 by market cap (e.g. USDT, USDC) — use
  --coins to filter to specific assets if desired.
