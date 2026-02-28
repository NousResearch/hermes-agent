---
name: crypto-price-tracker
description: Track cryptocurrency prices, market data, and portfolio values using the CoinGecko API. Use this skill when the user asks about crypto prices, market cap, 24h change, top coins, or wants to check the value of their holdings. Triggers on queries like "what's the price of bitcoin", "check ETH price", "show me top 10 coins", "how is my crypto portfolio doing", "what's BTC dominance today".
version: 1.0.0
metadata:
  hermes:
    tags: [crypto, finance, coingecko, prices, market-data]
    related_skills: []
---

# Crypto Price Tracker

Use the CoinGecko public API (no API key required) to fetch live cryptocurrency data via terminal or web_extract.

## Base URL
https://api.coingecko.com/api/v3
---

## 1. Get Price of One or More Coins

Endpoint: GET /simple/price

`bash
curl -s "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true&include_market_cap=true"
Parameters:
| Parameter | Description | Example |
|---|---|---|
| ids | Coin ID(s), comma-separated | bitcoin,ethereum |
| vs_currencies | Target currency | usd, eur, btc |
| include_24hr_change | Include 24h % change | true |
| include_market_cap | Include market cap | true |
| include_24hr_vol | Include 24h volume | true |
Example response:
{
  "bitcoin": {
    "usd": 67420.0,
    "usd_24h_change": 2.35,
    "usd_market_cap": 1327000000000
  }
}
Endpoint: GET /coins/markets
curl -s "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false"
Key fields returned: id, symbol, name, current_price, market_cap, market_cap_rank, price_change_percentage_24h, total_volume, high_24h, low_24h
curl -s "https://api.coingecko.com/api
curl -s "https://api.coingecko.com/api/v3/global" | python3 -c "
import sys, json
data = json.load(sys.stdin)['data']
print(f'Total Market Cap: USD {data[\"total_market_cap\"][\"usd\"]:,.0f}')
print(f'BTC Dominance: {data[\"market_cap_percentage\"][\"btc\"]:.1f}%')
print(f'ETH Dominance: {data[\"market_cap_percentage\"][\"eth\"]:.1f}%')
"
import requests

holdings = {
    "bitcoin": 0.5,
    "ethereum": 3.2,
    "solana": 50
}

ids = ",".join(holdings.keys())
url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true"
data = requests.get(url).json()

total = 0
print("=== Portfolio Summary ===")
for coin, amount in holdings.items():
    price = data[coin]["usd"]
    change = data[coin].get("usd_24h_change", 0)
    value = price * amount
    total += value
    print(f"{coin.upper()}: {amount} x ${price:,.2f} = ${value:,.2f} ({change:+.2f}% 24h)")

print(f"Total Value: ${total:,.2f}")
Common Coin IDs
Ticker
CoinGecko ID
BTC
bitcoin
ETH
ethereum
BNB
binancecoin
SOL
solana
XRP
ripple
ADA
cardano
DOGE
dogecoin
AVAX
avalanche-2
MATIC
matic-network
SHIB
shiba-inu
Rate Limits
Free tier: ~30 calls/minute, no API key needed
If rate limited (429): wait 60 seconds and retry
For unknown coins, search by ticker first using endpoint 3
Show 24h change: positive = up, negative = down
Example Interactions
User: "What's the price of Bitcoin?"
Use endpoint 1 with ids=bitcoin, report price and 24h change.
User: "Show me top 5 cryptocurrencies"
Use endpoint 2 with per_page=5, display ranked table.
User: "What's BTC dominance?"
Use endpoint 4 global data.
User: "I have 2 ETH and 0.1 BTC, what is my portfolio worth?"
Use endpoint 1 with both coin IDs, calculate and display totals.
