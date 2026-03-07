---
name: crypto-price-tracker
description: >
  Track cryptocurrency prices, market data, portfolio values, trending coins,
  Fear & Greed index, price history, and Ethereum gas prices using free public APIs.
  Use this skill when the user asks about crypto prices, market cap, 24h change,
  top coins, portfolio value, market sentiment, trending coins, gas fees, or
  price history. Triggers on queries like "what's the price of bitcoin",
  "check ETH price", "show me top 10 coins", "how is my crypto portfolio doing",
  "what's BTC dominance today", "is the market bullish or bearish",
  "what coins are trending", "show me ETH price last 7 days",
  "what's the ethereum gas price right now".
version: 2.0.0
metadata:
  hermes:
    tags: [crypto, finance, coingecko, prices, market-data, gas, trending, sentiment]
    related_skills: []
---

# Crypto Price Tracker

Use free public APIs (no API key required) to fetch live cryptocurrency data via terminal or web_extract.

---

## Base URLs

| API | Base URL | Rate Limit |
|-----|----------|------------|
| CoinGecko | https://api.coingecko.com/api/v3 | ~30 calls/min |
| Alternative.me | https://api.alternative.me | Unlimited |
| Etherscan Gas | https://api.etherscan.io/api | No key needed for gas |

---

## 1. Get Price of One or More Coins

Endpoint: GET /simple/price

curl -s "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true&include_market_cap=true"
| Parameter | Description | Example |
|-----------|-------------|---------|
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
---

## 2. Top Coins by Market Cap

Endpoint: GET /coins/markets

curl -s "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false"
Key fields returned: id, symbol, name, current_price, market_cap, market_cap_rank, price_change_percentage_24h, total_volume, high_24h, low_24h

---

## 3. Global Market Data & BTC Dominance

curl -s "https://api.coingecko.com/api/v3/global"
```python
import sys, json
data = json.load(sys.stdin)['data']
print(f'Total Market Cap: USD {data["total_market_cap"]["usd"]:,.0f}')
print(f'BTC Dominance: {data["market_cap_percentage"]["btc"]:.1f}%')
print(f'ETH Dominance: {data["market_cap_percentage"]["eth"]:.1f}%')
print(f'24h Volume: USD {data["total_volume"]["usd"]:,.0f}')

---

## 4. Portfolio Value Calculator

python
import requests

holdings = {"bitcoin": 0.5, "ethereum": 3.2, "solana": 50}

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

---

## 5. Fear & Greed Index (Market Sentiment)

bash
curl -s "https://api.alternative.me/fng/?limit=1"


python
import requests, json

data = requests.get("https://api.alternative.me/fng/?limit=1").json()
index = data["data"][0]
value = int(index["value"])
label = index["value_classification"]

print(f"Fear & Greed Index: {value}/100 — {label}")
if value <= 25:
    print("🔴 Extreme Fear — possible buying opportunity")
elif value <= 45:
    print("🟠 Fear — market is nervous")
elif value <= 55:
    print("🟡 Neutral — balanced sentiment")
elif value <= 75:
    print("🟢 Greed — market is optimistic")
else:
    print("🚀 Extreme Greed — caution advised")

Example response:
Fear & Greed Index: 72/100 — Greed
🟢 Greed — market is optimistic

---

## 6. Trending Coins (Last 24h Most Searched)

bash
curl -s "https://api.coingecko.com/api/v3/search/trending"


python
import requests

data = requests.get("https://api.coingecko.com/api/v3/search/trending").json()

print("=== Trending Coins (Most Searched) ===")
for i, item in enumerate(data["coins"][:7], 1):
    coin = item["item"]
    print(f"{i}. {coin['name']} ({coin['symbol'].upper()}) — Rank #{coin['market_cap_rank']}")

---

## 7. Price History (7 or 30 Days)

bash
# 7-day hourly data
curl -s "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=7"

# 30-day daily data
curl -s "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30&interval=daily"


python
import requests

def get_price_history(coin_id, days=7):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    data = requests.get(url).json()
    prices = data["prices"]

    print(f"=== {coin_id.upper()} Price History ({days} days) ===")
    for timestamp, price in prices:
        from datetime import datetime
        date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
        print(f"{date}: ${price:,.2f}")

    first_price = prices[0][1]
    last_price = prices[-1][1]
    change = ((last_price - first_price) / first_price) * 100
    print(f"\n{days}d Change: {change:+.2f}%")

get_price_history("ethereum", 7)

---

## 8. Ethereum Gas Tracker

bash
curl -s "https://api.etherscan.io/api?module=gastracker&action=gasoracle"


python
import requests

data = requests.get("https://api.etherscan.io/api?module=gastracker&action=gasoracle").json()
result = data["result"]

print("=== Ethereum Gas Prices ===")
print(f"🟢 Low (slow):    {result['SafeGasPrice']} Gwei")
print(f"🟡 Average:       {result['ProposeGasPrice']} Gwei")
print(f"🔴 Fast:          {result['FastGasPrice']} Gwei")
print(f"\nEstimated costs (ETH transfer ~21,000 gas):")

for speed, gwei_key in [("Slow", "SafeGasPrice"), ("Average", "ProposeGasPrice"), ("Fast", "FastGasPrice")]:
    gwei = int(result[gwei_key])
    cost_eth = (gwei * 21000) / 1e9
    print(f"  {speed}: {cost_eth:.6f} ETH")
`

---

## Common Coin IDs

| Ticker | CoinGecko ID |
|--------|-------------|
| BTC | bitcoin |
| ETH | ethereum |
| BNB | binancecoin |
| SOL | solana |
| XRP | ripple |
| ADA | cardano |
| DOGE | dogecoin |
| AVAX | avalanche-2 |
| MATIC | matic-network |
| SHIB | shiba-inu |
| DOT | polkadot |
| LINK | chainlink |
| UNI | uniswap |
| LTC | litecoin |

> For unknown coins, search by ticker first:
> `curl -s "https://api.coingecko.com/api/v3/search?query=TICKER"`

---

## Rate Limits

- Free tier: ~30 calls/minute, no API key needed
- If rate limited (429): wait 60 seconds and retry
- For heavy usage, consider CoinGecko Demo API (free key)

---

## Example Interactions

User: "What's the price of Bitcoin?"
→ Use endpoint 1 with `ids=bitcoin`, report price and 24h change.

User: "Show me top 5 cryptocurrencies"
→ Use endpoint 2 with `per_page=5`, display ranked table.

User: "What's BTC dominance?"
→ Use endpoint 3, show global data and display totals.

User: "I have 2 ETH and 0.1 BTC, what is my portfolio worth?"
→ Use endpoint 1 with both coin IDs, calculate and display total.

User: "Is the market bullish or bearish right now?"
→ Use endpoint 5 (Fear & Greed Index), explain sentiment.

User: "What coins are trending today?"
→ Use endpoint 6, list top 7 trending coins.

User: "Show me ETH price for the last 7 days"
→ Use endpoint 7 with `coin_id=ethereum, days=7`.
User: "What's the Ethereum gas price?"
→ Use endpoint 8, show slow/average/fast gas prices.
