python
    #!/usr/bin/env python3
    """
    Simple CLI for fetching market data (Stocks, Crypto, Forex)
    """
    import sys
    import json
    import argparse
    import time

    try:
        import yfinance as yf
        import requests
    except ImportError:
        print(json.dumps({"error": "Missing dependencies. Run setup.py"}))
        sys.exit(1)

    def get_stock(symbol: str) -> dict:
        """Fetch structured data for a stock/ETF ticker."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            data = {
                "symbol": symbol.upper(),
                "name": info.get("longName"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "currency": info.get("currency"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "change_pct": None
            }

            prev_close = info.get("previousClose")
            if data["price"] and prev_close:
                change = data["price"] - prev_close
                data["change_pct"] = round((change / prev_close) * 100, 2)

            return data
        except Exception as e:
            return {"error": f"Failed to fetch stock {symbol}: {str(e)}"}

    def get_crypto_price(query: str) -> dict:
        """Fetch crypto price from CoinGecko (simple price)."""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            ids = query.lower().replace(" ", "-")
            params = {
                "ids": ids,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    return {"error": f"No crypto found for '{query}'. Try the full name."}
                return data
            return {"error": f"CoinGecko API Error: {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def get_crypto_trending() -> dict:
        """Fetch trending coins on CoinGecko."""
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                coins = data.get("coins", [])
                trending = []
                for item in coins:
                    c = item["item"]
                    trending.append({
                        "name": c.get("name"),
                        "symbol": c.get("symbol"),
                        "rank": c.get("market_cap_rank"),
                    })
                return {"trending": trending}
            return {"error": f"CoinGecko API Error: {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def main():
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        stock_p = subparsers.add_parser("stock")
        stock_p.add_argument("symbol")

        crypto_p = subparsers.add_parser("crypto")
        crypto_p.add_argument("query")

        subparsers.add_parser("trending")

        args = parser.parse_args()

        if args.command == "stock":
            print(json.dumps(get_stock(args.symbol), indent=2))
        elif args.command == "crypto":
            print(json.dumps(get_crypto_price(args.query), indent=2))
        elif args.command == "trending":
            print(json.dumps(get_crypto_trending(), indent=2))
        else:
            parser.print_help()

    if __name__ == "__main__":
        main()
