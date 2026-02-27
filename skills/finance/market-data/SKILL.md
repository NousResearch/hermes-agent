markdown
    ---
    name: market-data
    description: Fetch real-time stock prices (Yahoo Finance) and cryptocurrency data (CoinGecko).
    version: 1.0.0
    author: Nous Research
    license: MIT
    metadata:
      hermes:
        tags: [Finance, Stocks, Crypto, API]
    ---

    # Market Data Skill

    This skill allows Hermes to fetch real-time financial market data.

    ## Usage

    ### 1. Stock / ETF Data
    ```bash
    python ~/.hermes/skills/finance/market-data/scripts/market_data.py stock NVDA
    ```

    ### 2. Cryptocurrency Prices
    ```bash
    python ~/.hermes/skills/finance/market-data/scripts/market_data.py crypto bitcoin
    ```

    ### 3. Trending Cryptos
    ```bash
    python ~/.hermes/skills/finance/market-data/scripts/market_data.py trending
