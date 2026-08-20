# Python SDK — QuoteContext

`QuoteContext` (sync) / `AsyncQuoteContext` (async) — market data, subscriptions, watchlist.

## Creation

```python
# Sync
ctx = QuoteContext(config)

# Async — use classmethod, NOT constructor
ctx = AsyncQuoteContext.create(config)
# or with explicit loop for async callbacks:
ctx = AsyncQuoteContext.create(config, loop_=asyncio.get_running_loop())
```

## Push Subscriptions

### Set callbacks (both sync and async contexts)

```python
ctx.set_on_quote(lambda symbol, event: print(symbol, event))
ctx.set_on_depth(lambda symbol, event: print(symbol, event))
ctx.set_on_brokers(lambda symbol, event: print(symbol, event))
ctx.set_on_trades(lambda symbol, event: print(symbol, event))
ctx.set_on_candlestick(lambda symbol, event: print(symbol, event))
```

`AsyncQuoteContext` callbacks may be `async def` — they are scheduled on the running event loop.

Push types: `PushQuote`, `PushDepth`, `PushBrokers`, `PushTrades`, `PushCandlestick`

### subscribe / unsubscribe

```python
ctx.subscribe(["700.HK", "AAPL.US"], [SubType.Quote, SubType.Depth])
ctx.unsubscribe(["AAPL.US"], [SubType.Quote])
resp = ctx.subscriptions()  # List[Subscription]
```

### subscribe_candlesticks

```python
# Returns initial snapshot; push arrives via set_on_candlestick
candles = ctx.subscribe_candlesticks("700.HK", Period.Min_1, TradeSessions.Intraday)
ctx.unsubscribe_candlesticks("700.HK", Period.Min_1)
```

## Market Data (pull)

### static_info

```python
resp = ctx.static_info(["700.HK", "AAPL.US"])  # List[SecurityStaticInfo]
# Fields: symbol, name_en, name_zh, exchange, currency, lot_size,
#         total_shares, circulating_shares, eps, eps_ttm, bps, dividend_yield
```

### quote

```python
resp = ctx.quote(["700.HK", "AAPL.US", "TSLA.US"])  # List[SecurityQuote]
# Fields: symbol, last_done, prev_close_price, open, high, low, volume, turnover,
#         trade_session, trade_status, pre_market_quote, post_market_quote
```

### option_quote / warrant_quote

```python
resp = ctx.option_quote(["AAPL230317P160000.US"])   # List[OptionQuote]
resp = ctx.warrant_quote(["21125.HK"])              # List[WarrantQuote]
```

### depth

```python
resp = ctx.depth("700.HK")     # SecurityDepth
# resp.asks: List[Depth], resp.bids: List[Depth]
# Depth fields: position, price, volume, order_num
```

### brokers

```python
resp = ctx.brokers("700.HK")   # SecurityBrokers
# resp.ask_brokers: List[Brokers], resp.bid_brokers: List[Brokers]
```

### participants

```python
resp = ctx.participants()      # List[ParticipantInfo]  (HK only)
```

### trades

```python
resp = ctx.trades("700.HK", 50)   # List[Trade], max count=1000
```

### intraday

```python
resp = ctx.intraday("700.HK")                                 # List[IntradayLine]
resp = ctx.intraday("700.HK", TradeSessions.All)              # include pre/post
```

### candlesticks (recent N)

```python
resp = ctx.candlesticks("700.HK", Period.Day, 100, AdjustType.NoAdjust)
resp = ctx.candlesticks("700.HK", Period.Min_5, 200, AdjustType.ForwardAdjust, TradeSessions.All)
# Returns: List[Candlestick]
# Fields: close, open, high, low, volume, turnover, trade_session, timestamp
```

### history_candlesticks_by_offset

```python
# forward=True: query forward from `time`; forward=False: backward
resp = ctx.history_candlesticks_by_offset(
    "700.HK", Period.Day, AdjustType.NoAdjust,
    forward=False, count=100, time=datetime(2024, 1, 1)
)
```

### history_candlesticks_by_date

```python
resp = ctx.history_candlesticks_by_date(
    "700.HK", Period.Day, AdjustType.ForwardAdjust,
    start=date(2024, 1, 1), end=date(2024, 12, 31)
)
```

## Options

```python
dates = ctx.option_chain_expiry_date_list("AAPL.US")    # List[date]
strikes = ctx.option_chain_info_by_date("AAPL.US", date(2024, 1, 19))
# List[StrikePriceInfo]: price, call_symbol, put_symbol, standard
```

## Warrants

```python
issuers = ctx.warrant_issuers()  # List[IssuerInfo]

resp = ctx.warrant_list(
    "700.HK",
    sort_by=WarrantSortBy.LastDone,
    sort_order=SortOrderType.Ascending,
    warrant_type=[WarrantType.Call],     # optional filters
)  # List[WarrantInfo]
```

## Trading Calendar

```python
sessions = ctx.trading_session()   # List[MarketTradingSession]
days = ctx.trading_days(Market.HK, date(2024, 1, 1), date(2024, 3, 31))
# MarketTradingDays: trading_days: List[date], half_trading_days: List[date]
# Constraint: interval < 1 month, only last year supported
```

## Capital & Indexes

```python
flow = ctx.capital_flow("700.HK")          # List[CapitalFlowLine]
dist = ctx.capital_distribution("700.HK")  # CapitalDistributionResponse

indexes = ctx.calc_indexes(
    ["700.HK", "AAPL.US"],
    [CalcIndex.LastDone, CalcIndex.PeTtmRatio, CalcIndex.PbRatio, CalcIndex.TotalMarketValue]
)  # List[SecurityCalcIndex]
```

## Watchlist

```python
groups = ctx.watchlist()   # List[WatchlistGroup]

group_id = ctx.create_watchlist_group("My Group", securities=["700.HK", "AAPL.US"])

ctx.update_watchlist_group(
    group_id,
    name="Updated Name",
    securities=["TSLA.US"],
    mode=SecuritiesUpdateMode.Add    # Add | Remove | Replace
)

ctx.delete_watchlist_group(group_id, purge=False)
```

## Security List & Market Temperature

```python
securities = ctx.security_list(Market.HK)                                   # List[Security]
securities = ctx.security_list(Market.US, SecurityListCategory.Overnight)

temp = ctx.market_temperature(Market.HK)          # MarketTemperature: temperature (0-100)
hist = ctx.history_market_temperature(            # HistoryMarketTemperatureResponse
    Market.HK, date(2024, 1, 1), date(2024, 3, 31)
)
```

## Realtime Cache (after subscribe)

These return data from the local push cache without making a network call:

```python
ctx.subscribe(["700.HK"], [SubType.Quote, SubType.Depth, SubType.Brokers, SubType.Trade])
from time import sleep; sleep(2)

quotes   = ctx.realtime_quote(["700.HK"])          # List[RealtimeQuote]
depth    = ctx.realtime_depth("700.HK")            # SecurityDepth
brokers  = ctx.realtime_brokers("700.HK")          # SecurityBrokers
trades   = ctx.realtime_trades("700.HK", 100)      # List[Trade]
candles  = ctx.realtime_candlesticks("AAPL.US", Period.Min_1, 50)  # List[Candlestick]
```

## Filings

```python
items = ctx.filings("700.HK")   # List of filing objects
# Each item: symbol, name, title, lang, type, url, published_at
```

## Account Info

```python
member_id = ctx.member_id()                    # int
level     = ctx.quote_level()                  # str (e.g. "2")
packages  = ctx.quote_package_details()        # List[QuotePackageDetail]
```
