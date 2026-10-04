# Rust SDK — QuoteContext

All methods are `async` and return `Result<T>`.

## Creation

```rust
let (ctx, push_rx) = QuoteContext::new(Arc::new(config));
```

## Subscriptions

### subscribe / unsubscribe

```rust
use longbridge::quote::SubFlags;

// SubFlags are bit-flags, combine with |
ctx.subscribe(["700.HK", "AAPL.US"], SubFlags::QUOTE | SubFlags::DEPTH).await?;
ctx.unsubscribe(["AAPL.US"], SubFlags::QUOTE).await?;

let subs = ctx.subscriptions().await?;  // Vec<Subscription>
```

**SubFlags:**
```rust
SubFlags::QUOTE    // Real-time quote
SubFlags::DEPTH    // Level 2 order book
SubFlags::BROKER   // HK broker queue
SubFlags::TRADE    // Tick-by-tick trades
```

### subscribe_candlesticks

```rust
use longbridge::quote::{Period, TradeSessions};

// Returns initial snapshot; push arrives via push_rx
let candles = ctx.subscribe_candlesticks("700.HK", Period::Day).await?;
ctx.unsubscribe_candlesticks("700.HK", Period::Day).await?;
```

## Market Data

### static_info

```rust
let infos = ctx.static_info(["700.HK", "AAPL.US"]).await?;
// Vec<SecurityStaticInfo>: symbol, name_en, name_zh, exchange, currency, lot_size, etc.
```

### quote

```rust
let quotes = ctx.quote(["700.HK", "AAPL.US"]).await?;
// Vec<SecurityQuote>: symbol, last_done, prev_close_price, open, high, low, volume, turnover
```

### option_quote / warrant_quote

```rust
let opt = ctx.option_quote(["AAPL230317P160000.US"]).await?;  // Vec<OptionQuote>
let war = ctx.warrant_quote(["21125.HK"]).await?;              // Vec<WarrantQuote>
```

### depth

```rust
let depth = ctx.depth("700.HK").await?;
// SecurityDepth { asks: Vec<Depth>, bids: Vec<Depth> }
// Depth { position, price, volume, order_num }
```

### brokers

```rust
let brokers = ctx.brokers("700.HK").await?;
// SecurityBrokers { ask_brokers: Vec<Brokers>, bid_brokers: Vec<Brokers> }
```

### participants

```rust
let participants = ctx.participants().await?;  // Vec<ParticipantInfo> (HK only)
```

### trades

```rust
let trades = ctx.trades("700.HK", 50).await?;  // Vec<Trade>, max 1000
```

### intraday

```rust
use longbridge::quote::TradeSessions;

let lines = ctx.intraday("700.HK", TradeSessions::Intraday).await?;  // Vec<IntradayLine>
let lines = ctx.intraday("700.HK", TradeSessions::All).await?;       // include pre/post
```

### candlesticks (recent N)

```rust
use longbridge::quote::{Period, AdjustType, TradeSessions};

let candles = ctx.candlesticks("700.HK", Period::Day, 100, AdjustType::NoAdjust, TradeSessions::Intraday).await?;
// Vec<Candlestick>: close, open, high, low, volume, turnover, trade_session, timestamp
```

### history_candlesticks_by_offset

```rust
use time::macros::datetime;

let candles = ctx.history_candlesticks_by_offset(
    "700.HK",
    Period::Day,
    AdjustType::NoAdjust,
    false,      // forward: false = look backward from `time`
    100,
    Some(datetime!(2024-01-01 00:00 UTC)),
    TradeSessions::Intraday,
).await?;
```

### history_candlesticks_by_date

```rust
use time::macros::date;

let candles = ctx.history_candlesticks_by_date(
    "700.HK",
    Period::Day,
    AdjustType::ForwardAdjust,
    Some(date!(2024-01-01)),
    Some(date!(2024-12-31)),
    TradeSessions::Intraday,
).await?;
```

## Options

```rust
use time::macros::date;

let dates   = ctx.option_chain_expiry_date_list("AAPL.US").await?;   // Vec<Date>
let strikes = ctx.option_chain_info_by_date("AAPL.US", date!(2024-01-19)).await?;
// Vec<StrikePriceInfo>: price, call_symbol, put_symbol, standard
```

## Warrants

```rust
use longbridge::quote::{WarrantSortBy, SortOrderType};

let issuers = ctx.warrant_issuers().await?;   // Vec<IssuerInfo>

let warrants = ctx.warrant_list(
    "700.HK",
    WarrantSortBy::LastDone,
    SortOrderType::Ascending,
    Default::default(),  // WarrantListOptions (optional filters)
).await?;  // Vec<WarrantInfo>
```

## Trading Calendar

```rust
use longbridge::Market;
use time::macros::date;

let sessions = ctx.trading_session().await?;
// Vec<MarketTradingSession>

let days = ctx.trading_days(Market::HK, date!(2024-01-01), date!(2024-03-31)).await?;
// MarketTradingDays { trading_days, half_trading_days }
```

## Capital & Indexes

```rust
use longbridge::quote::CalcIndex;

let flow = ctx.capital_flow("700.HK").await?;          // Vec<CapitalFlowLine>
let dist = ctx.capital_distribution("700.HK").await?;  // CapitalDistributionResponse

let indexes = ctx.calc_indexes(
    ["700.HK", "AAPL.US"],
    [CalcIndex::LastDone, CalcIndex::PeTtmRatio, CalcIndex::PbRatio],
).await?;  // Vec<SecurityCalcIndex>
```

## Watchlist

```rust
use longbridge::quote::{RequestCreateWatchlistGroup, RequestUpdateWatchlistGroup};

let groups = ctx.watchlist().await?;   // Vec<WatchlistGroup>

let group_id = ctx.create_watchlist_group(RequestCreateWatchlistGroup {
    name: "My Group".into(),
    securities: vec!["700.HK".into(), "AAPL.US".into()],
}).await?;  // i64

ctx.update_watchlist_group(RequestUpdateWatchlistGroup {
    id: group_id,
    name: Some("Updated".into()),
    securities: vec!["TSLA.US".into()],
    mode: Some(SecuritiesUpdateMode::Add),
}).await?;

ctx.delete_watchlist_group(group_id, false).await?;
```

## Security List & Market Temperature

```rust
use longbridge::{Market, quote::SecurityListCategory};
use time::macros::date;

let secs = ctx.security_list(Market::HK, None).await?;   // Vec<Security>
let temp = ctx.market_temperature(Market::HK).await?;    // MarketTemperature
let hist = ctx.history_market_temperature(
    Market::HK, date!(2024-01-01), date!(2024-03-31)
).await?;
```

## Realtime Cache

After subscribing, get cached data without a network call:

```rust
let quotes   = ctx.realtime_quote(["700.HK"]).await?;      // Vec<RealtimeQuote>
let depth    = ctx.realtime_depth("700.HK").await?;         // SecurityDepth
let brokers  = ctx.realtime_brokers("700.HK").await?;       // SecurityBrokers
let trades   = ctx.realtime_trades("700.HK", 100).await?;   // Vec<Trade>
```

## Account Info

```rust
let id       = ctx.member_id().await?;               // i64
let level    = ctx.quote_level().await?;              // String
let packages = ctx.quote_package_details().await?;    // Vec<QuotePackageDetail>
```
