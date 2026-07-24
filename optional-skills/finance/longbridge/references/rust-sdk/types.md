# Rust SDK — Types & Enums

All types are in the `longbridge` crate. Key modules: `longbridge::quote`, `longbridge::trade`.

## SubFlags — Quote subscription (bit-flags)

```rust
use longbridge::quote::SubFlags;

SubFlags::QUOTE    // Real-time quote
SubFlags::DEPTH    // Level 2 order book
SubFlags::BROKER   // HK broker queue
SubFlags::TRADE    // Tick-by-tick trades

// Combine:
SubFlags::QUOTE | SubFlags::DEPTH
```

## Period — Candlestick periods

```rust
use longbridge::quote::Period;

Period::OneMinute    Period::TwoMinute     Period::ThreeMinute
Period::FiveMinute   Period::TenMinute     Period::FifteenMinute
Period::TwentyMinute Period::ThirtyMinute  Period::FortyFiveMinute
Period::SixtyMinute  Period::TwoHour       Period::ThreeHour    Period::FourHour
Period::Day          Period::Week          Period::Month
Period::Quarter      Period::Year
```

**MCP string equivalents:** `"1m"`, `"2m"`, `"3m"`, `"5m"`, `"10m"`, `"15m"`, `"20m"`, `"30m"`, `"45m"`, `"60m"`, `"120m"`, `"180m"`, `"240m"`, `"day"`, `"week"`, `"month"`, `"quarter"`, `"year"`

## AdjustType

```rust
use longbridge::quote::AdjustType;

AdjustType::NoAdjust       // Actual (unadjusted)
AdjustType::ForwardAdjust  // Forward-adjusted for splits/dividends
```

## TradeSessions

```rust
use longbridge::quote::TradeSessions;

TradeSessions::Intraday   // Regular trading hours only
TradeSessions::All        // All sessions (pre, intraday, post, overnight)
```

## Market

```rust
use longbridge::Market;

Market::HK    // Hong Kong
Market::US    // United States
Market::CN    // China (SH/SZ)
Market::SG    // Singapore
```

## OrderSide

```rust
use longbridge::trade::OrderSide;

OrderSide::Buy
OrderSide::Sell
```

## OrderType

```rust
use longbridge::trade::OrderType;

OrderType::LO        // Limit Order
OrderType::ELO       // Enhanced Limit Order (HK only)
OrderType::MO        // Market Order
OrderType::AO        // At-Auction Order
OrderType::ALO       // At-Auction Limit Order
OrderType::ODD       // Odd Lots Order
OrderType::LIT       // Limit If Touched
OrderType::MIT       // Market If Touched
OrderType::TSLPAMT   // Trailing Limit (Trailing Amount)
OrderType::TSLPPCT   // Trailing Limit (Trailing Percent)
OrderType::TSMAMT    // Trailing Market (Trailing Amount)
OrderType::TSMPCT    // Trailing Market (Trailing Percent)
OrderType::SLO       // Special Limit Order (HK only)
```

## OrderStatus

```rust
use longbridge::trade::OrderStatus;

OrderStatus::NotReported      OrderStatus::New             OrderStatus::WaitToNew
OrderStatus::PartialFilled    OrderStatus::Filled          OrderStatus::WaitToReplace
OrderStatus::PendingReplace   OrderStatus::Replaced        OrderStatus::WaitToCancel
OrderStatus::PendingCancel    OrderStatus::Rejected        OrderStatus::Canceled
OrderStatus::Expired          OrderStatus::PartialWithdrawal
```

## TimeInForceType

```rust
use longbridge::trade::TimeInForceType;

TimeInForceType::Day              // Day order
TimeInForceType::GoodTilCanceled  // GTC
TimeInForceType::GoodTilDate      // GTD — use .expire_date()
```

## OutsideRTH (US only)

```rust
use longbridge::trade::OutsideRTH;

OutsideRTH::RTH_Only   // Regular hours only (default)
OutsideRTH::AnyTime    // Pre and post market
OutsideRTH::Overnight  // Overnight session
```

## TopicType (trade push)

```rust
use longbridge::trade::TopicType;

TopicType::Private  // Order change notifications
```

## CalcIndex

```rust
use longbridge::quote::CalcIndex;

// Key indexes:
CalcIndex::LastDone          CalcIndex::ChangeValue       CalcIndex::ChangeRate
CalcIndex::Volume            CalcIndex::Turnover          CalcIndex::TotalMarketValue
CalcIndex::PeTtmRatio        CalcIndex::PbRatio           CalcIndex::DividendRatioTtm
CalcIndex::YtdChangeRate     CalcIndex::ImpliedVolatility CalcIndex::Delta
CalcIndex::Gamma             CalcIndex::Theta             CalcIndex::Vega
// ... see Python types.md for full list
```

## Push Events

### QuoteContext push

```rust
use longbridge::quote::PushEvent;

PushEvent::Quote(PushQuote)          // last_done, open, high, low, volume, turnover
PushEvent::Depth(PushDepth)          // asks: Vec<Depth>, bids: Vec<Depth>
PushEvent::Brokers(PushBrokers)      // ask_brokers, bid_brokers
PushEvent::Trade(PushTrades)         // trades: Vec<Trade>
PushEvent::Candlestick(PushCandlestick)  // candlestick, period
```

### TradeContext push

```rust
use longbridge::trade::PushEvent;

PushEvent::OrderChanged(PushOrderChanged)
// Fields: order_id, symbol, status, side, filled_qty, price, msg
```

## Language

```rust
use longbridge::Language;

Language::EN     // English (default)
Language::ZH_CN  // Simplified Chinese
Language::ZH_HK  // Traditional Chinese
```

## PushCandlestickMode

```rust
use longbridge::PushCandlestickMode;

PushCandlestickMode::Realtime   // Push every tick update
PushCandlestickMode::Confirmed  // Push only after candle closes
```

## Decimal

Rust SDK uses `rust_decimal::Decimal` for all prices and quantities:

```rust
use longbridge::Decimal;  // re-export of rust_decimal::Decimal
use std::str::FromStr;

let price = Decimal::from_str("50.00")?;
let qty   = Decimal::from(200u32);
```

## SecuritiesUpdateMode (watchlist)

```rust
use longbridge::quote::SecuritiesUpdateMode;

SecuritiesUpdateMode::Add     // Append securities
SecuritiesUpdateMode::Remove  // Remove securities
SecuritiesUpdateMode::Replace // Replace all
```

## Error Type

```rust
use longbridge::Error;

match result {
    Err(Error::OpenApi { code, message, trace_id, .. }) => { /* API error */ }
    Err(Error::Http { .. })  => { /* network/HTTP error */ }
    Err(e) => { /* other */ }
}
```
