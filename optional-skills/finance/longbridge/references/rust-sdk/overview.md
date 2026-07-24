# Rust SDK Overview

**Crate:** `longbridge` v4.0.5
**Docs:** https://longbridge.github.io/openapi/rust/longbridge/

## Add to Cargo.toml

```toml
[dependencies]
longbridge = "4.0.5"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
rust_decimal = "1"
time = "0.3"
```

> **Note:** Previously named `longport`. The crate was renamed to `longbridge`.

## Import

```rust
use longbridge::{
    Config, Market, QuoteContext, TradeContext,
    quote::{Period, SubFlags, AdjustType, TradeSessions, PushEvent},
    trade::{OrderType, OrderSide, TimeInForceType, OutsideRTH, SubmitOrderOptions},
    Decimal,
};
```

## Authentication

### OAuth 2.0 (Recommended)

```rust
use longbridge::{Config, QuoteContext};
use longbridge_oauth::OAuthBuilder;  // in oauth crate

// Token cached at ~/.longbridge/openapi/tokens/<client_id>
let oauth = OAuthBuilder::new("your-client-id")
    .build(|url| println!("Open URL to authorize: {url}"))
    .await?;
let config = Config::from_oauth(oauth);
```

## Config Options

```rust
use std::sync::Arc;
use longbridge::{Config, Language, PushCandlestickMode};

let config = Arc::new(
    Config::from_oauth(oauth)
        .language(Language::EN)
        .enable_overnight(false)
        .push_candlestick_mode(PushCandlestickMode::Realtime)
);
```

## Creating Contexts

The Rust SDK is **async-only** (tokio). Both contexts return a channel receiver for push events.

```rust
use std::sync::Arc;
use longbridge::{Config, QuoteContext, TradeContext};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let oauth = OAuthBuilder::new("your-client-id")
        .build(|url| println!("Open URL to authorize: {url}"))
        .await?;
    let config = Arc::new(Config::from_oauth(oauth));

    // Returns (context, push_receiver)
    let (quote_ctx, mut push_rx) = QuoteContext::new(config.clone());
    let (trade_ctx, mut order_rx) = TradeContext::new(config);

    // Spawn push handler
    tokio::spawn(async move {
        while let Some(event) = push_rx.recv().await {
            println!("Push: {:?}", event);
        }
    });

    Ok(())
}
```

## Push Event Handling

`QuoteContext::new` returns `mpsc::UnboundedReceiver<PushEvent>`.

```rust
use longbridge::quote::PushEvent;

while let Some(event) = push_rx.recv().await {
    match event {
        PushEvent::Quote(e)       => println!("{}: last={}", e.symbol, e.last_done),
        PushEvent::Depth(e)       => println!("{}: {} asks", e.symbol, e.asks.len()),
        PushEvent::Brokers(e)     => println!("brokers: {}", e.symbol),
        PushEvent::Trade(e)       => println!("trade: {}", e.symbol),
        PushEvent::Candlestick(e) => println!("candle: {}", e.symbol),
    }
}
```

`TradeContext::new` returns `mpsc::UnboundedReceiver<longbridge::trade::PushEvent>`.

```rust
use longbridge::trade::PushEvent as TradePushEvent;

while let Some(event) = order_rx.recv().await {
    match event {
        TradePushEvent::OrderChanged(o) => println!("Order {} -> {:?}", o.order_id, o.status),
    }
}
```

## Error Handling

```rust
use longbridge::Error;

match ctx.quote(["INVALID.XX"]).await {
    Ok(quotes) => { /* ... */ }
    Err(Error::OpenApi { code, message, .. }) => eprintln!("API error {code}: {message}"),
    Err(e) => eprintln!("Other error: {e}"),
}
```

## Environment Variables

Same as Python SDK — see [Python overview](../python-sdk/overview.md#environment-variables).
