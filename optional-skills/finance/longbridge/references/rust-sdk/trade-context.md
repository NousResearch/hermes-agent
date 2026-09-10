# Rust SDK — TradeContext

All methods are `async` and return `Result<T>`.

## Creation

```rust
let (ctx, order_rx) = TradeContext::new(Arc::new(config));

// Spawn push handler
tokio::spawn(async move {
    use longbridge::trade::PushEvent;
    while let Some(event) = order_rx.recv().await {
        match event {
            PushEvent::OrderChanged(o) => println!("Order {} -> {:?}", o.order_id, o.status),
        }
    }
});
```

## Subscribe to Order Push

```rust
use longbridge::trade::TopicType;

ctx.subscribe([TopicType::Private]).await?;
ctx.unsubscribe([TopicType::Private]).await?;
```

## Submit Order

> ⚠️ **State-changing — confirm before executing.** `submit_order` places a real order against
> the user's brokerage account. Read back symbol, side, quantity, and price and get an explicit
> "yes, place it" from the user immediately before calling this — never chain a submission
> straight off a data lookup. Same rule as `mcp.md`'s "always ask for confirmation before
> executing."

```rust
use longbridge::{Decimal, trade::{SubmitOrderOptions, OrderType, OrderSide, TimeInForceType}};

let opts = SubmitOrderOptions::new(
    "700.HK",
    OrderType::LO,
    OrderSide::Buy,
    Decimal::from(200),
    TimeInForceType::Day,
)
.submitted_price(Decimal::from_str("50.00").unwrap())
.remark("my order".into());

let resp = ctx.submit_order(opts).await?;
println!("Order ID: {}", resp.order_id);
```

**Builder methods for special order types:**

```rust
// LIT / MIT
opts.trigger_price(Decimal::from_str("48.00")?)

// Trailing orders (TSLPAMT)
opts.limit_offset(Decimal::from_str("1.00")?)
    .trailing_amount(Decimal::from_str("2.00")?)

// GTD
opts.expire_date(date!(2024-12-31))

// US pre/post market
opts.outside_rth(OutsideRTH::AnyTime)
```

## Replace / Cancel Order

> ⚠️ **State-changing — confirm before executing.** Both calls immediately mutate a live order.
> Confirm the target order ID and the new price/quantity (for `replace_order`) with the user
> right before calling — same confirmation gate as Submit Order above.

```rust
use longbridge::trade::ReplaceOrderOptions;

let opts = ReplaceOrderOptions::new("709043056541253632", Decimal::from(100))
    .price(Decimal::from_str("100.00")?);
ctx.replace_order(opts).await?;

ctx.cancel_order("709043056541253632").await?;
```

## Query Orders

```rust
use longbridge::trade::{GetHistoryOrdersOptions, GetTodayOrdersOptions, OrderStatus, OrderSide};

// Today's orders
let orders = ctx.today_orders(None).await?;  // Vec<Order>

// With filters
let orders = ctx.today_orders(Some(GetTodayOrdersOptions::new()
    .symbol("700.HK")
    .status([OrderStatus::Filled, OrderStatus::New])
    .side(OrderSide::Buy)
)).await?;

// Historical orders (does not include today)
let orders = ctx.history_orders(
    GetHistoryOrdersOptions::new()
        .symbol("700.HK")
        .start_at(datetime!(2024-01-01 00:00 UTC))
        .end_at(datetime!(2024-12-31 23:59 UTC))
).await?;

// Order detail
let detail = ctx.order_detail("701276261045858304").await?;  // OrderDetail
```

## Executions

```rust
use longbridge::trade::GetHistoryExecutionsOptions;

// Today's fills
let execs = ctx.today_executions(None).await?;    // Vec<Execution>

// Historical fills
let execs = ctx.history_executions(
    GetHistoryExecutionsOptions::new()
        .symbol("700.HK")
        .start_at(datetime!(2024-01-01 00:00 UTC))
        .end_at(datetime!(2024-12-31 23:59 UTC))
).await?;
// Execution: order_id, trade_id, symbol, trade_done_at, quantity, price
```

## Account Balance

```rust
let balances = ctx.account_balance(None).await?;         // Vec<AccountBalance>
let balances = ctx.account_balance(Some("HKD")).await?;  // filter by currency
```

## Cash Flow

```rust
use longbridge::trade::GetCashFlowOptions;

let flows = ctx.cash_flow(
    GetCashFlowOptions::new(
        datetime!(2024-01-01 00:00 UTC),
        datetime!(2024-12-31 23:59 UTC),
    )
    .symbol("700.HK")     // optional
).await?;  // Vec<CashFlow>
```

## Positions

```rust
let stock = ctx.stock_positions(None).await?;
// StockPositionsResponse { channels: Vec<StockPositionChannel> }
// StockPositionChannel { account_channel, positions: Vec<StockPosition> }
// StockPosition: symbol, symbol_name, quantity, available_quantity, currency, cost_price

let fund = ctx.fund_positions(None).await?;
// FundPositionsResponse { channels: Vec<FundPositionChannel> }
```

## Margin & Estimation

```rust
let ratio = ctx.margin_ratio("TSLA.US").await?;
// MarginRatio { im_factor, mm_factor, fm_factor }

let est = ctx.estimate_max_purchase_quantity(
    "700.HK",
    OrderType::LO,
    OrderSide::Buy,
    Some(Decimal::from_str("50.00")?),
    None,   // currency
    None,   // order_id
    false,  // fractional_shares
).await?;
// EstimateMaxPurchaseQuantityResponse { cash_max_qty, margin_max_qty }
```

## Complete Example

```rust
use std::sync::Arc;
use longbridge::{Config, TradeContext, Decimal, trade::{SubmitOrderOptions, OrderType, OrderSide, TimeInForceType}};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Arc::new(Config::from_apikey_env()?);
    let (ctx, _rx) = TradeContext::new(config);

    // Check balance before trading
    let balances = ctx.account_balance(None).await?;
    println!("Balance: {:?}", balances);

    // Place a limit buy order
    let resp = ctx.submit_order(
        SubmitOrderOptions::new(
            "700.HK",
            OrderType::LO,
            OrderSide::Buy,
            Decimal::from(100),
            TimeInForceType::Day,
        )
        .submitted_price(Decimal::from_str("45.00")?)
    ).await?;
    println!("Order placed: {}", resp.order_id);

    Ok(())
}
```
