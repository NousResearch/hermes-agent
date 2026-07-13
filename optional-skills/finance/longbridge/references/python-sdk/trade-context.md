# Python SDK — TradeContext

`TradeContext` (sync) / `AsyncTradeContext` (async) — order management, positions, account.

## Creation

```python
# Sync
ctx = TradeContext(config)

# Async
ctx = AsyncTradeContext.create(config)
```

## Order Push

```python
def on_order_changed(event: PushOrderChanged):
    print(event.symbol, event.status, event.filled_qty)

ctx.set_on_order_changed(on_order_changed)
ctx.subscribe([TopicType.Private])    # start receiving push
ctx.unsubscribe([TopicType.Private])
```

## Submit Order

> ⚠️ **State-changing — confirm before executing.** `submit_order` places a real order against
> the user's brokerage account. Read back symbol, side, quantity, and price and get an explicit
> "yes, place it" from the user immediately before calling this — never chain a submission
> straight off a data lookup. Same rule as `mcp.md`'s "always ask for confirmation before
> executing."

```python
from decimal import Decimal
from longbridge.openapi import TradeContext, Config, OrderSide, OrderType, TimeInForceType

ctx = TradeContext(config)

resp = ctx.submit_order(
    symbol="700.HK",
    order_type=OrderType.LO,
    side=OrderSide.Buy,
    submitted_quantity=Decimal(200),
    time_in_force=TimeInForceType.Day,
    submitted_price=Decimal("50.00"),  # required for LO/ELO/ALO/ODD
    remark="my order",                 # optional, max 64 chars
)
# resp.order_id: str
```

**Optional parameters by order type:**

| Parameter | Required for |
|-----------|-------------|
| `submitted_price` | LO, ELO, ALO, ODD, LIT |
| `trigger_price` | LIT, MIT |
| `limit_offset` | TSLPAMT, TSLPPCT |
| `trailing_amount` | TSLPAMT |
| `trailing_percent` | TSLPPCT |
| `expire_date` | GTD time_in_force |
| `outside_rth` | US only: `OutsideRTH.RTHOnly / AnyTime / Overnight` |

## Replace / Cancel Order

> ⚠️ **State-changing — confirm before executing.** Both calls immediately mutate a live order.
> Confirm the target `order_id` and the new price/quantity (for `replace_order`) with the user
> right before calling — same confirmation gate as Submit Order above.

```python
ctx.replace_order(
    order_id="709043056541253632",
    quantity=Decimal(100),
    price=Decimal("100.00"),
)

ctx.cancel_order("709043056541253632")
```

## Query Orders

```python
# Today's orders
orders = ctx.today_orders(
    symbol="700.HK",                              # optional
    status=[OrderStatus.Filled, OrderStatus.New], # optional
    side=OrderSide.Buy,                           # optional
    market=Market.HK,                             # optional
    order_id="123456",                            # optional
)  # List[Order]

# Historical orders (no today)
orders = ctx.history_orders(
    symbol="700.HK",
    status=[OrderStatus.Filled],
    side=OrderSide.Buy,
    market=Market.HK,
    start_at=datetime(2024, 1, 1),
    end_at=datetime(2024, 12, 31),
)  # List[Order]

# Order detail (includes charge breakdown)
detail = ctx.order_detail("701276261045858304")  # OrderDetail
```

**Order fields:** `order_id`, `symbol`, `order_type`, `side`, `status`, `submitted_quantity`,
`submitted_price`, `executed_qty`, `executed_price`, `submitted_at`, `updated_at`, `tag`,
`time_in_force`, `expire_date`, `outside_rth`, `remark`

## Executions

```python
# Today's fills
execs = ctx.today_executions(symbol="700.HK", order_id=None)  # List[Execution]

# Historical fills
execs = ctx.history_executions(
    symbol="700.HK",
    start_at=datetime(2024, 1, 1),
    end_at=datetime(2024, 12, 31),
)  # List[Execution]
# Execution fields: order_id, trade_id, symbol, trade_done_at, quantity, price
```

## Account Balance

```python
balances = ctx.account_balance()              # List[AccountBalance]
balances = ctx.account_balance("HKD")        # filter by currency

# AccountBalance fields:
# currency, total_cash, max_finance_amount, remaining_finance_amount,
# risk_level, margin_call, net_assets, buy_power, cash_infos: List[CashInfo]
```

## Cash Flow

```python
flows = ctx.cash_flow(
    start_at=datetime(2024, 1, 1),
    end_at=datetime(2024, 12, 31),
    business_type=BalanceType.Cash,  # optional: Cash | Stock | Fund
    symbol="700.HK",                 # optional
    page=1,                          # optional, default 1
    size=50,                         # optional, default 50
)  # List[CashFlow]
```

## Positions

```python
stock_pos = ctx.stock_positions()                       # StockPositionsResponse
stock_pos = ctx.stock_positions(symbols=["700.HK"])     # filter

fund_pos  = ctx.fund_positions()                        # FundPositionsResponse
fund_pos  = ctx.fund_positions(symbols=["HK123"])
```

**StockPositionsResponse:** `.channels: List[StockPositionChannel]`
Each channel: `account_channel`, `positions: List[StockPosition]`

**StockPosition fields:** `symbol`, `symbol_name`, `quantity`, `available_quantity`,
`currency`, `cost_price`, `market`, `init_quantity`

## Margin & Estimation

```python
ratio = ctx.margin_ratio("TSLA.US")   # MarginRatio
# Fields: im_factor (initial), mm_factor (maintenance), fm_factor (forced liq.)

est = ctx.estimate_max_purchase_quantity(
    symbol="700.HK",
    order_type=OrderType.LO,
    side=OrderSide.Buy,
    price=Decimal("50.00"),
    currency="HKD",          # optional
    fractional_shares=False, # optional
)  # EstimateMaxPurchaseQuantityResponse
# Fields: cash_max_qty, margin_max_qty
```

## Async Example (FastAPI / asyncio)

```python
import asyncio
from decimal import Decimal
from longbridge.openapi import OAuthBuilder, Config, AsyncTradeContext, OrderSide, OrderType, TimeInForceType

async def main():
    oauth = await OAuthBuilder("your-client-id").build_async(
        lambda url: print("Visit:", url)
    )
    config = Config.from_oauth(oauth)
    ctx = AsyncTradeContext.create(config)

    resp = await ctx.submit_order(
        symbol="700.HK",
        order_type=OrderType.LO,
        side=OrderSide.Buy,
        submitted_quantity=Decimal(100),
        time_in_force=TimeInForceType.Day,
        submitted_price=Decimal("50.00"),
    )
    print(resp.order_id)

asyncio.run(main())
```
