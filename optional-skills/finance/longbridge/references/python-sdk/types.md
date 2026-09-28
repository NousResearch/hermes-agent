# Python SDK — Types & Enums

All types imported from `longbridge.openapi`.

## SubType — Quote subscription flags

```python
SubType.Quote    # Real-time quote (price, volume)
SubType.Depth    # Level 2 order book
SubType.Brokers  # HK broker queue
SubType.Trade    # Tick-by-tick trades
```

## Period — Candlestick periods

```python
Period.Min_1    Period.Min_2    Period.Min_3
Period.Min_5    Period.Min_10   Period.Min_15
Period.Min_20   Period.Min_30   Period.Min_45
Period.Min_60   Period.Min_120  Period.Min_180   Period.Min_240
Period.Day      Period.Week     Period.Month
Period.Quarter  Period.Year
```

## AdjustType

```python
AdjustType.NoAdjust       # Actual (unadjusted)
AdjustType.ForwardAdjust  # Forward-adjusted for splits/dividends
```

## TradeSessions

```python
TradeSessions.Intraday   # Regular trading hours only (default)
TradeSessions.All        # All sessions (pre, intraday, post, overnight)
```

## Market

```python
Market.HK   # Hong Kong
Market.US   # United States
Market.CN   # China (SH/SZ)
Market.SG   # Singapore
```

## OrderSide

```python
OrderSide.Buy
OrderSide.Sell
```

## OrderType

```python
OrderType.LO       # Limit Order (price required)
OrderType.ELO      # Enhanced Limit Order (HK only)
OrderType.MO       # Market Order
OrderType.AO       # At-Auction Order
OrderType.ALO      # At-Auction Limit Order
OrderType.ODD      # Odd Lots Order
OrderType.LIT      # Limit If Touched (price + trigger_price)
OrderType.MIT      # Market If Touched (trigger_price)
OrderType.TSLPAMT  # Trailing Limit If Touched (Trailing Amount)
OrderType.TSLPPCT  # Trailing Limit If Touched (Trailing Percent)
OrderType.TSMAMT   # Trailing Market If Touched (Trailing Amount)
OrderType.TSMPCT   # Trailing Market If Touched (Trailing Percent)
OrderType.SLO      # Special Limit Order (HK only, no replace)
```

## OrderStatus

```python
OrderStatus.NotReported          # Pending broker submission
OrderStatus.New                  # Accepted by exchange
OrderStatus.WaitToNew            # In transit to exchange
OrderStatus.PartialFilled        # Partially executed
OrderStatus.Filled               # Fully executed
OrderStatus.WaitToReplace        # Replace request in transit
OrderStatus.PendingReplace       # Replace pending on exchange
OrderStatus.Replaced             # Successfully replaced
OrderStatus.WaitToCancel         # Cancel request in transit
OrderStatus.PendingCancel        # Cancel pending on exchange
OrderStatus.Rejected             # Rejected
OrderStatus.Canceled             # Canceled
OrderStatus.Expired              # Expired
OrderStatus.PartialWithdrawal    # Partially canceled
```

## TimeInForceType

```python
TimeInForceType.Day              # Valid today only
TimeInForceType.GoodTilCanceled  # GTC
TimeInForceType.GoodTilDate      # GTD — requires expire_date
```

## OutsideRTH (US only)

```python
OutsideRTH.RTHOnly    # Regular trading hours only (default)
OutsideRTH.AnyTime    # Pre and post market allowed
OutsideRTH.Overnight  # Overnight session
```

## TopicType (trade push)

```python
TopicType.Private   # Order change notifications
```

## CalcIndex — Financial indexes

```python
# Price & volume
CalcIndex.LastDone           CalcIndex.ChangeValue       CalcIndex.ChangeRate
CalcIndex.Volume             CalcIndex.Turnover          CalcIndex.Amplitude
CalcIndex.VolumeRatio        CalcIndex.TurnoverRate      CalcIndex.TotalMarketValue
CalcIndex.CapitalFlow        CalcIndex.YtdChangeRate

# Period returns
CalcIndex.FiveDayChangeRate  CalcIndex.TenDayChangeRate
CalcIndex.HalfYearChangeRate CalcIndex.FiveMinutesChangeRate

# Valuation
CalcIndex.PeTtmRatio         CalcIndex.PbRatio           CalcIndex.DividendRatioTtm

# Options / Warrants
CalcIndex.ExpiryDate         CalcIndex.StrikePrice       CalcIndex.UpperStrikePrice
CalcIndex.LowerStrikePrice   CalcIndex.OutstandingQty    CalcIndex.OutstandingRatio
CalcIndex.Premium            CalcIndex.ItmOtm            CalcIndex.ImpliedVolatility
CalcIndex.WarrantDelta       CalcIndex.CallPrice         CalcIndex.ToCallPrice
CalcIndex.EffectiveLeverage  CalcIndex.LeverageRatio     CalcIndex.ConversionRatio
CalcIndex.BalancePoint       CalcIndex.OpenInterest
CalcIndex.Delta              CalcIndex.Gamma             CalcIndex.Theta
CalcIndex.Vega               CalcIndex.Rho
```

## SecuritiesUpdateMode (watchlist)

```python
SecuritiesUpdateMode.Add      # Append securities
SecuritiesUpdateMode.Remove   # Remove securities
SecuritiesUpdateMode.Replace  # Replace all securities
```

## SecurityListCategory

```python
SecurityListCategory.Overnight  # Overnight-eligible securities
```

## BalanceType (cash flow)

```python
BalanceType.Cash   # Cash transactions
BalanceType.Stock  # Stock transactions
BalanceType.Fund   # Fund transactions
```

## WarrantSortBy / SortOrderType

```python
WarrantSortBy.LastDone   WarrantSortBy.ChangeRate  WarrantSortBy.Volume
WarrantSortBy.Price      WarrantSortBy.Premium     WarrantSortBy.Leverage
# ... and more

SortOrderType.Ascending
SortOrderType.Descending
```

## Push Types

| Class | Carrier | Fields |
|-------|---------|--------|
| `PushQuote` | `set_on_quote` | `last_done`, `open`, `high`, `low`, `volume`, `turnover`, `trade_session` |
| `PushDepth` | `set_on_depth` | `asks: List[Depth]`, `bids: List[Depth]` |
| `PushBrokers` | `set_on_brokers` | `ask_brokers: List[Brokers]`, `bid_brokers: List[Brokers]` |
| `PushTrades` | `set_on_trades` | `trades: List[Trade]` |
| `PushCandlestick` | `set_on_candlestick` | `candlestick: Candlestick`, `period: Period` |
| `PushOrderChanged` | `set_on_order_changed` | `order_id`, `symbol`, `status`, `side`, `filled_qty`, `price`, `msg` |

## Error Handling

```python
from longbridge.openapi import OpenApiException, ErrorKind

try:
    resp = ctx.quote(["INVALID.XX"])
except OpenApiException as e:
    print(e.kind)       # ErrorKind.Http | ErrorKind.OpenApi | ErrorKind.Other
    print(e.code)       # API error code (int)
    print(e.message)    # Human-readable message
    print(e.trace_id)   # Request trace ID for support
```
