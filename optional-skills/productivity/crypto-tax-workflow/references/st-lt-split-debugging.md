# Debugging the ST/LT Split Mismatch

## Symptom

`capital_gains_summary.short_term_gain_loss_usd + long_term_gain_loss_usd` does NOT equal `total_realized_gain_loss_usd`. Example: total = $510,526 but ST ($934) + LT ($0) = $934.

## Root Cause

The original `calculate_crypto_gains` only classified gains from *matched* lots. When a SELL exceeded available buy lots (no cost basis), `lots_used` was empty for the unmatched portion. The `for li in lots_used:` loop never executed, so the unmatched gain was silently dropped from the ST/LT split while still correctly included in `total_realized_gain_loss_usd`.

## Reproduction

A wallet with 37 incoming transfers (BUYs) and 15 outgoing (SELLs). The SELLs total 224+ ETH but the BUYs only provide ~20 ETH of cost-basis lots. Two large SELLs (64 ETH and 160 ETH) exceed the FIFO queue:

```
SELL 64.000000 ETH proceeds=$122,807 basis=$794 gain=$122,013
  lots_used: 19 (covering ~0.4 ETH total)
  ST portion: $787  LT portion: $0    ← only matched 0.4 ETH worth!

SELL 160.000000 ETH proceeds=$307,019 basis=$0 gain=$307,019
  lots_used: 0                        ← no lots at all!
  ST portion: $0  LT portion: $0      ← ENTIRE $307K gain lost from split!
```

## Fix (in v1.2.0)

After the `for li in lots_used:` loop, add unmatched-portion fallback:

```python
# After processing all matched lots:
if remaining_to_sell > 1e-12 and amount > 0:
    unmatched_ratio = remaining_to_sell / amount
    unmatched_gain = gain_loss * unmatched_ratio
    sale_st_gain += unmatched_gain  # no basis → can't prove LT → short-term
```

## Verification

Always run this check after `calculate_crypto_gains` returns:

```python
cg = result["capital_gains_summary"]
match = abs(cg["total_realized_gain_loss_usd"] -
            (cg["short_term_gain_loss_usd"] + cg["long_term_gain_loss_usd"])) < 0.02
assert match, f"ST/LT split mismatch: total={cg['total']} vs ST+LT={cg['st']+cg['lt']}"
```

## Instrumentation Pattern

To trace the lot-matching logic, replicate the core loop with print statements. Key points to log per SELL: amount, proceeds, basis, gain, number of lots used, ST/LT portions, and remaining unmatched amount.
