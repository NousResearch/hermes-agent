# Params

Use this file for field semantics, order-shape mapping, defaults, validation, and local normalization.

1. Required: `chainId`, `swapper`, `input.token`, `input.amount`, `output.token`.
2. Optional: `input.maxAmount`, `nonce`, `start`, `deadline`, `epoch`, `freshness`, `slippage`, `output.limit`, `output.triggerLower`, `output.triggerUpper`, `output.recipient`.
3. Common order-shape fields: market = `output.limit = 0`; limit = `output.limit > 0`; stop-loss = `output.triggerLower > 0`; take-profit = `output.triggerUpper > 0`; delayed-start = future `start`; chunked or TWAP = `input.amount < input.maxAmount`; recurring chunked = `epoch > 0`; back to native output = `output.token = 0x0000000000000000000000000000000000000000`.
4. `input.amount` is the fixed per-chunk size. `input.maxAmount` is the total requested size. If omitted, it defaults to `input.amount`. If it is not divisible by `input.amount`, round it down to a whole number of chunks before building `typedData`.
5. `output.limit`, `output.triggerLower`, and `output.triggerUpper` are output-token amounts per chunk, encoded in the output token's smallest unit. When `output.token` is an ERC-20, encode them in that token's decimals. When `output.token = 0x0000000000000000000000000000000000000000`, encode them in native `wei` with 18 decimals.
6. Input and output units are independent. Always encode `input.amount` and `input.maxAmount` in `input.token` decimals, even for back-to-native orders; encode `output.limit`, `output.triggerLower`, and `output.triggerUpper` in `output.token` units. For example, USDC-to-native uses USDC decimals for input amounts and native `wei` for output triggers.
7. Future `start` delays the first fill. `epoch` is the delay between chunks, but it is not exact: each chunk can fill anywhere inside its epoch window, only once. Large `epoch` is not a delayed order by itself.
8. Chunked orders should use `epoch > 0`; with `epoch = 0`, only the first chunk can fill.
9. Defaults:
   `input.maxAmount = input.amount`, `nonce = now`, `start = now`, `epoch = 0` for single orders, `epoch = 60` for chunked orders, `freshness = 50`, `deadline = start + 300 + chunkCount * epoch`, `slippage = 500`, `output.limit = 0`, `output.triggerLower = 0`, `output.triggerUpper = 0`, `output.recipient = swapper`. When replacing `<EPOCH_SECONDS>` for a recurring order, choose a value greater than `freshness`.
10. Validation:
   `start != 0`, `input.amount != 0`, `input.amount <= input.maxAmount`, `input.token != output.token`, `output.triggerLower <= output.triggerUpper` when `triggerUpper != 0`, `slippage <= 5000`, `freshness != 0`, `freshness < epoch` when `epoch != 0`, and non-zero `epoch >= 60`.
11. Higher slippage is still protected by oracle pricing and offchain executors.
12. `output.recipient` is dangerous to change away from `swapper`.
13. Native input is not supported. Wrap to WNATIVE first. Native output, including back-to-native orders, is supported directly with `output.token = 0x0000000000000000000000000000000000000000`.
14. Derived execution values:
    `chunkCount = floor(input.maxAmount / input.amount)` after any rounding down.
15. JSON typing:
    keep addresses and byte strings as `0x` strings,
    keep large integer fields as decimal strings,
    use JSON numbers for `domain.chainId`, `message.witness.chainid`, `message.witness.exclusivity`, `message.witness.epoch`, `message.witness.slippage`, `message.witness.freshness`, and `message.witness.exchange.share`.
16. For full mock relay payloads, see [examples.md](examples.md).
