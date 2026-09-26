# Bitcoin Skill — Reference Notes

## API Quirks and Implementation Notes

### mempool.space `/block/{hash}` endpoint

As of mid-2026, the `/block/{hash}` endpoint no longer returns `reward` or
`fees`. The client calculates the coinbase subsidy locally from the block
height (50 BTC halved every 210,000 blocks). Total block reward would be
`subsidy + fees`, but fees are not available without paginating the full
transaction list of the block, which is too expensive for a quick lookup.

The client leaves `fees_sats`, `fees_btc`, `reward_sats`, `reward_btc` as
`null` and includes a `note` explaining this.

### Hashrate units

mempool.space returns hashrate in H/s (hashes per second). The client
formats these into SI units (EH/s, ZH/s) for human readability and exposes
the raw H/s values as `*_hs` fields.

### Address `first_seen` / `last_seen`

mempool.space does not return these fields on the `/address/{addr}`
endpoint. The client does not fabricate them. If activity timestamps are
needed, inspect individual transactions with the `tx` command or an
explorer that exposes this data.

### `/mempool/recent` whale watch

This endpoint returns the last ~10 transactions that arrived in the
mempool. It is a convenience sampler, not a full mempool scan. Large
confirmed transactions require inspecting blocks or using a different API.

### CoinGecko currency handling

The client supports any CoinGecko-compatible currency code. The
`fiat_currency` field is returned in upper case, and fiat formatting uses
`$` for USD, `€` for EUR, and the code prefix for other currencies.

## Useful Endpoints

- Address: `GET /api/address/{addr}`
- Transaction: `GET /api/tx/{txid}`
- Block by height: `GET /api/block-height/{height}` (returns hash as text)
- Block: `GET /api/block/{hash}`
- Block status: `GET /api/block/{hash}/status`
- Mempool: `GET /api/mempool`
- Recommended fees: `GET /api/v1/fees/recommended`
- Hashrate: `GET /api/v1/mining/hashrate/{interval}`
- Difficulty adjustments: `GET /api/v1/mining/difficulty-adjustments/{interval}`
- Recent mempool txs: `GET /api/mempool/recent`

## Trusted Sources

- mempool.space (on-chain data, mempool, fees, hashrate, difficulty)
- blockstream.info (fallback on-chain API, same endpoints as mempool.space)
- CoinGecko (BTC price, market cap, volume, 24h change)
