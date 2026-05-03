# Gacha — public API

One unauthenticated JSON endpoint on `https://api.cat.town` serves a user's capsule-item history. It's the workhorse for detecting when an async pull result has landed.

## `GET /v2/items/capsule/<address>`

Returns every capsule-pool NFT the address has ever received, newest first. Used by the frontend `useGachaItemPolling` hook on a 1-second interval to detect new mints after a pull.

### Request

- Path parameter: `<address>` — 0x wallet address (mixed case OK; API lowercases).
- Method: `GET`. No auth, no headers required.

### Response

Plain JSON array of items:

```ts
[
  {
    id: number               // token id, monotonically increasing — the key field for polling
    name: string             // e.g. "Royal Blue Varsity"
    rarity: string           // "Common" | "Uncommon" | "Rare" | "Epic" | "Legendary"
    imageUrl: string
    traitNames: string[]     // parallel with traitValues
    traitValues: string[]
    source: string           // "Gacha" or "Boutique" depending on pool; capsule endpoint is gacha-specific
    // …other metadata fields mirror the /v2/items/master shape
  },
  …
]
```

### Known quirks

- **Cold wallet → 500.** Addresses that have never pulled return a 500 with `{ "status": 500, "error": "Failed to fetch capsule drops from database" }` rather than an empty `[]`. Agents should treat 500 as "no capsule history yet" (equivalent to `[]`) and not surface the error to the user.
- **Newest-first ordering.** Reliable as far as I've tested; still, use `max(item.id)` rather than `response[0].id` if you need a ground-truth "latest id".

### Polling pattern

```
# Before the pay tx
latestId = max( item.id for item in GET /v2/items/capsule/<user> )  # or 0 if no history

# Submit N pay txs (gacha.purchaseAndOpenCapsule, one per pull)

# After the pay txs confirm, poll:
while len(newItems) < N:
  response = GET /v2/items/capsule/<user>
  newItems = [ item for item in response if item.id > latestId ]
  sleep 1 s
# newItems now contains all N new capsule items
```

Pair this with a timeout (60 s is generous) to avoid hanging if a VRF callback gets delayed. On timeout, surface "still waiting — I'll check again shortly" and retry.

### Polling cadence

The frontend polls every **1 second** for 60 seconds max, then backs off. For chat use, every 2–3 seconds is plenty — the VRF callback usually lands within a few seconds of the pay tx on Base.

### Caching

Do not cache this endpoint between polls of the same pull — the whole point is to catch new mints promptly. Between unrelated requests, a short cache (10–30 s) is fine.

## Related — `/v2/items/master`

Not gacha-specific, but useful: the general item catalog at `GET /v2/items/master?limit=1000` (also public, no auth) contains every gacha item with full drop metadata. Filter `source == "Gacha"` to see the full pool; further filter by `dropConditions.seasons` including the current season to narrow to what's actually pullable right now.

Use this when a user asks "what can I pull right now?" or "show me the rarest thing in the summer gacha."

Full shape in [../fishing/drops.md](../fishing/drops.md) (same endpoint, same shape).
