# What a lease costs and when the money moves

USDG is a six-decimal token, so one base unit is 0.000001 USDG. Offers quote
`rate_per_second` in base units. Every figure below is in those units unless it says USDG.

## The three moments

**Funding.** `lease()` or `fund_quote()` moves `rate_per_second × duration_seconds` into the
escrow contract in one transaction. That is the whole window, paid before the machine boots.
It is also the most the lease can ever cost.

**Metering.** The meter starts when access opens, which is after provisioning, and stops at
the release. Provisioning time is not charged.

**Settlement.** The escrow charges the metered seconds at the quoted rate, pays the node
operator its share, and returns the remainder to the paying wallet. A receipt is published
and its hash is committed on chain by a `LeaseFinalized` event.

Unused seconds come back at settlement, and settlement is triggered by the release. There is
no automatic refund for a lease left open: it bills to the end of its window.

## A settled example

Lease 34, RTX 6000 Ada, quoted at 222 base units per second for a 600-second window.

| Line | Base units | USDG |
|---|---|---|
| Deposit at funding (222 × 600) | 133,200 | 0.133200 |
| Metered runtime, 28 seconds (222 × 28) | 6,216 | 0.006216 |
| Returned to the wallet | 126,984 | 0.126984 |
| Paid to the node operator | 5,595 | 0.005595 |

Receipt `8f3e0c1d-391c-8510-9f77-ebc574905ffe`, settled by escrow
`0xfD4228eEEfC49e4b76A0CD40af9fdd546220B2FD` in transaction
`0x1de4eba627f8ffc6c0ce3f628b648c5f13d9f2815843d1fc6979a9b731309d88`. Fetch and check it with
`python3 scripts/verify_receipt.py --lease 34`, which scopes the id to that escrow. Two
retired escrows also published a lease 34, one of them a 1800-second run charged 0.399600
USDG, so the number on its own is not a reference to this run.

The lease held 4.7% of what it reserved. Plan against the deposit anyway, because the deposit
is what the wallet and the day's budget have to clear at the moment of funding.

## Ceilings

Two ceilings bound every lease. A call can lower either for itself and can never raise one.

| Setting | Default | Bounds |
|---|---|---|
| `PRISM_MAX_USDG` | 1 | The deposit of any single lease. |
| `PRISM_DAILY_BUDGET_USDG` | 5 | Everything funded in a rolling 24 hours. `0` removes it. |

A per-call cap above the daily one is refused at startup as the configuration mistake it is.

The ledger at `~/.prism/spend.json` (override with `PRISM_LEDGER_PATH`) records the
reservation before the transaction is signed, then settles it to the amount the escrow
actually pulled. A failure that never reached the chain hands the reservation back. A failure
that may have reached it keeps the reservation, so a bad hour on an RPC endpoint cannot let
one wallet fund escrow after escrow while the day reports nothing spent.

Every Prism client on the machine draws down the same file.

## Reading the receipt

| Field | Meaning |
|---|---|
| `charged_base_units` | Metered seconds at the quoted rate. |
| `refunded_base_units` | Returned to the paying wallet. Add the two for the deposit. |
| `provider_paid_base_units` | The node operator's share of the charge. |
| `runtime_seconds` | Seconds billed. `charged / runtime` reproduces the rate. |
| `credited_seconds` | Seconds held and **not** charged. Present on interrupted runs only. |
| `failure_class` | `interrupted`, `provisioning_timeout`, or null on a clean run. |
| `outcome` | `finalized`, `refunded`, or `disputed`. Only the first two are evidence. |
| `receipt_hash` | SHA-256 of the canonical payload, committed on chain. |

Canonicalisation is field-declaration order with absent fields dropped and `failure_class`
kept as an explicit null. Re-serialising with sorted keys produces a different digest and a
false mismatch. `scripts/verify_receipt.py` implements the rule; reuse it rather than hashing
the JSON as it arrived.

A `refunded` outcome means the run never produced a charge, most often
`provisioning_timeout`. Its receipt hash is self-consistency evidence: the `LeaseRefunded`
event carries a reason hash rather than the receipt hash.

## Public feed

- Index of every settled lease: `https://api.prismnetwork.tech/proof/index.json`
- One receipt: `https://api.prismnetwork.tech/proof/receipts/<receipt_id>.json`
- Human view: `https://prismnetwork.tech/proof`

As of 2026-09-10 the feed carries 300 settlements, 226 finalized and 74 refunded, across
seven card names: `RTX A6000` and `NVIDIA RTX A6000`, `RTX 6000Ada`, `RTX 5880Ada`, `L40S`,
`H100 PCIe` and `A40`. Nodes publish the model string themselves, so the same card appears
under more than one spelling and matching on it exactly will miss machines. A card in the
feed is a card the network has run, which is a weaker claim than a card that can be rented
today. Only `offers()` says what is rentable now.
