# Lifecycle

Use this file for relay query semantics, status polling, and exact-match cancellation.

## Query

1. Primary query path: `GET https://agents-sink.orbs.network/orders?hash=<orderHash>`.
2. Canonical response shape is a top-level object with an `orders` array. When you query by `hash`, use the first element of `orders`.
3. Canonical order status field is `.orders[0].metadata.status`. If it is absent, fall back to `.orders[0].status`.
4. Canonical chunk status field is `.orders[0].metadata.chunks[].status` when `metadata.chunks` exists.
5. Known non-terminal statuses are `pending` and `eligible`.
6. Known terminal statuses are `filled`, `completed`, `partially_completed`, `cancelled`, `expired`, `failed`, and `rejected`.
7. Cancellation can surface as `cancelled`, or `failed` with cancellation text, depending on relay normalization. Treat them as terminal outcomes after you have confirmed the onchain cancel transaction.
8. `partially_completed` means at least one chunk reached a successful final state and at least one chunk did not. Treat it as terminal for polling and report the chunk statuses.
9. Poll every 5 seconds until the order reaches a terminal status.
10. Canonical CLI query path:

```sh
curl -fsS "https://agents-sink.orbs.network/orders?hash=$order_hash" \
  | jq -r '.orders[0] | (.metadata.status // .status // "pending")'
```

11. Canonical CLI chunk-status path:

```sh
curl -fsS "https://agents-sink.orbs.network/orders?hash=$order_hash" \
  | jq -r '.orders[0].metadata.chunks[]?.status'
```

12. Secondary recovery path: `GET https://agents-sink.orbs.network/orders?swapper=<swapper>`. Use it only when you do not have `orderHash` yet or need to recover from an ambiguous submit response.
13. If you query by `swapper` and receive multiple orders, disambiguate by `hash` before polling or reporting status.

## Cancel

1. Cancellation is exact-match and onchain. It invalidates only the digest you pass, not every order by the same `swapper`.
2. Always keep the exact populated `typedData` used for signing and submission. The cancellation digest must be derived from that same populated payload.
3. Compute the EIP-712 digest for that exact populated `typedData`, then call `cancel(bytes32[] digests)` on `typedData.domain.verifyingContract` from `swapper` with `[digest]`.
4. Canonical CLI cancel call once you already have the digest:

```sh
cast send "$typed_data_domain_verifying_contract" 'cancel(bytes32[])' "[$order_digest]"
```

5. Canonical JavaScript cancel path with `ethers` once you already have the digest:

```js
import { Contract } from "ethers";

const repermit = new Contract(typedData.domain.verifyingContract, ["function cancel(bytes32[] digests)"], signer);

const tx = await repermit.cancel([orderDigest]);
await tx.wait();
```

6. In both examples, the contract address is the exact `typedData.domain.verifyingContract` value from the submitted order.
7. Use the same signer that created the order. The caller must equal `swapper`.
8. After the cancel transaction is confirmed, keep polling the relay query endpoint until the order reaches a terminal status.
