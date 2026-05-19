# Template, Sign, And Submit

1. Load [../assets/repermit.template.json](../assets/repermit.template.json) and the normalized params. Respect the JSON typing rules from [params.md](params.md).
2. Confirm `chainId` is listed in `## Supported Chains` in [../SKILL.md](../SKILL.md), replace `<ADAPTER>` with that chain's listed adapter, then replace the remaining `<...>` placeholders. Keep the fixed protocol fields already present in the template unchanged.
3. If allowance for `input.token` to `typedData.domain.verifyingContract` is lower than `input.maxAmount`, the default suggestion is a standard ERC-20 `approve(typedData.domain.verifyingContract, input.maxAmount)` transaction first.
4. If you explicitly want a standing approval instead, `maxUint256` is often more convenient for repeat orders, but keep it opt-in rather than the default suggestion.
5. Sign `typedData` with any EIP-712-capable wallet or library. The signer must equal `swapper`.
6. Canonical CLI signing path with Foundry `cast`:

```sh
cast wallet sign --data --from-file ./typed-data.json
```

7. Canonical browser-wallet signing path:

```js
const signature = await ethereum.request({
  method: "eth_signTypedData_v4",
  params: [swapper, JSON.stringify(typedData)],
});
```

8. Submit this exact relay payload to `https://agents-sink.orbs.network/orders/new`:

```json
{
  "order": "<typedData.message>",
  "signature": "<full signature or { r, s, v }>",
  "status": "pending"
}
```

9. Relay accepts either one full signature hex string or the exact `{ "r": "...", "s": "...", "v": "..." }` object returned by the signer.
10. Send the signature exactly as returned. Do not split, normalize, or rewrite it.
11. Canonical CLI submit path:

```sh
sig=$(cast wallet sign --data --from-file ./typed-data.json --interactive)
jq -n --slurpfile typed ./typed-data.json --arg sig "$sig" \
  '{order: $typed[0].message, signature: $sig, status: "pending"}' \
  > ./relay-payload.json

curl -fsS -X POST 'https://agents-sink.orbs.network/orders/new' \
  -H 'content-type: application/json' \
  --data @./relay-payload.json
```

12. Canonical JavaScript submit path with `ethers` plus standard `fetch`:

```js
const relayPayload = {
  order: typedData.message,
  signature,
  status: "pending",
};

const response = await fetch("https://agents-sink.orbs.network/orders/new", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(relayPayload),
});

const body = await response.json();
const orderHash = body.orderHash ?? body.signedOrder?.hash ?? "";
```

13. Capture `orderHash` from the submit response when available. Canonical extraction:

```sh
curl -fsS -X POST 'https://agents-sink.orbs.network/orders/new' \
  -H 'content-type: application/json' \
  --data @./relay-payload.json \
  | jq -r '.orderHash // .signedOrder.hash // empty'
```

14. After an ambiguous relay failure such as a timeout or `5xx`, persist and reuse the exact populated `typedData` and signature for any retry. Do not rebuild the order from fresh balances or refreshed prices until you have resolved whether the previous submit created an order.
15. See [lifecycle.md](lifecycle.md) for follow-up query and cancellation flow.
16. See [examples.md](examples.md) for full payload examples.
