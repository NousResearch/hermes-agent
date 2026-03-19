import test from "node:test";
import assert from "node:assert/strict";

import {
  KaspaWalletClient,
  buildCandidateUtxoPlans,
  buildRawDirectedSpendTransaction,
  buildRawSelfSpendTransaction,
  deriveWalletIdentity,
  makeOutpointKey,
  normalizeFeePolicy,
  normalizeSendState,
  previewRawDirectedSpend,
  previewRawSelfSpend,
  reconcileSendState,
  selectFeeRateFromEstimate,
  selectDirectedRawEntries,
  selectContextualRawEntries,
  shouldCompactSend,
} from "../lib/kaspa_wallet.js";

const TEST_SEED =
  "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
const TEST_IDENTITY = deriveWalletIdentity(TEST_SEED, "mainnet");
const ALT_IDENTITY = deriveWalletIdentity(
  "legal winner thank year wave sausage worth useful legal winner thank yellow",
  "mainnet"
);
const hexId = (char) => String(char).repeat(64);

function utxo({ txId, index, amount, blockDaaScore = 1, isCoinbase = false }) {
  return {
    address: TEST_IDENTITY.address,
    outpoint: {
      transactionId: txId,
      index,
    },
    amount: BigInt(amount),
    blockDaaScore: BigInt(blockDaaScore),
    isCoinbase,
    scriptPublicKey: TEST_IDENTITY.scriptPublicKey,
  };
}

test("normalizeFeePolicy defaults invalid values back to priority", () => {
  assert.equal(normalizeFeePolicy("priority"), "priority");
  assert.equal(normalizeFeePolicy("NORMAL"), "normal");
  assert.equal(normalizeFeePolicy(""), "priority");
  assert.equal(normalizeFeePolicy("weird"), "priority");
  assert.equal(normalizeFeePolicy("bogus", "low"), "low");
});

test("selectFeeRateFromEstimate maps low normal priority and auto to live node buckets", () => {
  const estimate = {
    estimate: {
      priorityBucket: {
        feerate: 6.393,
        estimatedSeconds: 0.1,
      },
      normalBuckets: [
        {
          feerate: 1.913,
          estimatedSeconds: 3.7,
        },
        {
          feerate: 1.5,
          estimatedSeconds: 8.5,
        },
      ],
      lowBuckets: [
        {
          feerate: 1.15,
          estimatedSeconds: 17.1,
        },
      ],
    },
  };

  assert.equal(selectFeeRateFromEstimate(estimate, "low"), 1.15);
  assert.equal(selectFeeRateFromEstimate(estimate, "normal"), 1.913);
  assert.equal(selectFeeRateFromEstimate(estimate, "priority"), 6.393);
  assert.equal(selectFeeRateFromEstimate(estimate, "auto"), 1.913);
  assert.equal(
    selectFeeRateFromEstimate(estimate, "auto", { autoTargetSeconds: 2 }),
    6.393
  );
});

test("previewRawSelfSpend charges more fee when the selected fee rate is higher", () => {
  const entries = [utxo({ txId: hexId("a"), index: 0, amount: 1_000_000 })];
  const payloadBytes = new Uint8Array(Buffer.from("hello", "utf8"));

  const normalPreview = previewRawSelfSpend({
    entries,
    payloadBytes,
    networkId: TEST_IDENTITY.networkId,
    scriptPublicKey: TEST_IDENTITY.scriptPublicKey,
    feeRateSompiPerGram: 2,
  });
  const priorityPreview = previewRawSelfSpend({
    entries,
    payloadBytes,
    networkId: TEST_IDENTITY.networkId,
    scriptPublicKey: TEST_IDENTITY.scriptPublicKey,
    feeRateSompiPerGram: 6,
  });

  assert.ok(priorityPreview.fee > normalPreview.fee);
  assert.ok(priorityPreview.outputAmount < normalPreview.outputAmount);
});

test("KaspaWalletClient caches fee estimates and falls back to the selected policy default", async () => {
  let nowMs = 10_000;
  let feeEstimateCalls = 0;
  const client = new KaspaWalletClient({
    seedPhrase: TEST_SEED,
    nodeUrl: "wss://example.invalid",
    network: "mainnet",
    feePolicy: "priority",
    feeEstimateTtlMs: 5_000,
    nowFn: () => nowMs,
  });

  client.rpc = {
    getFeeEstimate: async () => {
      feeEstimateCalls += 1;
      return {
        estimate: {
          priorityBucket: { feerate: 6.2, estimatedSeconds: 0.2 },
          normalBuckets: [{ feerate: 1.8, estimatedSeconds: 4.5 }],
          lowBuckets: [{ feerate: 1.1, estimatedSeconds: 15 }],
        },
      };
    },
  };

  assert.equal(await client.resolveFeeRate("priority"), 6.2);
  assert.equal(await client.resolveFeeRate("priority"), 6.2);
  assert.equal(feeEstimateCalls, 1);

  nowMs += 6_000;
  assert.equal(await client.resolveFeeRate("normal"), 1.8);
  assert.equal(feeEstimateCalls, 2);

  client.rpc.getFeeEstimate = async () => {
    throw new Error("fee estimates unavailable");
  };
  assert.equal(await client.resolveFeeRate("priority"), 6.2);

  const fallbackClient = new KaspaWalletClient({
    seedPhrase: TEST_SEED,
    nodeUrl: "wss://example.invalid",
    network: "mainnet",
    feePolicy: "priority",
  });
  fallbackClient.rpc = {
    getFeeEstimate: async () => {
      throw new Error("offline");
    },
  };
  assert.equal(await fallbackClient.resolveFeeRate("priority"), 6);
  assert.equal(await fallbackClient.resolveFeeRate("normal"), 2);
  assert.equal(await fallbackClient.resolveFeeRate("low"), 1);
});

test("buildCandidateUtxoPlans prefers tracked pending self-change before confirmed prefixes", () => {
  const pending = utxo({
    txId: "pending-a",
    index: 0,
    amount: 6000,
    blockDaaScore: 0,
  });
  const matureLarge = utxo({ txId: "confirmed-b", index: 0, amount: 4000 });
  const matureSmall = utxo({ txId: "confirmed-c", index: 1, amount: 2000 });

  const plans = buildCandidateUtxoPlans({
    trackedPendingUtxos: [pending],
    matureUtxos: [matureSmall, matureLarge],
    maxConfirmedPlans: 3,
  });

  assert.equal(plans[0].name, "pending-single");
  assert.equal(makeOutpointKey(plans[0].entries[0]), "pending-a:0");
  assert.equal(plans[1].name, "confirmed-1");
  assert.equal(makeOutpointKey(plans[1].entries[0]), "confirmed-b:0");
  assert.equal(plans[2].name, "confirmed-2");
});

test("reconcileSendState keeps recent invisible pending outputs but drops stale reservations", () => {
  const nowMs = 5_000;
  const state = normalizeSendState({
    reserved_outpoints: [
      { key: "old:0", reserved_at_ms: nowMs - 200_000 },
      { key: "live:1", reserved_at_ms: nowMs - 100 },
    ],
    pending_outputs: [
      {
        key: "pending-visible:0",
        tx_id: "pending-visible",
        index: 0,
        amount: "1000",
        created_ms: nowMs - 100,
      },
      {
        key: "pending-invisible:0",
        tx_id: "pending-invisible",
        index: 0,
        amount: "1000",
        created_ms: nowMs - 100,
      },
    ],
    last_compaction_ms: 99,
  });

  const reconciled = reconcileSendState(
    state,
    [
      utxo({ txId: "live", index: 1, amount: 1200 }),
      utxo({
        txId: "pending-visible",
        index: 0,
        amount: 1000,
        blockDaaScore: 0,
      }),
    ],
    { nowMs }
  );

  assert.deepEqual(
    reconciled.reserved_outpoints.map((entry) => entry.key),
    ["live:1"]
  );
  assert.deepEqual(
    reconciled.pending_outputs.map((entry) => entry.key).sort(),
    ["pending-invisible:0", "pending-visible:0"]
  );
});

test("reconcileSendState keeps recent reservations for spent outputs and prunes matching pending outputs", () => {
  const nowMs = 5_000;
  const state = normalizeSendState({
    reserved_outpoints: [{ key: "pending-spent:0", reserved_at_ms: nowMs - 100 }],
    pending_outputs: [
      {
        key: "pending-spent:0",
        tx_id: "pending-spent",
        index: 0,
        amount: "1000",
        created_ms: nowMs - 100,
      },
      {
        key: "pending-live:0",
        tx_id: "pending-live",
        index: 0,
        amount: "1200",
        created_ms: nowMs - 100,
      },
    ],
  });

  const reconciled = reconcileSendState(
    state,
    [
      utxo({
        txId: "pending-live",
        index: 0,
        amount: 1200,
        blockDaaScore: 0,
      }),
    ],
    { nowMs }
  );

  assert.deepEqual(
    reconciled.reserved_outpoints.map((entry) => entry.key),
    ["pending-spent:0"]
  );
  assert.deepEqual(
    reconciled.pending_outputs.map((entry) => entry.key),
    ["pending-live:0"]
  );
});

test("reconcileSendState removes tracked pending outputs once they mature", () => {
  const state = normalizeSendState({
    pending_outputs: [
      {
        key: "ready:0",
        tx_id: "ready",
        index: 0,
        amount: "1000",
        created_ms: 10,
      },
    ],
  });

  const reconciled = reconcileSendState(state, [
    utxo({
      txId: "ready",
      index: 0,
      amount: 1000,
      blockDaaScore: 25,
    }),
  ]);

  assert.equal(reconciled.pending_outputs.length, 0);
});

test("shouldCompactSend matches KaChat-style no-pending multi-utxo condition", () => {
  assert.equal(
    shouldCompactSend({
      matureUtxos: [utxo({ txId: "a", index: 0, amount: 1 }), utxo({ txId: "b", index: 0, amount: 1 })],
      trackedPendingUtxos: [],
      lastCompactionMs: 0,
      nowMs: 20_000,
      threshold: 3,
      cooldownMs: 1_000,
    }),
    false
  );

  assert.equal(
    shouldCompactSend({
      matureUtxos: [
        utxo({ txId: "a", index: 0, amount: 1 }),
        utxo({ txId: "b", index: 0, amount: 1 }),
        utxo({ txId: "c", index: 0, amount: 1 }),
      ],
      trackedPendingUtxos: [],
      lastCompactionMs: 0,
      nowMs: 20_000,
      threshold: 3,
      cooldownMs: 1_000,
    }),
    true
  );

  assert.equal(
    shouldCompactSend({
      matureUtxos: [
        utxo({ txId: "a", index: 0, amount: 1 }),
        utxo({ txId: "b", index: 0, amount: 1 }),
        utxo({ txId: "c", index: 0, amount: 1 }),
      ],
      trackedPendingUtxos: [utxo({ txId: "pending", index: 0, amount: 2, blockDaaScore: 0 })],
      lastCompactionMs: 0,
      nowMs: 20_000,
      threshold: 3,
      cooldownMs: 1_000,
    }),
    false
  );
});

test("raw self-spend preview and signing preserve a dust-safe self output", () => {
  const entry = utxo({ txId: hexId("a"), index: 0, amount: 50_000_000 });
  const payloadBytes = new Uint8Array(Buffer.from("hello raw"));

  const preview = previewRawSelfSpend({
    entries: [entry],
    payloadBytes,
    networkId: TEST_IDENTITY.networkId,
    scriptPublicKey: TEST_IDENTITY.scriptPublicKey,
  });

  assert.equal(preview.inputCount, 1);
  assert.equal(preview.usesPendingInputs, false);
  assert.ok(preview.outputAmount > 10_000n);
  assert.ok(preview.fee > 0n);

  const signed = buildRawSelfSpendTransaction({
    entries: [entry],
    payloadBytes,
    networkId: TEST_IDENTITY.networkId,
    scriptPublicKey: TEST_IDENTITY.scriptPublicKey,
    privateKey: TEST_IDENTITY.privateKey,
  });

  assert.equal(String(signed.transaction.outputs[0].value), String(preview.outputAmount));
  assert.match(String(signed.transaction.inputs[0].signatureScript), /^[0-9a-f]+$/i);
});

test("raw directed spend preview and signing preserve recipient amount with change", () => {
  const entry = utxo({ txId: hexId("1"), index: 0, amount: 90_000_000 });
  const amountSompi = 20_000_000n;

  const preview = previewRawDirectedSpend({
    entries: [entry],
    amountSompi,
    payloadBytes: new Uint8Array(Buffer.from("hello direct")),
    networkId: TEST_IDENTITY.networkId,
    destinationScriptPublicKey: ALT_IDENTITY.scriptPublicKey,
    changeScriptPublicKey: TEST_IDENTITY.scriptPublicKey,
  });

  assert.equal(preview.inputCount, 1);
  assert.equal(preview.usesPendingInputs, false);
  assert.ok(preview.fee > 0n);
  assert.ok(preview.changeAmount > 10_000n);

  const signed = buildRawDirectedSpendTransaction({
    entries: [entry],
    amountSompi,
    payloadBytes: new Uint8Array(Buffer.from("hello direct")),
    networkId: TEST_IDENTITY.networkId,
    destinationScriptPublicKey: ALT_IDENTITY.scriptPublicKey,
    changeScriptPublicKey: TEST_IDENTITY.scriptPublicKey,
    privateKey: TEST_IDENTITY.privateKey,
  });

  assert.equal(String(signed.transaction.outputs[0].value), String(amountSompi));
  assert.equal(String(signed.transaction.outputs[1].value), String(preview.changeAmount));
});

test("contextual raw selection prefers a single tracked pending self-change when it covers the fee", () => {
  const trackedPending = utxo({
    txId: hexId("b"),
    index: 0,
    amount: 60_000_000,
    blockDaaScore: 0,
  });
  const mature = utxo({ txId: hexId("c"), index: 0, amount: 30_000_000 });

  const selected = selectContextualRawEntries({
    availablePendingUtxos: [trackedPending],
    trackedPendingUtxos: [trackedPending],
    matureUtxos: [mature],
    payloadBytes: new Uint8Array(Buffer.from("hello pending")),
    networkId: TEST_IDENTITY.networkId,
    scriptPublicKey: TEST_IDENTITY.scriptPublicKey,
  });

  assert.deepEqual(selected.map((entry) => makeOutpointKey(entry)), [`${hexId("b")}:0`]);
});

test("directed raw selection uses the smallest largest-first prefix that covers amount plus fee", () => {
  const pending = utxo({ txId: hexId("2"), index: 0, amount: 25_000_000, blockDaaScore: 0 });
  const mature = utxo({ txId: hexId("3"), index: 0, amount: 70_000_000 });
  const selected = selectDirectedRawEntries({
    availablePendingUtxos: [pending],
    trackedPendingUtxos: [],
    matureUtxos: [mature],
    amountSompi: 20_000_000n,
    payloadBytes: new Uint8Array(Buffer.from("hello direct")),
    networkId: TEST_IDENTITY.networkId,
    destinationScriptPublicKey: TEST_IDENTITY.scriptPublicKey,
    changeScriptPublicKey: TEST_IDENTITY.scriptPublicKey,
  });

  assert.deepEqual(selected.map((entry) => makeOutpointKey(entry)), [`${hexId("3")}:0`]);
});

test("directed raw selection prefers a tracked pending self-change when one output covers the spend", () => {
  const trackedPending = utxo({
    txId: hexId("a"),
    index: 0,
    amount: 60_000_000,
    blockDaaScore: 0,
  });
  const mature = utxo({ txId: hexId("b"), index: 0, amount: 70_000_000 });

  const selected = selectDirectedRawEntries({
    availablePendingUtxos: [trackedPending],
    trackedPendingUtxos: [trackedPending],
    matureUtxos: [mature],
    amountSompi: 20_000_000n,
    payloadBytes: new Uint8Array(Buffer.from("hello direct pending")),
    networkId: TEST_IDENTITY.networkId,
    destinationScriptPublicKey: ALT_IDENTITY.scriptPublicKey,
    changeScriptPublicKey: TEST_IDENTITY.scriptPublicKey,
  });

  assert.deepEqual(selected.map((entry) => makeOutpointKey(entry)), [`${hexId("a")}:0`]);
});

test("direct sends use the raw directed path with string destination addresses", async () => {
  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
      });
      this.identity = TEST_IDENTITY;
      this.utxoContext = {};
      this.rpc = {};
      this.directCalls = [];
      this.rebuildCalls = 0;
    }

    async _rebuildUtxoContext() {
      this.rebuildCalls += 1;
    }

    _loadSendContext() {
      return {
        trackedPendingUtxos: [],
        availablePendingUtxos: [],
        availableMatureUtxos: [utxo({ txId: hexId("4"), index: 0, amount: 90_000_000 })],
      };
    }

    async _submitRawDirectedSpend({ entries, destinationAddress, amountSompi }) {
      this.directCalls.push({
        entries: entries.map((entry) => makeOutpointKey(entry)),
        destinationAddress,
        amountSompi,
      });
      return {
        txId: "tx-ok",
        inputCount: entries.length,
        usesPendingInputs: false,
      };
    }
  }

  const wallet = new ProbeWalletClient();
  const result = await wallet._sendPayloadTransaction({
    destinationAddress: ALT_IDENTITY.address,
    amountSompi: 20_000_000n,
    payloadBytes: new Uint8Array(),
    strategy: "direct",
  });

  assert.equal(result.txId, "tx-ok");
  assert.equal(wallet.rebuildCalls, 1);
  assert.equal(wallet.directCalls.length, 1);
  assert.equal(typeof wallet.directCalls[0].destinationAddress, "string");
  assert.equal(wallet.directCalls[0].amountSompi, 20_000_000n);
});

test("direct sends can spend tracked pending change through the raw directed path", async () => {
  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
      });
      this.identity = TEST_IDENTITY;
      this.utxoContext = {};
      this.rpc = {};
      this.directCalls = [];
    }

    async _rebuildUtxoContext() {}

    _loadSendContext() {
      return {
        trackedPendingUtxos: [utxo({ txId: hexId("5"), index: 0, amount: 60_000_000, blockDaaScore: 0 })],
        availablePendingUtxos: [utxo({ txId: hexId("5"), index: 0, amount: 60_000_000, blockDaaScore: 0 })],
        availableMatureUtxos: [utxo({ txId: hexId("6"), index: 0, amount: 30_000_000 })],
      };
    }

    async _submitRawDirectedSpend({ entries }) {
      this.directCalls.push(entries.map((entry) => makeOutpointKey(entry)));
      return {
        txId: "tx-context",
        inputCount: entries.length,
        usesPendingInputs: true,
      };
    }
  }

  const wallet = new ProbeWalletClient();
  const result = await wallet._sendPayloadTransaction({
    destinationAddress: ALT_IDENTITY.address,
    amountSompi: 20_000_000n,
    payloadBytes: new Uint8Array(),
    strategy: "direct",
  });

  assert.equal(wallet.directCalls.length, 1);
  assert.deepEqual(wallet.directCalls[0], [`${hexId("5")}:0`]);
  assert.equal(result.txId, "tx-context");
});

test("contextual sends use the raw self-spend path and can spend tracked pending outputs", async () => {
  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
      });
      this.identity = TEST_IDENTITY;
      this.utxoContext = {};
      this.rpc = {};
      this.rawCalls = [];
      this.sendState = normalizeSendState({
        reserved_outpoints: [{ key: "spent:0", reserved_at_ms: 10 }],
        pending_outputs: [
          {
            key: "pending-1:0",
            tx_id: "pending-1",
            index: 0,
            amount: "1000",
            created_ms: 11,
          },
        ],
      });
    }

    async _rebuildUtxoContext() {}

    _loadSendContext() {
      return {
        trackedPendingUtxos: [
          utxo({ txId: hexId("d"), index: 0, amount: 60_000_000, blockDaaScore: 0 }),
        ],
        availablePendingUtxos: [
          utxo({ txId: hexId("d"), index: 0, amount: 60_000_000, blockDaaScore: 0 }),
        ],
        availableMatureUtxos: [utxo({ txId: hexId("e"), index: 0, amount: 30_000_000 })],
      };
    }

    async _submitRawSelfSpend(entries, payloadBytes) {
      this.rawCalls.push({
        entries: entries.map((entry) => makeOutpointKey(entry)),
        payloadSize: payloadBytes.length,
      });
      return {
        txId: "tx-pending",
        inputCount: entries.length,
        usesPendingInputs: true,
      };
    }
  }

  const wallet = new ProbeWalletClient();
  const result = await wallet._sendPayloadTransaction({
    destinationAddress: "kaspa:qselfaddressxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    amountSompi: 1000n,
    payloadBytes: new Uint8Array(),
    strategy: "contextual",
  });

  assert.equal(result.txId, "tx-pending");
  assert.equal(wallet.rawCalls.length, 1);
  assert.deepEqual(wallet.rawCalls[0].entries, [`${hexId("d")}:0`]);
});

test("wallet rebuild hydrates reserved spends and pending self-change from mempool", async () => {
  const identity = TEST_IDENTITY;

  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
      });
      this.identity = identity;
      this.rpc = {
        getMempoolEntriesByAddresses: async () => ({
          entries: [
            {
              address: identity.address,
              sending: [
                {
                  transaction: {
                    inputs: [
                      {
                        previousOutpoint: {
                          transactionId: "prev-1",
                          index: 0,
                        },
                      },
                    ],
                    outputs: [
                      {
                        value: 1234n,
                        scriptPublicKey: identity.scriptPublicKey,
                      },
                    ],
                    verboseData: {
                      transactionId: "pending-1",
                    },
                  },
                },
              ],
            },
          ],
        }),
      };
      this.utxoContext = {
        matureLength: 1,
        getMatureRange() {
          return [utxo({ txId: "prev-1", index: 0, amount: 5000 })];
        },
        getPending() {
          return [];
        },
      };
    }
  }

  const wallet = new ProbeWalletClient();
  await wallet._hydrateMempoolSendState();
  const context = wallet._loadSendContext();

  assert.deepEqual(
    wallet.sendState.reserved_outpoints.map((entry) => entry.key),
    ["prev-1:0"]
  );
  assert.deepEqual(
    wallet.sendState.pending_outputs.map((entry) => entry.key),
    ["pending-1:0"]
  );
  assert.equal(context.trackedPendingUtxos.length, 1);
  assert.equal(context.availablePendingUtxos.length, 1);
  assert.equal(makeOutpointKey(context.availablePendingUtxos[0]), "pending-1:0");
});

test("wallet mempool hydration clears stale persisted send state when mempool is empty", async () => {
  const identity = TEST_IDENTITY;

  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
      });
      this.identity = identity;
      this.rpc = {
        getMempoolEntriesByAddresses: async () => ({
          entries: [
            {
              address: identity.address,
              sending: [],
              receiving: [],
            },
          ],
        }),
      };
      this.sendState = normalizeSendState({
        reserved_outpoints: [{ key: "stale:0", reserved_at_ms: 10 }],
        pending_outputs: [
          { key: "stale-pending:0", tx_id: "stale-pending", index: 0, amount: "42", created_ms: 11 },
        ],
      });
    }
  }

  const wallet = new ProbeWalletClient();
  await wallet._hydrateMempoolSendState();

  assert.deepEqual(wallet.sendState.reserved_outpoints, []);
  assert.deepEqual(wallet.sendState.pending_outputs, []);
});

test("wallet mempool hydration preserves fresh local pending state until mempool catches up", async () => {
  const identity = TEST_IDENTITY;

  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
        nowFn: () => 20_000,
      });
      this.identity = identity;
      this.rpc = {
        getMempoolEntriesByAddresses: async () => ({
          entries: [
            {
              address: identity.address,
              sending: [],
              receiving: [],
            },
          ],
        }),
      };
      this.sendState = normalizeSendState({
        reserved_outpoints: [{ key: "fresh-input:0", reserved_at_ms: 10_000 }],
        pending_outputs: [
          {
            key: "fresh-output:0",
            tx_id: "fresh-output",
            index: 0,
            amount: "42",
            created_ms: 10_000,
            observed_in_mempool: false,
          },
        ],
      });
    }
  }

  const wallet = new ProbeWalletClient();
  await wallet._hydrateMempoolSendState();

  assert.deepEqual(
    wallet.sendState.reserved_outpoints.map((entry) => entry.key),
    ["fresh-input:0"]
  );
  assert.deepEqual(
    wallet.sendState.pending_outputs.map((entry) => entry.key),
    ["fresh-output:0"]
  );
});

test("wallet mempool hydration preserves fresh observed local pending state until mempool catches up", async () => {
  const identity = TEST_IDENTITY;

  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
        nowFn: () => 20_000,
      });
      this.identity = identity;
      this.rpc = {
        getMempoolEntriesByAddresses: async () => ({
          entries: [
            {
              address: identity.address,
              sending: [],
              receiving: [],
            },
          ],
        }),
      };
      this.sendState = normalizeSendState({
        reserved_outpoints: [{ key: "fresh-input:0", reserved_at_ms: 10_000 }],
        pending_outputs: [
          {
            key: "fresh-output:0",
            tx_id: "fresh-output",
            index: 0,
            amount: "42",
            created_ms: 10_000,
            observed_in_mempool: true,
          },
        ],
      });
    }
  }

  const wallet = new ProbeWalletClient();
  await wallet._hydrateMempoolSendState();

  assert.deepEqual(
    wallet.sendState.reserved_outpoints.map((entry) => entry.key),
    ["fresh-input:0"]
  );
  assert.deepEqual(
    wallet.sendState.pending_outputs.map((entry) => entry.key),
    ["fresh-output:0"]
  );
});

test("wallet mempool hydration prunes a pending self-change once a newer mempool tx spends it", async () => {
  const identity = TEST_IDENTITY;

  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
        nowFn: () => 20_000,
      });
      this.identity = identity;
      this.rpc = {
        getMempoolEntriesByAddresses: async () => ({
          entries: [
            {
              address: identity.address,
              sending: [
                {
                  transaction: {
                    inputs: [],
                    outputs: [
                      {
                        value: 1234n,
                        scriptPublicKey: identity.scriptPublicKey,
                      },
                    ],
                    verboseData: {
                      transactionId: "pending-1",
                    },
                  },
                },
                {
                  transaction: {
                    inputs: [
                      {
                        previousOutpoint: {
                          transactionId: "pending-1",
                          index: 0,
                        },
                      },
                    ],
                    outputs: [
                      {
                        value: 1200n,
                        scriptPublicKey: identity.scriptPublicKey,
                      },
                    ],
                    verboseData: {
                      transactionId: "pending-2",
                    },
                  },
                },
              ],
              receiving: [],
            },
          ],
        }),
      };
      this.sendState = normalizeSendState({
        reserved_outpoints: [{ key: "pending-1:0", reserved_at_ms: 19_000 }],
        pending_outputs: [
          {
            key: "pending-1:0",
            tx_id: "pending-1",
            index: 0,
            amount: "1234",
            created_ms: 19_000,
            observed_in_mempool: true,
          },
          {
            key: "pending-2:0",
            tx_id: "pending-2",
            index: 0,
            amount: "1200",
            created_ms: 19_500,
            observed_in_mempool: false,
          },
        ],
      });
    }
  }

  const wallet = new ProbeWalletClient();
  await wallet._hydrateMempoolSendState();

  assert.deepEqual(
    wallet.sendState.reserved_outpoints.map((entry) => entry.key),
    ["pending-1:0"]
  );
  assert.deepEqual(
    wallet.sendState.pending_outputs.map((entry) => entry.key),
    ["pending-2:0"]
  );
});

test("local pending outputs are immediately reusable as synthetic chained inputs", async () => {
  const identity = TEST_IDENTITY;

  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
      });
      this.identity = identity;
      this.utxoContext = {
        matureLength: 0,
        getMatureRange() {
          return [];
        },
        getPending() {
          return [];
        },
      };
      this.sendState = normalizeSendState({
        pending_outputs: [
          {
            key: `${hexId("f")}:0`,
            tx_id: hexId("f"),
            index: 0,
            amount: "1000",
            created_ms: 12_000,
            observed_in_mempool: false,
          },
        ],
      });
      this.nowFn = () => 12_100;
    }
  }

  const wallet = new ProbeWalletClient();
  const context = wallet._loadSendContext();

  assert.equal(context.trackedPendingUtxos.length, 1);
  assert.equal(context.availablePendingUtxos.length, 1);
  assert.equal(makeOutpointKey(context.availablePendingUtxos[0]), `${hexId("f")}:0`);
});

test("direct sends can compact before retrying a fragmented raw spend", async () => {
  class ProbeWalletClient extends KaspaWalletClient {
    constructor() {
      super({
        seedPhrase: TEST_SEED,
        nodeUrl: "ws://node.invalid",
        network: "mainnet",
      });
      this.identity = TEST_IDENTITY;
      this.utxoContext = {};
      this.rpc = {};
      this.didCompact = false;
      this.compactionCount = 0;
    }

    async _rebuildUtxoContext() {}

    _loadSendContext() {
      return {
        trackedPendingUtxos: [],
        availablePendingUtxos: [],
        availableMatureUtxos: [
          utxo({ txId: hexId("7"), index: 0, amount: 25_000_000 }),
          utxo({ txId: hexId("8"), index: 0, amount: 25_000_000 }),
          utxo({ txId: hexId("9"), index: 0, amount: 25_000_000 }),
        ],
      };
    }

    async _submitRawDirectedSpend() {
      if (!this.didCompact) {
        throw new Error("insufficient funds");
      }
      return {
        txId: "tx-after-compact",
        inputCount: 1,
        usesPendingInputs: true,
      };
    }

    async _compactSpendableUtxosRaw() {
      this.didCompact = true;
      this.compactionCount += 1;
    }
  }

  const wallet = new ProbeWalletClient();
  const result = await wallet._sendPayloadTransaction({
    destinationAddress: ALT_IDENTITY.address,
    amountSompi: 20_000_000n,
    payloadBytes: new Uint8Array(),
    strategy: "direct",
  });

  assert.equal(wallet.compactionCount, 1);
  assert.equal(result.txId, "tx-after-compact");
  assert.equal(result.usedPendingInput, true);
});
