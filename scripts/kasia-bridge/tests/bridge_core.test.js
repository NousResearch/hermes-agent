import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { KasiaBridgeCore } from "../lib/bridge_core.js";
import { deriveWalletIdentity } from "../lib/kaspa_wallet.js";
import {
  buildBroadcastTransactionPayload,
  buildContextualMessageTransactionPayload,
  buildHandshakePayload,
  buildHandshakeTransactionPayload,
  HANDSHAKE_PREFIX,
} from "../lib/protocol.js";

const VALID_CONTACT_ADDRESS =
  "kaspa:qr9ssytsv8gsw5wrmp4lhnxdhprlg5g9ct9m37ngq9x9nhr7wm3ycxcrzs5e7";

class FakeWalletClient {
  constructor() {
    this.isConnected = true;
    this.nodeUrl = "ws://node.invalid";
    this.sentTransactions = [];
    this.loadedSendState = null;
    this.sendState = {
      reserved_outpoints: [],
      pending_outputs: [],
      last_compaction_ms: 0,
    };
    this.info = {
      address: "kaspa:qpeerwalletaddressxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      publicKeyHex: "02".padEnd(66, "1"),
      privateKeyHex: "11".repeat(32),
      network: "mainnet",
    };
    this.balanceSnapshot = {
      onChainBalanceSompi: 500000000n,
      availableMatureBalanceSompi: 500000000n,
      availablePendingBalanceSompi: 0n,
      trackedPendingBalanceSompi: 0n,
      matureUtxoCount: 1,
      pendingUtxoCount: 0,
      trackedPendingUtxoCount: 0,
      updatedAtMs: 123,
    };
  }

  async init() {
    return this.info;
  }

  async close() {}

  getWalletInfo() {
    return this.info;
  }

  getNodeUrl() {
    return this.nodeUrl;
  }

  getBalanceSnapshot() {
    return this.balanceSnapshot;
  }

  async switchNodeUrl(nextNodeUrl) {
    this.nodeUrl = nextNodeUrl;
    this.isConnected = true;
    return this.info;
  }

  loadSendState(state) {
    this.loadedSendState = state;
    this.sendState = state || this.sendState;
  }

  exportSendState() {
    return this.sendState;
  }

  async hydrateSendState() {}

  canFitContextualPayload() {
    return true;
  }

  async getAddressMempoolEntries() {
    return { entries: [] };
  }

  async sendPayloadTransaction(payload) {
    this.sentTransactions.push(payload);
    return `tx-${this.sentTransactions.length}`;
  }
}

class PreflightChunkingWalletClient extends FakeWalletClient {
  constructor() {
    super();
    this.maxPayloadBytes = 420;
  }

  canFitContextualPayload(payloadBytes) {
    return payloadBytes.length <= this.maxPayloadBytes;
  }
}

class SlowWalletClient extends FakeWalletClient {
  async sendPayloadTransaction(payload) {
    this.sentTransactions.push(payload);
    await new Promise((resolve) => setTimeout(resolve, 25));
    return `tx-${this.sentTransactions.length}`;
  }
}

class FailingWalletClient extends FakeWalletClient {
  async sendPayloadTransaction(payload) {
    this.sentTransactions.push(payload);
    throw new Error("node offline");
  }
}

class HydratingWalletClient extends FakeWalletClient {
  async hydrateSendState() {
    this.sendState = {
      reserved_outpoints: [{ key: "mempool:0", reserved_at_ms: 21 }],
      pending_outputs: [
        { key: "mempool-pending:0", tx_id: "mempool-pending", index: 0, amount: "77", created_ms: 22 },
      ],
      last_compaction_ms: 0,
    };
  }
}

class FailoverWalletClient extends FakeWalletClient {
  constructor() {
    super();
    this.failedNodeUrls = new Set(["ws://node-primary.invalid"]);
  }

  async sendPayloadTransaction(payload) {
    if (this.failedNodeUrls.has(this.nodeUrl)) {
      throw new Error(`rpc connection failed for ${this.nodeUrl}`);
    }
    return await super.sendPayloadTransaction(payload);
  }
}

function response(jsonPayload) {
  return {
    ok: true,
    async json() {
      return jsonPayload;
    },
    async text() {
      return JSON.stringify(jsonPayload);
    },
  };
}

function mempoolResponseForAddress(address, transactions = []) {
  return {
    entries: [
      {
        address,
        sending: transactions.map((transaction) => ({
          transaction,
        })),
      },
    ],
  };
}

test("send rejects when there is no active conversation", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    fetchImpl: async () => response([]),
  });

  await bridge.init();

  await assert.rejects(
    bridge.send({
      chatId: VALID_CONTACT_ADDRESS,
      message: "hello",
    }),
    /No active Kasia conversation/
  );
});

test("preflight keeps long contextual messages as one Kasia part when they fit", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };

  const result = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "A".repeat(2000),
    waitMs: 250,
  });

  assert.equal(result.status, "submitted");
  assert.equal(result.partCount, 1);
  assert.deepEqual(result.txIds, ["tx-1"]);
  assert.equal(walletClient.sentTransactions.length, 1);
});

test("explicit target message chars still force chunking even when one payload would fit", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async () => response([]),
    contextualMessageTargetChars: 80,
    respectContextualMessageTarget: true,
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };

  const result = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "A".repeat(200),
    waitMs: 250,
  });

  assert.equal(result.status, "submitted");
  assert.equal(result.partCount > 1, true);
  assert.equal(walletClient.sentTransactions.length, result.partCount);
});

test("preflight chunking splits oversized contextual messages before send", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new PreflightChunkingWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };

  const result = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "A".repeat(700),
    waitMs: 250,
  });

  assert.equal(result.status, "submitted");
  assert.ok(result.partCount > 1);
  assert.equal(result.txIds.length, result.partCount);
  assert.equal(result.txId, result.txIds[result.txIds.length - 1]);
  assert.equal(walletClient.sentTransactions.length, result.partCount);
  assert.match(result.statusMessage, /waiting for indexer visibility/i);
});

test("multipart search does not skip viable Kasia chunk sizes", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new PreflightChunkingWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };

  const result = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "A".repeat(1336),
    waitMs: 250,
  });

  assert.equal(result.status, "submitted");
  assert.equal(result.partCount <= 8, true);
  assert.equal(result.partCount > 1, true);
  assert.equal(walletClient.sentTransactions.length, result.partCount);
});

test("non-size wallet send errors surface as failed send jobs", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FailingWalletClient(),
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };

  const result = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "hello",
    waitMs: 100,
  });

  assert.equal(result.status, "failed");
  assert.match(result.error, /node offline/);
});

test("send returns a queued job when waitMs expires before submission completes", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new SlowWalletClient(),
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };

  const queued = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "hello async kasia",
    waitMs: 1,
  });

  assert.ok(["queued", "submitting"].includes(queued.status));
  assert.ok(queued.jobId);

  const completed = await bridge.waitForSendJob(queued.jobId, 500);
  assert.equal(completed.status, "submitted");
  assert.equal(completed.partCount, 1);
  assert.equal(completed.completedParts, 1);
  assert.equal(completed.indexedParts, 0);
});

test("sync marks submitted jobs as processed once the indexer shows all tx ids", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const requestedUrls = [];
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async (url) => {
      requestedUrls.push(String(url));
      if (
        String(url).includes("/contextual-messages/by-sender") &&
        String(url).includes("alias=303031313232333334343535")
      ) {
        return response([
          {
            tx_id: "tx-1",
            block_time: 1234,
            message_payload: "ignored",
          },
        ]);
      }
      return response([]);
    },
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };

  const initial = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "hello indexed kasia",
    waitMs: 250,
  });
  assert.equal(initial.status, "submitted");

  await bridge.syncOnce();

  const processed = bridge.getSendJob(initial.jobId);
  assert.equal(processed.status, "processed");
  assert.equal(processed.indexedParts, 1);
  assert.equal(processed.indexedBlockTimeMs, 1234);
  assert.equal(processed.indexedMs != null, true);
  assert.match(processed.statusMessage, /visible through the kasia indexer/i);
  assert.ok(
    requestedUrls.some((url) => url.includes("alias=303031313232333334343535"))
  );
});

test("preflight rejects messages that exceed the Kasia multipart cap", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new PreflightChunkingWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async () => response([]),
    contextualMessageTargetChars: 80,
    maxMultipartParts: 2,
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };

  const result = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "B".repeat(500),
    waitMs: 50,
  });

  assert.equal(result.status, "rejected");
  assert.match(result.error, /caps Kasia sends at 2 parts/i);
  assert.equal(walletClient.sentTransactions.length, 0);
});

test("state persists across restart and preserves conversation cursors", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  bridge.state.conversations["kaspa:qcontactaddressxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"] = {
    conversation_id: "kasia:test",
    peer_address: "kaspa:qcontactaddressxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
  };
  bridge.state.cursors.handshakes_block_time = 100;
  bridge.state.last_sync_ms = 999;
  await bridge.close();

  const restarted = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    fetchImpl: async () => response([]),
  });
  await restarted.init();

  const conversation =
    restarted.state.conversations[
      "kaspa:qcontactaddressxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ];
  assert.equal(conversation.status, "active");
  assert.equal(conversation.last_context_block_time, 200);
  assert.equal(restarted.state.cursors.handshakes_block_time, 100);
});

test("bridge serializes concurrent state saves without temp-file collisions", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    fetchImpl: async () => response([]),
  });

  await bridge.init();

  bridge.state.send_jobs["job-1"] = {
    job_id: "job-1",
    chat_id: VALID_CONTACT_ADDRESS,
    status: "queued",
    created_ms: 1,
    updated_ms: 1,
    total_parts: 1,
    completed_parts: 0,
    tx_ids: [],
    last_tx_id: null,
    error: null,
    message_preview: "first",
  };
  const firstSave = bridge._saveState();

  bridge.state.send_jobs["job-2"] = {
    job_id: "job-2",
    chat_id: VALID_CONTACT_ADDRESS,
    status: "queued",
    created_ms: 2,
    updated_ms: 2,
    total_parts: 1,
    completed_parts: 0,
    tx_ids: [],
    last_tx_id: null,
    error: null,
    message_preview: "second",
  };
  const secondSave = bridge._saveState();

  await Promise.all([firstSave, secondSave]);

  const savedState = JSON.parse(
    await readFile(join(stateDir, "state.json"), "utf8")
  );
  assert.ok(savedState.send_jobs["job-1"]);
  assert.ok(savedState.send_jobs["job-2"]);
});

test("duplicate transactions are deduplicated by tx id", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  bridge.state.processed_tx_ids.push("tx-dup");
  await bridge.close();

  const stateJson = JSON.parse(
    await readFile(join(stateDir, "state.json"), "utf8")
  );
  assert.deepEqual(stateJson.processed_tx_ids, ["tx-dup"]);
});

test("retryable malformed handshakes are not marked processed", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const identity = deriveWalletIdentity(
    "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
    "mainnet"
  );
  walletClient.info = {
    address: identity.address,
    publicKeyHex: identity.publicKeyHex,
    privateKeyHex: identity.privateKeyHex,
    network: "mainnet",
  };
  const handshakePayload = buildHandshakePayload({
    alias: "001122334455",
    timestamp: 1,
    version: 1,
  });
  const messagePayload = buildHandshakeTransactionPayload({
    recipientAddress: identity.address,
    payload: handshakePayload,
  })
    .subarray(Buffer.byteLength(HANDSHAKE_PREFIX))
    .toString("hex");
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async (url) => {
      if (String(url).includes("/handshakes/by-receiver")) {
        return response([
          {
            tx_id: "tx-missing-sender",
            block_time: 123,
            message_payload: messagePayload,
          },
        ]);
      }
      return response([]);
    },
  });

  await bridge.init();
  await bridge._pollHandshakes();

  assert.equal(bridge.state.processed_tx_ids.includes("tx-missing-sender"), false);
  assert.equal(bridge.state.cursors.handshakes_block_time, 123);
});

test("wallet send state is loaded from disk and persisted back on save", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const firstWallet = new FakeWalletClient();

  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: firstWallet,
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  firstWallet.sendState = {
    reserved_outpoints: [{ key: "old:0", reserved_at_ms: 10 }],
    pending_outputs: [{ key: "new:1", tx_id: "new", index: 1, amount: "42", created_ms: 11 }],
    last_compaction_ms: 12,
  };
  await bridge.close();

  const restartedWallet = new FakeWalletClient();
  const restarted = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: restartedWallet,
    fetchImpl: async () => response([]),
  });

  await restarted.init();
  assert.deepEqual(restartedWallet.loadedSendState, firstWallet.sendState);
  assert.equal(restarted.health().pendingOutputCount, 1);
  assert.equal(restarted.health().reservedOutpointCount, 1);
});

test("bridge init rehydrates wallet send state after loading persisted disk state", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new HydratingWalletClient(),
    fetchImpl: async () => response([]),
  });

  await bridge.init();

  assert.equal(bridge.health().pendingOutputCount, 1);
  assert.equal(bridge.health().reservedOutpointCount, 1);
});

test("contextual polling encodes aliases for the live indexer query shape", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const requestedUrls = [];
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    fetchImpl: async (url) => {
      requestedUrls.push(String(url));
      return response([]);
    },
  });

  await bridge.init();
  bridge.state.conversations["kaspa:qcontactaddressxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"] = {
    conversation_id: "kasia:test",
    peer_address: "kaspa:qcontactaddressxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    our_alias: "001122334455",
    their_alias: "a1b2c3d4e5f6",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 0,
    pending_handshake: null,
  };

  await bridge.syncOnce();

  const contextualUrl = requestedUrls.find((url) =>
    url.includes("/contextual-messages/by-sender")
  );
  assert.ok(contextualUrl);
  assert.match(contextualUrl, /alias=613162326333643465356636/);
});

test("outbound handshake initiation persists state, deduplicates duplicates, and survives restart", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async () => response([]),
  });

  await bridge.init();

  const first = await bridge.initiateHandshake({
    chatId: VALID_CONTACT_ADDRESS,
    displayName: "Friendly Peer",
  });
  assert.equal(first.status, "sent");
  assert.equal(walletClient.sentTransactions.length, 1);

  const duplicate = await bridge.initiateHandshake({
    chatId: VALID_CONTACT_ADDRESS,
  });
  assert.equal(duplicate.status, "pending");
  assert.equal(walletClient.sentTransactions.length, 1);

  const retried = await bridge.initiateHandshake({
    chatId: VALID_CONTACT_ADDRESS,
    retry: true,
  });
  assert.equal(retried.status, "sent");
  assert.equal(walletClient.sentTransactions.length, 2);

  await bridge.close();

  const restarted = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    fetchImpl: async () => response([]),
  });
  await restarted.init();

  const conversation =
    restarted.state.conversations[VALID_CONTACT_ADDRESS];
  assert.equal(conversation.nickname, "Friendly Peer");
  assert.ok(conversation.pending_outbound_handshake?.tx_id);
});

test("kns resolution caches forward and reverse lookups and chat info prefers the KNS name", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const requests = [];
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    knsUrl: "https://kns.invalid/api/v1",
    walletClient: new FakeWalletClient(),
    fetchImpl: async (url) => {
      requests.push(String(url));
      if (String(url).includes("friend.kas/owner")) {
        return response({
          success: true,
          data: {
            asset: "friend.kas",
            owner: VALID_CONTACT_ADDRESS,
          },
        });
      }
      if (String(url).includes(`/primary-name/${encodeURIComponent(VALID_CONTACT_ADDRESS)}`)) {
        return response({
          success: true,
          data: {
            ownerAddress: VALID_CONTACT_ADDRESS,
            domain: { name: "friend.kas" },
          },
        });
      }
      return response([]);
    },
  });

  await bridge.init();

  const initiated = await bridge.initiateHandshake({
    chatId: "friend.kas",
  });
  assert.equal(initiated.chatId, VALID_CONTACT_ADDRESS);
  assert.equal(bridge.getChatInfo(VALID_CONTACT_ADDRESS).name, "friend.kas");

  await bridge.resolveTarget("friend.kas");
  assert.equal(
    requests.filter((value) => value.includes("friend.kas/owner")).length,
    1
  );
});

test("live contextual receive is deduplicated against later indexer backfill", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const identity = deriveWalletIdentity(
    "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
    "mainnet"
  );
  walletClient.info = {
    address: identity.address,
    publicKeyHex: identity.publicKeyHex,
    privateKeyHex: identity.privateKeyHex,
    network: "mainnet",
  };
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async (url) => {
      if (String(url).includes("/contextual-messages/by-sender")) {
        return response([
          {
            tx_id: "tx-live-1",
            block_time: 999,
            message_payload: "ignored",
          },
        ]);
      }
      return response([]);
    },
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 0,
    pending_handshake: null,
    pending_outbound_handshake: null,
  };

  const payload = buildContextualMessageTransactionPayload({
    recipientAddress: walletClient.info.address,
    alias: "aabbccddeeff",
    message: "hello over live path",
  });
  walletClient.getAddressMempoolEntries = async () =>
    mempoolResponseForAddress(VALID_CONTACT_ADDRESS, [
      {
        id: "tx-live-1",
        payload: payload.toString("utf8"),
      },
    ]);

  await bridge.syncOnce();
  const firstBatch = bridge.dequeueMessages();
  assert.equal(firstBatch.length, 1);
  assert.equal(firstBatch[0].body, "hello over live path");
  assert.ok(firstBatch[0].raw.delivery.liveObservedMs);
  assert.ok(firstBatch[0].raw.delivery.deliveredToHermesMs);

  await bridge.syncOnce();
  assert.equal(bridge.dequeueMessages().length, 0);
});

test("restart recovery keeps a live-delivered tx from being replayed by indexer catch-up", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const identity = deriveWalletIdentity(
    "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
    "mainnet"
  );
  walletClient.info = {
    address: identity.address,
    publicKeyHex: identity.publicKeyHex,
    privateKeyHex: identity.privateKeyHex,
    network: "mainnet",
  };
  const payload = buildContextualMessageTransactionPayload({
    recipientAddress: identity.address,
    alias: "aabbccddeeff",
    message: "hello before restart",
  });
  const fetchImpl = async (url) => {
    if (String(url).includes("/contextual-messages/by-sender")) {
      return response([
        {
          tx_id: "tx-live-restart",
          block_time: 777,
          message_payload: "ignored",
        },
      ]);
    }
    return response([]);
  };
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl,
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 0,
    pending_handshake: null,
    pending_outbound_handshake: null,
  };
  walletClient.getAddressMempoolEntries = async () =>
    mempoolResponseForAddress(VALID_CONTACT_ADDRESS, [
      {
        id: "tx-live-restart",
        payload: payload.toString("utf8"),
      },
    ]);

  await bridge.syncOnce();
  assert.equal(bridge.dequeueMessages().length, 1);
  await bridge.close();

  const restarted = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    fetchImpl,
  });
  await restarted.init();
  restarted.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 0,
    pending_handshake: null,
    pending_outbound_handshake: null,
  };

  await restarted.syncOnce();
  assert.equal(restarted.dequeueMessages().length, 0);
});

test("indexer and node pools fail over and surface degraded health", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FailoverWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrls: ["http://indexer-primary.invalid", "http://indexer-secondary.invalid"],
    nodeUrls: ["ws://node-primary.invalid", "ws://node-secondary.invalid"],
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async (url) => {
      if (String(url).startsWith("http://indexer-primary.invalid")) {
        throw new Error("indexer primary offline");
      }
      return response([]);
    },
  });

  await bridge.init();
  bridge.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
    pending_outbound_handshake: null,
  };

  const result = await bridge.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "hello after failover",
    waitMs: 250,
  });

  assert.equal(result.status, "submitted");
  assert.equal(walletClient.getNodeUrl(), "ws://node-secondary.invalid");
  assert.equal(bridge.health().indexerPool.degraded, true);
  assert.equal(bridge.health().nodePool.degraded, true);
  assert.equal(bridge.health().indexerPool.activeUrl, "http://indexer-secondary.invalid");
  assert.equal(bridge.health().nodePool.activeUrl, "ws://node-secondary.invalid");
  assert.equal(bridge.health().walletFundingState, "ready");
  assert.equal(bridge.health().walletBalanceSompi, "500000000");
});

test("health surfaces low Kasia wallet funding details", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  walletClient.balanceSnapshot = {
    onChainBalanceSompi: 27881431n,
    availableMatureBalanceSompi: 27881431n,
    availablePendingBalanceSompi: 0n,
    trackedPendingBalanceSompi: 0n,
    matureUtxoCount: 1,
    pendingUtxoCount: 0,
    trackedPendingUtxoCount: 0,
    updatedAtMs: 456,
  };
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    fetchImpl: async () => response([]),
  });

  await bridge.init();

  const health = bridge.health();
  assert.equal(health.walletFundingState, "low");
  assert.equal(health.walletBalanceSompi, "27881431");
  assert.equal(health.availableMatureBalanceSompi, "27881431");
  assert.equal(health.recommendedMinBalanceSompi, "40000000");
  assert.equal(health.walletBalanceUpdatedAtMs, 456);
});

test("broadcast receive is deduplicated and unauthorized publish is rejected", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    broadcastSubscriptions: {
      news: [VALID_CONTACT_ADDRESS],
    },
    allowedBroadcastChannels: [],
    fetchImpl: async () => response([]),
  });

  await bridge.init();

  await assert.rejects(
    bridge.sendBroadcast({
      channelName: "news",
      message: "ship it",
      waitMs: 10,
    }),
    /not allowed/i
  );

  const payload = buildBroadcastTransactionPayload({
    channelName: "news",
    message: "hello broadcast",
  });
  walletClient.getAddressMempoolEntries = async () =>
    mempoolResponseForAddress(VALID_CONTACT_ADDRESS, [
      {
        id: "tx-broadcast-1",
        payload: payload.toString("utf8"),
      },
    ]);

  await bridge.syncOnce();
  const messages = bridge.dequeueMessages();
  assert.equal(messages.length, 1);
  assert.equal(messages[0].eventType, "broadcast");
  assert.equal(messages[0].chatId, "broadcast:news");
  assert.equal(messages[0].body, "hello broadcast");

  await bridge.syncOnce();
  assert.equal(bridge.dequeueMessages().length, 0);
  assert.equal(
    bridge.state.broadcasts.channels.news.recent_messages.length,
    1
  );
});

test("broadcast send succeeds for an allowed announcement channel", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const walletClient = new FakeWalletClient();
  const bridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient,
    allowedBroadcastChannels: ["news"],
    fetchImpl: async () => response([]),
  });

  await bridge.init();
  const result = await bridge.sendBroadcast({
    channelName: "news",
    message: "ship announcement",
    waitMs: 250,
  });

  assert.equal(result.status, "submitted");
  assert.equal(result.jobKind, "broadcast");
  assert.equal(walletClient.sentTransactions.length, 1);
});

test("configured broadcast channels refresh persisted publishers and permissions on restart", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const firstBridge = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    broadcastSubscriptions: {
      news: ["kaspa:qpublisherold"],
    },
    allowedBroadcastChannels: [],
    fetchImpl: async () => response([]),
  });

  await firstBridge.init();
  const firstChannel = firstBridge.state.broadcasts.channels.news;
  firstChannel.recent_messages.push({
    tx_id: "tx-old",
    sender_address: "kaspa:qpublisherold",
    content: "hello",
    observed_live_ms: 123,
    block_time_ms: null,
  });
  firstChannel.last_seen_block_time = 456;
  await firstBridge.close();

  const restarted = new KasiaBridgeCore({
    stateDir,
    indexerUrl: "http://indexer.invalid",
    nodeUrl: "ws://node.invalid",
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FakeWalletClient(),
    broadcastSubscriptions: {
      news: ["kaspa:qpublishernew"],
    },
    allowedBroadcastChannels: ["news"],
    fetchImpl: async () => response([]),
  });

  await restarted.init();

  const refreshed = restarted.state.broadcasts.channels.news;
  assert.deepEqual(refreshed.publishers, ["kaspa:qpublishernew"]);
  assert.equal(refreshed.allow_publish, true);
  assert.equal(refreshed.last_seen_block_time, 456);
  assert.equal(refreshed.recent_messages.length, 1);
  assert.equal(refreshed.recent_messages[0].tx_id, "tx-old");
});

test("restart during degraded mode keeps the surviving endpoints active", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "kasia-bridge-"));
  const fetchImpl = async (url) => {
    if (String(url).startsWith("http://indexer-primary.invalid")) {
      throw new Error("primary down");
    }
    return response([]);
  };

  const first = new KasiaBridgeCore({
    stateDir,
    indexerUrls: ["http://indexer-primary.invalid", "http://indexer-secondary.invalid"],
    nodeUrls: ["ws://node-primary.invalid", "ws://node-secondary.invalid"],
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FailoverWalletClient(),
    fetchImpl,
  });
  await first.init();
  first.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
    pending_outbound_handshake: null,
  };
  await first.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "first send",
    waitMs: 250,
  });
  await first.close();

  const restarted = new KasiaBridgeCore({
    stateDir,
    indexerUrls: ["http://indexer-primary.invalid", "http://indexer-secondary.invalid"],
    nodeUrls: ["ws://node-primary.invalid", "ws://node-secondary.invalid"],
    network: "mainnet",
    seedPhrase: "seed",
    walletClient: new FailoverWalletClient(),
    fetchImpl,
  });
  await restarted.init();
  restarted.state.conversations[VALID_CONTACT_ADDRESS] = {
    conversation_id: "kasia:test",
    peer_address: VALID_CONTACT_ADDRESS,
    our_alias: "001122334455",
    their_alias: "aabbccddeeff",
    status: "active",
    updated_at: new Date().toISOString(),
    last_handshake_block_time: 100,
    last_context_block_time: 200,
    pending_handshake: null,
    pending_outbound_handshake: null,
  };
  const result = await restarted.send({
    chatId: VALID_CONTACT_ADDRESS,
    message: "second send",
    waitMs: 250,
  });

  assert.equal(result.status, "submitted");
  assert.equal(restarted.health().indexerPool.activeUrl, "http://indexer-secondary.invalid");
  assert.equal(restarted.health().nodePool.activeUrl, "ws://node-secondary.invalid");
});
