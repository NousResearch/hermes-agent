import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { KasiaBridgeCore } from "../lib/bridge_core.js";

class FakeWalletClient {
  constructor() {
    this.isConnected = true;
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
  }

  async init() {
    return this.info;
  }

  async close() {}

  loadSendState(state) {
    this.loadedSendState = state;
    this.sendState = state || this.sendState;
  }

  exportSendState() {
    return this.sendState;
  }

  async sendPayloadTransaction(payload) {
    this.sentTransactions.push(payload);
    return `tx-${this.sentTransactions.length}`;
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
      chatId: "kaspa:qcontactaddressxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
      message: "hello",
    }),
    /No active Kasia conversation/
  );
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
