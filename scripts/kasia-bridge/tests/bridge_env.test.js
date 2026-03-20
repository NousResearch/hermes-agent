import test from "node:test";
import assert from "node:assert/strict";

import { readBridgeEnv, validateBridgeEnv } from "../lib/bridge_env.js";

test("readBridgeEnv accepts plural-only endpoint configuration", () => {
  const config = readBridgeEnv({
    KASIA_SEED_PHRASE: "seed phrase",
    KASIA_INDEXER_URLS: "https://indexer-a.example.com,https://indexer-b.example.com",
    KASIA_NODE_WBORSH_URLS: "ws://127.0.0.1:17110,ws://127.0.0.1:17111",
  });

  assert.equal(config.indexerUrl, "https://indexer-a.example.com");
  assert.deepEqual(config.indexerUrls, [
    "https://indexer-a.example.com",
    "https://indexer-b.example.com",
  ]);
  assert.equal(config.nodeUrl, "ws://127.0.0.1:17110");
  assert.deepEqual(config.nodeUrls, [
    "ws://127.0.0.1:17110",
    "ws://127.0.0.1:17111",
  ]);

  assert.doesNotThrow(() => validateBridgeEnv(config));
});

test("validateBridgeEnv still rejects missing Kasia endpoints", () => {
  const config = readBridgeEnv({
    KASIA_SEED_PHRASE: "seed phrase",
  });

  assert.throws(() => validateBridgeEnv(config), /KASIA_SEED_PHRASE plus either/);
});
