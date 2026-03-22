import test from "node:test";
import assert from "node:assert/strict";

import { KnsClient, defaultKnsUrlForNetwork } from "../lib/kns_client.js";

test("KnsClient falls back to the network default base URL", () => {
  const client = new KnsClient({
    network: "testnet-10",
  });

  assert.equal(client.isEnabled(), true);
  assert.equal(client.baseUrl, defaultKnsUrlForNetwork("testnet-10"));
});

test("KnsClient enables lookups when a base URL is configured", () => {
  const client = new KnsClient({
    baseUrl: "https://kns.invalid/api/v1",
  });

  assert.equal(client.isEnabled(), true);
});
