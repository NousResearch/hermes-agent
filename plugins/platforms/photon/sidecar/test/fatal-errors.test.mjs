import assert from "node:assert/strict";
import test from "node:test";

import { classifyRecoverableOutboundError } from "../fatal-errors.mjs";

test("classifies upstream connection drops as recoverable outbound send errors", () => {
  const error = new Error("ConnectionError: [upstream] Connection dropped");

  assert.equal(classifyRecoverableOutboundError(error), "upstream_connection_dropped");
});