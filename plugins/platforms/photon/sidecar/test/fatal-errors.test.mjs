import assert from "node:assert/strict";
import test from "node:test";

import { errorToText, isFatalInboundStreamError, classifyRecoverableOutboundError } from "../fatal-errors.mjs";

test("detects Photon catchUpEvents concurrency-limit errors", () => {
  const error = {
    message: "internalError",
    grpcCode: 8,
    cause: {
      message:
        "/photon.imessage.v1.EventService/CatchUpEvents RESOURCE_EXHAUSTED: [spectrum-imessage] catchUpEvents concurrency limit (16) reached",
      details: "[spectrum-imessage] catchUpEvents concurrency limit (16) reached",
    },
  };

  assert.equal(isFatalInboundStreamError(error), true);
});

test("detects fatal retry text emitted by spectrum-ts internal stream logger", () => {
  assert.equal(
    isFatalInboundStreamError(
      "[spectrum.stream] stream persistently failing; still retrying RateLimitError: [spectrum-imessage] catchUpEvents concurrency limit (16) reached"
    ),
    true
  );
});

test("does not treat outbound target authorization failures as fatal inbound stream errors", () => {
  const error = new Error("AuthenticationError: [spectrum-imessage] Target not allowed for this project");

  assert.equal(isFatalInboundStreamError(error), false);
  assert.match(errorToText(error), /Target not allowed/);
});

test("classifies upstream connection drops as recoverable outbound send errors", () => {
  const error = new Error("ConnectionError: [upstream] Connection dropped");

  assert.equal(classifyRecoverableOutboundError(error), "upstream_connection_dropped");
});
