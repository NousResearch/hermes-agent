import test from "node:test";
import assert from "node:assert/strict";
import {
  isFindMyLocationMessage,
  normalizeLocationSnapshot,
  resolveFindMyLocation,
} from "../findmy-location.mjs";

const findMyMessage = {
  balloonBundleId:
    "com.apple.messages.MSMessageExtensionBalloonPlugin:0000000000:com.apple.findmy.FindMyMessagesApp",
  content: { type: "custom", raw: { imessage_type: "unsupported-message" } },
  sender: { id: "+15551234567" },
  space: { phone: "shared" },
};

test("recognizes Find My location cards only", () => {
  assert.equal(isFindMyLocationMessage(findMyMessage), true);
  assert.equal(isFindMyLocationMessage({ ...findMyMessage, balloonBundleId: "other" }), false);
});

test("normalizes a valid shared-location snapshot", () => {
  assert.deepEqual(
    normalizeLocationSnapshot({ latitude: 29.7, longitude: -95.4, accuracy: 8, shortAddress: "Houston" }),
    { type: "location", latitude: 29.7, longitude: -95.4, accuracy: 8, name: "", address: "", shortAddress: "Houston", longAddress: "", locationType: "" },
  );
});

test("resolves Find My through a one-shot shared client", async () => {
  let closed = false;
  let requested = "";
  const result = await resolveFindMyLocation({
    message: findMyMessage,
    projectId: "project",
    projectSecret: "secret",
    issueTokens: async () => ({ type: "shared", token: "token" }),
    createClient: () => ({
      locations: {
        get: async (address) => {
          requested = address;
          return { latitude: 29.7, longitude: -95.4, locationType: "legacy" };
        },
      },
      close: async () => { closed = true; },
    }),
  });
  assert.equal(requested, "+15551234567");
  assert.equal(result.type, "location");
  assert.equal(result.locationType, "legacy");
  assert.equal(closed, true);
});
