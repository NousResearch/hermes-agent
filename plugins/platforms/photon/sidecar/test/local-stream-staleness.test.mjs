import assert from "node:assert/strict";
import test from "node:test";
import { silenceProbeThreshold, shouldProbe, isZombieSuspect } from "../stream-staleness.mjs";

test("a quiet local provider cannot be declared a zombie by a no-op read", () => {
  for (const configured of [undefined, "600000", "1"]) {
    const threshold = silenceProbeThreshold(true, configured);
    assert.equal(shouldProbe(86400000, threshold, 86400000, 120000), false);
    assert.equal(isZombieSuspect(86400000, threshold, { alive: true }), false);
  }
});

test("cloud streams retain the configured watchdog and explicit disable", () => {
  const threshold = silenceProbeThreshold(false, "5000");
  assert.equal(shouldProbe(6000, threshold, 6000, 1000), true);
  assert.equal(isZombieSuspect(6000, threshold, { alive: true }), true);
  assert.equal(isZombieSuspect(6000, threshold, { alive: false }), false);
  assert.equal(shouldProbe(86400000, silenceProbeThreshold(false, "0"), 86400000, 1), false);
  assert.ok(silenceProbeThreshold(false, undefined) > 0);
});
