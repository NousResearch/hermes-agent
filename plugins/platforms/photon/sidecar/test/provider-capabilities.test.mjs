import assert from "node:assert/strict";
import test from "node:test";

import { supportsProviderCapability } from "../provider-capabilities.mjs";

test("Spectrum 12.7 local mode rejects unavailable native operations", () => {
  for (const capability of ["effect", "poll", "reaction", "read"]) {
    assert.equal(supportsProviderCapability(true, capability), false);
  }
});

test("cloud mode retains native operations", () => {
  for (const capability of ["effect", "poll", "reaction", "read"]) {
    assert.equal(supportsProviderCapability(false, capability), true);
  }
});

test("local mode permits capabilities not listed as unsupported", () => {
  for (const capability of ["attachment", "text", "typing"]) {
    assert.equal(supportsProviderCapability(true, capability), true);
  }
});
