import assert from "node:assert/strict";
import test from "node:test";
import { spectrumStartupError } from "../startup-error.mjs";

test("local database-open failures explain launch-specific access, not dependency reinstall", () => {
  for (const message of ["Failed to open database: unable to open database file", "SQLITE_CANTOPEN"]) {
    const error = new Error(message);
    const result = spectrumStartupError(error, { localMode: true });
    assert.match(result, /Full Disk Access/);
    assert.match(result, /\/usr\/bin\/osascript/);
    assert.match(result, /terminal app/);
    assert.match(result, /not Hermes-only/);
    assert.match(result, /hermes gateway restart/);
    assert.match(result, /does not verify background-service access/);
    assert.ok(result.includes(error.stack));
    assert.doesNotMatch(result, /install-sidecar|npm install/);
  }
});

test("only dependency failures recommend reinstall; cloud and other failures keep their cause", () => {
  for (const localMode of [true, false]) {
    for (const message of ["Cannot find package '@spectrum-ts/core'", "NODE_MODULE_VERSION mismatch", "Could not locate the bindings file"]) {
      const result = spectrumStartupError(new Error(message), { localMode });
      assert.match(result, /hermes photon install-sidecar/);
      assert.doesNotMatch(result, /Full Disk Access/);
    }
  }
  for (const [message, localMode] of [["unable to open database file", false], ["Invalid credentials", true]]) {
    const result = spectrumStartupError(new Error(message), { localMode });
    assert.ok(result.includes(message));
    assert.doesNotMatch(result, /Full Disk Access|install-sidecar/);
  }
});
