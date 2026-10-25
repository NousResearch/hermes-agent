import fs from "node:fs";

import {
  assertDisposableDirectory,
  assertLockedDependencies,
  assertManagedProfile,
  assertNewPrivateOutputPath,
  claimNewPrivateOutputPath,
  assertSubscriptionAuth,
} from "./contract.mjs";

const mode = process.argv[2];
if (mode === "auth") {
  let input = "";
  for await (const chunk of process.stdin) input += chunk;
  assertSubscriptionAuth(JSON.parse(input), process.env);
  process.stdout.write("subscription-auth-ok\n");
} else if (mode === "lock") {
  assertLockedDependencies(JSON.parse(fs.readFileSync(new URL("./package-lock.json", import.meta.url))));
  process.stdout.write("dependency-lock-ok\n");
} else if (mode === "location") {
  assertDisposableDirectory(process.cwd());
  process.stdout.write("disposable-location-ok\n");
} else if (mode === "profile") {
  assertManagedProfile(process.argv[3], process.argv[4], process.argv[5]);
  process.stdout.write("managed-profile-ok\n");
} else if (mode === "sessions") {
  let input = "";
  for await (const chunk of process.stdin) input += chunk;
  const sessions = JSON.parse(input);
  if (!Array.isArray(sessions) || sessions.length !== 0) {
    throw new Error("managed provider-auth profile already has an active session");
  }
  process.stdout.write("managed-profile-idle-ok\n");
} else if (mode === "output") {
  assertNewPrivateOutputPath(process.argv[3], process.argv[4]);
  process.stdout.write("private-output-path-ok\n");
} else if (mode === "claim-output") {
  claimNewPrivateOutputPath(process.argv[3], process.argv[4]);
  process.stdout.write("private-output-claimed\n");
} else {
  throw new Error("usage: node preflight.mjs auth|claim-output|lock|location|output|profile|sessions");
}
