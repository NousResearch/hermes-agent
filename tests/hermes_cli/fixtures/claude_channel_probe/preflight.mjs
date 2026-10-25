import fs from "node:fs";

import { assertLockedDependencies, assertSubscriptionAuth } from "./contract.mjs";

const mode = process.argv[2];
if (mode === "auth") {
  let input = "";
  for await (const chunk of process.stdin) input += chunk;
  assertSubscriptionAuth(JSON.parse(input), process.env);
  process.stdout.write("subscription-auth-ok\n");
} else if (mode === "lock") {
  assertLockedDependencies(JSON.parse(fs.readFileSync(new URL("./package-lock.json", import.meta.url))));
  process.stdout.write("dependency-lock-ok\n");
} else {
  throw new Error("usage: node preflight.mjs auth|lock");
}
