#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import process from "node:process";

const checks = [
  ["npm", ["run", "dashboard:usage:audit:strict"], "strict dashboard usage audit"],
  ["npm", ["run", "dashboard:registry:validate"], "dashboard registry validation"],
  ["npm", ["run", "dashboard:v4:validate"], "V4 standardization validation"],
  ["npm", ["run", "build", "--workspace", "@hermes/dashboard-kit"], "dashboard kit build"],
  ["npm", ["run", "build", "--workspace", "web"], "web build"],
  [
    "npm",
    ["run", "dashboard:visual:check", "--", "--reporter=line"],
    "Playwright visual and accessibility checks",
  ],
];

const failures = [];

for (const [command, args, label] of checks) {
  try {
    execFileSync(command, args, {
      cwd: process.cwd(),
      stdio: "inherit",
    });
  } catch {
    failures.push(label);
  }
}

if (failures.length) {
  console.error(`V5 dashboard quality validation failed (${failures.length})`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log("V5 dashboard quality validation passed");
