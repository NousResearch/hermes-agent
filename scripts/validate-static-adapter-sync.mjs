#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const source = path.resolve(root, "packages/hermes-dashboard-kit/static/hermes-dashboard-kit.css");
const targets = [
  "../khashi-vc/public/roc/hermes-dashboard-kit.css",
  "../media-engine/core/operations/hermes-dashboard-kit.css",
];

function hash(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

if (!fs.existsSync(source)) {
  console.error(`Missing source adapter: ${source}`);
  process.exit(1);
}

const sourceHash = hash(source);
const failures = [];

for (const target of targets) {
  const absolute = path.resolve(root, target);
  if (!fs.existsSync(absolute)) {
    failures.push(`${target} is missing`);
    continue;
  }
  const targetHash = hash(absolute);
  if (targetHash !== sourceHash) {
    failures.push(`${target} has drifted from packages/hermes-dashboard-kit/static/hermes-dashboard-kit.css`);
  }
}

if (failures.length) {
  console.error(`Static dashboard adapter sync failed (${failures.length})`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Static dashboard adapter sync passed (${targets.length} targets)`);
