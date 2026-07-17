#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const mediaEngineRoot = path.resolve(root, "../media-engine");
const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), "media-engine-v3-dashboard-"));

const requiredMarkers = [
  "hdk-body",
  "hdk-shell",
  "hdk-sidebar",
  "hdk-main",
  "hdk-header",
  "hdk-section",
  "hdk-metric-grid",
  "hdk-card",
  "hdk-button",
  "hdk-pill",
  "hdk-table-wrap",
  "hdk-table",
  "hdk-empty",
  "data-autopilot-control",
  "data-discord-preview",
  "Media Engine Unified Ops",
];

if (!fs.existsSync(path.join(mediaEngineRoot, "package.json"))) {
  console.error(`Media Engine project not found at ${mediaEngineRoot}`);
  process.exit(1);
}

execFileSync(
  "node",
  [
    "tasks/build-unified-ops-dashboard.js",
    "--skip-live-stores",
    "--output-dir",
    outputDir,
    "--now",
    "2026-07-16T00:00:00.000Z",
  ],
  {
    cwd: mediaEngineRoot,
    stdio: "pipe",
  },
);

const htmlPath = path.join(outputDir, "index.html");
const snapshotPath = path.join(outputDir, "unified-publishing-snapshot.json");

if (!fs.existsSync(htmlPath)) {
  console.error(`Generated dashboard HTML missing: ${htmlPath}`);
  process.exit(1);
}
if (!fs.existsSync(snapshotPath)) {
  console.error(`Generated dashboard snapshot missing: ${snapshotPath}`);
  process.exit(1);
}

const html = fs.readFileSync(htmlPath, "utf8");
const missing = requiredMarkers.filter((marker) => !html.includes(marker));

if (missing.length) {
  console.error(`Generated Media Engine dashboard validation failed (${missing.length})`);
  for (const marker of missing) console.error(`- missing marker ${marker}`);
  process.exit(1);
}

console.log(`Generated Media Engine dashboard validation passed: ${htmlPath}`);
