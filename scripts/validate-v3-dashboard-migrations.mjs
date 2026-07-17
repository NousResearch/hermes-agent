#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { execFileSync } from "node:child_process";

const root = process.cwd();

const targets = [
  {
    id: "hermes-os",
    label: "Hermes OS reference dashboard",
    files: ["web/src/pages/HermesOsPage.tsx"],
    required: [
      "DashboardHeader",
      "MetricGrid",
      "KpiCard",
      "DataTable",
      "StatusPill",
      "DashboardLoadingState",
      "DashboardErrorState",
      "/design-system",
    ],
  },
  {
    id: "khashi-vc",
    label: "Khashi VC ROC static adapter",
    files: [
      "../khashi-vc/public/roc/index.html",
      "../khashi-vc/public/roc/app.js",
      "../khashi-vc/public/roc/hermes-dashboard-kit.css",
    ],
    required: [
      "hdk-body",
      "hdk-shell",
      "hdk-sidebar",
      "hdk-main",
      "hdk-button",
      "hdk-metric-grid",
      "hdk-card",
      "hdk-pill",
      "hdk-table",
      "hdk-table-wrap",
      "hdk-empty",
      "hdk-loading",
    ],
  },
  {
    id: "media-engine",
    label: "Media Engine Ops static adapter",
    files: [
      "../media-engine/core/operations/unified-publishing-dashboard.js",
      "../media-engine/core/operations/hermes-dashboard-kit.css",
      "../media-engine/hermes.dashboards.json",
    ],
    required: [
      "hdk-body",
      "hdk-shell",
      "hdk-sidebar",
      "hdk-main",
      "hdk-header",
      "hdk-section",
      "hdk-button",
      "hdk-metric-grid",
      "hdk-card",
      "hdk-pill",
      "hdk-table",
      "hdk-table-wrap",
      "hdk-empty",
      "data-autopilot-control",
      "data-discord-preview",
      "media-engine.ops",
      "ops:dashboard:server",
    ],
  },
];

function readTarget(target) {
  return target.files.map((relativePath) => {
    const absolutePath = path.resolve(root, relativePath);
    return {
      relativePath,
      absolutePath,
      exists: fs.existsSync(absolutePath),
      text: fs.existsSync(absolutePath) ? fs.readFileSync(absolutePath, "utf8") : "",
    };
  });
}

const failures = [];

for (const target of targets) {
  const files = readTarget(target);
  const missingFiles = files.filter((file) => !file.exists);
  for (const file of missingFiles) {
    failures.push(`${target.label}: missing file ${file.relativePath}`);
  }

  const combined = files.map((file) => file.text).join("\n");
  for (const needle of target.required) {
    if (!combined.includes(needle)) {
      failures.push(`${target.label}: missing required migration marker "${needle}"`);
    }
  }
}

if (failures.length) {
  console.error(`V3 dashboard migration validation failed (${failures.length})`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

execFileSync("node", ["scripts/validate-static-adapter-sync.mjs"], {
  cwd: root,
  stdio: "inherit",
});
execFileSync("node", ["scripts/validate-media-engine-generated-dashboard.mjs"], {
  cwd: root,
  stdio: "inherit",
});

console.log(`V3 dashboard migration validation passed (${targets.length} targets)`);
