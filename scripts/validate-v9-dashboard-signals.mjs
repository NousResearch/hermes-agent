#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];

check("dashboard signal contracts", () => {
  const text = read("packages/hermes-dashboard-kit/src/signals.ts");
  for (const name of [
    "DashboardSignalSource",
    "DashboardSnapshot",
    "HealthSnapshot",
    "CostSnapshot",
    "CapacitySnapshot",
    "QueueSnapshot",
    "ActionNeeded",
    "ResearchSignal",
    "DeploymentSignal",
    "dashboardToneForHealth",
    "dashboardHealthScore",
  ]) {
    requireIncludes(text, name, name);
  }
  requireIncludes(read("packages/hermes-dashboard-kit/src/index.ts"), 'export * from "./signals"', "signals export");
});

check("web signal adapter", () => {
  const text = read("web/src/pages/dashboard-signals.ts");
  for (const id of [
    "khashi-vc.roc",
    "media-engine.ops",
    "media-business-operations.main",
    "business-mapper.workspace",
    "meal-assistant.main",
    "investing-system.roc",
    "hermes.workspace",
    "nous-hermes-agent.dashboard",
  ]) {
    requireIncludes(text, id, `known source ${id}`);
  }
  requireIncludes(text, "pluginToDashboardSnapshot", "plugin adapter");
  requireIncludes(text, "standard cost signal", "missing cost message");
});

check("executive summary consumes snapshots", () => {
  const text = read("web/src/pages/executive-data.ts");
  requireIncludes(text, "DashboardSnapshot", "snapshot type");
  requireIncludes(text, "getDashboardSnapshots", "server snapshot endpoint");
  requireIncludes(text, "mergeSnapshots", "snapshot merge");
  requireIncludes(text, "aggregateQueues", "queue rollup");
  requireIncludes(text, "dashboardHealthScore", "health scoring");
});

check("server snapshot endpoints", () => {
  const webServer = read("hermes_cli/web_server.py");
  requireIncludes(webServer, "/api/dashboard/snapshots", "aggregate snapshots endpoint");
  requireIncludes(webServer, "/dashboard-snapshot", "project snapshot endpoint");
  const adapter = read("hermes_cli/dashboard_snapshots.py");
  requireIncludes(adapter, "build_registry_dashboard_snapshots", "registry snapshot builder");
  requireIncludes(adapter, "snapshotUrl", "project snapshot URL support");
  requireIncludes(adapter, "normalize_project_snapshot", "project snapshot normalizer");
  requireIncludes(adapter, "health_for_dashboard", "health snapshot adapter");
});

check("project-owned snapshot endpoints", () => {
  const mediaServer = read("../media-engine/tasks/ops-dashboard-server.js");
  const mediaSnapshot = read("../media-engine/core/operations/unified-publishing-dashboard.js");
  requireIncludes(mediaServer, "/dashboard-snapshot", "Media Engine dashboard snapshot route");
  requireIncludes(mediaSnapshot, "sourceDashboardId", "Media Engine action source dashboard");
  for (const field of ["amountUsd", "tokenCount", "apiCalls", "storageBytes", "capacity", "queue", "actions", "research", "deployment"]) {
    requireIncludes(mediaSnapshot, field, `Media Engine ${field}`);
  }

  const khashiServer = read("../khashi-vc/src/web/server.ts");
  const khashiApi = read("../khashi-vc/src/web/roc-api.ts");
  requireIncludes(khashiServer, "/dashboard-snapshot", "Khashi dashboard snapshot route");
  requireIncludes(khashiApi, "dashboardSnapshot()", "Khashi dashboard snapshot method");
  for (const field of ["tokenCount", "apiCalls", "storageBytes", "capacity", "queue", "actions", "research", "deployment"]) {
    requireIncludes(khashiApi, field, `Khashi ${field}`);
  }
});

check("package script", () => {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v9:validate"] !== "node scripts/validate-v9-dashboard-signals.mjs") {
    throw new Error("missing dashboard:v9:validate script");
  }
});

finish("V9 dashboard signals validation");

function check(label, fn) {
  try {
    fn();
    console.log(`ok ${label}`);
  } catch (error) {
    failures.push(`${label}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function finish(label) {
  if (failures.length) {
    console.error(`${label} failed (${failures.length})`);
    for (const failure of failures) console.error(`- ${failure}`);
    process.exit(1);
  }
  console.log(`${label} passed`);
}

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function requireIncludes(text, needle, label) {
  if (!text.includes(needle)) throw new Error(`missing ${label}`);
}
