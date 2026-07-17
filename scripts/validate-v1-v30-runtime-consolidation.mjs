#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();

const required = [
  {
    file: "docs/design/v1-v30-runtime-consolidation.md",
    includes: [
      "SQLite runtime evidence",
      "permission-aware readiness checks",
      "Rich project-owned production snapshot endpoints",
      "Permission middleware",
      "Post-Consolidation Runtime Progress",
      "production route checks still need real network/screenshot execution",
    ],
  },
  {
    file: "docs/design/hermes-dashboard-design-system-build-plan.md",
    includes: [
      "V1-V30 Runtime Consolidation",
      "Hermes SQLite runtime bridge",
      "Surface runtime evidence on V22-V30 pages",
    ],
  },
  {
    file: "hermes_cli/operating_runtime.py",
    includes: [
      "sqlite3",
      "runtime_evidence",
      "runtime_audit",
      "runtime_workbench",
      "decide_permission",
      "run_readiness_check",
      "require_permission",
      "record_incident",
      "record_deployment",
      "create_workbench_item",
      "summary",
    ],
  },
  {
    file: "hermes_cli/web_server.py",
    includes: [
      "/api/operating-runtime/summary",
      "/api/operating-runtime/evidence",
      "/api/operating-runtime/readiness-check",
      "/api/operating-runtime/permission-decision",
      "/api/operating-runtime/incidents",
      "/api/operating-runtime/deployments",
      "/api/operating-runtime/workbench",
      "/api/dashboard/snapshots",
      "/dashboard-snapshot",
    ],
  },
  {
    file: "hermes_cli/dashboard_snapshots.py",
    includes: [
      "build_hermes_dashboard_snapshot",
      "build_registry_dashboard_snapshots",
      "discover_dashboard_registries",
      "health_for_dashboard",
      "DashboardSnapshot",
    ],
  },
  {
    file: "web/src/pages/executive-data.ts",
    includes: [
      "getDashboardSnapshots",
      "buildExecutiveSummary(response.snapshots",
    ],
  },
  {
    file: "web/src/pages/operating-runtime.ts",
    includes: [
      "OperatingRuntimeState",
      "RuntimeEvidenceRecord",
      "RuntimeAuditRecord",
      "loadOperatingRuntimeState",
      "loadOperatingRuntimeStateFromServer",
      "saveOperatingRuntimeState",
      "runRuntimeReadinessCheck",
      "runServerRuntimeReadinessCheck",
      "permissionForAction",
      "localStorage",
    ],
  },
  {
    file: "web/src/pages/OperatingSystemStagePage.tsx",
    includes: [
      "Runtime Evidence",
      "Run readiness check",
      "runRuntimeReadinessCheck",
      "runtimeSummaryForStage",
      "recordsForStage",
    ],
  },
  {
    file: "tests/dashboard/design-system.spec.ts",
    includes: [
      "/project-snapshots",
      "/autonomy-readiness",
      "V22-V30 autonomy-readiness layer",
    ],
  },
];

for (const item of required) {
  const content = read(item.file);
  for (const needle of item.includes) {
    if (!content.includes(needle)) {
      fail(`${item.file} missing "${needle}"`);
    }
  }
}

console.log("V1-V30 runtime consolidation validation passed.");

function read(relativePath) {
  const fullPath = path.join(root, relativePath);
  if (!fs.existsSync(fullPath)) {
    fail(`Missing required file: ${relativePath}`);
  }
  return fs.readFileSync(fullPath, "utf8");
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
