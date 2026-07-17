#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const targetVersion = (process.argv[2] || "ALL").toUpperCase();

const versions = [
  { version: "V22", route: "/project-snapshots", page: "web/src/pages/ProjectSnapshotsPage.tsx", title: "Live Project Snapshot Contracts", keys: ["DashboardSnapshot", "Snapshot Contract Registry", "project-owned endpoint"] },
  { version: "V23", route: "/durable-memory", page: "web/src/pages/DurableMemoryPage.tsx", title: "Durable Memory And Decision Store", keys: ["Durable Memory And Decision Store", "Memory Store Readiness", "database-backed"] },
  { version: "V24", route: "/permission-runtime", page: "web/src/pages/PermissionRuntimePage.tsx", title: "Permission Enforcement Runtime", keys: ["Permission Enforcement Runtime", "approval", "audit"] },
  { version: "V25", route: "/cost-governor", page: "web/src/pages/CostGovernorPage.tsx", title: "Model Router And Cost Governor", keys: ["Model Router And Cost Governor", "premium approval", "provider"] },
  { version: "V26", route: "/loop-runner", page: "web/src/pages/LoopRunnerPage.tsx", title: "Operating Loop Runner", keys: ["Operating Loop Runner", "dry-run", "scheduler"] },
  { version: "V27", route: "/business-command", page: "web/src/pages/BusinessCommandPage.tsx", title: "Cross-Business Command Center", keys: ["Cross-Business Command Center", "RevenueSignal", "business rollup"] },
  { version: "V28", route: "/agent-workbench", page: "web/src/pages/AgentWorkbenchPage.tsx", title: "Agent Workbench", keys: ["Agent Workbench", "ApprovalGate", "artifact"] },
  { version: "V29", route: "/evaluation-gates", page: "web/src/pages/EvaluationGatesPage.tsx", title: "Evaluation And Quality Gates", keys: ["Evaluation And Quality Gates", "QualityGate", "ModelEval"] },
  { version: "V30", route: "/autonomy-readiness", page: "web/src/pages/AutonomyReadinessPage.tsx", title: "Production Autonomy Readiness", keys: ["Production Autonomy Readiness", "KillSwitch", "BudgetBreaker"] },
];

const selected = targetVersion === "ALL" ? versions : versions.filter((entry) => entry.version === targetVersion);

if (selected.length === 0) {
  fail(`Unknown version "${targetVersion}". Expected one of ${versions.map((entry) => entry.version).join(", ")} or ALL.`);
}

const files = {
  plan: read("docs/design/v22-v30-autonomy-readiness-build-plan.md"),
  mainPlan: read("docs/design/hermes-dashboard-design-system-build-plan.md"),
  registry: read("web/src/dashboard-route-registry.tsx"),
  metadata: read("web/src/dashboard-page-metadata.ts"),
  data: read("web/src/pages/operating-system-data.ts"),
  runtime: read("web/src/pages/operating-runtime.ts"),
  sharedPage: read("web/src/pages/OperatingSystemStagePage.tsx"),
};

for (const entry of selected) {
  assertIncludes(files.plan, entry.version, `plan includes ${entry.version}`);
  assertIncludes(files.plan, entry.title, `plan includes ${entry.title}`);
  assertIncludes(files.mainPlan, "Version 22-30", "main plan links V22-V30");
  assertIncludes(files.registry, entry.route, `${entry.version} route is registered`);
  assertIncludes(files.metadata, entry.route, `${entry.version} metadata route exists`);
  assertIncludes(files.metadata, `dashboard:${entry.version.toLowerCase()}:validate`, `${entry.version} metadata validation exists`);
  assertIncludes(files.metadata, entry.title, `${entry.version} metadata title exists`);
  assertIncludes(files.data, entry.version, `${entry.version} data record exists`);
  assertIncludes(files.data, entry.route, `${entry.version} data route exists`);
  assertIncludes(files.data, entry.title, `${entry.version} data title exists`);
  assertIncludes(files.sharedPage, "OperatingSystemStagePage", "shared stage page exists");
  assertIncludes(files.sharedPage, "DataTable", "shared page uses DataTable");
  assertIncludes(files.sharedPage, "Runtime Evidence", "shared page surfaces runtime evidence");
  assertIncludes(files.runtime, "loadOperatingRuntimeState", "runtime state loader exists");
  assertIncludes(files.runtime, "runRuntimeReadinessCheck", "runtime readiness check exists");
  assertIncludes(files.runtime, "RuntimeAuditRecord", "runtime audit records exist");

  const page = read(entry.page);
  assertIncludes(page, `version="${entry.version}"`, `${entry.version} page selects correct version`);
  for (const key of entry.keys) {
    assertIncludes(files.plan + files.metadata + files.data, key, `${entry.version} exposes ${key}`);
  }
}

console.log(`V22-V30 autonomy-readiness validation passed for ${selected.map((entry) => entry.version).join(", ")}.`);

function read(relativePath) {
  const fullPath = path.join(root, relativePath);
  if (!fs.existsSync(fullPath)) {
    fail(`Missing required file: ${relativePath}`);
  }
  return fs.readFileSync(fullPath, "utf8");
}

function assertIncludes(content, needle, label) {
  if (!content.includes(needle)) {
    fail(`${label}: missing "${needle}"`);
  }
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
