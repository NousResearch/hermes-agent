#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const targetVersion = (process.argv[2] || "ALL").toUpperCase();

const versions = [
  { version: "V51", route: "/production-sweep", page: "web/src/pages/ProductionSweepPage.tsx", title: "Production DNS And Health Sweep", keys: ["ProductionSweepRun", "production-sweep", "run_production_sweep"] },
  { version: "V52", route: "/hetzner-promotion-execution", page: "web/src/pages/HetznerPromotionExecutionPage.tsx", title: "Hetzner Promotion Execution", keys: ["PromotionExecution", "promotion-execution", "plan_promotion_execution"] },
  { version: "V53", route: "/command-gate-coverage", page: "web/src/pages/CommandGateCoveragePage.tsx", title: "Command Gate Coverage Auditor", keys: ["GateCoverageRecord", "gate-coverage", "record_gate_coverage"] },
  { version: "V54", route: "/project-adapter-rollout", page: "web/src/pages/ProjectAdapterRolloutPage.tsx", title: "Project Adapter Rollout", keys: ["AdapterRolloutRecord", "adapter-rollout", "record_adapter_rollout"] },
  { version: "V55", route: "/incident-automation", page: "web/src/pages/IncidentAutomationPage.tsx", title: "Incident Automation Engine", keys: ["IncidentAutomationRule", "incident-automation", "run_incident_automation"] },
  { version: "V56", route: "/live-secret-scan", page: "web/src/pages/LiveSecretScanPage.tsx", title: "Live Secret Presence Scan", keys: ["SecretPresenceScan", "secret-presence-scan", "scan_secret_presence"] },
  { version: "V57", route: "/cost-reconciliation", page: "web/src/pages/CostReconciliationPage.tsx", title: "Cost Reconciliation Import", keys: ["CostRateSheet", "cost-reconciliation", "import_cost_reconciliation"] },
  { version: "V58", route: "/outcome-learning-feeds", page: "web/src/pages/OutcomeLearningFeedsPage.tsx", title: "Outcome Learning Feeds", keys: ["OutcomeFeed", "learning-batch", "ingest_learning_batch"] },
  { version: "V59", route: "/golden-eval-execution", page: "web/src/pages/GoldenEvalExecutionPage.tsx", title: "Golden Eval Execution", keys: ["GoldenTaskRun", "golden-eval-batch", "run_golden_eval_batch"] },
  { version: "V60", route: "/hard-breaker-enforcement", page: "web/src/pages/HardBreakerEnforcementPage.tsx", title: "Hard Circuit Breaker Enforcement", keys: ["BreakerCheck", "breaker-check", "check_circuit_breakers"] },
];

const selected = targetVersion === "ALL" ? versions : versions.filter((entry) => entry.version === targetVersion);
if (selected.length === 0) {
  fail(`Unknown version "${targetVersion}". Expected one of ${versions.map((entry) => entry.version).join(", ")} or ALL.`);
}

const files = {
  plan: read("docs/design/v51-v60-boundary-closure-build-plan.md"),
  mainPlan: read("docs/design/hermes-dashboard-design-system-build-plan.md"),
  registry: read("web/src/dashboard-route-registry.tsx"),
  metadata: read("web/src/dashboard-page-metadata.ts"),
  data: read("web/src/pages/operating-system-data.ts"),
  runtime: read("web/src/pages/operating-runtime.ts"),
  runtimeStore: read("hermes_cli/operating_runtime.py"),
  server: read("hermes_cli/web_server.py"),
  packageJson: read("package.json"),
  visualTests: read("tests/dashboard/design-system.spec.ts"),
};

for (const entry of selected) {
  assertIncludes(files.plan, entry.version, `plan includes ${entry.version}`);
  assertIncludes(files.plan, entry.title, `plan includes ${entry.title}`);
  assertIncludes(files.mainPlan, "Version 51-60", "main plan links V51-V60");
  assertIncludes(files.registry, entry.route, `${entry.version} route is registered`);
  assertIncludes(files.metadata, entry.route, `${entry.version} metadata route exists`);
  assertIncludes(files.metadata, `dashboard:${entry.version.toLowerCase()}:validate`, `${entry.version} metadata validation exists`);
  assertIncludes(files.metadata, entry.title, `${entry.version} metadata title exists`);
  assertIncludes(files.data, entry.version, `${entry.version} data record exists`);
  assertIncludes(files.data, entry.route, `${entry.version} data route exists`);
  assertIncludes(files.data, entry.title, `${entry.version} data title exists`);
  assertIncludes(files.runtime, entry.version, `${entry.version} runtime mapping exists`);
  assertIncludes(files.server, entry.keys[1], `${entry.version} runtime endpoint exists`);
  assertIncludes(files.runtimeStore, entry.keys[2], `${entry.version} runtime store function exists`);
  assertIncludes(files.packageJson, `dashboard:${entry.version.toLowerCase()}:validate`, `${entry.version} package script exists`);
  assertIncludes(files.visualTests, entry.route, `${entry.version} visual test route exists`);

  const page = read(entry.page);
  assertIncludes(page, `version="${entry.version}"`, `${entry.version} page selects correct version`);
  for (const key of entry.keys) {
    assertIncludes(files.plan + files.metadata + files.data + files.runtime + files.server + files.runtimeStore, key, `${entry.version} exposes ${key}`);
  }
}

console.log(`V51-V60 boundary closure validation passed for ${selected.map((entry) => entry.version).join(", ")}.`);

function read(relativePath) {
  const fullPath = path.join(root, relativePath);
  if (!fs.existsSync(fullPath)) fail(`Missing required file: ${relativePath}`);
  return fs.readFileSync(fullPath, "utf8");
}

function assertIncludes(content, needle, label) {
  if (!content.includes(needle)) fail(`${label}: missing "${needle}"`);
}

function fail(message) {
  console.error(message);
  process.exit(1);
}
