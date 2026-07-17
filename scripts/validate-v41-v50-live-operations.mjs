#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const targetVersion = (process.argv[2] || "ALL").toUpperCase();

const versions = [
  { version: "V41", route: "/production-verification", page: "web/src/pages/ProductionVerificationPage.tsx", title: "Live Production Verification Runner", keys: ["ProductionCheck", "production-checks", "registry"] },
  { version: "V42", route: "/command-gates", page: "web/src/pages/CommandGateRuntimePage.tsx", title: "Command Gate Runtime", keys: ["PermissionDecision", "permission-decision", "permission"] },
  { version: "V43", route: "/telemetry-adapters", page: "web/src/pages/TelemetryAdapterKitPage.tsx", title: "Project Telemetry Adapter Kit", keys: ["TelemetryAdapter", "dashboard/snapshots", "telemetry"] },
  { version: "V44", route: "/incident-ingestion", page: "web/src/pages/IncidentIngestionPage.tsx", title: "Incident Ingestion And Escalation", keys: ["IncidentRule", "incidents", "incident"] },
  { version: "V45", route: "/promotion-runner", page: "web/src/pages/PromotionRunnerPage.tsx", title: "Shared Deployment Promotion Runner", keys: ["PromotionRun", "deployments", "deployment"] },
  { version: "V46", route: "/secret-scanner", page: "web/src/pages/SecretScannerPage.tsx", title: "Secrets Posture Scanner", keys: ["SecretPresence", "permission-decision", "secrets"] },
  { version: "V47", route: "/cost-attribution-engine", page: "web/src/pages/CostAttributionEnginePage.tsx", title: "Cost Attribution Engine", keys: ["CostRecord", "costs", "finance"] },
  { version: "V48", route: "/learning-ingestion", page: "web/src/pages/LearningIngestionPage.tsx", title: "Learning Ingestion Pipeline", keys: ["LearningEvent", "learning", "learning"] },
  { version: "V49", route: "/model-eval-harness", page: "web/src/pages/ModelEvalHarnessPage.tsx", title: "Agent And Model Eval Harness", keys: ["GoldenTask", "evals", "eval"] },
  { version: "V50", route: "/circuit-breakers", page: "web/src/pages/CircuitBreakersPage.tsx", title: "Runtime Circuit Breakers", keys: ["KillSwitch", "autonomy-controls", "autonomy"] },
];

const selected = targetVersion === "ALL" ? versions : versions.filter((entry) => entry.version === targetVersion);
if (selected.length === 0) {
  fail(`Unknown version "${targetVersion}". Expected one of ${versions.map((entry) => entry.version).join(", ")} or ALL.`);
}

const files = {
  plan: read("docs/design/v41-v50-live-operations-build-plan.md"),
  mainPlan: read("docs/design/hermes-dashboard-design-system-build-plan.md"),
  registry: read("web/src/dashboard-route-registry.tsx"),
  metadata: read("web/src/dashboard-page-metadata.ts"),
  data: read("web/src/pages/operating-system-data.ts"),
  runtime: read("web/src/pages/operating-runtime.ts"),
  runtimeStore: read("hermes_cli/operating_runtime.py"),
  server: read("hermes_cli/web_server.py"),
  visualTests: read("tests/dashboard/design-system.spec.ts"),
};

for (const entry of selected) {
  assertIncludes(files.plan, entry.version, `plan includes ${entry.version}`);
  assertIncludes(files.plan, entry.title, `plan includes ${entry.title}`);
  assertIncludes(files.mainPlan, "Version 41-50", "main plan links V41-V50");
  assertIncludes(files.registry, entry.route, `${entry.version} route is registered`);
  assertIncludes(files.metadata, entry.route, `${entry.version} metadata route exists`);
  assertIncludes(files.metadata, `dashboard:${entry.version.toLowerCase()}:validate`, `${entry.version} metadata validation exists`);
  assertIncludes(files.metadata, entry.title, `${entry.version} metadata title exists`);
  assertIncludes(files.data, entry.version, `${entry.version} data record exists`);
  assertIncludes(files.data, entry.route, `${entry.version} data route exists`);
  assertIncludes(files.data, entry.title, `${entry.version} data title exists`);
  assertIncludes(files.runtime, entry.keys[2], `${entry.version} runtime evidence kind exists`);
  assertIncludes(files.server, entry.keys[1], `${entry.version} runtime endpoint exists`);
  assertIncludes(files.visualTests, entry.route, `${entry.version} visual test route exists`);

  const page = read(entry.page);
  assertIncludes(page, `version="${entry.version}"`, `${entry.version} page selects correct version`);
  for (const key of entry.keys) {
    assertIncludes(files.plan + files.metadata + files.data + files.runtime + files.server + files.runtimeStore, key, `${entry.version} exposes ${key}`);
  }
}

console.log(`V41-V50 live operations validation passed for ${selected.map((entry) => entry.version).join(", ")}.`);

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
