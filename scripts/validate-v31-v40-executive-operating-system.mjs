#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const targetVersion = (process.argv[2] || "ALL").toUpperCase();

const versions = [
  { version: "V31", route: "/project-registry", page: "web/src/pages/ProjectRegistryPage.tsx", title: "Production Project Registry", keys: ["Production project registry", "DashboardRegistryEntry", "registry"], endpoint: "/api/operating-runtime/production-checks", helper: "record_production_check" },
  { version: "V32", route: "/telemetry-fabric", page: "web/src/pages/TelemetryFabricPage.tsx", title: "Telemetry Fabric", keys: ["Telemetry fabric", "TelemetrySignal", "telemetry"], endpoint: "/api/operating-runtime/evidence", helper: "telemetry-fabric" },
  { version: "V33", route: "/incident-command", page: "web/src/pages/IncidentCommandPage.tsx", title: "Incident Command", keys: ["Incident command", "IncidentRecord", "incident"], endpoint: "/api/operating-runtime/incidents", helper: "record_incident" },
  { version: "V34", route: "/deployment-promotion", page: "web/src/pages/DeploymentPromotionPage.tsx", title: "Deployment Promotion Rail", keys: ["Deployment promotion rail", "PromotionGate", "deployment"], endpoint: "/api/operating-runtime/deployments", helper: "record_deployment" },
  { version: "V35", route: "/secrets-posture", page: "web/src/pages/SecretsPosturePage.tsx", title: "Secrets And Access Posture", keys: ["Secrets and access posture", "SecretPresence", "secrets"], endpoint: "/api/operating-runtime/permission-decision", helper: "require_permission" },
  { version: "V36", route: "/data-source-catalog", page: "web/src/pages/DataSourceCatalogPage.tsx", title: "Data Source Catalog", keys: ["Data source catalog", "DataSource", "catalog"], endpoint: "/api/operating-runtime/data-sources", helper: "record_data_source" },
  { version: "V37", route: "/finance-attribution", page: "web/src/pages/FinanceAttributionPage.tsx", title: "Finance And Cost Attribution", keys: ["Finance and cost attribution", "CostBucket", "finance"], endpoint: "/api/operating-runtime/costs", helper: "record_cost" },
  { version: "V38", route: "/learning-engine", page: "web/src/pages/LearningEnginePage.tsx", title: "Learning Engine", keys: ["Learning engine", "LearningRecord", "learning"], endpoint: "/api/operating-runtime/learning", helper: "record_learning" },
  { version: "V39", route: "/agent-eval-lab", page: "web/src/pages/AgentEvalLabPage.tsx", title: "Agent Evaluation Lab", keys: ["Agent evaluation lab", "ProviderEval", "eval"], endpoint: "/api/operating-runtime/evals", helper: "record_eval" },
  { version: "V40", route: "/executive-cockpit", page: "web/src/pages/ExecutiveCockpitPage.tsx", title: "Executive Cockpit", keys: ["Executive cockpit", "ExecutiveSignal", "executive"], endpoint: "/api/operating-runtime/autonomy-controls", helper: "set_autonomy_control" },
];

const selected = targetVersion === "ALL" ? versions : versions.filter((entry) => entry.version === targetVersion);

if (selected.length === 0) {
  fail(`Unknown version "${targetVersion}". Expected one of ${versions.map((entry) => entry.version).join(", ")} or ALL.`);
}

const files = {
  plan: read("docs/design/v31-v40-executive-operating-system-build-plan.md"),
  mainPlan: read("docs/design/hermes-dashboard-design-system-build-plan.md"),
  registry: read("web/src/dashboard-route-registry.tsx"),
  metadata: read("web/src/dashboard-page-metadata.ts"),
  data: read("web/src/pages/operating-system-data.ts"),
  runtime: read("web/src/pages/operating-runtime.ts"),
  server: read("hermes_cli/web_server.py"),
  runtimeStore: read("hermes_cli/operating_runtime.py"),
  sharedPage: read("web/src/pages/OperatingSystemStagePage.tsx"),
  visualTests: read("tests/dashboard/design-system.spec.ts"),
};

for (const entry of selected) {
  assertIncludes(files.plan, entry.version, `plan includes ${entry.version}`);
  assertIncludes(files.plan, entry.title, `plan includes ${entry.title}`);
  assertIncludes(files.mainPlan, "Version 31-40", "main plan links V31-V40");
  assertIncludes(files.registry, entry.route, `${entry.version} route is registered`);
  assertIncludes(files.metadata, entry.route, `${entry.version} metadata route exists`);
  assertIncludes(files.metadata, `dashboard:${entry.version.toLowerCase()}:validate`, `${entry.version} metadata validation exists`);
  assertIncludes(files.metadata, entry.title, `${entry.version} metadata title exists`);
  assertIncludes(files.data, entry.version, `${entry.version} data record exists`);
  assertIncludes(files.data, entry.route, `${entry.version} data route exists`);
  assertIncludes(files.data, entry.title, `${entry.version} data title exists`);
  assertIncludes(files.sharedPage, "OperatingSystemStagePage", "shared stage page exists");
  assertIncludes(files.sharedPage, "Runtime Evidence", "shared page surfaces runtime evidence");
  assertIncludes(files.runtime, "RuntimeEvidenceKind", "runtime evidence kind exists");
  assertIncludes(files.runtime, entry.keys[2], `${entry.version} runtime kind exists`);
  assertIncludes(files.visualTests, entry.route, `${entry.version} visual test route exists`);
  if (entry.endpoint) assertIncludes(files.server, entry.endpoint, `${entry.version} runtime endpoint exists`);
  if (entry.helper) assertIncludes(files.runtimeStore, entry.helper, `${entry.version} runtime helper exists`);

  const page = read(entry.page);
  assertIncludes(page, `version="${entry.version}"`, `${entry.version} page selects correct version`);
  for (const key of entry.keys) {
    assertIncludes(files.plan + files.metadata + files.data + files.runtime, key, `${entry.version} exposes ${key}`);
  }
}

console.log(`V31-V40 executive operating-system validation passed for ${selected.map((entry) => entry.version).join(", ")}.`);

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
