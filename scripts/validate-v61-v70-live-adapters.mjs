#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const targetVersion = (process.argv[2] || "ALL").toUpperCase();

const versions = [
  { version: "V61", route: "/network-runner-adapter", page: "web/src/pages/NetworkRunnerAdapterPage.tsx", title: "Network Runner Adapter", keys: ["NetworkRunnerAdapter", "adapter-run", "record_adapter_run"] },
  { version: "V62", route: "/hetzner-ssh-adapter", page: "web/src/pages/HetznerSshAdapterPage.tsx", title: "Hetzner SSH Adapter", keys: ["HetznerSshAdapter", "adapter-run", "record_adapter_run"] },
  { version: "V63", route: "/secret-provider-adapter", page: "web/src/pages/SecretProviderAdapterPage.tsx", title: "Secret Provider Adapter", keys: ["SecretProviderAdapter", "adapter-run", "record_adapter_run"] },
  { version: "V64", route: "/billing-provider-adapter", page: "web/src/pages/BillingProviderAdapterPage.tsx", title: "Billing Provider Adapter", keys: ["BillingProviderAdapter", "adapter-run", "record_adapter_run"] },
  { version: "V65", route: "/project-outcome-emitter", page: "web/src/pages/ProjectOutcomeEmitterPage.tsx", title: "Project Outcome Emitter", keys: ["ProjectOutcomeEmitter", "adapter-run", "record_adapter_run"] },
  { version: "V66", route: "/provider-eval-runner", page: "web/src/pages/ProviderEvalRunnerPage.tsx", title: "Provider Eval Runner", keys: ["ProviderEvalRunner", "adapter-run", "record_adapter_run"] },
  { version: "V67", route: "/breaker-middleware", page: "web/src/pages/BreakerMiddlewarePage.tsx", title: "Breaker Middleware SDK", keys: ["BreakerMiddleware", "adapter-run", "record_adapter_run"] },
  { version: "V68", route: "/incident-subscriptions", page: "web/src/pages/IncidentSubscriptionPage.tsx", title: "Incident Subscription Bus", keys: ["IncidentSubscription", "incident-subscriptions", "record_incident_subscription"] },
  { version: "V69", route: "/evidence-artifact-store", page: "web/src/pages/EvidenceArtifactStorePage.tsx", title: "Evidence Artifact Store", keys: ["EvidenceArtifact", "evidence-artifacts", "index_evidence_artifact"] },
  { version: "V70", route: "/release-train-orchestrator", page: "web/src/pages/ReleaseTrainOrchestratorPage.tsx", title: "Release Train Orchestrator", keys: ["ReleaseTrain", "release-trains", "plan_release_train"] },
];

const selected = targetVersion === "ALL" ? versions : versions.filter((entry) => entry.version === targetVersion);
if (selected.length === 0) {
  fail(`Unknown version "${targetVersion}". Expected one of ${versions.map((entry) => entry.version).join(", ")} or ALL.`);
}

const files = {
  plan: read("docs/design/v61-v70-live-adapter-build-plan.md"),
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
  assertIncludes(files.mainPlan, "Version 61-70", "main plan links V61-V70");
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

console.log(`V61-V70 live adapter validation passed for ${selected.map((entry) => entry.version).join(", ")}.`);

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
