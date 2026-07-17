#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const targetVersion = (process.argv[2] || "ALL").toUpperCase();

const versions = [
  { version: "V71", route: "/production-screenshot-runner", page: "web/src/pages/ProductionScreenshotRunnerPage.tsx", title: "Production Screenshot Runner", keys: ["ProductionScreenshotRunner", "production-screenshots", "record_production_screenshot_run"] },
  { version: "V72", route: "/hetzner-promotion-transport", page: "web/src/pages/HetznerPromotionTransportPage.tsx", title: "Hetzner Promotion Transport", keys: ["HetznerPromotionTransport", "hetzner-promotion-transport", "record_hetzner_transport_run"] },
  { version: "V73", route: "/server-secret-posture-scanner", page: "web/src/pages/ServerSecretPostureScannerPage.tsx", title: "Server Secret Posture Scanner", keys: ["ServerSecretManifest", "server-secret-posture", "record_server_secret_posture"] },
  { version: "V74", route: "/incident-notification-fanout", page: "web/src/pages/IncidentNotificationFanoutPage.tsx", title: "Incident Notification Fanout", keys: ["IncidentNotificationTarget", "incident-fanout", "record_incident_notification_target"] },
  { version: "V75", route: "/durable-artifact-backend", page: "web/src/pages/DurableArtifactBackendPage.tsx", title: "Durable Artifact Backend", keys: ["ArtifactBackend", "artifact-backends", "record_artifact_backend"] },
  { version: "V76", route: "/remaining-project-outcome-adapters", page: "web/src/pages/RemainingProjectOutcomeAdaptersPage.tsx", title: "Remaining Project Outcome Adapters", keys: ["OutcomeAdapterAdoption", "outcome-adapter-adoption", "record_project_outcome_adapter_adoption"] },
  { version: "V77", route: "/breaker-middleware-rollout", page: "web/src/pages/BreakerMiddlewareRolloutPage.tsx", title: "Breaker Middleware Rollout", keys: ["BreakerRolloutRecord", "breaker-rollout", "record_breaker_middleware_rollout"] },
  { version: "V78", route: "/provider-eval-execution", page: "web/src/pages/ProviderEvalExecutionPage.tsx", title: "Provider Eval Execution", keys: ["ProviderEvalExecution", "provider-eval-execution", "record_provider_eval_execution"] },
  { version: "V79", route: "/billing-provider-integrations", page: "web/src/pages/BillingProviderIntegrationsPage.tsx", title: "Billing Provider Integrations", keys: ["BillingProviderIntegration", "billing-provider-integrations", "record_billing_provider_integration"] },
  { version: "V80", route: "/release-train-execution", page: "web/src/pages/ReleaseTrainExecutionPage.tsx", title: "Release Train Execution", keys: ["ReleaseTrainExecution", "release-train-execution", "execute_release_train_record"] },
];

const selected = targetVersion === "ALL" ? versions : versions.filter((entry) => entry.version === targetVersion);
if (selected.length === 0) {
  fail(`Unknown version "${targetVersion}". Expected one of ${versions.map((entry) => entry.version).join(", ")} or ALL.`);
}

const files = {
  plan: read("docs/design/v71-v80-operational-readiness-build-plan.md"),
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
  assertIncludes(files.registry, entry.route, `${entry.version} route is registered`);
  assertIncludes(files.metadata, entry.route, `${entry.version} metadata route exists`);
  assertIncludes(files.metadata, `dashboard:${entry.version.toLowerCase()}:validate`, `${entry.version} metadata validation exists`);
  assertIncludes(files.data, entry.version, `${entry.version} data record exists`);
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

console.log(`V71-V80 operational readiness validation passed for ${selected.map((entry) => entry.version).join(", ")}.`);

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
