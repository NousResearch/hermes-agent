#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const targetVersion = (process.argv[2] || "ALL").toUpperCase();

const versions = [
  {
    version: "V15",
    route: "/live-signals",
    page: "web/src/pages/LiveSignalsPage.tsx",
    title: "Live Project Signal Integration",
    dataKeys: ["LiveSignalIntegration", "liveSignalIntegrations", "DashboardSnapshot", "HealthSnapshot"],
    docKeys: ["V15 Live Project Signal Integration", "Signal Integrations"],
  },
  {
    version: "V16",
    route: "/task-routing",
    page: "web/src/pages/TaskRoutingPage.tsx",
    title: "Agent Task Routing",
    dataKeys: ["RoutedTask", "routedTasks", "priority", "status"],
    docKeys: ["V16 Agent Task Routing", "Work Intake Queue"],
  },
  {
    version: "V17",
    route: "/decision-ledger",
    page: "web/src/pages/DecisionLedgerPage.tsx",
    title: "Memory And Decision Ledger",
    dataKeys: ["DecisionRecord", "decisionLedger", "reason", "reviewedAt"],
    docKeys: ["V17 Memory And Decision Ledger", "Decision Records"],
  },
  {
    version: "V18",
    route: "/model-routing",
    page: "web/src/pages/ModelRoutingPage.tsx",
    title: "Model And Cost Routing",
    dataKeys: ["ModelRoutingPolicy", "modelRoutingPolicies", "premium-approval", "approvalRequired"],
    docKeys: ["V18 Model And Cost Routing", "approval gates"],
  },
  {
    version: "V19",
    route: "/operating-loops",
    page: "web/src/pages/OperatingLoopsPage.tsx",
    title: "Autonomous Operating Loops",
    dataKeys: ["OperatingLoop", "operatingLoops", "cadence", "output"],
    docKeys: ["V19 Autonomous Operating Loops", "Operating Loop Registry"],
  },
  {
    version: "V20",
    route: "/permission-security",
    page: "web/src/pages/PermissionSecurityPage.tsx",
    title: "Secure Tool And Permission Layer",
    dataKeys: ["PermissionPolicy", "permissionPolicies", "explicit", "audit"],
    docKeys: ["V20 Secure Tool And Permission Layer", "Permission Policies"],
  },
  {
    version: "V21",
    route: "/business-os",
    page: "web/src/pages/BusinessOSPage.tsx",
    title: "TLC Business Operating System",
    dataKeys: ["BusinessScorecard", "businessScorecards", "revenueSignal", "costSignal"],
    docKeys: ["V21 TLC Business Operating System", "Business Unit Scorecards"],
  },
];

const selected = targetVersion === "ALL" ? versions : versions.filter((entry) => entry.version === targetVersion);

if (selected.length === 0) {
  fail(`Unknown version "${targetVersion}". Expected one of ${versions.map((entry) => entry.version).join(", ")} or ALL.`);
}

const files = {
  plan: read("docs/design/v15-v21-operating-system-build-plan.md"),
  registry: read("web/src/dashboard-route-registry.tsx"),
  metadata: read("web/src/dashboard-page-metadata.ts"),
  data: read("web/src/pages/operating-system-data.ts"),
  mainPlan: read("docs/design/hermes-dashboard-design-system-build-plan.md"),
};

for (const entry of selected) {
  assertIncludes(files.plan, entry.version, `plan includes ${entry.version}`);
  assertIncludes(files.plan, "100% trackable infrastructure", "plan marks infrastructure completion");
  assertIncludes(files.mainPlan, "Version 15-21", "main plan links V15-V21");
  assertIncludes(files.registry, entry.route, `${entry.version} route is registered`);
  assertIncludes(files.metadata, entry.route, `${entry.version} metadata route exists`);
  assertIncludes(files.metadata, `dashboard:${entry.version.toLowerCase()}:validate`, `${entry.version} metadata validation is registered`);
  assertIncludes(files.metadata, entry.title, `${entry.version} metadata title exists`);

  const page = read(entry.page);
  assertIncludes(page, entry.title, `${entry.version} page heading exists`);
  for (const key of entry.docKeys) {
    assertIncludes(files.plan + page, key, `${entry.version} exposes ${key}`);
  }
  for (const key of entry.dataKeys) {
    assertIncludes(files.data + page + files.metadata, key, `${entry.version} data contract includes ${key}`);
  }
}

console.log(`V15-V21 operating-system validation passed for ${selected.map((entry) => entry.version).join(", ")}.`);

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
