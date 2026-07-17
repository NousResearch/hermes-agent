#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];

check("marketplace package exports", () => {
  const marketplace = read("packages/hermes-dashboard-kit/src/marketplace.ts");
  for (const phrase of ["DashboardPluginManifest", "DashboardPluginPanel", "DashboardPluginCommand", "DashboardPluginPermission", "dashboardPluginHasSignal", "dashboardPluginRequiresAdmin"]) {
    requireIncludes(marketplace, phrase, phrase);
  }
  requireIncludes(read("packages/hermes-dashboard-kit/src/index.ts"), 'export * from "./marketplace"', "marketplace export");
});

check("marketplace registry and route", () => {
  const data = read("web/src/pages/dashboard-marketplace-data.ts");
  for (const id of ["khashi-vc.roc", "media-engine.ops", "hermes.central-command"]) {
    requireIncludes(data, id, id);
  }
  requireIncludes(read("web/src/dashboard-route-registry.tsx"), '"/dashboard-marketplace": DashboardMarketplacePage', "marketplace route");
  requireIncludes(read("web/src/dashboard-page-metadata.ts"), 'route: "/dashboard-marketplace"', "marketplace metadata");
});

check("marketplace page", () => {
  const page = read("web/src/pages/DashboardMarketplacePage.tsx");
  for (const phrase of ["Dashboard Marketplace", "Permission-Aware Commands", "dashboardPluginRequiresAdmin"]) {
    requireIncludes(page, phrase, phrase);
  }
});

check("package script", () => {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v14:validate"] !== "node scripts/validate-v14-dashboard-marketplace.mjs") {
    throw new Error("missing dashboard:v14:validate script");
  }
});

finish("V14 dashboard marketplace validation");

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
