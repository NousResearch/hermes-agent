#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];

check("trackable V8-V14 plan", () => {
  const text = read("docs/design/v8-v14-trackable-build-plan.md");
  for (const version of ["V8", "V9", "V10", "V11", "V12", "V13", "V14"]) {
    requireIncludes(text, `## ${version}:`, `${version} section`);
  }
  requireIncludes(text, "Media Engine Ops Migration", "Media Engine migration");
  requireIncludes(text, "Khashi VC ROC Migration", "Khashi migration");
});

check("V8 migration route", () => {
  const route = read("web/src/dashboard-route-registry.tsx");
  requireIncludes(route, '"/dashboard-migrations": PackageNativeMigrationsPage', "route registry entry");
  requireIncludes(route, "Dashboard Migrations", "nav entry");
  const page = read("web/src/pages/PackageNativeMigrationsPage.tsx");
  for (const recipe of ["pipeline-workflow-dashboard", "operations-control-room", "market-asset-explorer", "executive-command-center"]) {
    requireIncludes(page, recipe, `recipe ${recipe}`);
  }
  requireIncludes(page, "Retire Adapter", "adapter retirement guardrail");
});

check("package script", () => {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v8:validate"] !== "node scripts/validate-v8-dashboard-migrations.mjs") {
    throw new Error("missing dashboard:v8:validate script");
  }
});

finish("V8 dashboard migration validation");

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
