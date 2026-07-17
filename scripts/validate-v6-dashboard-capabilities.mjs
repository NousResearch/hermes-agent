#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const checks = [
  ["dashboard route registry", validateRouteRegistry],
  ["dashboard scaffold registry integration", validateScaffoldIntegration],
  ["executive query adapter", validateExecutiveQueryAdapter],
  ["TanStack Query provider", validateQueryProvider],
  ["V5 quality gate", () => run("npm", ["run", "dashboard:v5:validate"])],
];

const failures = [];

for (const [label, check] of checks) {
  try {
    check();
    console.log(`ok ${label}`);
  } catch (error) {
    failures.push(`${label}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

if (failures.length) {
  console.error(`V6 dashboard capabilities validation failed (${failures.length})`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log("V6 dashboard capabilities validation passed");

function validateRouteRegistry() {
  const file = read("web/src/dashboard-route-registry.tsx");
  requireIncludes(file, "export const BUILTIN_ROUTES_CORE", "built-in route export");
  requireIncludes(file, "export const BUILTIN_NAV_REST", "built-in nav export");
  requireIncludes(file, '"/executive-summary": ExecutiveSummaryPage', "executive summary route");
  requireIncludes(file, '{ path: "/executive-summary"', "executive summary nav item");
}

function validateScaffoldIntegration() {
  const file = read("scripts/scaffold-dashboard-page.mjs");
  requireIncludes(file, "web/src/dashboard-route-registry.tsx", "route registry default");
  requireIncludes(file, "registerRouteRegistry", "registry registration function");
  requireIncludes(file, "Could not register", "anchor failure guard");
  if (file.includes("web/src/App.tsx")) {
    throw new Error("scaffolder still defaults to direct App.tsx edits");
  }
}

function validateExecutiveQueryAdapter() {
  const file = read("web/src/pages/executive-data.ts");
  requireIncludes(file, "useQuery", "TanStack Query hook");
  requireIncludes(file, "api.getPlugins", "live dashboard signal source");
  requireIncludes(file, "placeholderData", "offline/fallback data handling");
}

function validateQueryProvider() {
  const main = read("web/src/main.tsx");
  const pkg = JSON.parse(read("web/package.json"));
  requireIncludes(main, "QueryClientProvider", "React Query provider");
  if (!pkg.dependencies?.["@tanstack/react-query"]) {
    throw new Error("web package missing @tanstack/react-query dependency");
  }
}

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function requireIncludes(text, needle, label) {
  if (!text.includes(needle)) {
    throw new Error(`missing ${label}`);
  }
}

function run(command, args) {
  execFileSync(command, args, {
    cwd: root,
    stdio: "inherit",
  });
}
