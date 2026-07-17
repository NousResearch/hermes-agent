#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];
const recipeIds = [
  "executive-command-center",
  "operations-control-room",
  "research-intelligence-dashboard",
  "pipeline-workflow-dashboard",
  "cost-capacity-dashboard",
  "market-asset-explorer",
  "brand-business-performance",
  "system-health-deployment",
];

check("dashboard metadata covers governed routes", () => {
  const metadata = read("web/src/dashboard-page-metadata.ts");
  const routeRegistry = read("web/src/dashboard-route-registry.tsx");
  for (const route of ["/design-system", "/executive-summary", "/dashboard-migrations"]) {
    requireIncludes(routeRegistry, `"${route}"`, `route registry ${route}`);
    requireIncludes(metadata, `route: "${route}"`, `metadata ${route}`);
  }
});

check("dashboard metadata uses approved recipes", () => {
  const metadata = read("web/src/dashboard-page-metadata.ts");
  const recipeCatalog = read("web/src/pages/dashboard-recipes.ts");
  for (const recipeId of recipeIds) requireIncludes(recipeCatalog, `id: "${recipeId}"`, `recipe catalog ${recipeId}`);
  const declaredRecipes = [...metadata.matchAll(/recipe: "([^"]+)"/g)].map((match) => match[1]);
  if (!declaredRecipes.length) throw new Error("no recipes declared");
  for (const recipe of declaredRecipes) {
    if (!recipeIds.includes(recipe)) throw new Error(`unapproved recipe ${recipe}`);
  }
});

check("metadata enforces data, states, and validation", () => {
  const metadata = read("web/src/dashboard-page-metadata.ts");
  const entries = [...metadata.matchAll(/\{\n\s+route: "[^"]+"[\s\S]*?\n\s+\}/g)].map((match) => match[0]);
  if (entries.length < 3) throw new Error("expected at least three governed dashboard entries");
  for (const entry of entries) {
    for (const field of ["dataContracts", "requiredStates", "validation"]) {
      requireIncludes(entry, `${field}: [`, `${field} array`);
    }
    if (!entry.includes("normal")) throw new Error("metadata entry missing normal state");
    if (!entry.includes("mobile")) throw new Error("metadata entry missing mobile state");
  }
});

check("agent guidance references V11 governance", () => {
  const plan = read("docs/design/v8-v14-trackable-build-plan.md");
  requireIncludes(plan, "Require recipe metadata for every governed dashboard route", "recipe metadata requirement");
  requireIncludes(plan, "Require data contract metadata before dashboard implementation", "data contract requirement");
  requireIncludes(plan, "Require screenshot evidence in final handoff", "screenshot evidence requirement");
});

check("package script", () => {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v11:validate"] !== "node scripts/validate-v11-agent-governance.mjs") {
    throw new Error("missing dashboard:v11:validate script");
  }
});

finish("V11 agent governance validation");

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
