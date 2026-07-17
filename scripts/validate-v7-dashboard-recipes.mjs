#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const expectedIds = [
  "executive-command-center",
  "operations-control-room",
  "research-intelligence-dashboard",
  "pipeline-workflow-dashboard",
  "cost-capacity-dashboard",
  "market-asset-explorer",
  "brand-business-performance",
  "system-health-deployment",
];

const checks = [
  ["V7 recipe document", validateRecipeDocument],
  ["gallery recipe data", validateGalleryData],
  ["scaffolder recipe support", validateScaffolder],
  ["build plan V7 entry", validateBuildPlan],
  ["package script", validatePackageScript],
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
  console.error(`V7 dashboard recipe validation failed (${failures.length})`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log("V7 dashboard recipe validation passed");

function validateRecipeDocument() {
  const text = read("docs/design/v7-dashboard-layout-recipes.md");
  requireIncludes(text, "Status: `[x]` Complete", "complete status");
  for (const title of [
    "Executive Command Center",
    "Operations Control Room",
    "Research Intelligence Dashboard",
    "Pipeline Workflow Dashboard",
    "Cost And Capacity Dashboard",
    "Market Asset Explorer",
    "Brand Business Performance Dashboard",
    "System Health And Deployment Dashboard",
  ]) {
    requireIncludes(text, title, `recipe ${title}`);
  }
}

function validateGalleryData() {
  const text = read("web/src/pages/dashboard-recipes.ts");
  const ids = [...text.matchAll(/id: "([^"]+)"/g)].map((match) => match[1]);
  assertSameIds(ids, "web/src/pages/dashboard-recipes.ts");
  for (const field of ["layout", "dataContract", "components", "validation"]) {
    requireIncludes(text, `${field}: [`, `${field} arrays`);
  }
}

function validateScaffolder() {
  const text = read("scripts/scaffold-dashboard-page.mjs");
  requireIncludes(text, "--recipe", "recipe CLI option");
  requireIncludes(text, "printRecipes", "recipe listing helper");
  requireIncludes(text, "recipePageTemplate", "recipe page template");
  for (const id of expectedIds) requireIncludes(text, id, `scaffolder recipe ${id}`);
}

function validateBuildPlan() {
  const text = read("docs/design/hermes-dashboard-design-system-build-plan.md");
  requireIncludes(text, "## Version 7: Package-Native Premium Dashboard Adoption", "V7 plan section");
  requireIncludes(text, "dashboard:v7:validate", "V7 validation command");
}

function validatePackageScript() {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v7:validate"] !== "node scripts/validate-v7-dashboard-recipes.mjs") {
    throw new Error("missing dashboard:v7:validate script");
  }
}

function assertSameIds(ids, label) {
  const actual = [...new Set(ids)].sort();
  const expected = [...expectedIds].sort();
  if (actual.length !== expected.length || actual.some((id, index) => id !== expected[index])) {
    throw new Error(`${label} recipe ids mismatch: ${actual.join(", ")}`);
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
