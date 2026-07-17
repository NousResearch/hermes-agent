#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];

check("quality scorecard document", () => {
  const text = read("docs/design/dashboard-quality-scorecard.md");
  for (const phrase of [
    "Minimum passing score",
    "Recipe fit",
    "Data contract",
    "Decision clarity",
    "Production readiness",
    "Failure Conditions",
  ]) {
    requireIncludes(text, phrase, phrase);
  }
});

check("quality metadata exists", () => {
  const text = read("web/src/dashboard-page-metadata.ts");
  for (const route of ["/design-system", "/executive-summary", "/dashboard-migrations"]) {
    requireIncludes(text, `route: "${route}"`, `metadata for ${route}`);
  }
  for (const field of ["recipe", "dataContracts", "requiredStates", "validation", "owner", "category"]) {
    requireIncludes(text, field, `metadata field ${field}`);
  }
});

check("visual tests cover V10 routes", () => {
  const text = read("tests/dashboard/design-system.spec.ts");
  requireIncludes(text, "/dashboard-migrations", "dashboard migrations visual route");
  requireIncludes(text, "V7 Full-Page Dashboard Recipes", "recipe gallery assertion");
  requireIncludes(text, "Package-Native Dashboard Migrations", "migration dashboard assertion");
});

check("package script", () => {
  const pkg = JSON.parse(read("package.json"));
  if (pkg.scripts?.["dashboard:v10:validate"] !== "node scripts/validate-v10-dashboard-quality.mjs") {
    throw new Error("missing dashboard:v10:validate script");
  }
});

finish("V10 dashboard quality validation");

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
