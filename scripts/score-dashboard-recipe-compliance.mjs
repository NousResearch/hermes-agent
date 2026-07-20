#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const strict = process.argv.includes("--strict");
const minimumPassingScore = 80;
const metadataPath = "web/src/dashboard-page-metadata.ts";
const recipesPath = "web/src/pages/dashboard-recipes.ts";

const metadataText = read(metadataPath);
const recipesText = read(recipesPath);
const approvedRecipes = new Set([...recipesText.matchAll(/id: "([^"]+)"/g)].map((match) => match[1]));
const entries = [...metadataText.matchAll(/\{\n\s+route: "[^"]+"[\s\S]*?\n\s+\}/g)].map((match) => match[0]);
const rows = entries.map(scoreEntry);
const failed = rows.filter((row) => row.score < minimumPassingScore);

console.log(`Dashboard recipe compliance score: ${rows.length} governed routes`);
console.log(`Minimum passing score: ${minimumPassingScore}`);
console.table(rows.map(({ route, recipe, score, missing }) => ({
  route,
  recipe,
  score,
  missing: missing.join("; "),
})));

if (failed.length) {
  console.error(`Recipe compliance failed (${failed.length})`);
  for (const row of failed) {
    console.error(`- ${row.route}: ${row.score}/100 missing ${row.missing.join(", ")}`);
  }
  if (strict) process.exit(1);
}

if (!failed.length) {
  console.log("Dashboard recipe compliance passed.");
}

function scoreEntry(entry) {
  const route = stringField(entry, "route");
  const recipe = stringField(entry, "recipe");
  const dataContracts = arrayField(entry, "dataContracts");
  const requiredStates = arrayField(entry, "requiredStates");
  const validation = arrayField(entry, "validation");
  const missing = [];
  let score = 0;

  if (recipe && approvedRecipes.has(recipe)) {
    score += 20;
  } else {
    missing.push("approved recipe");
  }

  if (dataContracts.length >= 2) {
    score += 15;
  } else {
    missing.push("two or more data contracts");
  }

  if (requiredStates.length >= 4 && requiredStates.includes("mobile")) {
    score += 15;
  } else {
    missing.push("mobile plus at least three operational states");
  }

  if (validation.some((item) => item.startsWith("dashboard:v")) && validation.includes("dashboard:visual:check")) {
    score += 15;
  } else {
    missing.push("version validator plus visual check");
  }

  if (validation.some((item) => item.includes("build --workspace web"))) {
    score += 10;
  } else {
    missing.push("web build validation");
  }

  if (hasStringField(entry, "owner") && hasStringField(entry, "category") && hasStringField(entry, "title")) {
    score += 10;
  } else {
    missing.push("owner, category, and title");
  }

  if (hasStringField(entry, "productionUrl") || hasStringField(entry, "localOnlyReason")) {
    score += 5;
  } else {
    missing.push("production URL or local-only reason");
  }

  if (
    metadataText.includes("dashboardGovernanceDefaults") &&
    metadataText.includes("sourcePackage: \"@hermes/dashboard-kit\"") &&
    metadataText.includes("finalHandoffEvidence")
  ) {
    score += 10;
  } else {
    missing.push("design-system governance defaults");
  }

  return { route, recipe, score, missing };
}

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function stringField(entry, field) {
  return entry.match(new RegExp(`${field}: "([^"]+)"`))?.[1] ?? "";
}

function hasStringField(entry, field) {
  return Boolean(stringField(entry, field));
}

function arrayField(entry, field) {
  const arrayText = entry.match(new RegExp(`${field}: \\[(.*)\\],`))?.[1] ?? "";
  return [...arrayText.matchAll(/"([^"]+)"/g)].map((match) => match[1]);
}
