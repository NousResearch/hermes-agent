#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const args = new Set(process.argv.slice(2));
const strict = args.has("--strict");
const sync = args.has("--sync");
const allowMissingExternal = args.has("--allow-missing-external");
const registryPath = path.resolve(root, "docs/design/dashboard-kit-adoption.json");

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function hash(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function relative(file) {
  return path.relative(root, file) || ".";
}

if (!fs.existsSync(registryPath)) {
  console.error(`Missing adoption registry: ${relative(registryPath)}`);
  process.exit(1);
}

const registry = readJson(registryPath);
const sourcePath = path.resolve(root, registry.source.cssPath);

if (!fs.existsSync(sourcePath)) {
  console.error(`Missing canonical CSS source: ${relative(sourcePath)}`);
  process.exit(1);
}

const sourceHash = hash(sourcePath);
const rows = [];
const failures = [];

for (const dashboard of registry.dashboards ?? []) {
  const targetPath = path.resolve(root, dashboard.targetPath);
  const targetRelative = path.relative(root, targetPath);
  const isExternal = targetRelative.startsWith("..") || path.isAbsolute(targetRelative);
  let actual = "missing";
  let targetHash = "";

  if (fs.existsSync(targetPath)) {
    if (sync && dashboard.type === "static-adapter") {
      fs.copyFileSync(sourcePath, targetPath);
    }
    targetHash = hash(targetPath);
    actual = targetHash === sourceHash ? "synced" : "drifted";
  }

  if (actual === "missing" && allowMissingExternal && isExternal) {
    actual = "external-unavailable";
  }

  const expected = dashboard.status ?? "unknown";
  rows.push({
    project: dashboard.project,
    name: dashboard.name,
    type: dashboard.type,
    expected,
    actual,
    target: dashboard.targetPath,
    hash: targetHash ? targetHash.slice(0, 12) : "",
  });

  if (actual !== "synced" && actual !== "external-unavailable") {
    failures.push(`${dashboard.project}: ${actual} (${dashboard.targetPath})`);
  }
}

console.log(`Hermes dashboard design-system status`);
console.log(`Source: ${registry.source.package}@${registry.source.version}`);
console.log(`CSS: ${registry.source.cssPath}`);
console.log(`Source hash: ${sourceHash}`);
console.table(rows);

if (sync) {
  console.log("Sync mode completed for static-adapter targets.");
}

if (allowMissingExternal) {
  console.log("External dashboard targets may be unavailable in single-repo CI.");
}

if ((strict || sync) && failures.length) {
  console.error(`Design-system adoption check failed (${failures.length})`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

if (failures.length) {
  console.log(`Drift or missing adapters found: ${failures.length}`);
  process.exit(0);
}

console.log(`All registered dashboard adapters are synced.`);
