#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const failures = [];
const checks = [];

checkPackageExport();
checkPackageNativeConsumers();
checkStaticAdapterConsumers();
checkRegistries();
run("node", ["scripts/validate-dashboard-registry.mjs"], "dashboard registry schema");
run("node", ["scripts/validate-dashboard-health.mjs"], "dashboard health URLs");
run("node", ["scripts/validate-static-adapter-sync.mjs"], "static adapter sync");
run("node", ["scripts/validate-media-engine-generated-dashboard.mjs"], "Media Engine generated dashboard");
run("node", ["scripts/audit-dashboard-usage.mjs", "--strict"], "dashboard usage audit");

if (failures.length) {
  console.error(`V4 dashboard standardization validation failed (${failures.length})`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

for (const check of checks) console.log(`ok ${check}`);
console.log("V4 dashboard standardization validation passed");

function checkPackageExport() {
  const packagePath = path.join(root, "packages/hermes-dashboard-kit/package.json");
  const packageJson = JSON.parse(fs.readFileSync(packagePath, "utf8"));
  if (packageJson.name !== "@hermes/dashboard-kit") {
    failures.push("dashboard kit package must be named @hermes/dashboard-kit");
  }
  if (packageJson.exports?.["."]?.import !== "./dist/index.js") {
    failures.push("dashboard kit package must export the React entry from ./dist/index.js");
  }
  if (packageJson.exports?.["./static/hermes-dashboard-kit.css"] !== "./static/hermes-dashboard-kit.css") {
    failures.push("dashboard kit package must export ./static/hermes-dashboard-kit.css");
  }
  for (const requiredFile of ["dist", "static", "README.md", "CHANGELOG.md"]) {
    if (!packageJson.files?.includes(requiredFile)) {
      failures.push(`dashboard kit package files must include ${requiredFile}`);
    }
  }
  if (!fs.existsSync(path.join(root, "packages/hermes-dashboard-kit/static/hermes-dashboard-kit.css"))) {
    failures.push("dashboard kit static adapter CSS is missing");
  }
  checks.push("dashboard kit exports React and static adapter entries");
}

function checkPackageNativeConsumers() {
  const consumerFiles = [
    "web/src/pages/HermesOsPage.tsx",
    "web/src/pages/DesignSystemPage.tsx",
    "web/src/pages/ExecutiveSummaryPage.tsx",
  ];
  const missing = consumerFiles.filter((file) => {
    const fullPath = path.join(root, file);
    return !fs.existsSync(fullPath) || !fs.readFileSync(fullPath, "utf8").includes("@hermes/dashboard-kit");
  });
  if (missing.length) {
    failures.push(`package-native dashboard consumers missing @hermes/dashboard-kit imports: ${missing.join(", ")}`);
  }
  checks.push(`${consumerFiles.length} package-native dashboard consumers use @hermes/dashboard-kit`);
}

function checkStaticAdapterConsumers() {
  const adapterTargets = [
    "../khashi-vc/public/roc/hermes-dashboard-kit.css",
    "../media-engine/core/operations/hermes-dashboard-kit.css",
  ];
  for (const target of adapterTargets) {
    if (!fs.existsSync(path.resolve(root, target))) {
      failures.push(`static adapter consumer missing ${target}`);
    }
  }
  checks.push(`${adapterTargets.length} static adapter consumers are present`);
}

function checkRegistries() {
  const registries = discoverRegistries();
  const dashboardCount = registries.reduce((count, registryPath) => {
    const registry = JSON.parse(fs.readFileSync(registryPath, "utf8"));
    return count + (Array.isArray(registry.dashboards) ? registry.dashboards.length : 0);
  }, 0);
  if (registries.length < 8) {
    failures.push(`expected at least 8 dashboard registries, found ${registries.length}`);
  }
  if (dashboardCount < 10) {
    failures.push(`expected at least 10 dashboard registry entries, found ${dashboardCount}`);
  }
  checks.push(`${registries.length} registries and ${dashboardCount} dashboard entries discovered`);
}

function discoverRegistries() {
  const projectRoot = path.resolve(root, "..");
  const registries = [];
  for (const entry of fs.readdirSync(projectRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const candidate = path.join(projectRoot, entry.name, "hermes.dashboards.json");
    if (fs.existsSync(candidate)) registries.push(candidate);
  }
  return registries.sort();
}

function run(command, args, label) {
  try {
    execFileSync(command, args, {
      cwd: root,
      stdio: "pipe",
    });
    checks.push(label);
  } catch (error) {
    const stderr = error?.stderr?.toString()?.trim();
    const stdout = error?.stdout?.toString()?.trim();
    failures.push(`${label} failed${stderr || stdout ? `: ${stderr || stdout}` : ""}`);
  }
}
