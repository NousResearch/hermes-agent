#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const live = process.argv.includes("--live") || process.env.DASHBOARD_HEALTH_LIVE === "1";
const timeoutMs = Number(process.env.DASHBOARD_HEALTH_TIMEOUT_MS || 3000);
const registries = discoverDefaultRegistries();
const failures = [];
const healthTargets = [];

for (const registryPath of registries) {
  const registry = JSON.parse(fs.readFileSync(registryPath, "utf8"));
  for (const dashboard of registry.dashboards || []) {
    if (!dashboard.healthUrl) {
      failures.push(`${registryPath}:${dashboard.id} missing healthUrl`);
      continue;
    }
    healthTargets.push({
      registryPath,
      id: dashboard.id,
      healthUrl: dashboard.healthUrl,
    });
  }
}

if (live) {
  for (const target of healthTargets) {
    if (target.healthUrl.startsWith("/")) {
      console.log(`skip ${target.id} ${target.healthUrl} (relative health URL needs deployment base URL)`);
      continue;
    }
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      const response = await fetch(target.healthUrl, {
        method: "GET",
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (!response.ok) {
        failures.push(`${target.id} ${target.healthUrl} returned ${response.status}`);
      }
    } catch (error) {
      failures.push(`${target.id} ${target.healthUrl} failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

if (failures.length) {
  console.error(`Dashboard health validation failed (${failures.length})`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`Dashboard health validation passed (${healthTargets.length} targets${live ? ", live" : ", static"})`);

function discoverDefaultRegistries() {
  const here = path.resolve(root, "hermes.dashboards.json");
  const projectRoot = path.resolve(root, "..");
  const siblingRegistries = [];
  if (fs.existsSync(projectRoot)) {
    for (const entry of fs.readdirSync(projectRoot, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const candidate = path.join(projectRoot, entry.name, "hermes.dashboards.json");
      if (fs.existsSync(candidate) && path.resolve(candidate) !== here) {
        siblingRegistries.push(candidate);
      }
    }
  }
  return [here, ...siblingRegistries.sort()];
}
