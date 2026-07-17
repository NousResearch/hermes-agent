#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const files = process.argv.slice(2);
const targets = files.length ? files : discoverDefaultRegistries();
let failures = 0;

for (const target of targets) {
  const file = path.resolve(target);
  try {
    validateFile(file);
    console.log(`ok ${target}`);
  } catch (error) {
    failures += 1;
    console.error(`error ${target}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

if (failures) {
  process.exit(1);
}

function validateFile(file) {
  if (!fs.existsSync(file)) {
    throw new Error("file does not exist");
  }
  const registry = JSON.parse(fs.readFileSync(file, "utf8"));
  if (!registry || typeof registry !== "object" || Array.isArray(registry)) {
    throw new Error("registry must be an object");
  }
  if (!Array.isArray(registry.dashboards)) {
    throw new Error("registry.dashboards must be an array");
  }

  const ids = new Set();
  for (const [index, dashboard] of registry.dashboards.entries()) {
    const label = `dashboards[${index}]`;
    if (!dashboard || typeof dashboard !== "object" || Array.isArray(dashboard)) {
      throw new Error(`${label} must be an object`);
    }
    requireString(dashboard, "id", label);
    requireString(dashboard, "label", label);
    requireString(dashboard, "description", label);
    requireString(dashboard, "category", label);
    requireString(dashboard, "owner", label);

    if (ids.has(dashboard.id)) {
      throw new Error(`${label}.id duplicates ${dashboard.id}`);
    }
    ids.add(dashboard.id);

    const url = dashboard.productionUrl || dashboard.url || dashboard.localUrl;
    if (!url || typeof url !== "string") {
      throw new Error(`${label} must declare url, localUrl, or productionUrl`);
    }
    validateUrlLike(url, `${label}.url`);

    if (dashboard.healthUrl !== undefined) {
      requireString(dashboard, "healthUrl", label);
      validateUrlLike(dashboard.healthUrl, `${label}.healthUrl`);
    }

    if (dashboard.command !== undefined) {
      if (!Array.isArray(dashboard.command)) {
        throw new Error(`${label}.command must be an array when present`);
      }
      for (const [commandIndex, part] of dashboard.command.entries()) {
        if (typeof part !== "string") {
          throw new Error(`${label}.command[${commandIndex}] must be a string`);
        }
      }
    }

    if (dashboard.projectPath !== undefined && typeof dashboard.projectPath !== "string") {
      throw new Error(`${label}.projectPath must be a string when present`);
    }
    if (dashboard.projectName !== undefined && typeof dashboard.projectName !== "string") {
      throw new Error(`${label}.projectName must be a string when present`);
    }
  }
}

function requireString(object, key, label) {
  if (typeof object[key] !== "string" || !object[key].trim()) {
    throw new Error(`${label}.${key} must be a non-empty string`);
  }
}

function validateUrlLike(value, label) {
  if (value.startsWith("/") || value.startsWith("http://") || value.startsWith("https://")) {
    return;
  }
  throw new Error(`${label} must be an absolute http(s) URL or root-relative path`);
}

function discoverDefaultRegistries() {
  const here = path.resolve("hermes.dashboards.json");
  const projectRoot = path.resolve("..");
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
