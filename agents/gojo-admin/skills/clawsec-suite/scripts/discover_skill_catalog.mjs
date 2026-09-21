#!/usr/bin/env node

import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadTextFile } from "./local_file_io.mjs";

const DEFAULT_INDEX_URL = "https://clawsec.prompt.security/skills/index.json";
const DEFAULT_TIMEOUT_MS = 5000;

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const SUITE_DIR = path.resolve(SCRIPT_DIR, "..");
const SUITE_SKILL_JSON = path.join(SUITE_DIR, "skill.json");

function isObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeSkillId(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

function normalizeBoolean(value) {
  return value === true;
}

const ENVIRONMENT = (() => {
  const runtimeProcess = Reflect.get(globalThis, "process");
  if (!runtimeProcess || typeof runtimeProcess !== "object") return {};
  if (!("env" in runtimeProcess)) return {};
  const env = runtimeProcess.env;
  return env && typeof env === "object" ? env : {};
})();

function envVar(name) {
  const value = ENVIRONMENT[name];
  return typeof value === "string" ? value.trim() : "";
}

function parseTimeoutMs() {
  const raw = envVar("CLAWSEC_SKILLS_INDEX_TIMEOUT_MS");
  if (!raw) return DEFAULT_TIMEOUT_MS;

  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_TIMEOUT_MS;
  }
  return parsed;
}

function parseArgs(argv) {
  const args = {
    json: false,
  };

  for (const token of argv) {
    if (token === "--json") {
      args.json = true;
      continue;
    }
    if (token === "--help" || token === "-h") {
      printUsage();
      process.exit(0);
    }

    throw new Error(`Unknown argument: ${token}`);
  }

  return args;
}

function printUsage() {
  process.stdout.write(
    [
      "Usage:",
      "  node scripts/discover_skill_catalog.mjs [--json]",
      "",
      "Behavior:",
      "  - Fetches dynamic catalog from CLAWSEC_SKILLS_INDEX_URL (default: https://clawsec.prompt.security/skills/index.json)",
      "  - Falls back to suite-local catalog metadata in skill.json when remote index is unavailable/invalid",
      "",
      "Environment:",
      "  CLAWSEC_SKILLS_INDEX_URL         Override remote catalog index URL",
      "  CLAWSEC_SKILLS_INDEX_TIMEOUT_MS  HTTP timeout in milliseconds (default: 5000)",
      "",
    ].join("\n"),
  );
}

function normalizeRemoteSkills(payload) {
  if (!isObject(payload)) {
    throw new Error("Catalog index payload must be a JSON object");
  }

  const rawSkills = payload.skills;
  if (!Array.isArray(rawSkills)) {
    throw new Error("Catalog index missing skills array");
  }

  const dedup = new Map();

  for (const entry of rawSkills) {
    if (!isObject(entry)) continue;

    const id = normalizeSkillId(entry.id ?? entry.name);
    if (!id) continue;

    dedup.set(id, {
      id,
      name: String(entry.name ?? id),
      version: String(entry.version ?? "").trim() || null,
      description: String(entry.description ?? "").trim() || null,
      emoji: String(entry.emoji ?? "").trim() || null,
      category: String(entry.category ?? "").trim() || null,
      tag: String(entry.tag ?? "").trim() || null,
      trust: entry.trust ?? null,
      source: "remote",
    });
  }

  return {
    version: String(payload.version ?? "").trim() || null,
    updated: String(payload.updated ?? "").trim() || null,
    skills: [...dedup.values()].sort((a, b) => a.id.localeCompare(b.id)),
  };
}

async function loadFallbackCatalog() {
  const raw = await loadTextFile(SUITE_SKILL_JSON);
  const parsed = JSON.parse(raw);

  const catalogSkills = isObject(parsed?.catalog?.skills) ? parsed.catalog.skills : {};
  const dedup = new Map();

  for (const [rawId, meta] of Object.entries(catalogSkills)) {
    const id = normalizeSkillId(rawId);
    if (!id) continue;

    const safeMeta = isObject(meta) ? meta : {};

    dedup.set(id, {
      id,
      name: id,
      version: null,
      description: String(safeMeta.description ?? "").trim() || null,
      emoji: null,
      category: null,
      tag: null,
      trust: null,
      source: "fallback",
      integrated_in_suite: normalizeBoolean(safeMeta.integrated_in_suite),
      requires_explicit_consent: normalizeBoolean(safeMeta.requires_explicit_consent),
      default_install: normalizeBoolean(safeMeta.default_install),
    });
  }

  return {
    version: null,
    updated: null,
    skills: [...dedup.values()].sort((a, b) => a.id.localeCompare(b.id)),
  };
}

function mergeWithFallbackMetadata(remoteSkills, fallbackSkills) {
  const fallbackById = new Map(fallbackSkills.map((skill) => [skill.id, skill]));

  return remoteSkills.map((skill) => {
    const fallback = fallbackById.get(skill.id);
    if (!fallback) {
      return {
        ...skill,
        integrated_in_suite: false,
        requires_explicit_consent: false,
        default_install: false,
      };
    }

    return {
      ...skill,
      description: skill.description || fallback.description || null,
      integrated_in_suite: normalizeBoolean(fallback.integrated_in_suite),
      requires_explicit_consent: normalizeBoolean(fallback.requires_explicit_consent),
      default_install: normalizeBoolean(fallback.default_install),
    };
  });
}

async function loadRemoteCatalog(indexUrl, timeoutMs) {
  if (typeof globalThis.fetch !== "function") {
    throw new Error("fetch is unavailable in this runtime");
  }
  if (typeof globalThis.AbortController !== "function") {
    throw new Error("AbortController is unavailable in this runtime");
  }

  const controller = new globalThis.AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await globalThis.fetch(indexUrl, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while fetching catalog`);
    }

    const payload = await response.json();
    return normalizeRemoteSkills(payload);
  } finally {
    clearTimeout(timeout);
  }
}

function formatFlags(skill) {
  const flags = [];

  if (skill.id === "clawsec-suite") {
    flags.push("this suite");
  }
  if (skill.integrated_in_suite) {
    flags.push("already integrated in suite");
  }
  if (skill.requires_explicit_consent) {
    flags.push("explicit opt-in");
  }
  if (skill.default_install) {
    flags.push("recommended default");
  }

  return flags;
}

function printHumanSummary(result) {
  process.stdout.write("=== ClawSec Skill Catalog Discovery ===\n");
  process.stdout.write(`Source: ${result.source}\n`);
  process.stdout.write(`Index URL: ${result.index_url}\n`);
  if (result.updated) {
    process.stdout.write(`Catalog updated: ${result.updated}\n`);
  }
  if (result.warning) {
    process.stdout.write(`Fallback reason: ${result.warning}\n`);
  }

  process.stdout.write("\nAvailable installable skills:\n");

  if (!Array.isArray(result.skills) || result.skills.length === 0) {
    process.stdout.write("- none\n");
    return;
  }

  for (const skill of result.skills) {
    const label = skill.version ? `${skill.id} (v${skill.version})` : skill.id;
    process.stdout.write(`- ${label}\n`);
    if (skill.description) {
      process.stdout.write(`  ${skill.description}\n`);
    }

    const flags = formatFlags(skill);
    if (flags.length > 0) {
      process.stdout.write(`  notes: ${flags.join("; ")}\n`);
    }

    process.stdout.write(`  install: npx clawhub@latest install ${skill.id}\n`);
  }
}

async function discoverCatalog() {
  const indexUrl = envVar("CLAWSEC_SKILLS_INDEX_URL") || DEFAULT_INDEX_URL;
  const timeoutMs = parseTimeoutMs();
  const fallback = await loadFallbackCatalog();

  try {
    const remote = await loadRemoteCatalog(indexUrl, timeoutMs);

    return {
      source: "remote",
      index_url: indexUrl,
      version: remote.version,
      updated: remote.updated,
      skills: mergeWithFallbackMetadata(remote.skills, fallback.skills),
      warning: null,
    };
  } catch (error) {
    return {
      source: "fallback",
      index_url: indexUrl,
      version: fallback.version,
      updated: fallback.updated,
      skills: fallback.skills,
      warning: String(error),
    };
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const result = await discoverCatalog();

  if (args.json) {
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    return;
  }

  printHumanSummary(result);
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
