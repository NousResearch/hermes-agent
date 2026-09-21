import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { isObject, normalizeSkillName, resolveUserPath } from "./utils.mjs";

const DEFAULT_PRIMARY_PATH = path.join(os.homedir(), ".openclaw", "security-audit.json");
const DEFAULT_FALLBACK_PATH = ".clawsec/allowlist.json";

const EMPTY_CONFIG = Object.freeze({
  suppressions: [],
  enabledFor: [],
  source: "none",
});

/**
 * @param {unknown} entry
 * @param {number} index
 * @param {string} source
 * @returns {{ checkId: string, skill: string, reason: string, suppressedAt: string }}
 */
function normalizeRule(entry, index, source) {
  if (!isObject(entry)) {
    throw new Error(`Suppression entry at index ${index} in ${source} must be an object`);
  }

  const checkId = typeof entry.checkId === "string" ? entry.checkId.trim() : "";
  const skill = typeof entry.skill === "string" ? entry.skill.trim() : "";
  const reason = typeof entry.reason === "string" ? entry.reason.trim() : "";
  const suppressedAt = typeof entry.suppressedAt === "string" ? entry.suppressedAt.trim() : "";

  if (!checkId) throw new Error(`Suppression entry at index ${index} in ${source} missing required field: checkId`);
  if (!skill) throw new Error(`Suppression entry at index ${index} in ${source} missing required field: skill`);
  if (!reason) throw new Error(`Suppression entry at index ${index} in ${source} missing required field: reason`);
  if (!suppressedAt) throw new Error(`Suppression entry at index ${index} in ${source} missing required field: suppressedAt`);

  return { checkId, skill, reason, suppressedAt };
}

/**
 * @param {unknown} raw
 * @param {string} source
 * @returns {{ suppressions: Array, enabledFor: string[], source: string }}
 */
function parseConfig(raw, source) {
  if (!isObject(raw)) {
    throw new Error(`Config at ${source} must be a JSON object`);
  }

  if (!Array.isArray(raw.suppressions)) {
    throw new Error(`Config at ${source} missing 'suppressions' array`);
  }

  const suppressions = [];
  for (let i = 0; i < raw.suppressions.length; i++) {
    suppressions.push(normalizeRule(raw.suppressions[i], i, source));
  }

  const enabledFor = Array.isArray(raw.enabledFor)
    ? raw.enabledFor
        .filter((v) => typeof v === "string" && v.trim() !== "")
        .map((v) => v.trim().toLowerCase())
    : [];

  return { suppressions, enabledFor, source };
}

/**
 * @param {string} configPath
 * @returns {Promise<{ suppressions: Array, enabledFor: string[], source: string } | null>}
 */
async function loadConfigFromPath(configPath) {
  try {
    const raw = await fs.readFile(configPath, "utf8");
    return parseConfig(JSON.parse(raw), configPath);
  } catch (err) {
    if (err.code === "ENOENT") return null;
    if (err.code === "EACCES") throw new Error(`Permission denied reading config: ${configPath}`, { cause: err });
    if (err instanceof SyntaxError) throw new Error(`Malformed JSON in ${configPath}: ${err.message}`, { cause: err });
    throw err;
  }
}

/**
 * Load advisory suppression config using the same 4-tier path resolution
 * as the audit watchdog config loader.
 *
 * The config file must include "advisory" in its enabledFor sentinel
 * array for advisory suppression to activate. No CLI flag needed -- the
 * sentinel in the config file IS the gate.
 *
 * @param {string} [configPath] - Optional explicit config file path
 * @returns {Promise<{ suppressions: Array, enabledFor: string[], source: string }>}
 */
export async function loadAdvisorySuppression(configPath) {
  // Priority 1: Explicit path
  if (configPath) {
    const resolved = resolveUserPath(configPath, { label: "advisory suppression config path" });
    const config = await loadConfigFromPath(resolved);
    if (!config) throw new Error(`Advisory suppression config not found: ${resolved}`);
    if (!config.enabledFor.includes("advisory")) return { ...EMPTY_CONFIG };
    return config;
  }

  // Priority 2: Environment variable
  const envPath = process.env.OPENCLAW_AUDIT_CONFIG;
  if (typeof envPath === "string" && envPath.trim()) {
    const resolved = resolveUserPath(envPath.trim(), { label: "OPENCLAW_AUDIT_CONFIG" });
    const config = await loadConfigFromPath(resolved);
    if (config && config.enabledFor.includes("advisory")) return config;
    return { ...EMPTY_CONFIG };
  }

  // Priority 3: Primary default path
  const primary = await loadConfigFromPath(DEFAULT_PRIMARY_PATH);
  if (primary && primary.enabledFor.includes("advisory")) return primary;

  // Priority 4: Fallback path
  const fallback = await loadConfigFromPath(DEFAULT_FALLBACK_PATH);
  if (fallback && fallback.enabledFor.includes("advisory")) return fallback;

  return { ...EMPTY_CONFIG };
}

/**
 * Check if an advisory match should be suppressed.
 *
 * Matching requires BOTH:
 *   - advisory.id === rule.checkId (exact)
 *   - normalizeSkillName(skill.name) === normalizeSkillName(rule.skill) (case-insensitive)
 *
 * @param {{ advisory: { id?: string }, skill: { name: string } }} match
 * @param {Array<{ checkId: string, skill: string }>} suppressions
 * @returns {boolean}
 */
export function isAdvisorySuppressed(match, suppressions) {
  if (!Array.isArray(suppressions) || suppressions.length === 0) return false;

  const advisoryId = match.advisory.id ?? "";
  const skillName = normalizeSkillName(match.skill.name);

  return suppressions.some(
    (rule) => rule.checkId === advisoryId && normalizeSkillName(rule.skill) === skillName,
  );
}
