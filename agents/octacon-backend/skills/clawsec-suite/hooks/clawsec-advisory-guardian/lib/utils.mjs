import os from "node:os";
import path from "node:path";

/**
 * @param {unknown} value
 * @returns {value is Record<string, unknown>}
 */
export function isObject(value) {
  return typeof value === "object" && value !== null;
}

/**
 * @param {string} value
 * @returns {string}
 */
export function normalizeSkillName(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

/**
 * @param {string[]} values
 * @returns {string[]}
 */
export function uniqueStrings(values) {
  return Array.from(new Set(values));
}

function detectHomeDirectory(env = process.env) {
  if (typeof env.HOME === "string" && env.HOME.trim()) return env.HOME.trim();
  if (typeof env.USERPROFILE === "string" && env.USERPROFILE.trim()) return env.USERPROFILE.trim();
  if (
    typeof env.HOMEDRIVE === "string" &&
    env.HOMEDRIVE.trim() &&
    typeof env.HOMEPATH === "string" &&
    env.HOMEPATH.trim()
  ) {
    return `${env.HOMEDRIVE.trim()}${env.HOMEPATH.trim()}`;
  }
  return os.homedir();
}

const UNEXPANDED_HOME_TOKEN_PATTERN =
  /(?:^|[\\/])(?:\\?\$HOME|\\?\$\{HOME\}|\\?\$USERPROFILE|\\?\$\{USERPROFILE\}|%HOME%|%USERPROFILE%|\$env:HOME|\$env:USERPROFILE)(?:$|[\\/])/i;

/**
 * @param {string} value
 * @returns {string}
 */
function expandKnownHomeTokens(value) {
  const homeDir = detectHomeDirectory(process.env);
  if (!homeDir) return value;

  let expanded = String(value ?? "");

  if (expanded === "~") {
    expanded = homeDir;
  } else if (expanded.startsWith("~/") || expanded.startsWith("~\\")) {
    expanded = path.join(homeDir, expanded.slice(2));
  }

  expanded = expanded
    .replace(/(?<!\\)\$\{HOME\}/g, homeDir)
    .replace(/(?<!\\)\$HOME(?=$|[\\/])/g, homeDir)
    .replace(/(?<!\\)\$\{USERPROFILE\}/gi, homeDir)
    .replace(/(?<!\\)\$USERPROFILE(?=$|[\\/])/gi, homeDir)
    .replace(/%HOME%/gi, homeDir)
    .replace(/%USERPROFILE%/gi, homeDir)
    .replace(/(?<!\\)\$env:HOME/gi, homeDir)
    .replace(/(?<!\\)\$env:USERPROFILE/gi, homeDir);

  return expanded;
}

/**
 * @param {string} value
 * @returns {boolean}
 */
export function hasUnexpandedHomeToken(value) {
  return UNEXPANDED_HOME_TOKEN_PATTERN.test(String(value ?? "").trim());
}

/**
 * Expand `~` and known home env var patterns in user-provided path-like strings.
 * Also fails fast when unresolved home tokens remain.
 *
 * @param {string} inputPath
 * @param {{label?: string}} [options]
 * @returns {string}
 */
export function resolveUserPath(inputPath, { label = "path" } = {}) {
  const raw = String(inputPath ?? "").trim();
  if (!raw) return raw;

  const expanded = expandKnownHomeTokens(raw);
  const normalized = path.normalize(expanded);

  if (hasUnexpandedHomeToken(normalized)) {
    throw new Error(
      `Unexpanded home token detected in ${label}: ${raw}. ` +
        "Use an absolute path or an unquoted home-path expression.",
    );
  }

  return normalized;
}

/**
 * Resolve an optional explicit path; if invalid, fall back to a default path.
 *
 * @param {string | undefined} explicitPath
 * @param {string} fallbackPath
 * @param {{label?: string, onInvalid?: (error: unknown, rawValue: string) => void}} [options]
 * @returns {string}
 */
export function resolveConfiguredPath(
  explicitPath,
  fallbackPath,
  { label = "path", onInvalid } = {},
) {
  const explicit = typeof explicitPath === "string" ? explicitPath.trim() : "";
  if (!explicit) {
    return resolveUserPath(fallbackPath, { label });
  }

  try {
    return resolveUserPath(explicit, { label });
  } catch (error) {
    if (typeof onInvalid === "function") {
      onInvalid(error, explicit);
    }
    return resolveUserPath(fallbackPath, { label });
  }
}
