/**
 * @param {string} version
 * @returns {[number, number, number] | null}
 */
export function parseSemver(version) {
  const cleaned = String(version ?? "")
    .trim()
    .replace(/^v/i, "")
    .split("-")[0];
  const parts = cleaned.split(".");
  if (parts.length === 0) return null;

  const normalized = parts.slice(0, 3).map((part) => Number.parseInt(part, 10));
  while (normalized.length < 3) {
    normalized.push(0);
  }

  if (normalized.some((part) => Number.isNaN(part))) {
    return null;
  }
  return /** @type {[number, number, number]} */ (normalized);
}

/**
 * @param {string} left
 * @param {string} right
 * @returns {number | null}
 */
export function compareSemver(left, right) {
  const a = parseSemver(left);
  const b = parseSemver(right);
  if (!a || !b) return null;

  for (let index = 0; index < 3; index += 1) {
    if (a[index] > b[index]) return 1;
    if (a[index] < b[index]) return -1;
  }
  return 0;
}

/**
 * @param {string} value
 * @returns {string}
 */
export function escapeRegex(value) {
  return String(value ?? "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * @param {string | null} version
 * @param {string} rawSpec
 * @returns {boolean}
 */
export function versionMatches(version, rawSpec) {
  const spec = String(rawSpec ?? "").trim();
  if (!spec || spec === "*" || spec.toLowerCase() === "any") return true;
  if (!version || String(version).trim().toLowerCase() === "unknown") return false;

  const normalizedVersion = String(version).trim().replace(/^v/i, "");

  if (spec.includes("*")) {
    const regex = new RegExp(`^${escapeRegex(spec).replace(/\\\*/g, ".*")}$`);
    return regex.test(normalizedVersion);
  }

  const comparatorMatch = spec.match(/^(>=|<=|>|<|=)\s*(.+)$/);
  if (comparatorMatch) {
    const operator = comparatorMatch[1];
    const targetVersion = comparatorMatch[2].trim();
    const compared = compareSemver(normalizedVersion, targetVersion);
    if (compared === null) return false;
    if (operator === ">=") return compared >= 0;
    if (operator === "<=") return compared <= 0;
    if (operator === ">") return compared > 0;
    if (operator === "<") return compared < 0;
    return compared === 0;
  }

  if (spec.startsWith("^")) {
    const target = parseSemver(spec.slice(1));
    const current = parseSemver(normalizedVersion);
    if (!target || !current) return false;
    if (current[0] !== target[0]) return false;
    if (target[0] === 0 && current[1] !== target[1]) return false;
    return compareSemver(normalizedVersion, spec.slice(1)) !== -1;
  }

  if (spec.startsWith("~")) {
    const target = parseSemver(spec.slice(1));
    const current = parseSemver(normalizedVersion);
    if (!target || !current) return false;
    return (
      current[0] === target[0] &&
      current[1] === target[1] &&
      compareSemver(normalizedVersion, spec.slice(1)) !== -1
    );
  }

  return normalizedVersion === spec || normalizedVersion === spec.replace(/^v/i, "");
}
