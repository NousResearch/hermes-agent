const ADVISORY_APPLICATION_OPENCLAW = "openclaw";
const ADVISORY_APPLICATION_ALL = "all";

/**
 * @param {unknown} value
 * @returns {string[]}
 */
function normalizeApplicationValue(value) {
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    return normalized ? [normalized] : [];
  }

  if (Array.isArray(value)) {
    return value
      .filter((entry) => typeof entry === "string")
      .map((entry) => entry.trim().toLowerCase())
      .filter(Boolean);
  }

  return [];
}

/**
 * Decide whether an advisory should be considered by OpenClaw-facing flows.
 *
 * Backward compatibility rule:
 * - Advisories without `application` remain eligible.
 *
 * @param {{ application?: unknown }} advisory
 * @returns {boolean}
 */
export function advisoryAppliesToOpenclaw(advisory) {
  const application = advisory?.application;
  if (application === undefined || application === null) {
    return true;
  }

  const applications = normalizeApplicationValue(application);
  if (applications.length === 0) {
    return true;
  }

  return (
    applications.includes(ADVISORY_APPLICATION_OPENCLAW) ||
    applications.includes(ADVISORY_APPLICATION_ALL)
  );
}
