// Defaults must run before bundled modules resolve paths or onboarding flags.
// An esbuild define would replace reads, not populate the child process env.

/** Validate a bundle environment object: plain object of identifiers to
 * strings (defaults) or null (clears). Shared by the banner writer and the
 * smoke driver's stamp reader.
 * @param {unknown} values @returns {Record<string, string|null>} */
export function validateBundleEnvironment(values) {
  if (!values || Array.isArray(values) || typeof values !== 'object') {
    throw new Error('Bundle environment must be a JSON object')
  }
  for (const [key, value] of Object.entries(values)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || (value !== null && (typeof value !== 'string' || value.includes('\0')))) {
      throw new Error('Bundle environment requires valid names and string values without NUL, or null to clear')
    }
  }
  return /** @type {Record<string, string|null>} */ (values)
}

/** Apply bundle environment defaults/clears to a base environment and return a
 * new object without mutating it. This is the pure twin of the side-effecting
 * loop the bundled banner emits at launch — keep the two in lockstep (pinned
 * by bundle-env.test.mjs). Clears (null) set the key to an empty string
 * unconditionally so a stale registry or ambient value cannot win; defaults
 * fill only an absent key, so an explicit runtime value always wins.
 * @param {Record<string, string|undefined>} env
 * @param {Record<string, string|null>} values */
export function applyBundleEnvironment(env, values) {
  const result = { ...env }
  for (const [key, value] of Object.entries(values)) {
    if (value === null) result[key] = ''
    else if (result[key] == null) result[key] = value
  }
  return result
}

/** @param {string} raw @returns {string} */
export function environmentDefaultsBanner(raw) {
  const values = validateBundleEnvironment(JSON.parse(raw))
  // Keep clears present-but-empty so path resolution cannot restore registry
  // defaults. The emitted loop is the side-effecting twin of
  // applyBundleEnvironment above; it cannot call it (the banner runs before
  // the bundle's imports), so the two must stay in lockstep — see the test.
  return `\nfor (const [key, value] of ${JSON.stringify(Object.entries(values))}) { if (value === null) process.env[key] = ''; else process.env[key] ??= value; }\n`
}
