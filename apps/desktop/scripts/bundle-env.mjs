// Defaults must run before bundled modules resolve paths or onboarding flags.
// An esbuild define would replace reads, not populate the child process env.
/** @param {string} raw @returns {string} */
export function environmentDefaultsBanner(raw) {
  const values = JSON.parse(raw)
  if (!values || Array.isArray(values) || typeof values !== 'object') {
    throw new Error('Bundle environment must be a JSON object')
  }
  for (const [key, value] of Object.entries(values)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || (value !== null && (typeof value !== 'string' || value.includes('\0')))) {
      throw new Error('Bundle environment requires valid names and string values without NUL, or null to clear')
    }
  }
  // Keep clears present-but-empty so path resolution cannot restore registry defaults.
  return `\nfor (const [key, value] of ${JSON.stringify(Object.entries(values))}) { if (value === null) process.env[key] = ''; else process.env[key] ??= value; }\n`
}
