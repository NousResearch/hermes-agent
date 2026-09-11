// Defaults must run before bundled modules resolve paths or onboarding flags.
// An esbuild define would replace reads, not populate the child process env.
/** @param {string} raw @returns {string} */
export function environmentDefaultsBanner(raw) {
  const values = JSON.parse(raw)
  if (!values || Array.isArray(values) || typeof values !== 'object') {
    throw new Error('Bundle environment must be a JSON object')
  }
  for (const [key, value] of Object.entries(values)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || typeof value !== 'string' || value.includes('\0')) {
      throw new Error('Bundle environment requires valid names and string values without NUL')
    }
  }
  return `\nfor (const [key, value] of ${JSON.stringify(Object.entries(values))}) { process.env[key] ??= value; }\n`
}
