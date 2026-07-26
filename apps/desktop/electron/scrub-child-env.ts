/**
 * Shared credential scrub for Desktop Electron child processes.
 *
 * Desktop often spreads `{ ...process.env }` into PTY / serve / updater children.
 * Provider and messaging secrets belong in HERMES_HOME/.env for the backend, not
 * in the parent Electron environment forwarded wholesale to every child.
 */

const CREDENTIAL_SUFFIXES = Object.freeze([
  '_API_KEY',
  '_TOKEN',
  '_SECRET',
  '_PASSWORD',
  '_CREDENTIALS',
  '_ACCESS_KEY',
  '_PRIVATE_KEY',
  '_OAUTH_TOKEN'
])

const CREDENTIAL_NAMES = new Set([
  'ANTHROPIC_BASE_URL',
  'ANTHROPIC_TOKEN',
  'AWS_ACCESS_KEY_ID',
  'AWS_SECRET_ACCESS_KEY',
  'AWS_SESSION_TOKEN',
  'CUSTOM_API_KEY',
  'GEMINI_BASE_URL',
  'OPENAI_BASE_URL',
  'OPENROUTER_BASE_URL',
  'OLLAMA_BASE_URL',
  'GROQ_BASE_URL',
  'XAI_BASE_URL'
])

export function isCredentialEnvVar(name: string): boolean {
  if (!name) {
    return false
  }

  if (CREDENTIAL_NAMES.has(name)) {
    return true
  }

  return CREDENTIAL_SUFFIXES.some(suffix => name.endsWith(suffix))
}

export type EnvMap = Record<string, string | undefined>

/**
 * Copy `source` while dropping credential-shaped keys. Optionally re-apply
 * explicit overrides afterwards (e.g. a minted dashboard session token).
 */
export function scrubDesktopChildEnv(source: EnvMap = {}, overrides: EnvMap = {}): Record<string, string> {
  const out: Record<string, string> = {}

  for (const [key, value] of Object.entries(source || {})) {
    if (value == null || value === '') {
      continue
    }

    if (isCredentialEnvVar(key)) {
      continue
    }

    out[key] = String(value)
  }

  for (const [key, value] of Object.entries(overrides || {})) {
    if (value == null || value === '') {
      delete out[key]
      continue
    }

    out[key] = String(value)
  }

  return out
}

export { CREDENTIAL_NAMES, CREDENTIAL_SUFFIXES }
