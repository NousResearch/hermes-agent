const IGNORE_EXISTING_ENV = 'HERMES_DESKTOP_IGNORE_EXISTING'

/**
 * The explicit developer overrides (HERMES_DESKTOP_HERMES_ROOT and an
 * unpackaged source checkout) name a runtime the operator typed by hand.
 * Everything below them on the resolve ladder — the managed install at
 * ACTIVE_HERMES_ROOT, `hermes` on PATH, the system-python hermes_cli
 * module — is *discovered*, and that is what "ignore existing" promises
 * to skip.
 */
export function shouldIgnoreDiscoveredLocalRuntimes(env: NodeJS.ProcessEnv = process.env): boolean {
  return env[IGNORE_EXISTING_ENV] === '1'
}
