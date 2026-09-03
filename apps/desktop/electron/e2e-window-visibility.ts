export const E2E_HIDDEN_WINDOW_ENV = 'HERMES_DESKTOP_E2E_HIDDEN'

/**
 * Keep Playwright's Electron windows rendered but off the active desktop.
 * This is an internal test-process switch, never a user-facing app setting.
 */
export function shouldKeepE2EWindowsHidden(env: NodeJS.ProcessEnv): boolean {
  return env[E2E_HIDDEN_WINDOW_ENV] === '1'
}
