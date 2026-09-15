export interface RuntimeModeEnvironment {
  HERMES_DESKTOP_FORCE_DEV?: string | undefined
  HERMES_DESKTOP_IS_PACKAGED?: string | undefined
}

/**
 * Electron 41 can report packaged mode while Playwright launches a development
 * checkout as `electron <desktop-dir>` on Windows. The isolated E2E fixture
 * sets the explicit test-only override; real packaged launches never do.
 */
export function isPackagedDesktopRuntime(
  electronReportsPackaged: boolean,
  env: RuntimeModeEnvironment,
): boolean {
  return env.HERMES_DESKTOP_FORCE_DEV !== '1' && (
    electronReportsPackaged || Boolean(env.HERMES_DESKTOP_IS_PACKAGED)
  )
}
