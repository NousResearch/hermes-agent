// app-updater.ts — the win32 update arm for bundled desktop installs.
//
//   win32  — the OS App Installer owns the apply. The package was installed
//            from an .appinstaller, which registered the feed URI as the
//            package's update source; the OS checks it and swaps the package
//            wholesale. The app's only job is the checker: ask the OS whether
//            an update is available (via the bundled payload python's winrt),
//            show its own prompt, run graceful teardown, then trigger
//            the downloaded App Installer file and quit. Installations that Windows manages
//            for us declare updateMechanism 'external'. They get no in-app
//            updater: the store or package manager owns the update loop.
//
// Source installs never reach this module. The callers gate on the install
// stamp first and fall through to the git-based update path.
//
// macOS packaged updates live in updater/mac.ts and updater/mac-client.ts.
//
// The win32 helpers are pure so vitest covers them; the impure pieces
// (electron shell, payload python) are injected.

// ─── feed hosting ───────────────────────────────────────────────────────────

/**
 * The App Installer feed dir for a channel. The .appinstaller for a channel
 * lives under this dir; the OS re-reads it on launch (HoursBetweenUpdateChecks)
 * and swaps the bundle when a newer version is there. The light variant feeds
 * from its own subtree so the two variants can never serve each other's
 * packages.
 */
export function win32AppInstallerFeedPath(
  channel: 'stable' | 'canary',
  light: boolean
): string {
  const variant = light ? 'light/' : ''

  return `win32/${variant}${channel}/`
}

// ─── win32 arm (OS App Installer checker + trigger) ────────────────────────

/** The result of an App Installer update check. */
export interface AppInstallerCheck {
  /** null when the check could not run (no winrt, not packaged, error). */
  available: boolean | null
  /** Human-readable availability string from the OS, when reported. */
  availability?: string
  error?: string
  sourceUri?: string
}

/**
 * The payload-python invocation the win32 arm needs. Injectable so vitest
 * covers the arm without a payload.
 */
export interface PayloadPythonRunner {
  /** Absolute path to the bundled payload python (tools/<entry>/python.exe). */
  python: string
  /** The checker script's absolute path. */
  script: string
  /** Run the script; resolve with {code, stdout}. */
  run: (python: string, script: string) => Promise<{ code: number; stdout: string }>
}

/**
 * win32 arm: ask the OS whether an App Installer update is available, via
 * the bundled payload python's winrt (Package.check_update_availability_async).
 * The OS compares the package's registered .appinstaller source against the
 * installed version; it does NOT download anything.
 */
export async function checkAppInstallerUpdate(runner: PayloadPythonRunner): Promise<AppInstallerCheck> {
  const { code, stdout } = await runner.run(runner.python, runner.script)

  return parseCheckOutput(code, stdout)
}

export function parseCheckOutput(code: number, stdout: string): AppInstallerCheck {
  let parsed: { available?: boolean | null; availability?: string; error?: string; source_uri?: string } | null = null

  try {
    parsed = stdout.trim() ? JSON.parse(stdout) : null
  } catch {
    return { available: null, error: code !== 0 ? `checker exited ${code}` : 'checker returned invalid JSON' }
  }

  if (typeof parsed?.available === 'boolean') {
    return { available: parsed.available, availability: parsed.availability, error: parsed.error, sourceUri: parsed.source_uri }
  }

  return { available: null, error: parsed?.error || (code !== 0 ? `checker exited ${code}` : 'checker returned no availability') }
}

/** Open a local descriptor. The ms-appinstaller protocol is disabled by default. */
export async function triggerAppInstallerUpdate(
  feedBaseUrl: string,
  channel: 'stable' | 'canary',
  light: boolean,
  installer: { prepare: (url: string) => Promise<string>; open: (file: string) => Promise<string> },
  beforeInstall?: () => void | Promise<void>,
  sourceUri?: string
): Promise<{ ok: true }> {
  const appinstallerUrl = sourceUri ||
    `${feedBaseUrl.replace(/\/+$/, '')}/${win32AppInstallerFeedPath(channel, light)}${channel}.appinstaller`

  const file = await installer.prepare(appinstallerUrl)

  if (beforeInstall) {
    await beforeInstall()
  }

  const error = await installer.open(file)

  if (error) { throw new Error(`App Installer could not open: ${error}`) }

  return { ok: true }
}
