// app-updater.ts — the win32 update arm for bundled desktop installs.
//
//   win32  — the OS App Installer owns the apply. The package was installed
//            from an .appinstaller, which registered the feed URI as the
//            package's update source; the OS checks it and swaps the package
//            wholesale. The app's only job is the checker: ask the OS whether
//            an update is available (via the bundled payload python's winrt),
//            show its own prompt, run graceful teardown, then trigger
//            ms-appinstaller: and quit. Installations that Windows manages
//            for us — Microsoft Store deployments (process.windowsStore) and
//            stamps whose updateMechanism is 'external' — get NO in-app
//            updater at all: the store/steward owns the update loop.
//
// Source installs never reach this module. The callers gate on the install
// stamp first and fall through to the git-based update path.
//
// NOTE: the darwin electron-updater arm (and its gate/channel/feed helpers —
// shouldUseAppUpdater, selectUpdaterArm, resolveUpdaterChannel,
// resolveFeedBaseUrl, feedSelection, describeFeedCheck) was ripped out in
// wt/darwin-updater: main.ts reimplemented the gate/channel/feed inline, so
// the module surface was test-only. To add macOS in-app updates back, restore
// the arm from git history (this file @ the parent of that commit) together
// with the `electron-updater` dependency and its test block — see the
// add-back plan in the wt/darwin-updater commit message.
//
// The win32 helpers are pure so vitest covers them; the impure pieces
// (electron shell, payload python) are injected.

// ─── feed hosting ───────────────────────────────────────────────────────────

/**
 * PLACEHOLDER — the default base URL of the desktop release feeds.
 *
 * W3 (R2 hosting) supplies the real value; until it lands, this default is
 * a documented dead end that config overrides (`updates.desktop_feed_base_url`
 * in config.yaml — read inline by main.ts). Layout under the base:
 *
 *   <base>/win32/<channel>/            — App Installer feed dir (.appinstaller
 *                                        + .msixbundle) for out-of-store MSIX.
 *
 * `<channel>` is 'stable' or 'canary'.
 */
export const PLACEHOLDER_FEED_BASE_URL = 'https://updates.invalid/hermes-desktop'

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
 * the bundled payload python's winrt (PackageManager.CheckPackageUpdateAvailabilityAsync).
 * The OS compares the package's registered .appinstaller source against the
 * installed version; it does NOT download anything.
 */
export async function checkAppInstallerUpdate(
  runner: PayloadPythonRunner
): Promise<AppInstallerCheck> {
  const { code, stdout } = await runner.run(runner.python, runner.script)
  const text = stdout.trim()
  let parsed: { available?: boolean | null; availability?: string; error?: string; reason?: string } | null = null

  try {
    parsed = text ? JSON.parse(text) : null
  } catch {
    parsed = null
  }

  if (parsed && typeof parsed.available === 'boolean') {
    return { available: parsed.available, availability: parsed.availability, error: parsed.error }
  }

  if (parsed && parsed.available === null) {
    return { available: null, error: parsed.error || 'checker returned unknown' }
  }

  if (code !== 0) {
    return { available: null, error: parsed?.error || `checker exited ${code}` }
  }

  return { available: null, error: 'checker returned no availability' }
}

/**
 * win32 arm: trigger the OS App Installer to apply the update and quit.
 * `ms-appinstaller:?source=<feed .appinstaller URL>` makes the OS re-read the
 * package's update source and install the newer bundle; the app then exits so
 * the package swap is not contested. `beforeInstall` runs first (backend
 * teardown while the process is still alive).
 */
export async function triggerAppInstallerUpdate(
  feedBaseUrl: string,
  channel: 'stable' | 'canary',
  light: boolean,
  shell: { openExternal: (url: string) => Promise<void> },
  beforeInstall?: () => void | Promise<void>
): Promise<{ ok: true }> {
  if (beforeInstall) {
    await beforeInstall()
  }

  const appinstallerUrl =
    `${feedBaseUrl.replace(/\/+$/, '')}/${win32AppInstallerFeedPath(channel, light)}${channel}.appinstaller`

  await shell.openExternal(`ms-appinstaller:?source=${encodeURIComponent(appinstallerUrl)}`)

  return { ok: true }
}
