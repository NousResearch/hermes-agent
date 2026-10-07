import fs from 'node:fs'
import path from 'node:path'

function directoryExists(dir) {
  try {
    return fs.statSync(dir).isDirectory()
  } catch {
    return false
  }
}

function hasDashboardBundle(dir) {
  try {
    return fs.statSync(path.join(dir, 'index.html')).isFile()
  } catch {
    return false
  }
}

/**
 * Resolve the web bundle served by `rabbit dashboard` / `rabbit serve` when it
 * is spawned by Desktop as the local backend.
 *
 * This must be the dashboard bundle (`rabbit_cli/web_dist`), not the Electron
 * renderer bundle (`apps/desktop/dist` or the packaged `app.asar.unpacked/dist`).
 * The Electron renderer is loaded by BrowserWindow from file://; the dashboard
 * backend is also reachable in a normal browser, where serving the Desktop
 * renderer produces a broken page (`Desktop IPC bridge is unavailable`).
 */
function resolveDashboardWebDist(options: any = {}) {
  const env = options.env || process.env
  const override = env.RABBIT_DESKTOP_DASHBOARD_WEB_DIST

  if (override && directoryExists(path.resolve(override))) {
    return path.resolve(override)
  }

  const candidates = []

  if (options.activeRabbitRoot) {
    candidates.push(path.join(options.activeRabbitRoot, 'rabbit_cli', 'web_dist'))
  }

  if (options.appRoot) {
    // Dev/source layout: <repo>/apps/desktop/electron/main.ts has APP_ROOT at
    // <repo>/apps/desktop. Packaged layout can put APP_ROOT under resources/app.asar;
    // this candidate is harmless there and useful in tests/source checkouts.
    candidates.push(path.resolve(options.appRoot, '..', '..', 'rabbit_cli', 'web_dist'))
  }

  for (const candidate of candidates) {
    if (hasDashboardBundle(candidate)) {
      return candidate
    }
  }

  // Fall back to the canonical active install path even if it does not exist so
  // the child `rabbit dashboard --skip-build` fails with its explicit missing
  // dist error instead of silently serving the wrong Desktop bundle.
  if (options.activeRabbitRoot) {
    return path.join(options.activeRabbitRoot, 'rabbit_cli', 'web_dist')
  }

  return path.resolve('rabbit_cli', 'web_dist')
}

export { resolveDashboardWebDist }
