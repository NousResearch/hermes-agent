const fs = require('node:fs')
const path = require('node:path')

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
 * Resolve the web bundle served by `hermes dashboard` when it is spawned by
 * Desktop as the local backend.
 *
 * This must be the dashboard bundle (`hermes_cli/web_dist`), not the Electron
 * renderer bundle (`apps/desktop/dist` or the packaged `app.asar.unpacked/dist`).
 * The Electron renderer is loaded by BrowserWindow from file://; the dashboard
 * backend is also reachable in a normal browser, where serving the Desktop
 * renderer produces a broken page (`Desktop IPC bridge is unavailable`).
 */
function resolveDashboardWebDist(options = {}) {
  const env = options.env || process.env
  const override = env.HERMES_DESKTOP_DASHBOARD_WEB_DIST
  if (override && directoryExists(path.resolve(override))) {
    return path.resolve(override)
  }

  const candidates = []

  if (options.activeHermesRoot) {
    candidates.push(path.join(options.activeHermesRoot, 'hermes_cli', 'web_dist'))
  }

  if (options.appRoot) {
    // Dev/source layout: <repo>/apps/desktop/electron/main.cjs has APP_ROOT at
    // <repo>/apps/desktop. Packaged layout can put APP_ROOT under resources/app.asar;
    // this candidate is harmless there and useful in tests/source checkouts.
    candidates.push(path.resolve(options.appRoot, '..', '..', 'hermes_cli', 'web_dist'))
  }

  for (const candidate of candidates) {
    if (hasDashboardBundle(candidate)) {
      return candidate
    }
  }

  // Fall back to the canonical active install path even if it does not exist so
  // the child `hermes dashboard --skip-build` fails with its explicit missing
  // dist error instead of silently serving the wrong Desktop bundle.
  if (options.activeHermesRoot) {
    return path.join(options.activeHermesRoot, 'hermes_cli', 'web_dist')
  }

  return path.resolve('hermes_cli', 'web_dist')
}

module.exports = {
  resolveDashboardWebDist
}
