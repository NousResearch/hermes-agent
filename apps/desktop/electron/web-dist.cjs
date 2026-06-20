const path = require('node:path')

function resolveWebDist({
  activeHermesRoot,
  appRoot,
  directoryExists,
  env = process.env,
  isPackaged = false,
  rememberLog = () => undefined,
  sourceRepoRoot,
  unpackedPathFor
}) {
  const override = env.HERMES_DESKTOP_WEB_DIST
  if (override && directoryExists(path.resolve(override))) {
    return path.resolve(override)
  }

  const dashboardCandidates = [
    sourceRepoRoot && path.join(sourceRepoRoot, 'hermes_cli', 'web_dist'),
    activeHermesRoot && path.join(activeHermesRoot, 'hermes_cli', 'web_dist')
  ].filter(Boolean)

  for (const candidate of dashboardCandidates) {
    if (directoryExists(candidate)) {
      return candidate
    }
  }

  const unpackedDist = unpackedPathFor ? path.join(unpackedPathFor(appRoot), 'dist') : null
  if (unpackedDist && directoryExists(unpackedDist)) {
    return unpackedDist
  }

  // Final fallback: appRoot/dist. When packaged with asar:true this can live
  // inside app.asar rather than on the real filesystem; log that because the
  // backend cannot serve static files out of an asar path.
  const fallback = path.join(appRoot, 'dist')
  if (isPackaged && /app\.asar(?=$|[\\/])/.test(fallback) && !directoryExists(fallback)) {
    rememberLog(
      `[web-dist] dashboard frontend dir resolved to an asar-internal path that ` +
        `is not a real directory: ${fallback}. Static routes will 404. ` +
        `Ensure dist/** is unpacked (asarUnpack) or set HERMES_DESKTOP_WEB_DIST.`
    )
  }

  return fallback
}

module.exports = { resolveWebDist }

