'use strict'

// Historical desktop builds constructed argv for a frontend-managed local
// `hermes serve`/`dashboard` process. Reuben Desktop is remote-only, so these
// helpers intentionally reject any attempt to build local backend commands.

function localBackendDisabledError(feature) {
  return new Error(
    `${feature} is disabled. Reuben Desktop is a remote frontend; configure a gateway URL instead of spawning a local backend.`
  )
}

function serveBackendArgs() {
  throw localBackendDisabledError('Local backend argv construction')
}

function dashboardFallbackArgs() {
  throw localBackendDisabledError('Legacy dashboard fallback')
}

function sourceDeclaresServe() {
  return false
}

module.exports = {
  dashboardFallbackArgs,
  localBackendDisabledError,
  serveBackendArgs,
  sourceDeclaresServe
}
