'use strict'

// Remote-Only Mode (issue #54363).
//
// Hermes Desktop is a native local application, so its IPC surface can read,
// list, and preview files on the user's workstation even when the agent is
// connected to a remote gateway. Operators who run Desktop purely as a remote
// client want a hard guarantee that the local filesystem is never browsed,
// indexed, previewed, or exposed.
//
// When HERMES_DESKTOP_DISABLE_LOCAL_FILES is enabled (set by `hermes desktop
// --remote-only`, or directly in the environment) every local-filesystem IPC
// path funnels through assertLocalFilesAllowed() and is refused. Remote-backend
// browsing is unaffected because that traffic goes over the gateway API, not
// the local fs handlers.

const LOCAL_FILES_DISABLED_ENV = 'HERMES_DESKTOP_DISABLE_LOCAL_FILES'

// Accept the usual truthy spellings so the flag behaves the same whether it is
// exported by a shell, a launcher, or the CLI wrapper.
const TRUTHY = new Set(['1', 'true', 'yes', 'on'])

const LOCAL_FILES_DISABLED_CODE = 'local-files-disabled'
const LOCAL_FILES_DISABLED_MESSAGE =
  'Remote-only mode is enabled, so Hermes Desktop will not browse or open local workstation files. ' +
  `Unset ${LOCAL_FILES_DISABLED_ENV} (or relaunch without --remote-only) to allow local file access.`

function isLocalFilesDisabled(env = process.env) {
  const raw = env && env[LOCAL_FILES_DISABLED_ENV]
  if (raw === undefined || raw === null) {
    return false
  }
  return TRUTHY.has(String(raw).trim().toLowerCase())
}

function localFilesPolicy(env = process.env) {
  const disabled = isLocalFilesDisabled(env)
  return {
    disabled,
    reason: disabled ? LOCAL_FILES_DISABLED_MESSAGE : null
  }
}

function localFilesDisabledError(purpose = 'Local file access') {
  const error = new Error(`${purpose} blocked: ${LOCAL_FILES_DISABLED_MESSAGE}`)
  error.code = LOCAL_FILES_DISABLED_CODE
  return error
}

// Throw when the local filesystem is off-limits. Callers that funnel through
// hardening.cjs get this for free; native-dialog handlers call it directly.
function assertLocalFilesAllowed(purpose = 'Local file access', options = {}) {
  const env = options.env || process.env
  if (isLocalFilesDisabled(env)) {
    throw localFilesDisabledError(purpose)
  }
}

module.exports = {
  LOCAL_FILES_DISABLED_CODE,
  LOCAL_FILES_DISABLED_ENV,
  LOCAL_FILES_DISABLED_MESSAGE,
  assertLocalFilesAllowed,
  isLocalFilesDisabled,
  localFilesDisabledError,
  localFilesPolicy
}
