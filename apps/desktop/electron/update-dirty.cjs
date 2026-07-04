'use strict'

const DIRTY_UPDATE_ERROR = 'dirty-working-tree'

function isDirtyStatus(status) {
  return String(status || '').trim().length > 0
}

function dirtyUpdateMessage(hermesRoot) {
  const suffix = hermesRoot ? `\n\nCheckout: ${hermesRoot}` : ''
  return `Hermes paused the update because this source checkout has local changes. Commit or stash them first so working features are not hidden or replaced by the rebuild.${suffix}`
}

function dirtyUpdateResult(hermesRoot) {
  return {
    ok: false,
    dirty: true,
    error: DIRTY_UPDATE_ERROR,
    message: dirtyUpdateMessage(hermesRoot),
    hermesRoot
  }
}

module.exports = {
  DIRTY_UPDATE_ERROR,
  dirtyUpdateMessage,
  dirtyUpdateResult,
  isDirtyStatus
}
