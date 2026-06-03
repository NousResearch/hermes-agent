'use strict'

function shouldUseLocalUpdater(connection) {
  if (!connection) return true
  return connection.mode !== 'remote'
}

module.exports = { shouldUseLocalUpdater }
