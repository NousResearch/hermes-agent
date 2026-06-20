'use strict'

const INSTALLED_PIPE_GUARD = Symbol.for('hermes.desktop.stdioPipeGuardInstalled')

function isIgnorablePipeError(error) {
  if (!error) return false

  const code = error.code || error.errno
  if (code === 'EPIPE' || code === 'ERR_STREAM_DESTROYED') return true

  return typeof error.message === 'string' && /\b(?:broken pipe|EPIPE|ERR_STREAM_DESTROYED)\b/i.test(error.message)
}

function attachPipeGuard(stream) {
  if (!stream || typeof stream.on !== 'function') return false
  if (stream[INSTALLED_PIPE_GUARD]) return false

  const handler = error => {
    if (isIgnorablePipeError(error)) return
    throw error
  }

  Object.defineProperty(stream, INSTALLED_PIPE_GUARD, {
    configurable: false,
    enumerable: false,
    value: true
  })
  stream.on('error', handler)

  return true
}

function installStdioPipeErrorGuards({ stdout = process.stdout, stderr = process.stderr } = {}) {
  return [stdout, stderr].filter(attachPipeGuard).length
}

module.exports = {
  installStdioPipeErrorGuards,
  isIgnorablePipeError
}
