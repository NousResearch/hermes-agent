const fs = require('node:fs')
const path = require('node:path')

const DEFAULT_MAX_BYTES = 5 * 1024 * 1024
const DEFAULT_BACKUP_COUNT = 3
const MAX_DETAIL_DEPTH = 4
const MAX_STRING_LENGTH = 500
const SENSITIVE_KEY_RE = /(api[_-]?key|auth|authorization|bearer|cookie|credential|password|secret|session|token)/i

function diagnosticsBackupPath(filePath, n) {
  return `${filePath}.${n}`
}

function planDiagnosticsRotation(filePath, size, maxBytes = DEFAULT_MAX_BYTES, backupCount = DEFAULT_BACKUP_COUNT) {
  if (size < maxBytes) return []
  const ops = [['rm', diagnosticsBackupPath(filePath, backupCount)]]
  for (let i = backupCount - 1; i >= 1; i--) {
    ops.push(['mv', diagnosticsBackupPath(filePath, i), diagnosticsBackupPath(filePath, i + 1)])
  }
  ops.push(['mv', filePath, diagnosticsBackupPath(filePath, 1)])
  return ops
}

function redactDiagnosticValue(value, depth = 0) {
  if (value == null || typeof value === 'boolean' || typeof value === 'number') {
    return value
  }
  if (typeof value === 'string') {
    return value.length > MAX_STRING_LENGTH ? `${value.slice(0, MAX_STRING_LENGTH)}...` : value
  }
  if (depth >= MAX_DETAIL_DEPTH) {
    return '[Truncated]'
  }
  if (Array.isArray(value)) {
    return value.slice(0, 20).map(item => redactDiagnosticValue(item, depth + 1))
  }
  if (typeof value === 'object') {
    const next = {}
    for (const [key, inner] of Object.entries(value)) {
      next[key] = SENSITIVE_KEY_RE.test(key) ? '[Redacted]' : redactDiagnosticValue(inner, depth + 1)
    }
    return next
  }
  return String(value)
}

function normalizeDiagnosticEvent(input = {}) {
  const now = new Date().toISOString()
  const severity = ['debug', 'info', 'warn', 'error', 'fatal'].includes(input.severity) ? input.severity : 'info'
  const component = typeof input.component === 'string' && input.component ? input.component : 'desktop'
  const event = typeof input.event === 'string' && input.event ? input.event : 'event'
  const message = typeof input.message === 'string' ? input.message : ''
  const details = redactDiagnosticValue(input.details ?? {})

  return {
    ts: typeof input.ts === 'string' ? input.ts : now,
    severity,
    component,
    event,
    message,
    details
  }
}

function createDiagnosticsRecorder(options) {
  const filePath = options.filePath
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES
  const backupCount = options.backupCount ?? DEFAULT_BACKUP_COUNT
  const recentLimit = options.recentLimit ?? 300
  const recent = []

  function rotateIfNeeded() {
    let size
    try {
      size = fs.statSync(filePath).size
    } catch {
      return
    }

    for (const [op, src, dst] of planDiagnosticsRotation(filePath, size, maxBytes, backupCount)) {
      try {
        if (op === 'rm') fs.rmSync(src, { force: true })
        else fs.renameSync(src, dst)
      } catch {
        // Diagnostics must never crash the shell.
      }
    }
  }

  function record(input) {
    const entry = normalizeDiagnosticEvent(input)
    recent.push(entry)
    if (recent.length > recentLimit) {
      recent.splice(0, recent.length - recentLimit)
    }

    try {
      fs.mkdirSync(path.dirname(filePath), { recursive: true })
      rotateIfNeeded()
      fs.appendFileSync(filePath, `${JSON.stringify(entry)}\n`, 'utf8')
    } catch {
      // Diagnostics must never block app startup/shutdown.
    }

    return entry
  }

  return {
    filePath,
    record,
    recent: limit => recent.slice(-(limit ?? recentLimit))
  }
}

module.exports = {
  createDiagnosticsRecorder,
  normalizeDiagnosticEvent,
  planDiagnosticsRotation,
  redactDiagnosticValue
}
