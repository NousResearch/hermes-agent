'use strict'

const crypto = require('node:crypto')

const BROKER_PROTOCOL_VERSION = 1
const DEFAULT_TTL_MS = 30_000
const DEFAULT_CLOCK_SKEW_MS = 5_000

const MAC_PERMISSION_BROKER_METHODS = Object.freeze({
  'permission.status': Object.freeze({
    category: 'permissions',
    requiresUserGrant: false,
    description: 'Return broker registration and TCC permission status.'
  }),
  'permission.openSettings': Object.freeze({
    category: 'permissions',
    requiresUserGrant: false,
    description: 'Open the relevant macOS System Settings privacy pane.'
  }),
  'screen.snapshot': Object.freeze({
    category: 'screen',
    requiresUserGrant: true,
    tcc: 'ScreenCapture',
    description: 'Capture a single screen image through the signed broker.'
  }),
  'screen.record': Object.freeze({
    category: 'screen',
    requiresUserGrant: true,
    tcc: 'ScreenCapture',
    description: 'Record screen content through the signed broker.'
  }),
  'ui.click': Object.freeze({
    category: 'accessibility',
    requiresUserGrant: true,
    tcc: 'Accessibility',
    description: 'Perform an allowlisted click through Accessibility.'
  }),
  'ui.type': Object.freeze({
    category: 'accessibility',
    requiresUserGrant: true,
    tcc: 'Accessibility',
    description: 'Type text through Accessibility without granting generic Python/Node access.'
  }),
  'automation.appleEvent': Object.freeze({
    category: 'automation',
    requiresUserGrant: true,
    tcc: 'AppleEvents',
    description: 'Send a scoped Apple Event to an allowlisted target application.'
  }),
  'notification.send': Object.freeze({
    category: 'notifications',
    requiresUserGrant: true,
    tcc: 'UserNotifications',
    description: 'Post a macOS notification as Hermes.'
  }),
  'mic.status': Object.freeze({
    category: 'audio',
    requiresUserGrant: true,
    tcc: 'Microphone',
    description: 'Return microphone permission and availability status.'
  }),
  'system.run.approved': Object.freeze({
    category: 'exec',
    requiresUserGrant: false,
    requiresApproval: true,
    description: 'Run a command only after app-side exec approval policy allows it.'
  })
})

const SENSITIVE_PARAM_KEYS = new Set([
  'authorization',
  'cookie',
  'key',
  'password',
  'secret',
  'session',
  'signature',
  'token'
])

function assertToken(token) {
  if (typeof token !== 'string' || token.length < 32) {
    throw new Error('broker token must be a string of at least 32 characters')
  }
}

function stableStringify(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  const keys = Object.keys(value).sort()
  return `{${keys.map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(',')}}`
}

function signingPayload(envelope) {
  const unsigned = { ...envelope }
  delete unsigned.signature
  return stableStringify(unsigned)
}

function signBrokerEnvelope(envelope, token) {
  assertToken(token)
  return crypto.createHmac('sha256', token).update(signingPayload(envelope)).digest('hex')
}

function timingSafeEqualHex(left, right) {
  if (typeof left !== 'string' || typeof right !== 'string') return false
  if (!/^[0-9a-f]+$/i.test(left) || !/^[0-9a-f]+$/i.test(right)) return false
  const a = Buffer.from(left, 'hex')
  const b = Buffer.from(right, 'hex')
  if (a.length !== b.length) return false
  return crypto.timingSafeEqual(a, b)
}

function createBrokerRequest({
  method,
  params = {},
  token,
  caller = 'hermes-desktop',
  now = Date.now(),
  ttlMs = DEFAULT_TTL_MS,
  nonce = crypto.randomUUID(),
  id = crypto.randomUUID()
}) {
  assertToken(token)
  if (!Object.prototype.hasOwnProperty.call(MAC_PERMISSION_BROKER_METHODS, method)) {
    throw new Error(`unsupported macOS broker method: ${method}`)
  }
  if (!Number.isFinite(now) || !Number.isFinite(ttlMs) || ttlMs <= 0) {
    throw new Error('broker request requires finite now and positive ttlMs')
  }

  const envelope = {
    version: BROKER_PROTOCOL_VERSION,
    id,
    caller,
    method,
    params,
    issuedAt: Math.trunc(now),
    expiresAt: Math.trunc(now + ttlMs),
    nonce
  }
  envelope.signature = signBrokerEnvelope(envelope, token)
  return envelope
}

function verifyBrokerRequest(envelope, token, options = {}) {
  try {
    assertToken(token)
  } catch (error) {
    return { ok: false, error: error.message }
  }

  if (!envelope || typeof envelope !== 'object') {
    return { ok: false, error: 'broker request must be an object' }
  }
  if (envelope.version !== BROKER_PROTOCOL_VERSION) {
    return { ok: false, error: 'unsupported broker protocol version' }
  }
  if (!Object.prototype.hasOwnProperty.call(MAC_PERMISSION_BROKER_METHODS, envelope.method)) {
    return { ok: false, error: `unsupported broker method: ${envelope.method}` }
  }
  if (typeof envelope.id !== 'string' || !envelope.id) {
    return { ok: false, error: 'broker request id is required' }
  }
  if (typeof envelope.nonce !== 'string' || !envelope.nonce) {
    return { ok: false, error: 'broker request nonce is required' }
  }
  if (!Number.isFinite(envelope.issuedAt) || !Number.isFinite(envelope.expiresAt)) {
    return { ok: false, error: 'broker request timestamps are required' }
  }

  const now = Number.isFinite(options.now) ? options.now : Date.now()
  const skewMs = Number.isFinite(options.clockSkewMs) ? options.clockSkewMs : DEFAULT_CLOCK_SKEW_MS
  if (envelope.issuedAt - skewMs > now) {
    return { ok: false, error: 'broker request was issued in the future' }
  }
  if (envelope.expiresAt + skewMs < now) {
    return { ok: false, error: 'broker request expired' }
  }
  if (envelope.expiresAt <= envelope.issuedAt) {
    return { ok: false, error: 'broker request expiry must be after issue time' }
  }

  const expected = signBrokerEnvelope(envelope, token)
  if (!timingSafeEqualHex(envelope.signature, expected)) {
    return { ok: false, error: 'broker request signature mismatch' }
  }

  if (options.seenNonces) {
    if (options.seenNonces.has(envelope.nonce)) {
      return { ok: false, error: 'broker request nonce replayed' }
    }
    options.seenNonces.add(envelope.nonce)
  }

  return {
    ok: true,
    id: envelope.id,
    caller: envelope.caller,
    method: envelope.method,
    params: envelope.params || {},
    capability: MAC_PERMISSION_BROKER_METHODS[envelope.method]
  }
}

function redactForBrokerAudit(value) {
  if (Array.isArray(value)) return value.map(redactForBrokerAudit)
  if (!value || typeof value !== 'object') return value
  const out = {}
  for (const [key, child] of Object.entries(value)) {
    out[key] = SENSITIVE_PARAM_KEYS.has(key.toLowerCase()) ? '[REDACTED]' : redactForBrokerAudit(child)
  }
  return out
}

module.exports = {
  BROKER_PROTOCOL_VERSION,
  DEFAULT_TTL_MS,
  DEFAULT_CLOCK_SKEW_MS,
  MAC_PERMISSION_BROKER_METHODS,
  createBrokerRequest,
  redactForBrokerAudit,
  signBrokerEnvelope,
  stableStringify,
  verifyBrokerRequest
}
