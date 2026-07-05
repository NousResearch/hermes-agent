const REDACTED = '[REDACTED]'

const SENSITIVE_KEY_RE =
  /(?:^|[_-])(?:api[_-]?key|auth|authorization|bearer|cookie|password|passwd|private[_-]?key|refresh[_-]?token|secret|session|token)(?:$|[_-])/i

const NAMED_SECRET_RE =
  /(?:X-Plex-Token|Plex-Token|PLEX_TOKEN|[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY|AUTH)[A-Z0-9_]*|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|password|secret|token)/i

const QUERY_SECRET_RE =
  /([?&](?:X-Plex-Token|Plex-Token|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|password|secret|token)=)([^&\s"'`<>)]*)/gi

const BEARER_SECRET_RE = /\b(Authorization\s*:\s*Bearer\s+)([A-Za-z0-9._~+/-]+=*)/gi

const JSON_SECRET_RE =
  /((["'])(?:X-Plex-Token|Plex-Token|[A-Za-z0-9_]*(?:token|secret|password|passwd|api[_-]?key|access[_-]?key|private[_-]?key|auth)[A-Za-z0-9_]*|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|password|secret)\2\s*:\s*)(["'])([^"']+)(\3)/gi

const ASSIGNMENT_SECRET_RE =
  /(\b(?:X-Plex-Token|Plex-Token|[A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE[_-]?KEY)[A-Za-z0-9_]*|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|password|secret|token)\b\s*[:=]\s*)(["']?)([^\s"'`;&|)]+)/gi

function isSensitiveKey(key: string): boolean {
  return NAMED_SECRET_RE.test(key) || SENSITIVE_KEY_RE.test(key)
}

export function redactSensitiveText(value: string): string {
  if (!value) {
    return value
  }

  return value
    .replace(QUERY_SECRET_RE, (_match, prefix: string) => `${prefix}${REDACTED}`)
    .replace(BEARER_SECRET_RE, (_match, prefix: string) => `${prefix}${REDACTED}`)
    .replace(JSON_SECRET_RE, (_match, prefix: string, _keyQuote: string, valueQuote: string, _secret: string) => {
      return `${prefix}${valueQuote}${REDACTED}${valueQuote}`
    })
    .replace(ASSIGNMENT_SECRET_RE, (_match, prefix: string, quote: string) => `${prefix}${quote}${REDACTED}`)
}

function redactUnknown(value: unknown, seen: WeakSet<object>): unknown {
  if (typeof value === 'string') {
    return redactSensitiveText(value)
  }

  if (value === null || value === undefined || typeof value !== 'object') {
    return value
  }

  if (seen.has(value)) {
    return '[Circular]'
  }

  seen.add(value)

  if (Array.isArray(value)) {
    return value.map(item => redactUnknown(item, seen))
  }

  if (value instanceof Date) {
    return value
  }

  const next: Record<string, unknown> = {}

  for (const [key, rowValue] of Object.entries(value as Record<string, unknown>)) {
    next[key] = isSensitiveKey(key) ? REDACTED : redactUnknown(rowValue, seen)
  }

  return next
}

export function redactSensitiveValue<T>(value: T): T {
  return redactUnknown(value, new WeakSet()) as T
}
