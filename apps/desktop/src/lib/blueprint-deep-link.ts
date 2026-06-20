const SAFE_IDENTIFIER_RE = /^[A-Za-z0-9_-]{1,64}$/
const UNSAFE_NAME_RE = /[^A-Za-z0-9_-]/g
const MAX_NAME_LENGTH = 64
const MAX_VALUE_LENGTH = 512

interface BlueprintDeepLinkPayload {
  kind?: unknown
  name?: unknown
  params?: unknown
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

// Drop ASCII control characters (the C0 range plus DEL) without a control-char
// regex literal, which would trip eslint's no-control-regex.
function stripControlChars(value: string): string {
  let out = ''

  for (const char of value) {
    const code = char.codePointAt(0) ?? 0

    if (code > 0x1f && code !== 0x7f) {
      out += char
    }
  }

  return out
}

function formatValue(raw: unknown): string {
  const clean = stripControlChars(String(raw ?? '')).slice(0, MAX_VALUE_LENGTH)

  if (!/[\s"]/.test(clean)) {
    return clean
  }

  return `"${clean.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
}

// Build the `/blueprint <name> k=v …` composer command from a hermes://blueprint
// deep-link payload, or return null when the payload isn't a usable blueprint
// link. The name and every slot key/value are sanitized here because the result
// is interpolated verbatim into a slash command that the composer runs.
export function buildBlueprintDeepLinkCommand(rawPayload: unknown): string | null {
  if (!isRecord(rawPayload)) {
    return null
  }

  const payload = rawPayload as BlueprintDeepLinkPayload

  if (payload.kind !== 'blueprint') {
    return null
  }

  const name = String(payload.name ?? '')
    .replace(UNSAFE_NAME_RE, '')
    .slice(0, MAX_NAME_LENGTH)

  if (!name) {
    return null
  }

  const params = isRecord(payload.params) ? payload.params : {}

  const slots = Object.entries(params)
    .flatMap(([key, value]) => (SAFE_IDENTIFIER_RE.test(key) ? [`${key}=${formatValue(value)}`] : []))
    .join(' ')

  return `/blueprint ${name}${slots ? ' ' + slots : ''}`
}
