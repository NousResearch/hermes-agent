const SAFE_IDENTIFIER_RE = /^[A-Za-z0-9_-]{1,64}$/
const UNSAFE_BLUEPRINT_NAME_RE = /[^A-Za-z0-9_-]/g
const MAX_BLUEPRINT_NAME_LENGTH = 64
const MAX_PARAM_VALUE_LENGTH = 512

type BlueprintDeepLinkPayload = {
  kind?: unknown
  name?: unknown
  params?: unknown
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function sanitizeBlueprintName(value: unknown): string {
  return String(value ?? '').replace(UNSAFE_BLUEPRINT_NAME_RE, '').slice(0, MAX_BLUEPRINT_NAME_LENGTH)
}

function stripControlCharacters(value: string): string {
  return Array.from(value)
    .filter((char) => {
      const codePoint = char.codePointAt(0) ?? 0

      return codePoint > 0x1f && codePoint !== 0x7f
    })
    .join('')
}

function formatParamValue(value: unknown): string {
  const cleanValue = stripControlCharacters(String(value ?? '')).slice(0, MAX_PARAM_VALUE_LENGTH)

  if (!/[\s"]/.test(cleanValue)) {
    return cleanValue
  }

  return `"${cleanValue.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
}

export function buildBlueprintDeepLinkCommand(rawPayload: unknown): string | null {
  if (!isRecord(rawPayload)) {
    return null
  }

  const payload = rawPayload as BlueprintDeepLinkPayload

  if (payload.kind !== 'blueprint') {
    return null
  }

  const sanitizedName = sanitizeBlueprintName(payload.name)

  if (!sanitizedName) {
    return null
  }

  const params = isRecord(payload.params) ? payload.params : {}

  const slots = Object.entries(params)
    .flatMap(([rawKey, rawValue]) => {
      const key = String(rawKey ?? '')

      if (!SAFE_IDENTIFIER_RE.test(key)) {
        return []
      }

      return [`${key}=${formatParamValue(rawValue)}`]
    })
    .join(' ')

  return `/blueprint ${sanitizedName}${slots ? ' ' + slots : ''}`
}
