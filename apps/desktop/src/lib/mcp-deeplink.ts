/**
 * Parser/validator for the `hermes://mcp/install?name=NAME&config=B64` deep
 * link (the "Add to Hermes" button MCP vendors embed, mirroring Cursor's
 * `cursor://anysphere.cursor-deeplink/mcp/install` scheme). `config` is
 * base64url-encoded JSON of a single server config object; standard base64 is
 * accepted too.
 *
 * Everything here is HOSTILE INPUT — any web page can open the link — so this
 * module only classifies. It never installs: the caller must show the decoded
 * config to the user and require an explicit confirmation before saving.
 */

export const MCP_DEEPLINK_NAME_RE = /^[A-Za-z0-9._-]{1,64}$/

/** Hard cap on the DECODED config JSON (bytes). */
export const MCP_DEEPLINK_MAX_CONFIG_BYTES = 32 * 1024

// The base64 text is ~4/3 the decoded size; reject grossly oversized params
// before doing any decode work.
const MAX_ENCODED_LENGTH = Math.ceil((MCP_DEEPLINK_MAX_CONFIG_BYTES * 4) / 3) + 4

export type McpDeepLinkErrorCode =
  | 'bad_encoding'
  | 'invalid_name'
  | 'invalid_shape'
  | 'invalid_url'
  | 'missing_config'
  | 'not_an_object'
  | 'payload_too_large'

export interface McpInstallRequest {
  name: string
  config: Record<string, unknown>
  /** Inferred from the config shape: `url` -> http, `command` -> stdio. */
  transport: 'http' | 'stdio'
}

export type McpDeepLinkParseResult =
  | { ok: false; error: McpDeepLinkErrorCode }
  | { ok: true; request: McpInstallRequest }

/** i18n key (under `settings.mcp`) for each rejection, used by the toast. */
export const MCP_DEEPLINK_ERROR_KEYS: Record<McpDeepLinkErrorCode, string> = {
  bad_encoding: 'deepLinkErrorConfig',
  invalid_name: 'deepLinkErrorName',
  invalid_shape: 'deepLinkErrorShape',
  invalid_url: 'deepLinkErrorUrl',
  missing_config: 'deepLinkErrorConfig',
  not_an_object: 'deepLinkErrorConfig',
  payload_too_large: 'deepLinkErrorTooLarge'
}

/** base64url or standard base64 -> bytes, or null when not valid base64. */
function decodeBase64(text: string): null | Uint8Array {
  const normalized = text.replace(/-/g, '+').replace(/_/g, '/').replace(/\s/g, '')
  const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4)

  try {
    const binary = atob(padded)
    const bytes = new Uint8Array(binary.length)

    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i)
    }

    return bytes
  } catch {
    return null
  }
}

/**
 * Classify the deep link's query params into an installable request or a
 * rejection code. Pure and side-effect free; the confirmation UX is the
 * caller's responsibility.
 */
export function parseMcpInstallDeepLink(params: Record<string, string | undefined>): McpDeepLinkParseResult {
  const name = params.name ?? ''

  if (!MCP_DEEPLINK_NAME_RE.test(name)) {
    return { ok: false, error: 'invalid_name' }
  }

  const encoded = params.config ?? ''

  if (!encoded) {
    return { ok: false, error: 'missing_config' }
  }

  if (encoded.length > MAX_ENCODED_LENGTH) {
    return { ok: false, error: 'payload_too_large' }
  }

  const bytes = decodeBase64(encoded)

  if (!bytes) {
    return { ok: false, error: 'bad_encoding' }
  }

  if (bytes.length > MCP_DEEPLINK_MAX_CONFIG_BYTES) {
    return { ok: false, error: 'payload_too_large' }
  }

  let parsed: unknown

  try {
    parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)) as unknown
  } catch {
    return { ok: false, error: 'bad_encoding' }
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { ok: false, error: 'not_an_object' }
  }

  const config = parsed as Record<string, unknown>
  const hasUrl = typeof config.url === 'string'
  const hasCommand = typeof config.command === 'string' && (config.command as string).trim().length > 0

  // Exactly one transport: a config carrying BOTH `url` and `command` is
  // ambiguous (which one runs depends on loader precedence) — reject it so the
  // user can't be shown "just a URL" while a command rides along.
  if (hasUrl && 'command' in config) {
    return { ok: false, error: 'invalid_shape' }
  }

  if (hasUrl) {
    let target: URL

    try {
      target = new URL(config.url as string)
    } catch {
      return { ok: false, error: 'invalid_url' }
    }

    if (target.protocol !== 'http:' && target.protocol !== 'https:') {
      return { ok: false, error: 'invalid_url' }
    }

    return { ok: true, request: { name, config, transport: 'http' } }
  }

  if (hasCommand) {
    return { ok: true, request: { name, config, transport: 'stdio' } }
  }

  return { ok: false, error: 'invalid_shape' }
}
