/**
 * remote-url.ts
 *
 * Renderer-side twin of the scheme coercion in
 * electron/connection-config.ts `normalizeRemoteBaseUrl()`. Users routinely
 * paste scheme-less "host:port" (a Tailscale IP, a LAN hostname) into the
 * remote-gateway URL field; without coercion the renderer's `^https?://`
 * probe gates never fire and the field just sits idle with no feedback.
 *
 * Keep the opt-out regex in sync with the electron side: only a real
 * `scheme://` prefix skips the http:// prepend, so explicit non-http schemes
 * (ws://, ftp://) still reach main-process validation and get a clear error.
 */
export function coerceRemoteUrlScheme(rawUrl: string): string {
  const value = String(rawUrl || '').trim()

  if (!value || /^[a-z][a-z0-9+.-]*:\/\//i.test(value)) {
    return value
  }

  return `http://${value}`
}

export interface ParsedSshConnectionUrl {
  host: string
  user: string
  port: number | null
  remoteProfile: string
}

/**
 * Parse `ssh://user@host:port?profile=name` for first-run / Settings paste.
 * Trailing slashes are ignored. Port 22 is omitted (OpenSSH default).
 */
export function parseSshConnectionUrl(rawUrl: string): ParsedSshConnectionUrl | null {
  const value = String(rawUrl || '').trim()

  if (!/^ssh:\/\//i.test(value)) {
    return null
  }

  let parsed: URL

  try {
    parsed = new URL(value)
  } catch {
    return null
  }

  if (parsed.protocol !== 'ssh:') {
    return null
  }

  const host = parsed.hostname.replace(/^\[|]$/g, '').trim()

  if (!host) {
    return null
  }

  let port: number | null = null

  if (parsed.port) {
    const parsedPort = Number(parsed.port)

    if (Number.isInteger(parsedPort) && parsedPort > 0 && parsedPort <= 65535 && parsedPort !== 22) {
      port = parsedPort
    }
  }

  let user = ''

  try {
    user = decodeURIComponent(parsed.username || '').trim()
  } catch {
    user = String(parsed.username || '').trim()
  }

  const remoteProfile = String(parsed.searchParams.get('profile') || '').trim()

  return { host, port, remoteProfile, user }
}

