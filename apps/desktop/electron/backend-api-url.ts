/**
 * Join a Hermes backend base URL with a renderer-supplied API path.
 *
 * Callers previously did `connection.baseUrl + path`. A path that does not
 * start with `/` (notably `@host/...`, including whitespace-prefixed forms)
 * is parsed as URL userinfo and retargets the request to an attacker-controlled
 * host while fetchJson still attaches session / OAuth credentials. Reject those
 * paths before any network I/O.
 */

export function joinBackendApiUrl(baseUrl: string, path: unknown): string {
  if (typeof path !== 'string' || path.length === 0) {
    throw new Error('Hermes API path must be a non-empty string.')
  }

  // Single leading slash only: `@evil/` and `//evil/` must not retarget the host.
  if (!path.startsWith('/') || path.startsWith('//')) {
    throw new Error('Hermes API path must be a relative path starting with a single "/".')
  }

  const baseRaw = String(baseUrl || '').replace(/\/+$/, '')

  if (!baseRaw) {
    throw new Error('Hermes backend base URL is required.')
  }

  let base: URL
  let joined: URL

  try {
    base = new URL(baseRaw)
    joined = new URL(`${baseRaw}${path}`)
  } catch (error: any) {
    throw new Error(`Invalid Hermes backend API URL: ${error?.message || error}`)
  }

  if (joined.protocol !== 'http:' && joined.protocol !== 'https:') {
    throw new Error(`Unsupported Hermes backend URL protocol: ${joined.protocol}`)
  }

  if (joined.origin !== base.origin) {
    throw new Error('Hermes API path must not change the backend origin.')
  }

  if (joined.username || joined.password) {
    throw new Error('Hermes API URL must not include userinfo.')
  }

  return `${baseRaw}${path}`
}
