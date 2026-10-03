const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '::1'])

/**
 * Parse an HTTP Host authority into its hostname, or return an empty string.
 *
 * Host headers are authorities, not URLs. Keep this deliberately strict so the
 * all-interface bind exception accepts arbitrary valid hostnames, not malformed
 * values or values containing URL syntax.
 */
export function hostHeaderHostname(hostHeader) {
  const value = String(hostHeader || '').trim()
  if (!value) return ''
  if ([...value].some(char => ['"', "'", '<', '>', ' ', '\n', '\r', '\t'].includes(char))) return ''
  if (value.includes('://') || [...value].some(char => ['/', '?', '#', '@'].includes(char))) return ''

  if (value.startsWith('[')) {
    const close = value.indexOf(']')
    if (close === -1) return ''
    const hostname = value.slice(1, close)
    if (!hostname.includes(':')) return ''
    const suffix = value.slice(close + 1)
    if (suffix && !/^:\d+$/.test(suffix)) return ''
    return hostname.toLowerCase()
  }

  // An unbracketed IPv6 authority is ambiguous with a port separator.
  if ((value.match(/:/g) || []).length > 1) return ''
  if (value.includes(':')) {
    const [hostname, port] = value.split(':')
    if (!hostname || !/^\d+$/.test(port)) return ''
    return hostname.toLowerCase()
  }
  return value.toLowerCase()
}

/**
 * Match Dashboard's Host-header policy.
 *
 * Binding to all interfaces is an explicit operator choice. In that mode the
 * request still needs a valid Host authority, but its hostname may be the
 * reverse proxy's public hostname (or any other valid hostname). Loopback and
 * explicit non-loopback binds remain restricted to their intended hosts.
 */
export function isAcceptedHost(hostHeader, boundHost, trustedPublicHosts = new Set()) {
  const requestHost = hostHeaderHostname(hostHeader)
  if (!requestHost) return false

  const trusted = trustedPublicHosts instanceof Set
    ? trustedPublicHosts
    : new Set(trustedPublicHosts || [])
  if (trusted.has(requestHost)) return true

  const bound = String(boundHost || '').toLowerCase().replace(/^\[|\]$/g, '')
  if (bound === '0.0.0.0' || bound === '::') return true

  if (LOOPBACK_HOSTS.has(bound)) return LOOPBACK_HOSTS.has(requestHost)
  return requestHost === bound
}

export { LOOPBACK_HOSTS }
