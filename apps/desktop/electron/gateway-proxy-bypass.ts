/**
 * Chromium session proxy bypass for remote Hermes gateways.
 *
 * The desktop renderer opens `/api/ws` with Chromium's WebSocket, which honors
 * the macOS system proxy. Electron's main-process HTTPS (ticket mint, health)
 * uses a direct agent. When a TUN VPN already captures traffic, leaving the
 * system proxy on misroutes the upgrade over HTTP/2; a gated gateway then
 * answers JSON 401 `no_cookie` before the WebSocket handler runs.
 *
 * Bypass the saved remote gateway host(s) so renderer sockets take the same
 * direct/TUN path as the main process.
 *
 * Even with the system proxy off, Chromium still prefers RFC 8441 WebSocket
 * over HTTP/2 (and HTTP/3) to Cloudflare. That handshake is an HTTP request
 * to `/api/ws` as far as FastAPI's cookie gate is concerned — not a
 * WebSocket upgrade — so it 401s the same way. `--disable-http2` /
 * `--disable-quic` (applied in main before `ready`) force HTTP/1.1 Upgrade,
 * which is what `curl --http1.1` uses when it reaches the WS handler.
 */

export function hostnameFromGatewayUrl(raw: unknown): string | null {
  const value = String(raw || '').trim()

  if (!value) {
    return null
  }

  try {
    const href = /^[a-z][a-z0-9+.-]*:\/\//i.test(value) ? value : `https://${value}`
    const parsed = new URL(href)

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null
    }

    const host = parsed.hostname.trim().toLowerCase()

    if (!host || host === 'localhost' || host === '127.0.0.1' || host === '[::1]' || host === '::1') {
      return null
    }

    return host
  } catch {
    return null
  }
}

export function collectRemoteGatewayHosts(urls: unknown[]): string[] {
  const hosts = new Set<string>()

  for (const url of urls) {
    const host = hostnameFromGatewayUrl(url)

    if (host) {
      hosts.add(host)
    }
  }

  return [...hosts].sort()
}

export function proxyBypassRulesForGatewayUrls(urls: unknown[]): string | null {
  const hosts = collectRemoteGatewayHosts(urls)

  if (hosts.length === 0) {
    return null
  }

  return ['<local>', ...hosts].join(',')
}

/** Chromium `--proxy-bypass-list` value (comma-separated hostnames). */
export function chromiumProxyBypassListForGatewayUrls(urls: unknown[]): string | null {
  const hosts = collectRemoteGatewayHosts(urls)

  if (hosts.length === 0) {
    return null
  }

  return hosts.join(',')
}

/** True when scutil reports an active HTTP/HTTPS/SOCKS/PAC proxy. */
export function scutilProxyIsEnabled(scutilOutput: string): boolean {
  return (
    /HTTPEnable\s*:\s*1/.test(scutilOutput) ||
    /HTTPSEnable\s*:\s*1/.test(scutilOutput) ||
    /SOCKSEnable\s*:\s*1/.test(scutilOutput) ||
    /ProxyAutoConfigEnable\s*:\s*1/.test(scutilOutput)
  )
}
