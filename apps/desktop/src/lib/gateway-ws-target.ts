export function shouldDialGatewayFromMain(raw: unknown): boolean {
  if (typeof raw !== 'string' || !raw.trim()) {
    return false
  }

  try {
    const parsed = new URL(raw)
    const host = parsed.hostname.trim().toLowerCase()
    const path = parsed.pathname.replace(/\/+$/, '') || '/'

    if (parsed.protocol !== 'ws:' && parsed.protocol !== 'wss:') {
      return false
    }

    if (!path.endsWith('/api/ws')) {
      return false
    }

    return host !== 'localhost' && host !== '127.0.0.1' && host !== '[::1]' && host !== '::1'
  } catch {
    return false
  }
}
