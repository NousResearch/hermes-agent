/**
 * Whether a provider endpoint points at this machine.
 *
 * A user who already runs their own server (llama.cpp, LM Studio, vLLM) and wired it up as a
 * provider has a local model: the model menu's "Run locally · free, private" row would nudge them
 * to set up a runtime they deliberately replaced, so the offer reads this first.
 */

/** `127.0.0.0/8` is the IPv4 loopback block: every address in it is this machine. */
const IPV4_LOOPBACK = /^127(?:\.\d{1,3}){3}$/

/**
 * The host an endpoint string names, or null when it names none.
 *
 * Providers store endpoints in both shapes — a full URL and a bare `host:port` — and the bare shape
 * parses as a scheme (`localhost:11434` reads as protocol `localhost:`), so the plain read only
 * counts when it produced an http(s) URL.
 */
function hostnameOf(value: string): null | string {
  for (const candidate of [value, `http://${value}`]) {
    try {
      const url = new URL(candidate)

      if (url.protocol === 'http:' || url.protocol === 'https:') {
        return url.hostname
      }
    } catch {
      // Not a URL in this shape: the other candidate may still be one.
    }
  }

  return null
}

export function isLoopbackEndpoint(value: unknown): boolean {
  if (typeof value !== 'string') {
    return false
  }

  const raw = value.trim().toLowerCase()

  // A bare `::1` is neither a URL nor an authority `URL` can read; every other local spelling
  // (scheme, port, path, query, IPv6 brackets) is handled by the parse below.
  if (raw === '::1') {
    return true
  }

  if (!raw) {
    return false
  }

  const host = hostnameOf(raw)
    ?.replace(/^\[|\]$/g, '')
    .replace(/\.$/, '')

  if (!host) {
    return false
  }

  return (
    host === 'localhost' ||
    host === '0.0.0.0' ||
    host === '::1' ||
    host.endsWith('.localhost') ||
    IPV4_LOOPBACK.test(host)
  )
}
