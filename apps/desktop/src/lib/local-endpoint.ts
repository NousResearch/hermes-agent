/**
 * Local/private endpoint classification for model `base_url` values.
 *
 * Mirrors `agent/model_metadata.py::is_local_endpoint` (the canonical classifier
 * used for runtime timeout auto-bumps), so the desktop UI and the backend agree
 * on what counts as local: loopback, RFC-1918, link-local, Tailscale CGNAT,
 * mDNS (`*.local`), container-internal DNS and unqualified hostnames.
 *
 * A pin whose base_url is a private endpoint can never bill a provider — it is
 * the intended per-task base_url feature, not a stale pin.
 */

const CONTAINER_LOCAL_SUFFIXES = ['.docker.internal', '.containers.internal', '.lima.internal']
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '::1', '0.0.0.0'])

export function isLocalEndpointUrl(baseUrl: string): boolean {
  if (!baseUrl.trim()) {
    return false
  }

  const candidate = baseUrl.trim()
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(candidate) ? candidate : `http://${candidate}`

  let host: string
  try {
    host = new URL(withScheme).hostname
  } catch {
    return false
  }
  if (host.startsWith('[') && host.endsWith(']')) {
    host = host.slice(1, -1)
  }
  host = host.toLowerCase()

  // IPv6 first: loopback, ULA fc00::/7 and link-local fe80::/10 are local; a
  // global-scope address must never fall through to the unqualified-host rule.
  if (host.includes(':')) {
    return (
      LOOPBACK_HOSTS.has(host) ||
      /^f[cd][0-9a-f]*:/.test(host) || // ULA fc00::/7
      /^fe[89ab][0-9a-f]*:/.test(host) // link-local fe80::/10
    )
  }

  if (LOOPBACK_HOSTS.has(host)) {
    return true
  }
  if (host.endsWith('.local') || CONTAINER_LOCAL_SUFFIXES.some(suffix => host.endsWith(suffix))) {
    return true
  }
  // Unqualified hostnames (no dots) are local by definition — Docker Compose
  // service names, /etc/hosts entries, mDNS (e.g. an Ollama box at `byron`).
  if (!host.includes('.')) {
    return true
  }

  const parts = host.split('.')
  if (parts.length !== 4) {
    return false
  }
  const nums = parts.map(part => (part && /^\d+$/.test(part) ? Number(part) : Number.NaN))
  if (nums.some(Number.isNaN)) {
    return false
  }
  const [a, b] = nums
  return (
    a === 10 || // RFC-1918 10/8
    a === 127 || // loopback 127/8
    (a === 172 && b >= 16 && b <= 31) || // RFC-1918 172.16/12
    (a === 192 && b === 168) || // RFC-1918 192.168/16
    (a === 169 && b === 254) || // link-local 169.254/16 (cloud IMDS)
    (a === 100 && b >= 64 && b <= 127) // Tailscale CGNAT 100.64/10
  )
}
