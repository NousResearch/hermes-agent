// Parse any IPv4 representation that the platform resolver (inet_aton) would
// accept — dotted decimal, dotted hex/octal, and the shorthand forms where a
// trailing field fills the remaining bytes (e.g. `2130706433`, `0x7f000001`,
// `0177.0.0.1` all mean 127.0.0.1). Returns a uint32, or null when the host is
// not an IPv4 literal. This closes the alternate-encoding SSRF bypass that a
// plain `startsWith('127.')` string check misses.
function ipv4ToInt(host) {
  const parts = host.split('.')
  if (parts.length === 0 || parts.length > 4) return null
  const nums = []
  for (const part of parts) {
    if (part === '') return null
    let n
    if (/^0x[0-9a-f]+$/i.test(part)) n = parseInt(part, 16)
    else if (/^0[0-7]+$/.test(part)) n = parseInt(part, 8)
    else if (/^[0-9]+$/.test(part)) n = parseInt(part, 10)
    else return null
    if (!Number.isFinite(n) || n < 0) return null
    nums.push(n)
  }
  // Leading fields must fit a byte; the final field fills the remaining bytes.
  for (let i = 0; i < nums.length - 1; i++) {
    if (nums[i] > 0xff) return null
  }
  const last = nums[nums.length - 1]
  const maxLast = Math.pow(256, 4 - (nums.length - 1))
  if (last >= maxLast) return null
  let value = last
  for (let i = 0; i < nums.length - 1; i++) {
    value += nums[i] * Math.pow(256, 3 - i)
  }
  if (value < 0 || value > 0xffffffff) return null
  return value >>> 0
}

function isPrivateIPv4Int(v) {
  const a = (v >>> 24) & 0xff
  const b = (v >>> 16) & 0xff
  if (a === 127) return true // 127.0.0.0/8 loopback
  if (a === 10) return true // 10.0.0.0/8 private
  if (a === 0) return true // 0.0.0.0/8 ("this host")
  if (a === 169 && b === 254) return true // 169.254.0.0/16 link-local + cloud metadata
  if (a === 192 && b === 168) return true // 192.168.0.0/16 private
  if (a === 172 && b >= 16 && b <= 31) return true // 172.16.0.0/12 private
  if (a === 100 && b >= 64 && b <= 127) return true // 100.64.0.0/10 carrier-grade NAT
  return false
}

// Best-effort block of requests aimed at the local host / private networks.
// NOTE: this is a literal-host check only — it does NOT resolve DNS names, so a
// hostname that resolves to a private address (incl. DNS-rebinding) is not
// caught here. Callers that need that guarantee must validate the resolved
// socket address. Returns false (allowed) for non-http(s) schemes; callers are
// responsible for rejecting file:/data: themselves.
function isBlockedUrl(urlStr) {
  try {
    const u = new URL(urlStr)
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return false
    const h = u.hostname.toLowerCase()
    if (h === 'localhost' || h.endsWith('.localhost')) return true
    // IPv6 literal — URL.hostname strips the surrounding brackets.
    if (h.includes(':')) {
      const v6 = h.replace(/^\[|\]$/g, '')
      if (v6 === '::1' || v6 === '::') return true // loopback / unspecified
      if (/^fe[89ab]/.test(v6)) return true // fe80::/10 link-local
      if (/^f[cd]/.test(v6)) return true // fc00::/7 unique-local
      // IPv4-mapped (::ffff:127.0.0.1). WHATWG URL normalizes the embedded v4
      // to two hex groups (::ffff:7f00:1), so handle both the dotted tail and
      // the trailing 16-bit hex pair.
      const mapped = v6.match(/^::ffff:(.+)$/i)
      if (mapped) {
        const rest = mapped[1]
        if (rest.includes('.')) {
          const v = ipv4ToInt(rest)
          if (v !== null && isPrivateIPv4Int(v)) return true
        } else {
          const groups = rest.split(':')
          if (groups.length === 2 && groups.every(g => /^[0-9a-f]{1,4}$/.test(g))) {
            const v = ((parseInt(groups[0], 16) << 16) | parseInt(groups[1], 16)) >>> 0
            if (isPrivateIPv4Int(v)) return true
          }
        }
      }
      return false
    }
    const v4 = ipv4ToInt(h)
    if (v4 !== null) return isPrivateIPv4Int(v4)
    return false
  } catch {
    return false
  }
}

const MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
const MAX_FETCH_BYTES = 16 * 1024 * 1024

export { isBlockedUrl, ipv4ToInt, isPrivateIPv4Int, MAX_DOWNLOAD_BYTES, MAX_FETCH_BYTES }
