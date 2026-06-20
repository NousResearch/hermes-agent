function normalizeHostname(hostname) {
  return String(hostname || '')
    .trim()
    .replace(/^\[(.*)\]$/, '$1')
    .replace(/\.$/, '')
    .toLowerCase()
}

function isBlockedIpv4(hostname) {
  if (hostname === '127.0.0.1' || hostname === '0.0.0.0') return true
  if (hostname.startsWith('127.') || hostname.startsWith('10.') || hostname.startsWith('169.254.')) return true
  if (hostname.startsWith('192.168.')) return true
  const p = hostname.split('.')
  if (p.length === 4 && p[0] === '172') {
    const n = parseInt(p[1], 10)
    if (n >= 16 && n <= 31) return true
  }
  return false
}

function ipv4FromMappedIpv6Tail(tail) {
  if (tail.includes('.')) return tail
  const parts = tail.split(':').filter(Boolean)
  if (parts.length !== 2) return null
  const high = parseInt(parts[0], 16)
  const low = parseInt(parts[1], 16)
  if (!Number.isFinite(high) || !Number.isFinite(low) || high < 0 || high > 0xffff || low < 0 || low > 0xffff) {
    return null
  }
  return `${(high >> 8) & 0xff}.${high & 0xff}.${(low >> 8) & 0xff}.${low & 0xff}`
}

function isBlockedIpv6(hostname) {
  if (!hostname.includes(':')) return false
  if (hostname === '::' || hostname === '::1') return true

  if (hostname.startsWith('::ffff:')) {
    const mappedIpv4 = ipv4FromMappedIpv6Tail(hostname.slice('::ffff:'.length))
    if (mappedIpv4 && isBlockedIpv4(mappedIpv4)) return true
  }

  const firstHextet = parseInt(hostname.split(':')[0] || '0', 16)
  if (!Number.isFinite(firstHextet)) return false
  if (firstHextet >= 0xfc00 && firstHextet <= 0xfdff) return true
  if (firstHextet >= 0xfe80 && firstHextet <= 0xfebf) return true
  return false
}

function isBlockedUrl(urlStr) {
  try {
    const u = new URL(urlStr)
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return false
    const h = normalizeHostname(u.hostname)
    if (h === 'localhost') return true
    if (isBlockedIpv4(h)) return true
    if (isBlockedIpv6(h)) return true
    return false
  } catch (e) {
    return false
  }
}

module.exports = { isBlockedUrl, MAX_DOWNLOAD_BYTES: 50 * 1024 * 1024, MAX_FETCH_BYTES: 16 * 1024 * 1024 }
