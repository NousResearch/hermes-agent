const dns = require('node:dns')
const net = require('node:net')

const MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
const MAX_FETCH_BYTES = 16 * 1024 * 1024

function normalizeHostname(hostname) {
  return String(hostname || '')
    .trim()
    .replace(/^\[(.*)\]$/, '$1')
    .replace(/\.$/, '')
    .toLowerCase()
}

function isBlockedIpv4(hostname) {
  if (hostname === '0.0.0.0') return true
  const p = hostname.split('.')
  if (p.length !== 4) return false
  const octets = p.map(part => Number(part))
  if (octets.some(part => !Number.isInteger(part) || part < 0 || part > 255)) return false
  const [a, b] = octets

  if (a === 0) return true
  if (a === 10) return true
  if (a === 127) return true
  if (a === 169 && b === 254) return true
  if (a === 172 && b >= 16 && b <= 31) return true
  if (a === 192 && b === 168) return true
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

function isBlockedHostname(hostname) {
  const h = normalizeHostname(hostname)
  if (h === 'localhost') return true
  if (isBlockedIpv4(h)) return true
  if (isBlockedIpv6(h)) return true
  return false
}

function isBlockedUrl(urlStr) {
  try {
    const u = new URL(urlStr)
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return false
    return isBlockedHostname(u.hostname)
  } catch (e) {
    return false
  }
}

function privateUrlError() {
  return new Error('Blocked: private URL')
}

function lookupAll(hostname, lookup = dns.lookup) {
  return new Promise((resolve, reject) => {
    lookup(hostname, { all: true, verbatim: true }, (error, addresses, family) => {
      if (error) return reject(error)
      if (Array.isArray(addresses)) return resolve(addresses)
      if (addresses) return resolve([{ address: addresses, family }])
      resolve([])
    })
  })
}

async function resolveSafeHttpUrl(rawUrl, options = {}) {
  const parsed = rawUrl instanceof URL ? rawUrl : new URL(String(rawUrl))
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Unsupported URL protocol: ${parsed.protocol}`)
  }

  const hostname = normalizeHostname(parsed.hostname)
  if (!options.allowPrivateNetwork && isBlockedHostname(hostname)) throw privateUrlError()

  const literalFamily = net.isIP(hostname)
  const resolved = literalFamily
    ? [{ address: hostname, family: literalFamily }]
    : await lookupAll(hostname, options.lookup)

  if (!resolved.length) throw new Error(`Could not resolve ${parsed.hostname}`)

  const normalized = resolved.map(entry => ({
    address: normalizeHostname(entry.address),
    family: entry.family || net.isIP(entry.address)
  }))

  if (!options.allowPrivateNetwork && normalized.some(entry => isBlockedHostname(entry.address))) {
    throw privateUrlError()
  }

  const selected = normalized.find(entry => entry.family === 4 || entry.family === 6) || normalized[0]
  const pinnedLookup = (requestHostname, lookupOptions, callback) => {
    const requestHost = normalizeHostname(requestHostname)
    if (requestHost !== hostname) {
      const error = new Error(`Unexpected DNS lookup for ${requestHostname}`)
      if (typeof callback === 'function') return callback(error)
      throw error
    }
    if (lookupOptions?.all) return callback(null, [{ address: selected.address, family: selected.family }])
    return callback(null, selected.address, selected.family)
  }

  return { url: parsed, address: selected.address, family: selected.family, lookup: pinnedLookup }
}

function validateRedirectUrl(currentUrl, location) {
  const next = new URL(String(location || ''), currentUrl)
  if (next.protocol !== 'http:' && next.protocol !== 'https:') {
    throw new Error(`Unsupported URL protocol: ${next.protocol}`)
  }
  if (isBlockedUrl(next.toString())) throw privateUrlError()
  return next
}

module.exports = {
  isBlockedUrl,
  resolveSafeHttpUrl,
  validateRedirectUrl,
  MAX_DOWNLOAD_BYTES,
  MAX_FETCH_BYTES
}
