const ALLOWED_EXTERNAL_PROTOCOLS = new Set(['http:', 'https:', 'mailto:', 'tel:'])
const TELEPHONE_URL_RE = /^tel:\+?\d{7,15}$/iu

export function isExternalProtocolAllowed(protocol: string): boolean {
  return ALLOWED_EXTERNAL_PROTOCOLS.has(protocol.toLowerCase())
}

export function isExternalUrlAllowed(url: URL): boolean {
  if (!isExternalProtocolAllowed(url.protocol)) {
    return false
  }

  if (url.protocol !== 'tel:') {
    return true
  }

  return TELEPHONE_URL_RE.test(url.toString())
}
