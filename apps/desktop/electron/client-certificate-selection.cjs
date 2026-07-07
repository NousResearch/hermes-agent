/**
 * Helpers for selecting a TLS client certificate when a remote gateway asks
 * Chromium/Electron for one.
 *
 * Most users have zero or one matching client certificate after the server's
 * trusted CA filtering. Auto-selecting exactly one candidate lets Desktop work
 * with mTLS-protected gateways (Cloudflare Access, API Shield, private reverse
 * proxies) without needing a browser-style certificate picker in the main
 * process. When multiple candidates are available, callers can narrow the
 * choice with environment filters; otherwise we deliberately decline to guess.
 */

function normalizeString(value) {
  return String(value || '').trim()
}

function normalizeSerial(value) {
  return normalizeString(value).replace(/[^a-zA-Z0-9]/g, '').toLowerCase()
}

function certificateDisplayName(certificate) {
  return (
    certificate?.subjectName ||
    certificate?.issuerName ||
    certificate?.serialNumber ||
    'unknown certificate'
  )
}

function filtersFromEnv(env = process.env) {
  return {
    autoSelect: String(env.HERMES_DESKTOP_CLIENT_CERT_AUTO_SELECT || '1') !== '0',
    issuer: normalizeString(env.HERMES_DESKTOP_CLIENT_CERT_ISSUER).toLowerCase(),
    serial: normalizeSerial(env.HERMES_DESKTOP_CLIENT_CERT_SERIAL),
    subject: normalizeString(env.HERMES_DESKTOP_CLIENT_CERT_SUBJECT).toLowerCase()
  }
}

function certificateMatchesFilters(certificate, filters = {}) {
  const subject = normalizeString(certificate?.subjectName).toLowerCase()
  const issuer = normalizeString(certificate?.issuerName).toLowerCase()
  const serial = normalizeSerial(certificate?.serialNumber)

  if (filters.subject && !subject.includes(filters.subject)) return false
  if (filters.issuer && !issuer.includes(filters.issuer)) return false
  if (filters.serial && serial !== filters.serial) return false

  return true
}

function chooseClientCertificate(candidates, filters = filtersFromEnv()) {
  if (!filters.autoSelect) return null

  const list = Array.isArray(candidates) ? candidates.filter(Boolean) : []
  if (list.length === 0) return null

  const hasExplicitFilter = Boolean(filters.subject || filters.issuer || filters.serial)
  const filtered = hasExplicitFilter ? list.filter(cert => certificateMatchesFilters(cert, filters)) : list

  if (filtered.length === 1) return filtered[0]
  if (!hasExplicitFilter && list.length === 1) return list[0]

  return null
}

module.exports = {
  certificateDisplayName,
  certificateMatchesFilters,
  chooseClientCertificate,
  filtersFromEnv,
  normalizeSerial
}
