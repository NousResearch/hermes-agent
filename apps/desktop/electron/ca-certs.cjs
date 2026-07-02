const fs = require('node:fs')
const tls = require('node:tls')
const { execFileSync } = require('node:child_process')

// Node's `https` module trusts ONLY its bundled Mozilla root store plus
// whatever `NODE_EXTRA_CA_CERTS` points at — it never consults the macOS
// keychain. Chromium (and therefore the desktop renderer) DOES use the
// keychain, so a remote backend fronted by a private/homelab CA that the user
// has trusted in Keychain Access loads fine in the window but fails the
// main-process readiness probe (`waitForHermes` -> `fetchJson`) with
// `unable to get local issuer certificate`.
//
// This module closes that gap: on macOS it harvests the certificates in the
// login + System keychains, folds in `NODE_EXTRA_CA_CERTS`, and installs the
// merged bundle (default roots first) onto `https.globalAgent`. Because the
// boot fetch helpers all use the default agent, this makes the Node side trust
// exactly what the keychain-backed renderer already trusts.

const PEM_BLOCK_RE = /-----BEGIN CERTIFICATE-----[\s\S]+?-----END CERTIFICATE-----/g

// Keychains searched for user/admin-installed CAs. The default (login) keychain
// is queried by passing no keychain path; System.keychain holds machine-wide
// trust anchors. Apple's SystemRootCertificates is intentionally skipped — it
// duplicates Node's bundled Mozilla roots and would only bloat the bundle.
const MAC_KEYCHAINS = [null, '/Library/Keychains/System.keychain']

// Split a PEM blob that may contain any number of concatenated certificates
// into individual normalized PEM strings.
function splitPemCertificates(text) {
  if (!text) return []
  const matches = String(text).match(PEM_BLOCK_RE)
  if (!matches) return []
  return matches.map(block => `${block.trim()}\n`)
}

// Read every certificate the given keychains hold, as PEM. `-a` dumps all
// matches (not just the first); `-p` emits PEM. `find-certificate` ignores per
// cert trust settings and returns leaf certs too — acceptable here because we
// only ever ADD to the trust set, and everything returned is something the user
// deliberately placed in their keychain.
function collectMacKeychainCerts({ runSecurity, keychains = MAC_KEYCHAINS } = {}) {
  const certs = []
  for (const keychain of keychains) {
    try {
      const args = keychain ? ['find-certificate', '-a', '-p', keychain] : ['find-certificate', '-a', '-p']
      certs.push(...splitPemCertificates(runSecurity(args)))
    } catch {
      // A missing/locked keychain (or `security` being absent) must never break
      // boot — just skip that source.
    }
  }
  return certs
}

// Read the file(s) referenced by NODE_EXTRA_CA_CERTS. Node treats it as a single
// path, but we tolerate an unset/empty value and unreadable files.
function collectNodeExtraCaCerts({ env, readFile } = {}) {
  const raw = env && env.NODE_EXTRA_CA_CERTS
  if (!raw) return []
  try {
    return splitPemCertificates(readFile(raw))
  } catch {
    return []
  }
}

// Build the merged CA list: Node's default roots first, then keychain and
// NODE_EXTRA_CA_CERTS additions, deduplicated. Returns an array of PEM strings
// suitable for `https.Agent`'s `ca` option. Returns `null` when there is
// nothing extra to add, so the caller can leave the default trust store
// untouched.
function buildCaBundle({
  platform = process.platform,
  env = process.env,
  runSecurity,
  readFile = path => fs.readFileSync(path, 'utf8'),
  defaultRoots = tls.rootCertificates
} = {}) {
  const extras = []
  if (platform === 'darwin' && typeof runSecurity === 'function') {
    extras.push(...collectMacKeychainCerts({ runSecurity }))
  }
  extras.push(...collectNodeExtraCaCerts({ env, readFile }))

  if (extras.length === 0) return null

  const seen = new Set()
  const bundle = []
  for (const pem of [...defaultRoots, ...extras]) {
    const key = pem.replace(/\s+/g, '')
    if (!key || seen.has(key)) continue
    seen.add(key)
    bundle.push(pem)
  }
  return bundle
}

// Install the merged CA bundle onto the given https module's global agent so
// every default-agent request (the boot fetch helpers) trusts the same anchors
// the keychain-backed renderer does. Idempotent and defensive: any failure is
// swallowed so it can never block startup. Returns the number of certs
// installed (0 when nothing was added), for logging/tests.
function installNodeCaTrust(https, options = {}) {
  try {
    const runSecurity =
      options.runSecurity ||
      (args => execFileSync('/usr/bin/security', args, { encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 }))
    const bundle = buildCaBundle({ ...options, runSecurity })
    if (!bundle) return 0
    https.globalAgent.options.ca = bundle
    return bundle.length
  } catch {
    return 0
  }
}

module.exports = {
  splitPemCertificates,
  collectMacKeychainCerts,
  collectNodeExtraCaCerts,
  buildCaBundle,
  installNodeCaTrust,
  MAC_KEYCHAINS
}
