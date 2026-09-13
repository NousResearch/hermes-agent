// windows-bundled-helpers.mjs — pure helpers + small CLI for the Windows
// packaged-app (MSIX / App Installer) E2E arm (tests/install/windows-bundled-e2e.ps1).
//
// Everything OS-visible here derives from the PRODUCTION builders in
// scripts/msix-shared.mjs (buildAppInstaller, OUT_OF_STORE_PUBLISHER,
// contentTypeFor) so the test feed cannot drift from the real feed. The
// production module's path is a CLI arg (--msix-shared) so the primary
// checkout's copy (with the parent-owned MainBundle/descriptorFilename
// support) is used read-only; the in-repo file is the fallback.
//
// CLI (space-separated flag pairs — `node script.mjs -- --flag value` also
// works; see the repo AGENTS note on Node eating `--` args):
//   node windows-bundled-helpers.mjs validate-manifest --manifest <path> --msix-shared <path>
//   node windows-bundled-helpers.mjs descriptor --feed <dir> --base-url <url>
//        --identity <name> --version <4-part> --bundle <filename>
//        [--descriptor-filename update.appinstaller] [--msix-shared <path>]
//   node windows-bundled-helpers.mjs serve --feed <dir> --port-file <path>

import fs from 'node:fs'
import path from 'node:path'
import http from 'node:http'
import { pathToFileURL, fileURLToPath } from 'node:url'

// ── pure helpers (unit-tested in windows-bundled-helpers.test.mjs) ──────────

const MANIFEST_SIDES = ['old', 'new']
const REQUIRED_SIDE_FIELDS = ['tag', 'version', 'commit', 'identity', 'publisher', 'applicationId']
const REQUIRED_ARTIFACT_FIELDS = ['url', 'sha256']
const SUPPORTED_ARCHES = ['x64', 'arm64']
const COMMIT_RE = /^[0-9a-f]{40}$/
const SHA256_RE = /^[0-9a-fA-F]{64}$/
const FOUR_PART_RE = /^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}$/

/**
 * True when 4-part MSIX version `a` is strictly newer than `b`
 * (numeric, component-wise; MSIX swaps require strict monotonicity).
 * @param {string} a
 * @param {string} b
 * @returns {boolean}
 */
export function fourPartNewer(a, b) {
  if (!FOUR_PART_RE.test(a) || !FOUR_PART_RE.test(b)) return false
  const pa = a.split('.').map(Number)
  const pb = b.split('.').map(Number)
  for (let i = 0; i < 4; i++) {
    if (pa[i] !== pb[i]) return pa[i] > pb[i]
  }
  return false
}

/**
 * Validate the parent-owned resolver's normalized bundle-inputs manifest
 * (schema1). `expectedPublisher` is injected by the caller from the
 * production OUT_OF_STORE_PUBLISHER so the rule binds at runtime, and
 * `artifactPathsExist` (default true) lets unit tests pass without files.
 *
 * @param {unknown} raw parsed manifest JSON
 * @param {{ expectedPublisher: string, arch?: string, artifactPathsExist?: boolean }} o
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validateBundledManifest(raw, o) {
  const errors = []
  const arch = o.arch || 'x64'
  const m = raw
  if (!m || typeof m !== 'object' || Array.isArray(m)) {
    return { ok: false, errors: ['manifest is not an object'] }
  }
  if (m.schema !== 1) errors.push(`schema: expected 1, got ${JSON.stringify(m.schema)}`)
  if (m.platform !== 'windows') errors.push(`platform: expected "windows", got ${JSON.stringify(m.platform)}`)
  if (!SUPPORTED_ARCHES.includes(arch)) errors.push(`arch "${arch}" not supported (${SUPPORTED_ARCHES.join('|')})`)
  if (m.arch !== arch) errors.push(`arch: expected "${arch}", got ${JSON.stringify(m.arch)}`)

  for (const side of MANIFEST_SIDES) {
    const s = m?.[side]
    if (!s || typeof s !== 'object') {
      errors.push(`${side}: missing`)
      continue
    }
    for (const f of REQUIRED_SIDE_FIELDS) {
      if (typeof s[f] !== 'string' || !s[f]) errors.push(`${side}.${f}: missing or not a string`)
    }
    if (typeof s.version === 'string' && !FOUR_PART_RE.test(s.version)) {
      errors.push(`${side}.version: "${s.version}" is not a 4-part MSIX version`)
    }
    if (typeof s.commit === 'string' && !COMMIT_RE.test(s.commit)) {
      errors.push(`${side}.commit: "${s.commit}" is not a 40-hex commit`)
    }
    const art = s.artifact
    if (!art || typeof art !== 'object') {
      errors.push(`${side}.artifact: missing`)
    } else {
      for (const f of REQUIRED_ARTIFACT_FIELDS) {
        if (typeof art[f] !== 'string' || !art[f]) errors.push(`${side}.artifact.${f}: missing or not a string`)
      }
      if (typeof art.sha256 === 'string' && !SHA256_RE.test(art.sha256)) {
        errors.push(`${side}.artifact.sha256: not 64 hex chars`)
      }
      if (typeof art.path !== 'string' || !art.path) {
        // The parent resolver writes artifact.path (the downloaded, verified
        // local file). Its absence means the contract regressed — fail loudly
        // rather than serving an unverified package.
        errors.push(`${side}.artifact.path: missing (parent resolver must write the downloaded local path)`)
      } else if (o.artifactPathsExist !== false && !fs.existsSync(art.path)) {
        errors.push(`${side}.artifact.path: "${art.path}" does not exist`)
      }
    }
  }

  if (m?.old?.publisher && m?.new?.publisher && m.old.publisher !== m.new.publisher) {
    errors.push('old.publisher and new.publisher disagree — one manifest, one identity')
  }
  for (const side of MANIFEST_SIDES) {
    if (m?.[side]?.publisher && m[side].publisher !== o.expectedPublisher) {
      errors.push(`${side}.publisher does not equal the production OUT_OF_STORE_PUBLISHER — refusing to serve a package the OS would reject`)
    }
  }
  if (m?.old?.identity && m?.new?.identity && m.old.identity !== m.new.identity) {
    errors.push('old.identity and new.identity disagree — an App Installer update swaps within ONE identity')
  }
  if (m?.old?.applicationId && m?.new?.applicationId && m.old.applicationId !== m.new.applicationId) {
    errors.push('old.applicationId and new.applicationId disagree')
  }
  if (
    m?.old?.version && m?.new?.version &&
    FOUR_PART_RE.test(m.old.version) && FOUR_PART_RE.test(m.new.version) &&
    !fourPartNewer(m.new.version, m.old.version)
  ) {
    errors.push(`new.version ${m.new.version} is not strictly newer than old.version ${m.old.version}`)
  }
  if (m?.old?.commit && m?.old?.commit === m?.new?.commit) {
    errors.push('old.commit equals new.commit — no update to prove')
  }

  return { ok: errors.length === 0, errors }
}

/**
 * The feed layout for one manifest: per-side bundle file name (served under
 * <feedUrl>/<side>/) and the single swapped descriptor path.
 * @param {string} feedDir
 * @param {{ old: { artifact: { path: string } }, new: { artifact: { path: string } } }} manifest
 * @returns {{ oldBundlePath: string, newBundlePath: string, oldBundleName: string, newBundleName: string, descriptorPath: string }}
 */
export function feedLayout(feedDir, manifest) {
  const oldBundleName = path.basename(manifest.old.artifact.path)
  const newBundleName = path.basename(manifest.new.artifact.path)
  if (oldBundleName === newBundleName) {
    // Same filename at the same URL would leave App Installer's cache (and
    // any human reading the feed) unable to tell the swap happened.
    throw new Error(`old and new bundle filenames collide: ${oldBundleName} — stage per-side dirs`)
  }
  return {
    oldBundlePath: path.join(feedDir, 'old', oldBundleName),
    newBundlePath: path.join(feedDir, 'new', newBundleName),
    oldBundleName,
    newBundleName,
    descriptorPath: path.join(feedDir, 'update.appinstaller')
  }
}

/**
 * The arguments for the production buildAppInstaller for one feed side.
 * variantChannelPath is '' — the side is folded into baseUrl (context
 * contract), and the descriptor Uri is the optional descriptorFilename
 * (parent-owned fix; defaults to the bundle filename with .appinstaller).
 * @param {{ baseUrl: string, identityName: string, version: string, bundleFilename: string, descriptorFilename?: string }} o
 */
export function descriptorArgs(o) {
  return {
    baseUrl: o.baseUrl,
    variantChannelPath: '',
    identityName: o.identityName,
    version: o.version,
    bundleFilename: o.bundleFilename,
    ...(o.descriptorFilename ? { descriptorFilename: o.descriptorFilename } : {})
  }
}

/** Load the production msix-shared module from an explicit path. */
export async function loadMsixShared(msixSharedPath) {
  const resolved = path.resolve(msixSharedPath)
  if (!fs.existsSync(resolved)) {
    throw new Error(`production msix-shared.mjs not found at ${resolved} — pass --msix-shared`)
  }
  return import(pathToFileURL(resolved).href)
}

// ── CLI ──────────────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const out = {}
  for (let i = 0; i < argv.length; i += 2) {
    out[String(argv[i]).replace(/^--/, '')] = argv[i + 1]
  }
  return out
}

async function main() {
  const argv = process.argv[2] === '--' ? process.argv.slice(3) : process.argv.slice(2)
  const cmd = argv[0]
  const flags = parseArgs(argv.slice(1))
  const msixShared = flags['msix-shared'] ||
    path.resolve(fileURLToPath(new URL('../../../scripts/msix-shared.mjs', import.meta.url)))

  if (cmd === 'validate-manifest') {
    const shared = await loadMsixShared(msixShared)
    const manifest = JSON.parse(fs.readFileSync(flags.manifest, 'utf8'))
    const result = validateBundledManifest(manifest, {
      expectedPublisher: shared.OUT_OF_STORE_PUBLISHER,
      arch: flags.arch || undefined
    })
    console.log(JSON.stringify(result))
    process.exit(result.ok ? 0 : 1)
  }

  if (cmd === 'descriptor') {
    const shared = await loadMsixShared(msixShared)
    if (typeof shared.buildAppInstaller !== 'function') {
      throw new Error('production buildAppInstaller missing — contract regression')
    }
    const xml = shared.buildAppInstaller(descriptorArgs({
      baseUrl: flags['base-url'],
      identityName: flags.identity,
      version: flags.version,
      bundleFilename: flags.bundle,
      descriptorFilename: flags['descriptor-filename'] || undefined
    }))
    // CONTRACT CHECK: the descriptor's own Uri must be the descriptor
    // filename (the OS registers THIS uri as the update source), not a
    // derived-from-bundle name. If the parent's descriptorFilename support
    // is missing, the derived Uri would be <bundle>.appinstaller instead.
    if (flags['descriptor-filename']) {
      const m = /Uri="([^"]+)"/.exec(xml)
      if (!m || !m[1].endsWith(flags['descriptor-filename'])) {
        throw new Error(
          `buildAppInstaller ignored descriptorFilename: Uri=${m ? m[1] : '(none)'} ` +
          `does not end with ${flags['descriptor-filename']} — parent msix-shared fix required`
        )
      }
    }
    const feed = path.resolve(flags.feed)
    fs.mkdirSync(path.dirname(path.resolve(flags.out || path.join(feed, 'update.appinstaller'))), { recursive: true })
    const outPath = flags.out || path.join(feed, 'update.appinstaller')
    fs.writeFileSync(outPath, xml)
    console.log(JSON.stringify({ ok: true, path: outPath }))
    return
  }

  if (cmd === 'serve') {
    const shared = await loadMsixShared(msixShared)
    const feed = path.resolve(flags.feed)
    fs.mkdirSync(feed, { recursive: true })
    const server = http.createServer((req, res) => {
      let urlPath
      try { urlPath = decodeURIComponent((req.url || '/').split('?')[0]) }
      catch { res.writeHead(400).end('bad path'); return }
      const rel = urlPath.replace(/^\/+/, '')
      const target = path.resolve(feed, rel)
      if (!target.startsWith(feed + path.sep) && target !== feed) {
        res.writeHead(403).end('forbidden')
        return
      }
      fs.stat(target, (err, stat) => {
        if (err || !stat.isFile()) {
          console.log(`[feed] 404 ${urlPath}`)
          res.writeHead(404).end('not found')
          return
        }
        const mime = shared.contentTypeFor(path.basename(target)) || 'application/octet-stream'
        res.writeHead(200, { 'Content-Type': mime, 'Content-Length': stat.size, 'Cache-Control': 'no-store' })
        const stream = fs.createReadStream(target)
        stream.on('error', () => res.destroy())
        res.on('close', () => stream.destroy())
        stream.pipe(res)
        console.log(`[feed] 200 ${urlPath} (${mime})`)
      })
    })
    server.listen(0, '127.0.0.1', () => {
      const url = `http://127.0.0.1:${server.address().port}`
      fs.writeFileSync(flags['port-file'], url)
      console.log(`[feed] serving ${feed} at ${url}`)
    })
    return
  }

  console.error('usage: validate-manifest | descriptor | serve (see file header)')
  process.exit(2)
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(err => {
    console.error(err && err.stack || String(err))
    process.exit(1)
  })
}
