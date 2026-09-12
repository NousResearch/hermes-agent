#!/usr/bin/env node
// stage-msixbundle.mjs — the out-of-store MSIX distribution job.
//
// Runs on a Windows runner of the release workflow AFTER all legs built
// (needs: build). Two responsibilities:
//
//  1. OUT-OF-STORE FEED: bundle the x64 + arm64 per-arch .msix into one
//     universal .msixbundle, sign the bundle envelope, write the per-channel
//     .appinstaller, and upload both to the win32 feed dirs — bundle FIRST,
//     .appinstaller pointer LAST (a failed bundle upload leaves the previous
//     feed intact):
//         releases/win32/<stable|canary>/<name>-<ver>.win.msixbundle
//         releases/win32/<stable|canary>/stable.appinstaller (or canary.*)
//     The .appinstaller is the install + auto-update entry point; the bundle
//     is what the OS installs and swaps on update. Per-arch .msix files stay
//     in the immutable releases/tag/<tag>/ archive (uploaded by the legs).
//
//  2. STORE ARCHIVE: re-upload the Store-submission .msix files (built by
//     the win legs, prefixed Store-) to the tag archive. The Store is the
//     distribution for those — they never touch a feed dir.
//
// Usage (win runner, bash):
//   node scripts/stage-msixbundle.mjs --tag vX.Y.Z [--variant bundled|light]
// Reads HERMES_DESKTOP_VARIANT (bundled|light) from the environment; the
// workflow runs this job once per variant.
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'

import { appIdentity, buildAppInstaller } from './msix-shared.mjs'
import { ensureWindowsBundleTools } from '../apps/desktop/scripts/windows-bundle-tools.mjs'


const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const { values } = parseArgs({ options: {
  tag: { type: 'string' }, commit: { type: 'string' }, version: { type: 'string' },
  variant: { type: 'string' }, 'no-upload': { type: 'boolean' }, candidate: { type: 'boolean' },
} })
const tag = values.tag
const commitBuild = values.commit || ''
const commitVersion = values.version || ''
const noUpload = values['no-upload'] === true
const candidate = values.candidate === true
const variant = values.variant || process.env.HERMES_DESKTOP_VARIANT || 'bundled'

if (commitBuild && (tag || process.env.HERMES_PAYLOAD_TAG || candidate)) {
  throw new Error('Commit builds cannot select a release tag or candidate mode')
}
if (!commitBuild && (values.version !== undefined || noUpload)) {
  throw new Error('--version and --no-upload require --commit')
}

// product-identity.cjs keys the app name off HERMES_DESKTOP_VARIANT — the
// artifact filenames (HermesBundled-*-win-x64.msix) carry the bundled
// identity, so the env var MUST match the variant or the msix lookup
// fails. Set it before anything requires the identity.
process.env.HERMES_DESKTOP_VARIANT = variant
if (tag) process.env.HERMES_PAYLOAD_TAG = tag

if (commitBuild) {
  if (!/^[a-f0-9]{40}$/.test(commitBuild)) {
    console.error('[stage-msixbundle] --commit must be an exact full 40-hex SHA')
    process.exit(1)
  }
  if (!/^\d+\.\d+\.\d+$/.test(commitVersion)) {
    console.error('[stage-msixbundle] --version=X.Y.Z is required with --commit (the target pyproject version)')
    process.exit(1)
  }
  if (!noUpload) {
    console.error('[stage-msixbundle] commit mode must pass --no-upload (commit builds never write a feed)')
    process.exit(1)
  }
  if (tag) {
    console.error('[stage-msixbundle] --commit and --tag are mutually exclusive')
    process.exit(1)
  }
  process.env.HERMES_BUILD_COMMIT = commitBuild
  process.env.HERMES_PAYLOAD_VERSION = commitVersion
} else if (!tag) {
  console.error('[stage-msixbundle] --tag=<vX.Y.Z> is required')
  process.exit(1)
}
if (!['bundled', 'light'].includes(variant)) {
  console.error(`[stage-msixbundle] --variant must be 'bundled' or 'light', got '${variant}'`)
  process.exit(1)
}
if (process.platform !== 'win32') {
  console.error('[stage-msixbundle] this job must run on a Windows runner (makeappx + signtool)')
  process.exit(1)
}

const canary = /-canary\./.test(tag)
if (!commitBuild && !canary && !candidate) throw new Error('Stable bundles must use the staged stable-release workflow')
const channel = canary ? 'canary' : 'stable'
const channelDir = `releases/win32/${variant === 'light' ? 'light/' : ''}${channel}`

const desktop = path.join(REPO_ROOT, 'apps', 'desktop')
const releaseDir = path.join(desktop, 'release')
const { identity, version, name, fileVersion } = appIdentity(desktop, tag)

// Per-arch .msix files are found by the name electron-builder gave them
// (appInfo.version = the 3-part or full-canary string, NOT the 4-part feed
// version). The bundle /bv, .appinstaller Version and feed filenames all use
// the 4-part `version` — what Windows compares for updates.
function msixFile(arch) {
  return path.join(releaseDir, `${name}-${fileVersion}-win-${arch}.msix`)
}
function bundleFile() {
  return path.join(releaseDir, `${name}-${version}-win.msixbundle`)
}

const signing = Boolean(process.env.AZURE_SIGN_ENDPOINT && process.env.AZURE_SIGN_ACCOUNT && process.env.AZURE_SIGN_PROFILE)
const { makeappx, signtool, dlib, dotnetRoot } = await ensureWindowsBundleTools({ signing })

// ── 1. bundle ──────────────────────────────────────────────────────────────
const x64 = msixFile('x64')
const arm64 = msixFile('arm64')
const bundle = bundleFile()
if (!fs.existsSync(x64) || !fs.existsSync(arm64)) {
  console.error(`[stage-msixbundle] need both per-arch msix to bundle:\n  ${x64}\n  ${arm64}`)
  process.exit(1)
}

// makeappx bundle /d includes EVERY .msix in the dir — the Store-submission
// packages (Store-*.msix, same release dir after the legs merged their
// artifacts) must never ride inside the out-of-store bundle. Stage only the
// two per-arch packages into a clean dir before bundling.
const bundleStaging = path.join(releaseDir, '__bundle-staging')
fs.rmSync(bundleStaging, { recursive: true, force: true })
fs.mkdirSync(bundleStaging, { recursive: true })
fs.copyFileSync(x64, path.join(bundleStaging, path.basename(x64)))
fs.copyFileSync(arm64, path.join(bundleStaging, path.basename(arm64)))

if (fs.existsSync(bundle)) fs.rmSync(bundle, { force: true })
execFileSync(makeappx, ['bundle', '/o', '/bv', version, '/d', bundleStaging, '/p', bundle], { stdio: 'inherit' })

// Signtool signs the bundle and refreshes its inner package signatures.
// The source .msix files stay unchanged. Without Azure configuration this
// remains an unsigned local build, as on the build legs.
if (signing) {
  const metaPath = path.join(releaseDir, 'msixbundle-sign.json')
  fs.writeFileSync(metaPath, JSON.stringify({
    Endpoint: process.env.AZURE_SIGN_ENDPOINT,
    CodeSigningAccountName: process.env.AZURE_SIGN_ACCOUNT,
    CertificateProfileName: process.env.AZURE_SIGN_PROFILE
  }))
  const signEnv = { ...process.env }
  if (dotnetRoot) signEnv.DOTNET_ROOT = dotnetRoot
  // MSIX/appx packages REQUIRE a timestamp — signtool silently exits 3 on
  // a .msixbundle sign without /tr (untimestamped appx is invalid). And
  // the /tr URL must be one the ATS dlib can speak: the dlib handles the
  // RFC3161 exchange itself (@url: form) and cannot parse a third-party
  // server's response ("no content extracted" with digicert). The only
  // known-working timestamp server for the dlib is Microsoft's own
  // timestamp.acs.microsoft.com (electron-builder's default, and what the
  // build legs' .msix sign uses). acs is intermittently flaky, so retry
  // the whole sign — a retried sign beats a failed bundle, and signtool
  // replaces the signature on re-sign so a retry is safe.
  const sign = () =>
    execFileSync(signtool, [
      'sign', '/fd', 'SHA256', '/td', 'SHA256', '/tr', 'http://timestamp.acs.microsoft.com',
      '/dlib', dlib, '/dmdf', metaPath, bundle
    ], { stdio: 'inherit', env: signEnv })
  let attempt = 0
  for (;;) {
    try {
      sign()
      break
    } catch (err) {
      attempt += 1
      if (attempt >= 3) throw err
      console.warn(`[stage-msixbundle] sign attempt ${attempt} failed, retrying…`)
    }
  }
  execFileSync(signtool, ['verify', '/pa', bundle], { stdio: 'inherit' })
} else {
  console.warn('[stage-msixbundle] AZURE_SIGN_* not set — bundle will be UNSIGNED')
}

if (candidate) {
  execFileSync(signtool, ['verify', '/pa', bundle], { stdio: 'inherit' })
  console.log(`[stage-msixbundle] candidate ready: ${bundle}`)
  process.exit(0)
}

// Commit-only mode stops here: the workflow hands the bundle to R2 through
// scripts.releases.handoff (schema-2 receipt) — never a feed dir.
if (commitBuild) {
  console.log(`[stage-msixbundle] commit bundle ready (no upload): ${bundle}`)
  process.exit(0)
}

// ── 2. .appinstaller + uploads ─────────────────────────────────────────────
const baseUrl = String(process.env.CLOUDFLARE_R2_PUBLIC_URL || '').replace(/\/+$/, '')
if (!baseUrl) {
  console.error('[stage-msixbundle] CLOUDFLARE_R2_PUBLIC_URL is required (feed dir URLs come from it)')
  process.exit(1)
}

const appinstaller = buildAppInstaller({
  baseUrl,
  variantChannelPath: channelDir,
  identityName: identity.msixAppIdWithOrg,
  version,
  bundleFilename: `${name}-${version}-win.msixbundle`
})
const appinstallerName = `${channel}.appinstaller`
fs.writeFileSync(path.join(releaseDir, appinstallerName), appinstaller)

const upload = (key, file, keyIsFull = true) => {
  // NOTE: no fs.readFileSync here — the msixbundle can exceed Node's 2GiB
  // buffer limit (ERR_FS_FILE_TOO_LARGE). scripts.releases.r2 put reads + hashes
  // the file itself; log the size via stat instead.
  const { size } = fs.statSync(file)
  console.log(`[stage-msixbundle] upload ${key} (${size} bytes)`)
  // Feed-dir keys are FULL object keys (releases/win32/<ch>/…) — pass
  // --key-is-full so r2 put does NOT wrap them under releases/tag/<tag>/.
  // scripts.releases.r2 put derives Content-Type from the key extension.
  execFileSync(process.env.HERMES_PYTHON || 'python', ['-m', 'scripts.releases.r2', 'put', '--tag', tag, '--key', key, '--file', file, ...(keyIsFull ? ['--key-is-full'] : [])], {
    cwd: REPO_ROOT,
    stdio: 'inherit'
  })
}

// C22 ordering: bundle FIRST, pointer LAST. scripts.releases.r2 PUTs then
// HEAD-verifies the remote content-length — a failed/short upload throws
// and aborts this job before the pointer is written.
upload(`${channelDir}/${name}-${version}-win.msixbundle`, bundle)
upload(`${channelDir}/${appinstallerName}`, path.join(releaseDir, appinstallerName))

// The Store-submission .msix files were already uploaded to the tag archive
// by the win legs (Store- prefix); nothing for this job to re-upload.
console.log('[stage-msixbundle] done — feed manifests + bundle staged')
