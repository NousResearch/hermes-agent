#!/usr/bin/env node
// Native Windows adapter: bundle x64 + arm64 MSIX packages and sign the
// envelope. Every mode stops at the artifact; CI stages it for native smoke
// before the separate release Python publisher can write an App Installer feed.
// Usage: node scripts/stage-msixbundle.mjs --tag vX.Y.Z --candidate
//        node scripts/stage-msixbundle.mjs --tag vX.Y.Z-canary.STAMP
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'

import { appIdentity, channelBuildRequest } from './msix-shared.mjs'
import { ensureWindowsBundleTools } from '../apps/desktop/scripts/windows-bundle-tools.mjs'


const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const { values } = parseArgs({ options: {
  tag: { type: 'string' }, commit: { type: 'string' }, version: { type: 'string' },
  variant: { type: 'string' }, 'no-upload': { type: 'boolean' }, candidate: { type: 'boolean' },
  'channel-request': { type: 'string' },
} })
const channelRequest = values['channel-request']
  ? channelBuildRequest({ ...process.env, HERMES_DESKTOP_VARIANT: values.variant || 'bundled',
    _HERMES_CHANNEL_REQUEST_JSON: fs.readFileSync(values['channel-request'], 'utf8') }) : null
const tag = values.tag
const commitBuild = values.commit || ''
const commitVersion = values.version || ''
const noUpload = values['no-upload'] === true
const candidate = values.candidate === true
const variant = values.variant || process.env.HERMES_DESKTOP_VARIANT || 'bundled'

if (channelRequest && (tag || commitBuild || values.version || candidate || variant !== 'bundled'
    || process.env.HERMES_PAYLOAD_TAG || process.env.HERMES_BUILD_COMMIT)) {
  throw new Error('Channel requests cannot select tag, commit, Store or candidate inputs')
}
if (commitBuild && (tag || process.env.HERMES_PAYLOAD_TAG || candidate)) {
  throw new Error('Commit builds cannot select a release tag or candidate mode')
}
if (!commitBuild && !channelRequest && (values.version !== undefined || noUpload)) {
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
} else if (!tag && !channelRequest) {
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
if (!commitBuild && !channelRequest && !canary && !candidate) throw new Error('Stable bundles must use the staged stable-release workflow')

const desktop = path.join(REPO_ROOT, 'apps', 'desktop')
const releaseDir = path.join(desktop, 'release')
const { version, name, fileVersion } = channelRequest
  ? { version: channelRequest.windowsVersion, name: channelRequest.identity.artifactNamePascal, fileVersion: channelRequest.version }
  : appIdentity(desktop, tag)

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

// Local assembly may be unsigned. CI's native smoke requires a valid signature
// on the receipt-bound download before any release publication can proceed.
console.log(`[stage-msixbundle] bundle ready (no upload): ${bundle}`)
