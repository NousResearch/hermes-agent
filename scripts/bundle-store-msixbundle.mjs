#!/usr/bin/env node
// bundle-store-msixbundle.mjs — the Store-submission bundle step of the
// desktop-bundled-release workflow's store-publish job.
//
// The win32 build legs produce per-arch Store-submission packages
// (Store-<name>-<fileVersion>-win-<arch>.msix, built with
// HERMES_DESKTOP_VARIANT=store / the Partner Center identity). This script
// bundles the x64 + arm64 packages into ONE universal Store .msixbundle for
// the Windows Store submission, and prints the bundle's absolute path on
// stdout (the workflow captures it for `msstore publish`).
//
// The bundle is deliberately left UNSIGNED: the Store re-signs the package
// with the Microsoft Store certificate on ingestion (same posture as the
// build legs, whose Store-*.msix ship unsigned — see sign-msix.mjs).
//
// Usage (win runner, bash):
//   node scripts/bundle-store-msixbundle.mjs --tag vX.Y.Z
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { appIdentity, resolveWinSdkTools } from './msix-shared.mjs'

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// node strips the first '--' (and an immediately-following option) for its
// own use; parse space-separated flag pairs, not --flag=value.
const args = process.argv.slice(2)
const flagValue = (name) => {
  for (let i = 0; i < args.length - 1; i += 1) {
    if (args[i] === name) return args[i + 1]
  }
  return undefined
}
const tag = flagValue('--tag') || process.env.HERMES_PAYLOAD_TAG
if (!tag) {
  console.error('[bundle-store] --tag=<vX.Y.Z> is required')
  process.exit(1)
}
if (process.platform !== 'win32') {
  console.error('[bundle-store] this job must run on a Windows runner (makeappx)')
  process.exit(1)
}

// product-identity.cjs keys the app name off HERMES_DESKTOP_VARIANT — the
// Store-submission artifacts carry the Store- prefix + Partner Center
// identity, so the env var MUST be 'store' before the identity lookup.
process.env.HERMES_DESKTOP_VARIANT = 'store'

const desktop = path.join(REPO_ROOT, 'apps', 'desktop')
const releaseDir = path.join(desktop, 'release')
const { name, version, fileVersion } = appIdentity(desktop, tag)

// Per-arch Store-submission packages: electron-builder names them with the
// Store- prefix + appInfo.version (the 3-part or full-canary string), which
// appIdentity reports as `fileVersion` — same convention as the out-of-store
// bundle in stage-msixbundle.mjs.
const storeMsix = (arch) => path.join(releaseDir, `Store-${name}-${fileVersion}-win-${arch}.msix`)
const x64 = storeMsix('x64')
const arm64 = storeMsix('arm64')
if (!fs.existsSync(x64) || !fs.existsSync(arm64)) {
  console.error(`[bundle-store] need both per-arch Store-*.msix to bundle:\n  ${x64}\n  ${arm64}`)
  process.exit(1)
}

// makeappx bundle /d includes EVERY .msix in the dir — stage only the two
// Store packages into a clean dir (mirror stage-msixbundle.mjs).
const staging = path.join(releaseDir, '__store-bundle-staging')
fs.rmSync(staging, { recursive: true, force: true })
fs.mkdirSync(staging, { recursive: true })
fs.copyFileSync(x64, path.join(staging, path.basename(x64)))
fs.copyFileSync(arm64, path.join(staging, path.basename(arm64)))

const bundle = path.join(releaseDir, `Store-${name}-${version}-win.msixbundle`)
const makeappx = path.join(resolveWinSdkTools(), 'makeappx.exe')
if (fs.existsSync(bundle)) fs.rmSync(bundle, { force: true })
execFileSync(makeappx, ['bundle', '/o', '/bv', version, '/d', staging, '/p', bundle], {
  stdio: ['ignore', 'ignore', 'inherit'] // stdout stays clean: the path is the machine-readable result
})
fs.rmSync(staging, { recursive: true, force: true })

// stdout = the absolute bundle path, the ONLY thing the workflow reads back.
console.log(bundle)
