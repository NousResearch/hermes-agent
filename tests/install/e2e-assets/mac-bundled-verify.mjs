#!/usr/bin/env node
'use strict'
// mac-bundled-verify.mjs — verify an INSTALLED macOS bundle on the runner:
// real codesign validation, BOTH signing identifiers (CFBundleIdentifier
// exactly, TeamIdentifier exactly), the bundle version, the lipo arch, and
// the baked install stamp, against a manifest side from bundle-inputs.json.
// Parses success receipts — file existence is never success.
//
// macOS-only (codesign/plutil/lipo). Pure assertion logic lives in
// mac-bundled-manifest.cjs and is covered by tests-js.
//
// Usage:
//   node mac-bundled-verify.mjs verify-app --app <path> \
//     --expect-version 1.2.3 --expect-commit <sha> --expect-tag vX.Y.Z \
//     --expect-identity <CFBundleIdentifier> --expect-team <TeamID> \
//     --expect-arch arm64|x64 [--out <receipt.json>]

import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { parseArgs } from 'node:util'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const { codesignTeam, stampAssertions } = require('./mac-bundled-manifest.cjs')

const [, , command, ...rest] = process.argv

function fail(message) {
  console.error(`E2E ASSERTION FAILED: ${message}`)
  process.exit(1)
}

// codesign -dv prints its DISPLAY output on STDERR (stdout stays empty),
// so capture both streams and hand the combined text to the parser.
function codesignDisplay(app) {
  const result = spawnSync('codesign', ['-dv', app], { encoding: 'utf8' })
  const text = `${result.stdout || ''}\n${result.stderr || ''}`
  if (result.status !== 0 && !text.trim()) {
    fail(`codesign -dv failed for ${app}: ${result.stderr || result.error}`)
  }
  return text
}

// lipo -archs prints e.g. "arm64" or "x86_64" (space-separated for
// universal binaries) on stdout.
function bundleArchs(executable) {
  return execFileSync('lipo', ['-archs', executable], { encoding: 'utf8' }).split(/\s+/).filter(Boolean)
}

const ARCH_TOKENS = { arm64: 'arm64', x64: 'x86_64' }

function verifyApp() {
  const { values } = parseArgs({
    strict: false,
    args: rest,
    options: {
      app: { type: 'string' },
      'expect-version': { type: 'string' },
      'expect-commit': { type: 'string' },
      'expect-tag': { type: 'string' },
      'expect-identity': { type: 'string' },
      'expect-team': { type: 'string' },
      'expect-arch': { type: 'string' },
      out: { type: 'string' },
    },
  })
  const app = values.app
  if (!app || !fs.existsSync(path.join(app, 'Contents', 'Info.plist'))) {
    fail(`not a .app bundle: ${app}`)
  }

  // 1. The signature is valid, intact, and satisfies its own designated
  //    requirement — Squirrel's gate will re-run this at install time.
  try {
    execFileSync('codesign', ['--verify', '--deep', '--strict', '--verbose=2', app], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
  } catch (error) {
    fail(`codesign --verify rejected ${app}: ${String(error.stderr || error)}`)
  }

  // 2. BOTH identifiers, exactly (no namespace prefix acceptance):
  //    - the signing TEAM must match the manifest teamId — Squirrel.Mac
  //      only accepts an update signed by the SAME team;
  //    - the CFBundleIdentifier must equal the manifest identity — the
  //      updater only replaces the same application bundle.
  const displayOut = codesignDisplay(app)
  const team = codesignTeam(displayOut)
  if (!team) fail(`no TeamIdentifier in codesign -dv output for ${app}`)
  if (team !== values['expect-team']) {
    fail(`codesign TeamIdentifier ${team} != manifest teamId ${values['expect-team']}`)
  }

  // 3. Bundle version, identifier and executable from the real Info.plist.
  const plist = JSON.parse(execFileSync('plutil', ['-convert', 'json', '-o', '-', path.join(app, 'Contents', 'Info.plist')], { encoding: 'utf8' }))
  const receipt = {
    app,
    teamIdentifier: team,
    bundleVersion: plist.CFBundleShortVersionString,
    bundleIdentifier: plist.CFBundleIdentifier,
    executable: plist.CFBundleExecutable,
    verifiedAt: new Date().toISOString(),
  }
  if (receipt.bundleVersion !== values['expect-version']) {
    fail(`CFBundleShortVersionString ${receipt.bundleVersion} != expected ${values['expect-version']}`)
  }
  // The manifest identity IS the CFBundleIdentifier: exact equality.
  if (receipt.bundleIdentifier !== values['expect-identity']) {
    fail(`CFBundleIdentifier ${receipt.bundleIdentifier} != manifest identity ${values['expect-identity']}`)
  }

  // 4. The executable the plist NAMES must exist — the update driver and
  //    watcher launch exactly this path (never a hard-coded binary name).
  const execPath = path.join(app, 'Contents', 'MacOS', String(receipt.executable))
  if (!receipt.executable || !fs.existsSync(execPath)) {
    fail(`no executable at ${execPath} (CFBundleExecutable=${receipt.executable})`)
  }
  receipt.executablePath = execPath

  // 5. The installed bundle really IS built for the leg's architecture.
  const archs = bundleArchs(execPath)
  receipt.lipoArchs = archs
  const wantToken = ARCH_TOKENS[values['expect-arch']]
  if (!wantToken) fail(`--expect-arch must be arm64 or x64, got ${values['expect-arch']}`)
  if (!archs.includes(wantToken)) {
    fail(`lipo -archs of ${execPath} is [${archs.join(', ')}]; expected ${wantToken} for ${values['expect-arch']}`)
  }

  // 6. The baked install stamp (provenance: commit, tag, payload, mechanism).
  const stampPath = path.join(app, 'Contents', 'Resources', 'install-stamp.json')
  if (!fs.existsSync(stampPath)) fail(`no install stamp at ${stampPath}`)
  const stamp = JSON.parse(fs.readFileSync(stampPath, 'utf8'))
  receipt.stamp = stamp
  const problems = stampAssertions(stamp, {
    commit: values['expect-commit'],
    tag: values['expect-tag'],
  })
  if (problems.length) fail(`install stamp disagrees: ${problems.join('; ')}`)

  console.log(JSON.stringify(receipt, null, 2))
  if (values.out) fs.writeFileSync(values.out, JSON.stringify(receipt, null, 2) + '\n')
}

switch (command) {
  case 'verify-app': verifyApp(); break
  default:
    console.error(`unknown command: ${command}`)
    process.exit(3)
}
