#!/usr/bin/env node
// codesign-mac.cjs — mandatory ad-hoc bundle signing + hard verification gate
// for packaged macOS Hermes.app builds (BUILD-266).
//
// WHY THIS EXISTS
// ---------------
// The 2026-07-01 v0.17.0 local build shipped to /Applications/Hermes.app with
// NO valid bundle-level code signature: `Contents/_CodeSignature/` was absent
// and `codesign -dvvv` reported the generic Electron linker-signed stub
// (Identifier=Electron) instead of `com.nousresearch.hermes`. `codesign -vv`
// failed with "code has no resources but signature indicates they must be
// present", and Finder/Dock rejected the app as "damaged" on launch.
//
// electron-builder silently skips macOS code signing entirely when no
// Developer ID identity is configured (no CSC_LINK/CSC_NAME and no matching
// keychain identity, as on a bare dev machine) -- it leaves whatever stale
// signature the Electron binary shipped with. Our old `afterSign` hook
// (notarize.cjs) only ran real notarization and did nothing when
// notarization credentials were absent, so an unsigned/stub-signed bundle
// sailed straight through `npm run pack` / `dist:mac*` with no error, and the
// only signing step left was a MANUAL one in the deploy runbook -- easy to
// skip or run out of order, which is exactly what happened.
//
// This module closes that gap in code:
//   - `needsAdHocSign` inspects the bundle's *current* signature and only
//     reports true when it's missing, broken, or the bare Electron stub --
//     so a real Developer ID signature (from CSC_LINK or keychain
//     auto-discovery) is never clobbered.
//   - `signAdHoc` applies `codesign --force --deep --sign -` (no explicit
//     -i, so codesign derives the identifier from the bundle's
//     Info.plist CFBundleIdentifier, exactly like the historical manual
//     step that produced every known-good backup's `com.nousresearch.hermes`
//     identifier).
//   - `verifyCodesign` is the hard gate: `codesign -vv` must exit 0, and the
//     `codesign -dv` identifier must exist, must not be the bare Electron
//     stub, and (when `expectedIdentifier` is given) must match it exactly.
//     It throws on any failure -- callers must let that propagate so a
//     badly-signed bundle never becomes a build artifact, let alone gets
//     copied into /Applications.
//   - `signAndVerify` is the composed pipeline step: sign-if-needed, then
//     always verify.
//
// Every exported function accepts an injectable `exec` for unit testing.
// Also runnable standalone: `node scripts/codesign-mac.cjs <path-to-.app>`.

const fs = require('node:fs')
const { execFile } = require('node:child_process')

// The bare identifier Electron's own linker-signed stub carries when nothing
// has re-signed the bundle. Case variants are defensive; codesign itself is
// case-sensitive, but we'd rather over- than under-detect the stub.
const STUB_IDENTIFIERS = new Set(['Electron', 'electron'])

function defaultExec(command, args) {
  return new Promise((resolve, reject) => {
    execFile(command, args, { maxBuffer: 16 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        const err = new Error(
          `${command} ${args.join(' ')} failed (exit ${error.code ?? '?'}): ${
            stderr?.trim() || stdout?.trim() || error.message
          }`
        )
        err.stdout = stdout
        err.stderr = stderr
        err.code = error.code
        reject(err)
        return
      }
      resolve({ stdout, stderr })
    })
  })
}

function assertBundleExists(appPath) {
  if (!appPath || !fs.existsSync(appPath)) {
    throw new Error(`codesign-mac: bundle not found: ${appPath}`)
  }
}

// codesign writes `-d` display output to stderr, not stdout. Parse the
// `Identifier=` line out of whichever stream carries it.
function parseIdentifier(output) {
  const match = /^Identifier=(.+)$/m.exec(output || '')
  return match ? match[1].trim() : null
}

// Read the bundle's current signing identifier without throwing. Returns
// null when the bundle is unsigned or `codesign -dv` otherwise fails.
async function readIdentifier(appPath, { exec = defaultExec } = {}) {
  try {
    const { stderr, stdout } = await exec('codesign', ['-dv', appPath])
    return parseIdentifier(stderr) || parseIdentifier(stdout)
  } catch {
    return null
  }
}

// True when the bundle's current signature is missing, broken, or the bare
// Electron stub -- i.e. it needs our ad-hoc sign. False when it already
// carries a valid, non-stub signature (e.g. a real Developer ID signature
// electron-builder applied itself) that we must not clobber.
async function needsAdHocSign(appPath, { exec = defaultExec } = {}) {
  assertBundleExists(appPath)

  try {
    await exec('codesign', ['-vv', appPath])
  } catch {
    return true
  }

  const identifier = await readIdentifier(appPath, { exec })
  if (!identifier || STUB_IDENTIFIERS.has(identifier)) {
    return true
  }
  return false
}

// Ad-hoc sign `appPath` so it always carries a valid bundle-level signature
// under its own identity, even with no Developer ID configured. No explicit
// `-i` is passed, so codesign derives the identifier from the bundle's
// Info.plist (CFBundleIdentifier) -- matching how every known-good backup
// ended up with identifier `com.nousresearch.hermes`.
async function signAdHoc(appPath, { exec = defaultExec } = {}) {
  assertBundleExists(appPath)
  await exec('codesign', ['--force', '--deep', '--sign', '-', appPath])
}

// Hard verification gate. Throws (never returns falsy) if:
//   - `codesign -vv` reports the bundle's signature/resources don't validate
//   - the signed identifier is missing
//   - the signed identifier is the bare Electron linker-signed stub
//   - `expectedIdentifier` is given and the identifier doesn't match it
// Returns the resolved identifier on success.
async function verifyCodesign(appPath, { exec = defaultExec, expectedIdentifier } = {}) {
  assertBundleExists(appPath)

  try {
    await exec('codesign', ['-vv', appPath])
  } catch (err) {
    throw new Error(`codesign-mac: signature verification failed for ${appPath}: ${err.message}`)
  }

  const identifier = await readIdentifier(appPath, { exec })
  if (!identifier) {
    throw new Error(`codesign-mac: no signing identifier reported for ${appPath}`)
  }
  if (STUB_IDENTIFIERS.has(identifier)) {
    throw new Error(
      `codesign-mac: ${appPath} is signed with the generic Electron stub identifier ` +
        `("${identifier}") instead of a real app identity -- this is the BUILD-266 regression ` +
        `(bundle has no valid signature). Aborting before deploy.`
    )
  }
  if (expectedIdentifier && identifier !== expectedIdentifier) {
    throw new Error(
      `codesign-mac: ${appPath} signing identifier is "${identifier}", expected "${expectedIdentifier}"`
    )
  }

  return identifier
}

// Composed pipeline step: sign-if-needed, then always verify. Safe to call
// on an already-properly-signed bundle (a Developer ID signature is left
// alone); always safe to call on an unsigned/stub-signed one (it gets
// ad-hoc signed first). Throws on any verification failure.
async function signAndVerify(appPath, { exec = defaultExec, expectedIdentifier } = {}) {
  if (await needsAdHocSign(appPath, { exec })) {
    await signAdHoc(appPath, { exec })
  }
  return verifyCodesign(appPath, { exec, expectedIdentifier })
}

module.exports = {
  STUB_IDENTIFIERS,
  needsAdHocSign,
  parseIdentifier,
  readIdentifier,
  signAdHoc,
  signAndVerify,
  verifyCodesign
}

// CLI entry point: `node scripts/codesign-mac.cjs <path-to-.app> [--verify-only]`
if (require.main === module) {
  const appPath = process.argv[2]
  if (!appPath) {
    console.error('usage: codesign-mac.cjs <path-to-.app> [--verify-only]')
    process.exit(2)
  }
  const verifyOnly = process.argv.includes('--verify-only')
  const expectedIdentifier = process.env.HERMES_EXPECTED_APP_ID || undefined

  const task = verifyOnly ? verifyCodesign(appPath, { expectedIdentifier }) : signAndVerify(appPath, { expectedIdentifier })

  task
    .then(identifier => {
      console.log(`[codesign-mac] OK — ${appPath} verified (identifier=${identifier || 'n/a'})`)
    })
    .catch(err => {
      console.error(`[codesign-mac] ${err.message}`)
      process.exit(1)
    })
}
