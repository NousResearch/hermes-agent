'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const notarize = require('./notarize.cjs').default
const { needsAdHocSign, verifyCodesign } = require('./codesign-mac.cjs')
const { build: BUILD_CONFIG } = require('../package.json')

// notarize.cjs's afterSign hook is where BUILD-266 is fixed: every darwin
// build must come out ad-hoc signed (if it wasn't already properly signed)
// and verified before this hook returns, regardless of whether notarization
// credentials are configured. These tests exercise the real hook end to
// end against a fixture bundle -- no mocking of codesign-mac.cjs -- with
// notarization env vars cleared so no network calls happen.
const NOTARY_ENV_KEYS = [
  'APPLE_NOTARY_PROFILE',
  'APPLE_API_KEY',
  'APPLE_API_KEY_ID',
  'APPLE_API_ISSUER',
  'CSC_LINK',
  'CSC_NAME'
]

function withClearedNotaryEnv(fn) {
  const saved = {}
  for (const key of NOTARY_ENV_KEYS) {
    saved[key] = process.env[key]
    delete process.env[key]
  }
  return Promise.resolve()
    .then(fn)
    .finally(() => {
      for (const key of NOTARY_ENV_KEYS) {
        if (saved[key] === undefined) delete process.env[key]
        else process.env[key] = saved[key]
      }
    })
}

function buildFixtureApp(root, { identifier } = {}) {
  const appPath = path.join(root, 'Hermes.app')
  fs.mkdirSync(path.join(appPath, 'Contents', 'MacOS'), { recursive: true })
  fs.mkdirSync(path.join(appPath, 'Contents', 'Resources'), { recursive: true })
  fs.writeFileSync(path.join(appPath, 'Contents', 'MacOS', 'Hermes'), '#!/bin/sh\necho fixture\n', {
    mode: 0o755
  })
  fs.writeFileSync(
    path.join(appPath, 'Contents', 'Info.plist'),
    `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key>
  <string>${identifier || BUILD_CONFIG.appId}</string>
  <key>CFBundleExecutable</key>
  <string>Hermes</string>
  <key>CFBundleName</key>
  <string>Hermes</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
</dict>
</plist>
`,
    'utf8'
  )
  return appPath
}

function fakeContext(appOutDir, overrides = {}) {
  return {
    electronPlatformName: 'darwin',
    appOutDir,
    packager: { appInfo: { productFilename: 'Hermes' } },
    ...overrides
  }
}

test('notarize hook is a no-op on non-darwin platforms', async () => {
  const result = await notarize({ electronPlatformName: 'win32', appOutDir: '/nope', packager: {} })
  assert.equal(result, undefined)
})

test('notarize hook throws when the app bundle is missing', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-notarize-'))
  try {
    await assert.rejects(notarize(fakeContext(tempRoot)), /Cannot notarize missing app bundle/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test(
  'notarize hook signs and verifies an unsigned bundle, then skips notarization when creds are absent (BUILD-266)',
  { skip: process.platform !== 'darwin' && 'codesign is macOS-only' },
  async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-notarize-'))
    try {
      const appPath = buildFixtureApp(tempRoot)
      assert.equal(await needsAdHocSign(appPath), true, 'fixture should start unsigned')

      await withClearedNotaryEnv(() => notarize(fakeContext(tempRoot)))

      // The hook must leave the bundle validly signed under the real
      // (non-stub) identifier before it even reaches the "skip
      // notarization" branch.
      const identifier = await verifyCodesign(appPath, { expectedIdentifier: BUILD_CONFIG.appId })
      assert.equal(identifier, BUILD_CONFIG.appId)
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  }
)

test(
  'notarize hook re-signs and rejects the exact BUILD-266 regression (bare Electron stub identifier)',
  { skip: process.platform !== 'darwin' && 'codesign is macOS-only' },
  async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-notarize-'))
    try {
      const appPath = buildFixtureApp(tempRoot)
      // Reproduce the incident: signed, but with the bare "Electron" stub
      // identifier instead of com.nousresearch.hermes -- exactly what
      // electron-builder leaves behind when it skips signing and nothing
      // downstream fixes it.
      await new Promise((resolve, reject) => {
        require('node:child_process').execFile(
          'codesign',
          ['--force', '--sign', '-', '--identifier', 'Electron', appPath],
          err => (err ? reject(err) : resolve())
        )
      })

      await withClearedNotaryEnv(() => notarize(fakeContext(tempRoot)))

      // The hook must have detected the stub and re-signed with the real
      // identity before returning -- verify it's fixed, not still broken.
      const identifier = await verifyCodesign(appPath, { expectedIdentifier: BUILD_CONFIG.appId })
      assert.equal(identifier, BUILD_CONFIG.appId)
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  }
)

test(
  'notarize hook throws instead of returning when the bundle cannot be made to verify',
  { skip: process.platform !== 'darwin' && 'codesign is macOS-only' },
  async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-notarize-'))
    try {
      // A bundle with no Info.plist at all still gets ad-hoc signed by
      // codesign (it falls back to a path-derived identifier), so instead
      // force an expectedIdentifier mismatch scenario isn't reachable from
      // the hook's fixed appId -- so instead assert the hook's failure mode
      // by making the target not a valid bundle path (a plain file), which
      // codesign refuses to sign as a bundle.
      const notABundle = path.join(tempRoot, 'Hermes.app')
      fs.writeFileSync(notABundle, 'not a real bundle', 'utf8')

      await assert.rejects(withClearedNotaryEnv(() => notarize(fakeContext(tempRoot))))
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  }
)
