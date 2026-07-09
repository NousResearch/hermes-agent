'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  needsAdHocSign,
  parseIdentifier,
  signAdHoc,
  signAndVerify,
  verifyCodesign
} = require('./codesign-mac.cjs')

// ---------------------------------------------------------------------------
// Pure-logic tests -- run on every platform, no subprocess involved.
// ---------------------------------------------------------------------------

test('parseIdentifier extracts the Identifier= line from codesign -dv output', () => {
  const stderr = [
    'Executable=/tmp/Fixture.app/Contents/MacOS/Fixture',
    'Identifier=com.nousresearch.hermes',
    'Format=app bundle with Mach-O thin (arm64)',
    'CodeDirectory v=20500 size=185'
  ].join('\n')
  assert.equal(parseIdentifier(stderr), 'com.nousresearch.hermes')
})

test('parseIdentifier extracts the bare Electron stub identifier', () => {
  const stderr = 'Executable=/tmp/Fixture.app/Contents/MacOS/Fixture\nIdentifier=Electron\n'
  assert.equal(parseIdentifier(stderr), 'Electron')
})

test('parseIdentifier returns null when there is no Identifier= line', () => {
  assert.equal(parseIdentifier(''), null)
  assert.equal(parseIdentifier('code object is not signed at all'), null)
  assert.equal(parseIdentifier(undefined), null)
})

test('verifyCodesign rejects a missing bundle path without shelling out', async () => {
  await assert.rejects(verifyCodesign('/does/not/exist'), /bundle not found/)
})

test('needsAdHocSign rejects a missing bundle path without shelling out', async () => {
  await assert.rejects(needsAdHocSign('/does/not/exist'), /bundle not found/)
})

// ---------------------------------------------------------------------------
// Injected-exec tests -- exercise the branching logic without real codesign,
// so they run on every platform too.
// ---------------------------------------------------------------------------

function fakeExecSequence(responses) {
  const calls = []
  const exec = async (command, args) => {
    calls.push([command, ...args])
    const next = responses.shift()
    if (!next) throw new Error(`fakeExecSequence: no more scripted responses (call was ${command} ${args.join(' ')})`)
    if (next.reject) throw next.reject
    return next.resolve || { stdout: '', stderr: '' }
  }
  return { exec, calls }
}

test('signAndVerify skips signing when the bundle already has a valid non-stub signature', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-'))
  try {
    const appPath = path.join(tempRoot, 'Fixture.app')
    fs.mkdirSync(appPath, { recursive: true })

    const { exec, calls } = fakeExecSequence([
      { resolve: { stdout: '', stderr: '' } }, // needsAdHocSign: codesign -vv (valid)
      { resolve: { stdout: '', stderr: 'Identifier=com.nousresearch.hermes\n' } }, // needsAdHocSign: codesign -dv
      { resolve: { stdout: '', stderr: '' } }, // verifyCodesign: codesign -vv (valid)
      { resolve: { stdout: '', stderr: 'Identifier=com.nousresearch.hermes\n' } } // verifyCodesign: codesign -dv
    ])

    const identifier = await signAndVerify(appPath, { exec, expectedIdentifier: 'com.nousresearch.hermes' })

    assert.equal(identifier, 'com.nousresearch.hermes')
    // Exactly 4 codesign invocations (2x -vv, 2x -dv) -- no `--sign -` call,
    // proving a real Developer ID signature is never clobbered.
    assert.equal(calls.length, 4)
    assert.ok(calls.every(call => !call.includes('--sign')), `expected no --sign call, got: ${JSON.stringify(calls)}`)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('signAndVerify ad-hoc signs when the existing signature is the Electron stub', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-'))
  try {
    const appPath = path.join(tempRoot, 'Fixture.app')
    fs.mkdirSync(appPath, { recursive: true })

    const { exec, calls } = fakeExecSequence([
      { resolve: { stdout: '', stderr: '' } }, // needsAdHocSign: codesign -vv (valid but stub identity)
      { resolve: { stdout: '', stderr: 'Identifier=Electron\n' } }, // needsAdHocSign: codesign -dv -> stub
      { resolve: { stdout: '', stderr: '' } }, // signAdHoc: codesign --force --deep --sign -
      { resolve: { stdout: '', stderr: '' } }, // verifyCodesign: codesign -vv (valid)
      { resolve: { stdout: '', stderr: 'Identifier=com.nousresearch.hermes\n' } } // verifyCodesign: codesign -dv
    ])

    const identifier = await signAndVerify(appPath, { exec, expectedIdentifier: 'com.nousresearch.hermes' })

    assert.equal(identifier, 'com.nousresearch.hermes')
    const signCall = calls.find(call => call.includes('--sign'))
    assert.ok(signCall, `expected a --sign call, got: ${JSON.stringify(calls)}`)
    assert.deepEqual(signCall, ['codesign', '--force', '--deep', '--sign', '-', appPath])
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('verifyCodesign throws when the identifier is still the Electron stub after signing', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-'))
  try {
    const appPath = path.join(tempRoot, 'Fixture.app')
    fs.mkdirSync(appPath, { recursive: true })

    const { exec } = fakeExecSequence([
      { resolve: { stdout: '', stderr: '' } }, // codesign -vv (valid)
      { resolve: { stdout: '', stderr: 'Identifier=Electron\n' } } // codesign -dv -> stub
    ])

    await assert.rejects(verifyCodesign(appPath, { exec }), /Electron stub identifier/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('verifyCodesign throws when codesign -vv reports an invalid signature', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-'))
  try {
    const appPath = path.join(tempRoot, 'Fixture.app')
    fs.mkdirSync(appPath, { recursive: true })

    const { exec } = fakeExecSequence([
      { reject: new Error('codesign -vv failed (exit 1): a sealed resource is missing or invalid') }
    ])

    await assert.rejects(verifyCodesign(appPath, { exec }), /signature verification failed/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('verifyCodesign throws when the identifier does not match expectedIdentifier', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-'))
  try {
    const appPath = path.join(tempRoot, 'Fixture.app')
    fs.mkdirSync(appPath, { recursive: true })

    const { exec } = fakeExecSequence([
      { resolve: { stdout: '', stderr: '' } },
      { resolve: { stdout: '', stderr: 'Identifier=com.example.other\n' } }
    ])

    await assert.rejects(verifyCodesign(appPath, { exec, expectedIdentifier: 'com.nousresearch.hermes' }), /expected "com\.nousresearch\.hermes"/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

// ---------------------------------------------------------------------------
// Real-codesign fixture tests -- macOS only (codesign is a macOS system
// tool). Mirrors the real incident: build a tiny real .app bundle, sign it
// with the bare Electron stub identifier the way the broken build shipped,
// and prove the gate catches it; then prove our ad-hoc sign fixes it.
// ---------------------------------------------------------------------------

const isDarwin = process.platform === 'darwin'

function buildFixtureApp(root, { identifier } = {}) {
  const appPath = path.join(root, 'Fixture.app')
  fs.mkdirSync(path.join(appPath, 'Contents', 'MacOS'), { recursive: true })
  fs.mkdirSync(path.join(appPath, 'Contents', 'Resources'), { recursive: true })
  fs.writeFileSync(path.join(appPath, 'Contents', 'MacOS', 'Fixture'), '#!/bin/sh\necho fixture\n', {
    mode: 0o755
  })
  fs.writeFileSync(
    path.join(appPath, 'Contents', 'Info.plist'),
    `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key>
  <string>${identifier || 'com.hermes.fixture'}</string>
  <key>CFBundleExecutable</key>
  <string>Fixture</string>
  <key>CFBundleName</key>
  <string>Fixture</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
</dict>
</plist>
`,
    'utf8'
  )
  return appPath
}

test('real codesign: unsigned fixture bundle needs an ad-hoc sign', { skip: !isDarwin && 'codesign is macOS-only' }, async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-real-'))
  try {
    const appPath = buildFixtureApp(tempRoot)
    assert.equal(await needsAdHocSign(appPath), true)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test(
  'real codesign: verifyCodesign rejects the exact BUILD-266 regression (bare Electron stub identifier)',
  { skip: !isDarwin && 'codesign is macOS-only' },
  async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-real-'))
    try {
      const appPath = buildFixtureApp(tempRoot, { identifier: 'com.hermes.fixture' })
      // Reproduce the incident: signed, but with the bare "Electron" stub
      // identifier instead of the bundle's real one.
      await new Promise((resolve, reject) => {
        require('node:child_process').execFile(
          'codesign',
          ['--force', '--sign', '-', '--identifier', 'Electron', appPath],
          err => (err ? reject(err) : resolve())
        )
      })

      await assert.rejects(verifyCodesign(appPath), /Electron stub identifier/)
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  }
)

test(
  'real codesign: signAndVerify signs an unsigned fixture and produces a verifiable, non-stub identifier',
  { skip: !isDarwin && 'codesign is macOS-only' },
  async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-real-'))
    try {
      const appPath = buildFixtureApp(tempRoot, { identifier: 'com.hermes.fixture' })

      const identifier = await signAndVerify(appPath, { expectedIdentifier: 'com.hermes.fixture' })

      assert.equal(identifier, 'com.hermes.fixture')
      assert.equal(await needsAdHocSign(appPath), false)
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  }
)

test(
  'real codesign: verifyCodesign catches a stale signature after resources change post-sign',
  { skip: !isDarwin && 'codesign is macOS-only' },
  async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-real-'))
    try {
      const appPath = buildFixtureApp(tempRoot, { identifier: 'com.hermes.fixture' })
      await signAdHoc(appPath)
      await verifyCodesign(appPath) // sanity: passes right after signing

      // Reproduce "code has no resources but signature indicates they must
      // be present": mutate the bundle contents without re-signing.
      fs.writeFileSync(path.join(appPath, 'Contents', 'Resources', 'extra.txt'), 'unsealed change\n', 'utf8')

      await assert.rejects(verifyCodesign(appPath), /signature verification failed/)
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  }
)

test(
  'real codesign: signAndVerify does not reclobber an already-valid non-stub signature',
  { skip: !isDarwin && 'codesign is macOS-only' },
  async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-codesign-mac-real-'))
    try {
      const appPath = buildFixtureApp(tempRoot, { identifier: 'com.hermes.fixture' })
      await signAdHoc(appPath)
      const before = fs.statSync(path.join(appPath, 'Contents', '_CodeSignature', 'CodeResources')).mtimeMs

      // Small delay so a re-sign (if it happened) would produce a detectably
      // newer mtime than a skip would.
      await new Promise(resolve => setTimeout(resolve, 20))

      const identifier = await signAndVerify(appPath, { expectedIdentifier: 'com.hermes.fixture' })
      const after = fs.statSync(path.join(appPath, 'Contents', '_CodeSignature', 'CodeResources')).mtimeMs

      assert.equal(identifier, 'com.hermes.fixture')
      assert.equal(before, after, 'signAndVerify should not have re-signed an already-valid non-stub bundle')
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  }
)
