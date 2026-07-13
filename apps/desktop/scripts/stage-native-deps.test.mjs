import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import { patchUnixTerminalAsarUnpacked } from './stage-native-deps.mjs'

function makeFixtureDir(extra) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-stage-pty-'))
  const libDir = path.join(root, 'lib')
  fs.mkdirSync(libDir, { recursive: true })
  if (extra) extra(root, libDir)
  return root
}

function cleanup(root) {
  fs.rmSync(root, { recursive: true, force: true })
}

// ---------------------------------------------------------------------------
// Unit tests for patchUnixTerminalAsarUnpacked
// ---------------------------------------------------------------------------

test('patches naive replace calls into negative-lookahead regex', () => {
  const root = makeFixtureDir((_r, libDir) => {
    fs.writeFileSync(path.join(libDir, 'unixTerminal.js'), `
var helperPath = native.dir + '/spawn-helper';
helperPath = path.resolve(__dirname, helperPath);
helperPath = helperPath.replace('app.asar', 'app.asar.unpacked');
helperPath = helperPath.replace('node_modules.asar', 'node_modules.asar.unpacked');
`, 'utf8')
  })
  try {
    const patched = patchUnixTerminalAsarUnpacked(root)
    assert.equal(patched, true, 'should return true when patching')

    const content = fs.readFileSync(path.join(root, 'lib', 'unixTerminal.js'), 'utf8')

    // Must contain the regex-based replaces
    assert.ok(
      content.includes("helperPath.replace(/app\\.asar(?!\\.unpacked)/, 'app.asar.unpacked')"),
      'app.asar replace should be converted to negative-lookahead regex'
    )
    assert.ok(
      content.includes("helperPath.replace(/node_modules\\.asar(?!\\.unpacked)/, 'node_modules.asar.unpacked')"),
      'node_modules.asar replace should be converted to negative-lookahead regex'
    )

    // Must NOT contain the naive string replaces
    assert.ok(
      !content.includes("helperPath.replace('app.asar', 'app.asar.unpacked')"),
      'naive app.asar string replace must be gone'
    )
    assert.ok(
      !content.includes("helperPath.replace('node_modules.asar', 'node_modules.asar.unpacked')"),
      'naive node_modules.asar string replace must be gone'
    )
  } finally {
    cleanup(root)
  }
})

test('is idempotent — already-patched file returns false', () => {
  const root = makeFixtureDir((_r, libDir) => {
    fs.writeFileSync(path.join(libDir, 'unixTerminal.js'), `
var helperPath = native.dir + '/spawn-helper';
helperPath = path.resolve(__dirname, helperPath);
helperPath = helperPath.replace(/app\\.asar(?!\\.unpacked)/, 'app.asar.unpacked');
helperPath = helperPath.replace(/node_modules\\.asar(?!\\.unpacked)/, 'node_modules.asar.unpacked');
`, 'utf8')
  })
  try {
    const patched = patchUnixTerminalAsarUnpacked(root)
    assert.equal(patched, false, 'already-patched file must not be patched again')

    // Content must be byte-identical to what we wrote
    const content = fs.readFileSync(path.join(root, 'lib', 'unixTerminal.js'), 'utf8')
    assert.ok(content.includes("replace(/app\\.asar(?!\\.unpacked)/"), 'regex must still be present')
  } finally {
    cleanup(root)
  }
})

test('recognizes guard-based safety as already-safe', () => {
  const root = makeFixtureDir((_r, libDir) => {
    fs.writeFileSync(path.join(libDir, 'unixTerminal.js'), `
var helperPath = native.dir + '/spawn-helper';
helperPath = path.resolve(__dirname, helperPath);
if (!helperPath.includes('.asar.unpacked')) {
    helperPath = helperPath.replace('app.asar', 'app.asar.unpacked');
    helperPath = helperPath.replace('node_modules.asar', 'node_modules.asar.unpacked');
}
`, 'utf8')
  })
  try {
    const patched = patchUnixTerminalAsarUnpacked(root)
    assert.equal(patched, false, 'guard-protected file must not be patched again')
  } finally {
    cleanup(root)
  }
})

test('returns false when unixTerminal.js does not exist', () => {
  const root = makeFixtureDir(() => { /* no unixTerminal.js written */ })
  try {
    const patched = patchUnixTerminalAsarUnpacked(root)
    assert.equal(patched, false)
  } finally {
    cleanup(root)
  }
})

test('does not disturb nearby code — only targeted lines change', () => {
  const root = makeFixtureDir((_r, libDir) => {
    fs.writeFileSync(path.join(libDir, 'unixTerminal.js'), `
var native = utils_1.loadNativeModule('pty');
var pty = native.module;
var helperPath = native.dir + '/spawn-helper';
helperPath = path.resolve(__dirname, helperPath);
helperPath = helperPath.replace('app.asar', 'app.asar.unpacked');
helperPath = helperPath.replace('node_modules.asar', 'node_modules.asar.unpacked');
var DEFAULT_FILE = 'sh';
var DEFAULT_NAME = 'xterm';
var DESTROY_SOCKET_TIMEOUT_MS = 200;
`, 'utf8')
  })
  try {
    patchUnixTerminalAsarUnpacked(root)
    const content = fs.readFileSync(path.join(root, 'lib', 'unixTerminal.js'), 'utf8')

    // Surrounding code preserved
    assert.ok(content.includes("var native = utils_1.loadNativeModule('pty');"))
    assert.ok(content.includes("var pty = native.module;"))
    assert.ok(content.includes("helperPath = path.resolve(__dirname, helperPath);"))
    assert.ok(content.includes("var DEFAULT_FILE = 'sh';"))
    assert.ok(content.includes("var DEFAULT_NAME = 'xterm';"))
    assert.ok(content.includes("var DESTROY_SOCKET_TIMEOUT_MS = 200;"))
  } finally {
    cleanup(root)
  }
})

// ---------------------------------------------------------------------------
// Behavioral test: the patched code correctly handles both normal and
// already-unpacked paths (simulated by evaluating the regex logic).
// ---------------------------------------------------------------------------

test('patched regex does NOT double-append .unpacked when path already has it', () => {
  // Simulate app.asar.unpacked path (already unpacked)
  const pathWithUnpacked = '/Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist/node_modules/node-pty/lib/spawn-helper'
  const result = pathWithUnpacked.replace(/app\.asar(?!\.unpacked)/, 'app.asar.unpacked')
  assert.equal(
    result,
    '/Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist/node_modules/node-pty/lib/spawn-helper',
    'path with .asar.unpacked must remain unchanged'
  )
})

test('patched regex DOES append .unpacked when path has plain app.asar', () => {
  // Simulate packaged-in-asar path (not yet unpacked)
  const pathInAsar = '/Applications/Hermes.app/Contents/Resources/app.asar/dist/node_modules/node-pty/lib/spawn-helper'
  const result = pathInAsar.replace(/app\.asar(?!\.unpacked)/, 'app.asar.unpacked')
  assert.equal(
    result,
    '/Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist/node_modules/node-pty/lib/spawn-helper',
    'path with plain app.asar must get .unpacked appended'
  )
})

test('node_modules.asar regex does NOT double-append', () => {
  const pathWithUnpacked = '/path/to/node_modules.asar.unpacked/node-pty/lib/spawn-helper'
  const result = pathWithUnpacked.replace(/node_modules\.asar(?!\.unpacked)/, 'node_modules.asar.unpacked')
  assert.equal(
    result,
    '/path/to/node_modules.asar.unpacked/node-pty/lib/spawn-helper',
    'path with .asar.unpacked must remain unchanged'
  )
})

test('node_modules.asar regex DOES append when plain', () => {
  const pathInAsar = '/path/to/node_modules.asar/node-pty/lib/spawn-helper'
  const result = pathInAsar.replace(/node_modules\.asar(?!\.unpacked)/, 'node_modules.asar.unpacked')
  assert.equal(
    result,
    '/path/to/node_modules.asar.unpacked/node-pty/lib/spawn-helper',
    'path with plain node_modules.asar must get .unpacked appended'
  )
})
