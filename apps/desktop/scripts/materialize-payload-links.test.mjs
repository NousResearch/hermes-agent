import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { findPackedPayload, relativizePayloadLinks } from './materialize-payload-links.mjs'

function tempRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-relativize-'))
}

test('findPackedPayload locates the mac nested payload', () => {
  const root = tempRoot()
  try {
    const app = path.join(root, 'Hermes.app')
    const payload = path.join(app, 'Contents', 'Resources', 'agent-payload')
    fs.mkdirSync(payload, { recursive: true })
    assert.equal(findPackedPayload(app, 'darwin'), payload)
    assert.equal(findPackedPayload(root, 'darwin'), payload)
    assert.equal(findPackedPayload(root, 'linux'), null)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('relativizePayloadLinks rewrites an absolute build-staging symlink to a payload-relative one', () => {
  if (process.platform === 'win32') return
  const root = tempRoot()
  try {
    const store = path.join(root, 'tools', 'python', 'bin')
    const venv = path.join(root, 'venv', 'bin')
    fs.mkdirSync(store, { recursive: true })
    fs.mkdirSync(venv, { recursive: true })
    const real = path.join(store, 'python3.11')
    fs.writeFileSync(real, 'interpreter-bytes')
    const link = path.join(venv, 'python3')
    // The absolute build-staging form uv --relocatable writes: points at
    // build/agent-payload/tools/... which does NOT exist here — but the
    // store entry it names (tools/python/bin/python3.11) DOES exist in the
    // payload root, which is what the rewrite keys on.
    fs.symlinkSync('/somewhere/build/agent-payload/tools/python/bin/python3.11', link)
    assert.equal(fs.lstatSync(link).isSymbolicLink(), true)

    const n = relativizePayloadLinks(root)
    assert.equal(n, 1)
    assert.equal(fs.lstatSync(link).isSymbolicLink(), true)
    const target = fs.readlinkSync(link)
    assert.equal(target.startsWith(path.sep), false) // now relative
    assert.equal(path.resolve(venv, target), real)
    assert.equal(fs.readFileSync(link, 'utf8'), 'interpreter-bytes') // resolves
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('relativizePayloadLinks throws when the named store entry is missing from the payload', () => {
  if (process.platform === 'win32') return
  const root = tempRoot()
  try {
    const venv = path.join(root, 'venv', 'bin')
    fs.mkdirSync(venv, { recursive: true })
    // Names tools/python/... but no such entry exists under the payload.
    fs.symlinkSync('/somewhere/build/agent-payload/tools/python/bin/python3.11', path.join(venv, 'python3'))
    assert.throws(() => relativizePayloadLinks(root), /store entry .* is not present in the payload/)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('relativizePayloadLinks leaves a sibling link (python3 -> python) alone', () => {
  if (process.platform === 'win32') return
  const root = tempRoot()
  try {
    const store = path.join(root, 'tools', 'python', 'bin')
    const venv = path.join(root, 'venv', 'bin')
    fs.mkdirSync(store, { recursive: true })
    fs.mkdirSync(venv, { recursive: true })
    const real = path.join(store, 'python3.11')
    fs.writeFileSync(real, 'interpreter-bytes')
    // python -> store link; python3 -> python sibling.
    fs.symlinkSync('/somewhere/build/agent-payload/tools/python/bin/python3.11', path.join(venv, 'python'))
    fs.symlinkSync('python', path.join(venv, 'python3'))

    const n = relativizePayloadLinks(root)
    assert.equal(n, 1) // only the store link rewritten
    assert.equal(fs.readlinkSync(path.join(venv, 'python3')), 'python') // sibling untouched
    assert.equal(fs.readFileSync(path.join(venv, 'python3'), 'utf8'), 'interpreter-bytes') // resolves via chain
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('relativizePayloadLinks throws when the target does not name a store entry', () => {
  if (process.platform === 'win32') return
  const root = tempRoot()
  try {
    const venv = path.join(root, 'venv', 'bin')
    fs.mkdirSync(venv, { recursive: true })
    fs.symlinkSync('/usr/bin/python3.11', path.join(venv, 'python3'))
    assert.throws(() => relativizePayloadLinks(root), /does not name a store entry/)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('relativizePayloadLinks leaves already-relative links alone', () => {
  if (process.platform === 'win32') return
  const root = tempRoot()
  try {
    const store = path.join(root, 'tools', 'python', 'bin')
    const venv = path.join(root, 'venv', 'bin')
    fs.mkdirSync(store, { recursive: true })
    fs.mkdirSync(venv, { recursive: true })
    const real = path.join(store, 'python3.11')
    fs.writeFileSync(real, 'interpreter-bytes')
    const link = path.join(venv, 'python3')
    fs.symlinkSync(path.relative(venv, real), link)

    assert.equal(relativizePayloadLinks(root), 0)
    assert.equal(fs.readlinkSync(link), path.relative(venv, real))
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('relativizePayloadLinks does not touch a framework file-symlink (not under venv/bin)', () => {
  if (process.platform === 'win32') return
  const root = tempRoot()
  try {
    const versioned = path.join(root, 'tools', 'chromium-1208', 'F.framework', 'Versions', 'A')
    fs.mkdirSync(versioned, { recursive: true })
    const real = path.join(versioned, 'F')
    fs.writeFileSync(real, 'machO')
    const link = path.join(root, 'tools', 'chromium-1208', 'F.framework', 'F')
    fs.symlinkSync(path.join('Versions', 'A', 'F'), link)
    assert.equal(fs.lstatSync(link).isSymbolicLink(), true)

    assert.equal(relativizePayloadLinks(root), 0)
    assert.equal(fs.lstatSync(link).isSymbolicLink(), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
