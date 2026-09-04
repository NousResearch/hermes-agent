import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { adoptPayloadVenv, installIdForRoot, isBundledInstall, resolvePayload } from './payload-backend'

function tmpdir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'payload-test-'))
}

function writePayload(
  root: string,
  {
    manifest = { schema: 1, target: 'win32-x64', repo: 'hermes-agent', venv: 'venv', store: 'tools' },
    isWindows = true
  }: any = {}
) {
  const dir = path.join(root, 'agent-payload')

  fs.mkdirSync(path.join(dir, 'hermes-agent'), { recursive: true })
  fs.mkdirSync(path.join(dir, 'tools', 'python-3.11.16-win32-x64'), { recursive: true })

  // Store python (win: python.exe at the entry root; posix: bin/python3).
  if (isWindows) {
    fs.writeFileSync(path.join(dir, 'tools', 'python-3.11.16-win32-x64', 'python.exe'), '')
  } else {
    fs.mkdirSync(path.join(dir, 'tools', 'python-3.11.16-win32-x64', 'bin'), { recursive: true })
    fs.writeFileSync(path.join(dir, 'tools', 'python-3.11.16-win32-x64', 'bin', 'python3'), '')
  }

  // Venv site-packages (where the project deps are installed).
  const sp = isWindows
    ? path.join(dir, 'venv', 'Lib', 'site-packages')
    : path.join(dir, 'venv', 'lib', 'python3.11', 'site-packages')

  fs.mkdirSync(sp, { recursive: true })

  // The self-relative CLI entrypoint (minted launcher exe on win32, bash
  // trampoline on POSIX) — the bundled entry point.
  const binDir = path.join(dir, 'bin')
  fs.mkdirSync(binDir, { recursive: true })
  fs.writeFileSync(path.join(binDir, isWindows ? 'hermes.exe' : 'hermes'), '')

  fs.writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify(manifest))
  fs.writeFileSync(
    path.join(dir, 'tools', 'facts.json'),
    JSON.stringify({ packages: { python: { entry: 'python-3.11.16-win32-x64', version: '3.11.16' } } })
  )

  return dir
}

const fsDeps = {
  fileExists: (p: string) => {
    try {
      return fs.statSync(p).isFile()
    } catch {
      return false
    }
  },
  directoryExists: (p: string) => {
    try {
      return fs.statSync(p).isDirectory()
    } catch {
      return false
    }
  }
}

test('resolvePayload finds a complete payload (store python + venv site-packages)', () => {
  const root = tmpdir()

  writePayload(root)

  const payload = resolvePayload(root, { ...fsDeps, isWindows: true })

  assert.ok(payload)
  assert.equal(payload.repoDir, path.join(root, 'agent-payload', 'hermes-agent'))
  assert.ok(payload.storePython.endsWith(path.join('tools', 'python-3.11.16-win32-x64', 'python.exe')))
  assert.ok(payload.sitePackages.endsWith(path.join('venv', 'Lib', 'site-packages')))
  assert.ok(payload.shim.endsWith(path.join('bin', 'hermes.exe')))
})

test('resolvePayload resolves the posix store python + site-packages layout', () => {
  const root = tmpdir()

  writePayload(root, { isWindows: false })

  const payload = resolvePayload(root, { ...fsDeps, isWindows: false })

  assert.ok(payload)
  assert.ok(payload.storePython.endsWith(path.join('tools', 'python-3.11.16-win32-x64', 'bin', 'python3')))
  assert.ok(payload.sitePackages.endsWith(path.join('venv', 'lib', 'python3.11', 'site-packages')))
  assert.ok(payload.shim.endsWith(path.join('bin', 'hermes')))
})

test('resolvePayload returns null without a manifest, for external stubs, and for a broken payload', () => {
  assert.equal(resolvePayload(tmpdir(), { ...fsDeps, isWindows: true }), null)
  assert.equal(resolvePayload(undefined, { ...fsDeps, isWindows: true }), null)

  const externalRoot = tmpdir()

  writePayload(externalRoot, { manifest: { schema: 1, external: true } })
  assert.equal(resolvePayload(externalRoot, { ...fsDeps, isWindows: true }), null)

  const brokenRoot = tmpdir()
  const dir = writePayload(brokenRoot)

  fs.rmSync(path.join(dir, 'venv'), { recursive: true })
  assert.equal(resolvePayload(brokenRoot, { ...fsDeps, isWindows: true }), null)
})

test('adoptPayloadVenv verifies store python + site-packages without any cfg write', () => {
  const root = tmpdir()
  const dir = writePayload(root)
  const payload = resolvePayload(root, { ...fsDeps, isWindows: true })

  assert.ok(payload)
  // Bundled builds run the store python directly — there is NO pyvenv.cfg
  // write (the payload may be read-only, e.g. MSIX). The cfg, if present,
  // is left untouched.
  fs.writeFileSync(path.join(dir, 'venv', 'pyvenv.cfg'), 'home = C:\\ci\\build\\python\nversion_info = 3.11.16\n')
  assert.equal(adoptPayloadVenv(payload, { isWindows: true }), true)
  const text = fs.readFileSync(path.join(dir, 'venv', 'pyvenv.cfg'), 'utf8')
  assert.ok(text.includes('C:\\ci\\build'), 'the shipped cfg is not rewritten')

  // second call: still reports usable
  assert.equal(adoptPayloadVenv(payload, { isWindows: true }), true)
})

// adoptPayloadVenv is now a pure presence-verify; resolvePayload's own
// existence checks already reject a payload missing the store python or
// site-packages (covered above), so there is no separate fail-closed path
// to assert at this layer.

test('isBundledInstall is true for a real payload manifest', () => {
  const root = tmpdir()

  writePayload(root)

  assert.equal(isBundledInstall(root, fsDeps), true)
})

test('isBundledInstall is false for the external stub and for missing manifests', () => {
  const externalRoot = tmpdir()

  writePayload(externalRoot, { manifest: { schema: 1, external: true } })
  assert.equal(isBundledInstall(externalRoot, fsDeps), false)

  assert.equal(isBundledInstall(tmpdir(), fsDeps), false)
  assert.equal(isBundledInstall(undefined, fsDeps), false)
})

test('isBundledInstall is true even when the payload is damaged (never-install guard)', () => {
  const root = tmpdir()
  const dir = writePayload(root)

  // A broken payload must still read as bundled: isBundledInstall is the
  // guard that REFUSES the installer, and a damaged bundle is exactly the
  // case where the installer must not run.
  fs.rmSync(path.join(dir, 'venv'), { recursive: true })
  fs.rmSync(path.join(dir, 'tools', 'python-3.11.16-win32-x64'), { recursive: true })

  assert.equal(resolvePayload(root, { ...fsDeps, isWindows: true }), null)
  assert.equal(isBundledInstall(root, fsDeps), true)
})

test('isBundledInstall is false for a malformed manifest', () => {
  const root = tmpdir()
  const dir = path.join(root, 'agent-payload')

  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'manifest.json'), 'not json {')

  assert.equal(isBundledInstall(root, fsDeps), false)
})

// ─── update channel helpers ─────────────────────────────────────────

test('installIdForRoot matches the Python install id (sha16 of the canonical path)', () => {
  // sha256('/home/u/.hermes/hermes-agent')[:16] — recomputed independently.
  assert.equal(
    installIdForRoot('/home/u/.hermes/hermes-agent'),
    createHash('sha256').update('/home/u/.hermes/hermes-agent', 'utf8').digest('hex').slice(0, 16)
  )
  // The canonicalizer output is what gets hashed (symlinked homes).
  assert.equal(
    installIdForRoot('/link/hermes-agent', () => '/real/hermes-agent'),
    installIdForRoot('/real/hermes-agent')
  )
})

