import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { recoverLegacyMacosTccAnchor } from './macos-tcc-anchor-recovery'

const temporaryRoots: string[] = []

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function makeAnchoredVenv() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-tcc-recovery-'))
  temporaryRoots.push(root)

  const venvRoot = path.join(root, 'venv')
  const bin = path.join(venvRoot, 'bin')
  const source = path.join(root, 'uv', 'bin', 'python3.11')
  const python = path.join(bin, 'python')
  const marker = path.join(bin, '.tcc-anchor-source')

  fs.mkdirSync(path.dirname(source), { recursive: true })
  fs.mkdirSync(bin, { recursive: true })
  fs.writeFileSync(source, Buffer.from('cffaedfe00000000', 'hex'), { mode: 0o755 })
  fs.writeFileSync(python, 'broken anchored interpreter', { mode: 0o755 })
  fs.writeFileSync(marker, source)
  fs.symlinkSync('python', path.join(bin, 'python3'))
  fs.writeFileSync(path.join(bin, 'python3.11'), 'another anchored copy', { mode: 0o755 })

  return { root, venvRoot, bin, source, python, marker }
}

test('is a no-op outside macOS', () => {
  const fixture = makeAnchoredVenv()
  let probes = 0

  const result = recoverLegacyMacosTccAnchor({
    platform: 'linux',
    venvRoot: fixture.venvRoot,
    probePython: () => {
      probes += 1

      return true
    }
  })

  assert.deepEqual(result, { status: 'not-applicable', reason: 'platform' })
  assert.equal(probes, 0)
  assert.equal(fs.lstatSync(fixture.python).isSymbolicLink(), false)
  assert.equal(fs.existsSync(fixture.marker), true)
})

test('is a no-op when the legacy marker is absent', () => {
  const fixture = makeAnchoredVenv()
  fs.unlinkSync(fixture.marker)

  const result = recoverLegacyMacosTccAnchor({
    platform: 'darwin',
    venvRoot: fixture.venvRoot,
    probePython: () => true
  })

  assert.deepEqual(result, { status: 'not-applicable', reason: 'marker-absent' })
  assert.equal(fs.lstatSync(fixture.python).isSymbolicLink(), false)
})

test('does not trust a relative, multiline, or in-venv source path', () => {
  for (const markerValue of ['../python', '/safe/python\n/other/python', 'relative\0python']) {
    const fixture = makeAnchoredVenv()
    fs.writeFileSync(fixture.marker, markerValue)

    const result = recoverLegacyMacosTccAnchor({
      platform: 'darwin',
      venvRoot: fixture.venvRoot,
      probePython: () => true
    })

    assert.deepEqual(result, { status: 'failed', reason: 'invalid-marker' })
    assert.equal(fs.lstatSync(fixture.python).isSymbolicLink(), false)
    assert.equal(fs.existsSync(fixture.marker), true)
  }

  const fixture = makeAnchoredVenv()
  const inVenvSource = path.join(fixture.bin, 'recorded-python')
  fs.writeFileSync(inVenvSource, Buffer.from('cffaedfe00000000', 'hex'), { mode: 0o755 })
  fs.writeFileSync(fixture.marker, inVenvSource)

  const result = recoverLegacyMacosTccAnchor({
    platform: 'darwin',
    venvRoot: fixture.venvRoot,
    probePython: () => true
  })

  assert.deepEqual(result, { status: 'failed', reason: 'unsafe-source' })
  assert.equal(fs.lstatSync(fixture.python).isSymbolicLink(), false)
  assert.equal(fs.existsSync(fixture.marker), true)
})

test('leaves the venv untouched when the recorded interpreter fails its import probe', () => {
  const fixture = makeAnchoredVenv()
  const canonicalSource = fs.realpathSync(fixture.source)

  const result = recoverLegacyMacosTccAnchor({
    platform: 'darwin',
    venvRoot: fixture.venvRoot,
    probePython: candidate => candidate !== canonicalSource
  })

  assert.deepEqual(result, { status: 'failed', reason: 'source-probe-failed' })
  assert.equal(fs.readFileSync(fixture.python, 'utf8'), 'broken anchored interpreter')
  assert.equal(fs.existsSync(fixture.marker), true)
})

test('does not execute an executable marker target that is not a Mach-O binary', () => {
  const fixture = makeAnchoredVenv()
  fs.writeFileSync(fixture.source, '#!/bin/sh\nexit 0\n', { mode: 0o755 })
  let probes = 0

  const result = recoverLegacyMacosTccAnchor({
    platform: 'darwin',
    venvRoot: fixture.venvRoot,
    probePython: () => {
      probes += 1

      return true
    }
  })

  assert.deepEqual(result, { status: 'failed', reason: 'unsafe-source' })
  assert.equal(probes, 0)
  assert.equal(fs.lstatSync(fixture.python).isSymbolicLink(), false)
  assert.equal(fs.existsSync(fixture.marker), true)
})

test('atomically restores python and its aliases after both probes pass', () => {
  const fixture = makeAnchoredVenv()
  const canonicalSource = fs.realpathSync(fixture.source)
  const probed: string[] = []

  const result = recoverLegacyMacosTccAnchor({
    platform: 'darwin',
    venvRoot: fixture.venvRoot,
    probePython: candidate => {
      probed.push(candidate)

      return true
    }
  })

  assert.deepEqual(result, { status: 'recovered' })
  assert.deepEqual(probed, [canonicalSource, fixture.python])
  assert.equal(fs.readlinkSync(fixture.python), canonicalSource)
  assert.equal(fs.readlinkSync(path.join(fixture.bin, 'python3')), 'python')
  assert.equal(fs.readlinkSync(path.join(fixture.bin, 'python3.11')), 'python')
  assert.equal(fs.existsSync(fixture.marker), false)
  assert.deepEqual(
    fs.readdirSync(fixture.bin).filter(name => name.includes('tcc-recovery')),
    []
  )
})

test('rolls back every interpreter and keeps the marker when the repaired venv fails its probe', () => {
  const fixture = makeAnchoredVenv()
  const canonicalSource = fs.realpathSync(fixture.source)
  const originalPython3Target = fs.readlinkSync(path.join(fixture.bin, 'python3'))

  const result = recoverLegacyMacosTccAnchor({
    platform: 'darwin',
    venvRoot: fixture.venvRoot,
    probePython: candidate => candidate === canonicalSource
  })

  assert.deepEqual(result, { status: 'failed', reason: 'venv-probe-failed' })
  assert.equal(fs.readFileSync(fixture.python, 'utf8'), 'broken anchored interpreter')
  assert.equal(fs.readlinkSync(path.join(fixture.bin, 'python3')), originalPython3Target)
  assert.equal(fs.readFileSync(path.join(fixture.bin, 'python3.11'), 'utf8'), 'another anchored copy')
  assert.equal(fs.existsSync(fixture.marker), true)
  assert.deepEqual(
    fs.readdirSync(fixture.bin).filter(name => name.includes('tcc-recovery')),
    []
  )
})

test('finishes alias cleanup when python was already restored before a prior attempt stopped', () => {
  const fixture = makeAnchoredVenv()
  fs.unlinkSync(fixture.python)
  fs.symlinkSync(fixture.source, fixture.python)

  const result = recoverLegacyMacosTccAnchor({
    platform: 'darwin',
    venvRoot: fixture.venvRoot,
    probePython: () => true
  })

  assert.deepEqual(result, { status: 'recovered' })
  assert.equal(fs.readlinkSync(fixture.python), fixture.source)
  assert.equal(fs.readlinkSync(path.join(fixture.bin, 'python3')), 'python')
  assert.equal(fs.readlinkSync(path.join(fixture.bin, 'python3.11')), 'python')
  assert.equal(fs.existsSync(fixture.marker), false)
})

test('does not replace a symlinked python that points somewhere other than the recorded source', () => {
  const fixture = makeAnchoredVenv()
  const otherPython = path.join(fixture.root, 'other-python')
  fs.writeFileSync(otherPython, Buffer.from('cffaedfe00000000', 'hex'), { mode: 0o755 })
  fs.unlinkSync(fixture.python)
  fs.symlinkSync(otherPython, fixture.python)

  const result = recoverLegacyMacosTccAnchor({
    platform: 'darwin',
    venvRoot: fixture.venvRoot,
    probePython: () => true
  })

  assert.deepEqual(result, { status: 'not-applicable', reason: 'python-not-anchored' })
  assert.equal(fs.readlinkSync(fixture.python), otherPython)
  assert.equal(fs.existsSync(fixture.marker), true)
})
