import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { inspectClientOnlyUpdateSurface, isClientOnlyUpdateSurface } from './client-only-update'

test('remote mode with no venv is a client-only surface', () => {
  assert.equal(
    isClientOnlyUpdateSurface({ remoteMode: true, hasVenvHermes: false, hasVenvPython: false }),
    true
  )
})

test('a runnable local venv stays on the full-install update path', () => {
  assert.equal(
    isClientOnlyUpdateSurface({ remoteMode: true, hasVenvHermes: true, hasVenvPython: true }),
    false
  )
  assert.equal(
    isClientOnlyUpdateSurface({ remoteMode: false, hasVenvHermes: true, hasVenvPython: true }),
    false
  )
})

test('local mode with no venv is a broken install, not a thin client', () => {
  assert.equal(
    isClientOnlyUpdateSurface({ remoteMode: false, hasVenvHermes: false, hasVenvPython: false }),
    false
  )
})

test('a partial venv is a broken install even in remote mode', () => {
  assert.equal(
    isClientOnlyUpdateSurface({ remoteMode: true, hasVenvHermes: false, hasVenvPython: true }),
    false
  )
  assert.equal(
    isClientOnlyUpdateSurface({ remoteMode: true, hasVenvHermes: true, hasVenvPython: false }),
    false
  )
})

test('a .venv runtime and a partial runtime cannot be mistaken for a thin client', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-client-update-'))
  try {
    assert.equal(isClientOnlyUpdateSurface(inspectClientOnlyUpdateSurface(root, true)), true)
    const bin = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts' : 'bin')
    fs.mkdirSync(bin, { recursive: true })
    fs.writeFileSync(path.join(bin, process.platform === 'win32' ? 'python.exe' : 'python3'), 'fixture', { mode: 0o755 })
    assert.equal(isClientOnlyUpdateSurface(inspectClientOnlyUpdateSurface(root, true)), false)
    fs.writeFileSync(path.join(bin, process.platform === 'win32' ? 'hermes.exe' : 'hermes'), 'fixture', { mode: 0o755 })
    assert.equal(isClientOnlyUpdateSurface(inspectClientOnlyUpdateSurface(root, true)), false)
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})
