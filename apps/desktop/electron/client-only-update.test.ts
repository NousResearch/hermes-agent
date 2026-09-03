import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isClientOnlyUpdateSurface } from './client-only-update'

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
