import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  canActiveBackendResolve,
  shouldIgnoreDiscoveredLocalRuntimes,
  shouldUseActiveBackend,
  shouldUseSystemPythonBackend
} from './backend-resolution'

test('shouldIgnoreDiscoveredLocalRuntimes recognizes exactly "1"', () => {
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({ HERMES_DESKTOP_IGNORE_EXISTING: '1' }), true)
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({ HERMES_DESKTOP_IGNORE_EXISTING: 'true' }), false)
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({ HERMES_DESKTOP_IGNORE_EXISTING: '0' }), false)
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({ HERMES_DESKTOP_IGNORE_EXISTING: '' }), false)
  assert.equal(shouldIgnoreDiscoveredLocalRuntimes({}), false)
})

test('canActiveBackendResolve short-circuits probing when ignored or repair requested', () => {
  assert.equal(canActiveBackendResolve({ bootstrapRepairRequested: false, ignoreExisting: true }), false)
  assert.equal(canActiveBackendResolve({ bootstrapRepairRequested: true, ignoreExisting: false }), false)
  assert.equal(
    canActiveBackendResolve({ bootstrapRepairRequested: false, env: { HERMES_DESKTOP_IGNORE_EXISTING: '1' } }),
    false
  )
  assert.equal(
    canActiveBackendResolve({ bootstrapRepairRequested: false, env: { HERMES_DESKTOP_IGNORE_EXISTING: '0' } }),
    true
  )
  assert.equal(canActiveBackendResolve({ bootstrapRepairRequested: false, env: {} }), true)
})

test('ignore-existing skips a usable active runtime', () => {
  assert.equal(
    shouldUseActiveBackend({
      activeRuntimeUsable: true,
      bootstrapRepairRequested: false,
      ignoreExisting: true
    }),
    false
  )
  assert.equal(
    shouldUseActiveBackend({
      activeRuntimeUsable: true,
      bootstrapRepairRequested: false,
      env: { HERMES_DESKTOP_IGNORE_EXISTING: '1' }
    }),
    false
  )
})

test('without ignore-existing, active runtime precedence is preserved', () => {
  assert.equal(
    shouldUseActiveBackend({
      activeRuntimeUsable: true,
      bootstrapRepairRequested: false,
      ignoreExisting: false
    }),
    true
  )
  assert.equal(
    shouldUseActiveBackend({
      activeRuntimeUsable: true,
      bootstrapRepairRequested: false,
      env: { HERMES_DESKTOP_IGNORE_EXISTING: '0' }
    }),
    true
  )
  assert.equal(
    shouldUseActiveBackend({
      activeRuntimeUsable: true,
      bootstrapRepairRequested: false,
      env: {}
    }),
    true
  )
})

test('bootstrap repair still bypasses the active runtime', () => {
  assert.equal(
    shouldUseActiveBackend({
      activeRuntimeUsable: true,
      bootstrapRepairRequested: true,
      ignoreExisting: false
    }),
    false
  )
  assert.equal(
    shouldUseActiveBackend({
      activeRuntimeUsable: true,
      bootstrapRepairRequested: true,
      env: {}
    }),
    false
  )
})

test('unusable active runtime is never selected', () => {
  assert.equal(
    shouldUseActiveBackend({
      activeRuntimeUsable: false,
      bootstrapRepairRequested: false,
      ignoreExisting: false
    }),
    false
  )
})

test('system-Python fallback is skipped only when ignore-existing is enabled', () => {
  assert.equal(shouldUseSystemPythonBackend(true), false)
  assert.equal(shouldUseSystemPythonBackend(false), true)
  assert.equal(shouldUseSystemPythonBackend({ HERMES_DESKTOP_IGNORE_EXISTING: '1' }), false)
  assert.equal(shouldUseSystemPythonBackend({ HERMES_DESKTOP_IGNORE_EXISTING: '0' }), true)
  assert.equal(shouldUseSystemPythonBackend({ HERMES_DESKTOP_IGNORE_EXISTING: 'true' }), true)
  assert.equal(shouldUseSystemPythonBackend({}), true)
  assert.equal(shouldUseSystemPythonBackend({ ignoreExisting: true }), false)
  assert.equal(shouldUseSystemPythonBackend({ ignoreExisting: false }), true)
})
