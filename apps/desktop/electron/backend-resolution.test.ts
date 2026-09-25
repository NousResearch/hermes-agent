import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  type BackendResolutionInput,
  isIgnoreExisting,
  postBootstrapResolveOptions,
  selectBackendRung
} from './backend-resolution'

function discovered(overrides: Partial<BackendResolutionInput> = {}): BackendResolutionInput {
  return {
    ignoreExisting: false,
    justInstalled: false,
    bootstrapRepairRequested: false,
    hasExplicitRoot: false,
    hasUnpackagedSource: false,
    activeRuntimeUsable: false,
    hasExplicitCommand: false,
    hasPathHermes: false,
    hasSystemPython: false,
    ...overrides
  }
}

test('ignore-existing is exactly the string 1', () => {
  assert.equal(isIgnoreExisting('1'), true)
  assert.equal(isIgnoreExisting('true'), false)
  assert.equal(isIgnoreExisting('0'), false)
  assert.equal(isIgnoreExisting(''), false)
  assert.equal(isIgnoreExisting(undefined), false)
})

test('ignore-existing skips the active runtime and system python, not only hermes on PATH', () => {
  // The flag used to wrap only the PATH probe. A usable managed install at
  // ACTIVE_HERMES_ROOT, or a system Python that can import hermes_cli, still
  // started a local serve.
  assert.equal(
    selectBackendRung(
      discovered({
        ignoreExisting: true,
        activeRuntimeUsable: true,
        hasPathHermes: true,
        hasSystemPython: true
      })
    ),
    'onboarding'
  )
})

test('explicit root and an unpackaged source checkout stay above the ignore-existing gate', () => {
  const ignored = {
    ignoreExisting: true,
    activeRuntimeUsable: true,
    hasPathHermes: true,
    hasSystemPython: true
  }

  assert.equal(selectBackendRung(discovered({ ...ignored, hasExplicitRoot: true })), 'explicit-root')
  assert.equal(selectBackendRung(discovered({ ...ignored, hasUnpackagedSource: true })), 'unpackaged-source')
})

test('an explicit hermes command is not a discovered runtime', () => {
  assert.equal(
    selectBackendRung(
      discovered({
        ignoreExisting: true,
        activeRuntimeUsable: true,
        hasExplicitCommand: true,
        hasPathHermes: true,
        hasSystemPython: true
      })
    ),
    'explicit-command'
  )
})

test('the post-bootstrap re-resolve uses the runtime just installed instead of repeating onboarding', () => {
  const ignored = discovered({
    ignoreExisting: true,
    activeRuntimeUsable: true,
    hasPathHermes: true,
    hasSystemPython: true
  })

  assert.equal(selectBackendRung(ignored), 'onboarding')
  assert.equal(
    selectBackendRung({
      ...ignored,
      justInstalled: postBootstrapResolveOptions().justInstalled
    }),
    'active'
  )
})

test('a just-installed runtime that is not usable does not fall back to a discovered runtime or repeat onboarding', () => {
  assert.equal(
    selectBackendRung(
      discovered({
        ignoreExisting: true,
        justInstalled: true,
        activeRuntimeUsable: false,
        hasPathHermes: true,
        hasSystemPython: true
      })
    ),
    'installed-unusable'
  )
})

test('without the flag a usable active runtime still wins, and repair still bypasses it', () => {
  assert.equal(
    selectBackendRung(
      discovered({
        activeRuntimeUsable: true,
        hasPathHermes: true,
        hasSystemPython: true
      })
    ),
    'active'
  )
  assert.equal(
    selectBackendRung(
      discovered({
        bootstrapRepairRequested: true,
        activeRuntimeUsable: true,
        hasPathHermes: true
      })
    ),
    'path'
  )
  assert.equal(
    selectBackendRung(
      discovered({
        hasSystemPython: true
      })
    ),
    'system-python'
  )
})
