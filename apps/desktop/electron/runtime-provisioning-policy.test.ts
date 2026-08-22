import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  assertRuntimeProvisioningAllowed,
  runtimeNeedsProvisioning,
  RuntimeProvisioningDeniedError,
  SECONDARY_RUNTIME_NOT_INSTALLED
} from './runtime-provisioning-policy'

test('only the unresolved bootstrap sentinel requires provisioning', () => {
  assert.equal(runtimeNeedsProvisioning({ kind: 'bootstrap-needed' }), true)
  assert.equal(runtimeNeedsProvisioning({ kind: 'python' }), false)
  assert.equal(runtimeNeedsProvisioning({}), false)
  assert.equal(runtimeNeedsProvisioning(null), false)
  assert.equal(runtimeNeedsProvisioning(undefined), false)
})

test('primary startup may provision a missing local runtime', () => {
  assert.doesNotThrow(() => assertRuntimeProvisioningAllowed({ kind: 'bootstrap-needed' }, 'primary-bootstrap'))
})

test('secondary backends refuse implicit provisioning with a typed actionable error', () => {
  assert.throws(
    () => assertRuntimeProvisioningAllowed({ kind: 'bootstrap-needed' }, 'secondary-activate'),
    error => {
      assert.ok(error instanceof RuntimeProvisioningDeniedError)
      assert.equal(error.code, 'runtime-provisioning-not-allowed')
      assert.equal(error.intent, 'secondary-activate')
      assert.equal(error.message, SECONDARY_RUNTIME_NOT_INSTALLED)

      return true
    }
  )
})

test('secondary backends may activate an already-installed managed runtime', () => {
  // createActiveBackend() deliberately carries bootstrap=true while wiring the
  // already-installed venv. Policy is keyed to the bootstrap-needed sentinel,
  // not the broader bootstrap flag, so this remains usable.
  assert.doesNotThrow(() => assertRuntimeProvisioningAllowed({ kind: 'python', bootstrap: true }, 'secondary-activate'))
})

test('external runtimes are valid for both intents', () => {
  for (const intent of ['primary-bootstrap', 'secondary-activate'] as const) {
    assert.doesNotThrow(() => assertRuntimeProvisioningAllowed({ kind: 'command' }, intent))
  }
})
