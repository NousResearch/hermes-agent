/**
 * "Update all instances" never runs `hermes update` against a Hermes Cloud
 * agent, however it was saved: as a Cloud connection or as a plain URL remote.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { updateEligibility } from './connection-registry'

test('a Cloud agent is platform-managed whether saved as cloud or as a URL remote', () => {
  assert.deepEqual(updateEligibility({ id: 'c', kind: 'cloud', label: 'Cloud' }), {
    eligible: false,
    reason: 'cloud-managed'
  })
  assert.deepEqual(
    updateEligibility({ id: 'u', kind: 'remote', label: 'x', url: 'https://lab.agents.staging-nousresearch.com' }),
    { eligible: false, reason: 'cloud-managed' }
  )
  assert.equal(updateEligibility({ id: 'r', kind: 'remote', label: 'x', url: 'https://box.example.com' }).eligible, true)
})
