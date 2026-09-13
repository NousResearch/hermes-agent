import assert from 'node:assert/strict'

import { beforeEach, test } from 'vitest'

import { gatewaySharedProfiles, recordGatewaySharedProfiles } from './gateway-shared-profiles'

beforeEach(() => recordGatewaySharedProfiles(null))

test('a gateway that has not reported its shared profiles resolves to null', () => {
  assert.equal(gatewaySharedProfiles(), null)
})

test('the profiles a status response reports are kept in order', () => {
  recordGatewaySharedProfiles(['default', 'alpha'])
  assert.deepEqual(gatewaySharedProfiles(), ['default', 'alpha'])
})

test('non-string entries are dropped and an absent field clears the record', () => {
  recordGatewaySharedProfiles(['default', 7, null])
  assert.deepEqual(gatewaySharedProfiles(), ['default'])

  recordGatewaySharedProfiles(undefined)
  assert.equal(gatewaySharedProfiles(), null)
})
