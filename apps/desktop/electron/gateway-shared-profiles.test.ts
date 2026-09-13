import assert from 'node:assert/strict'

import { beforeEach, test } from 'vitest'

import {
  gatewaySharedProfiles,
  invalidateGatewaySharedProfiles,
  recordGatewaySharedProfiles
} from './gateway-shared-profiles'

beforeEach(() => invalidateGatewaySharedProfiles())

test('a gateway that has not reported its shared profiles resolves to null', () => {
  assert.equal(gatewaySharedProfiles(), null)
})

test('the profiles a status response reports are kept in order', () => {
  recordGatewaySharedProfiles(['default', 'alpha'])
  assert.deepEqual(gatewaySharedProfiles(), ['default', 'alpha'])
})

test('a later status response replaces the previous report', () => {
  recordGatewaySharedProfiles(['default', 'alpha'])
  recordGatewaySharedProfiles(['alpha'])
  assert.deepEqual(gatewaySharedProfiles(), ['alpha'])
})

test('non-string entries are dropped and an absent field clears the record', () => {
  recordGatewaySharedProfiles(['default', 7, null])
  assert.deepEqual(gatewaySharedProfiles(), ['default'])

  recordGatewaySharedProfiles(undefined)
  assert.equal(gatewaySharedProfiles(), null)
})

test('a primary restart whose status cannot be read drops the record', () => {
  recordGatewaySharedProfiles(['default'])
  invalidateGatewaySharedProfiles()
  assert.equal(gatewaySharedProfiles(), null)

  // The next readable poll starts a new record.
  recordGatewaySharedProfiles(['default', 'work'])
  assert.deepEqual(gatewaySharedProfiles(), ['default', 'work'])
})
