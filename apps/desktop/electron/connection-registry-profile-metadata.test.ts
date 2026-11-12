/**
 * Tests for connection-registry.ts::profileMetadataFromApiProfiles — the projection of a remote
 * connection's `/api/profiles` into the cross-connection roster metadata (#139548).
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { profileMetadataFromApiProfiles } from './connection-registry'

test('remote API profile descriptions are retained for the roster, including an empty description', () => {
  assert.deepEqual(
    profileMetadataFromApiProfiles([
      { name: 'builder', description: '  Personal project developer  ' },
      { name: 'empty', description: '' },
      { name: 'legacy', description: null }
    ]),
    {
      builder: { description: 'Personal project developer' },
      empty: { description: '' },
      legacy: {}
    }
  )
})
