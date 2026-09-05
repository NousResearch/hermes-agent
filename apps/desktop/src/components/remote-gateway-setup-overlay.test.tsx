import assert from 'node:assert/strict'

import { test } from 'vitest'

import { remoteSetupErrorText, shouldShowRemoteGatewaySetup } from './remote-gateway-setup-overlay'

test('remote setup is shown only while the main process requires remote configuration', () => {
  assert.equal(shouldShowRemoteGatewaySetup({ remoteOnly: false, remoteSetupRequired: true }), false)
  assert.equal(shouldShowRemoteGatewaySetup({ remoteOnly: true, remoteSetupRequired: false }), false)
  assert.equal(shouldShowRemoteGatewaySetup({ remoteOnly: true, remoteSetupRequired: true }), true)
})

test('remote setup remains recoverable when the persisted route is malformed', () => {
  const capabilities = {
    remoteOnly: true,
    remoteSetupRequired: true,
    remoteSetupError: 'Remote gateway URL needs attention.'
  }

  assert.equal(shouldShowRemoteGatewaySetup(capabilities), true)
  assert.equal(
    remoteSetupErrorText(capabilities, 'Connection test failed'),
    'Connection test failed: Remote gateway URL needs attention.'
  )
})
