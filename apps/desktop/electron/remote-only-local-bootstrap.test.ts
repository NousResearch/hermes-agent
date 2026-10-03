import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  REMOTE_ONLY_LOCAL_BOOTSTRAP_MESSAGE,
  RemoteOnlyLocalBootstrapError,
  remoteOnlyLocalBootstrapReason
} from './remote-only-local-bootstrap'

test('a remote-gateway primary skips the local install with actionable copy', () => {
  assert.equal(remoteOnlyLocalBootstrapReason(true), REMOTE_ONLY_LOCAL_BOOTSTRAP_MESSAGE)
  // The message has to tell the user how to get the local install back, not
  // just refuse (#112514 is a "no clear way" report).
  assert.match(REMOTE_ONLY_LOCAL_BOOTSTRAP_MESSAGE, /Settings → Gateway/)
})

test('the default local primary installs exactly as before', () => {
  assert.equal(remoteOnlyLocalBootstrapReason(false), null)
})

test('the skip is typed so callers can tell it from a failed install', () => {
  const error = new RemoteOnlyLocalBootstrapError()

  assert.equal(error.localBootstrapSkipped, true)
  assert.equal(error.name, 'RemoteOnlyLocalBootstrapError')
  assert.equal(error.message, REMOTE_ONLY_LOCAL_BOOTSTRAP_MESSAGE)
  assert.ok(error instanceof Error)
})
