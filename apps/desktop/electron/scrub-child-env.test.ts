import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isCredentialEnvVar, scrubDesktopChildEnv } from './scrub-child-env'

test('isCredentialEnvVar matches suffix and known names', () => {
  assert.equal(isCredentialEnvVar('OPENROUTER_API_KEY'), true)
  assert.equal(isCredentialEnvVar('HERMES_DESKTOP_REMOTE_TOKEN'), true)
  assert.equal(isCredentialEnvVar('AWS_SECRET_ACCESS_KEY'), true)
  assert.equal(isCredentialEnvVar('PATH'), false)
  assert.equal(isCredentialEnvVar('HERMES_HOME'), false)
  assert.equal(isCredentialEnvVar('HERMES_DESKTOP'), false)
})

test('scrubDesktopChildEnv drops secrets and keeps operational keys', () => {
  const scrubbed = scrubDesktopChildEnv(
    {
      PATH: '/usr/bin',
      HERMES_HOME: '/home/u/.hermes',
      OPENROUTER_API_KEY: 'sk-live',
      TELEGRAM_BOT_TOKEN: '123:abc',
      HERMES_DESKTOP_REMOTE_TOKEN: 'remote-secret',
      EMPTY: ''
    },
    {
      HERMES_DESKTOP: '1',
      HERMES_DASHBOARD_SESSION_TOKEN: 'minted-session'
    }
  )

  assert.equal(scrubbed.PATH, '/usr/bin')
  assert.equal(scrubbed.HERMES_HOME, '/home/u/.hermes')
  assert.equal(scrubbed.HERMES_DESKTOP, '1')
  assert.equal(scrubbed.HERMES_DASHBOARD_SESSION_TOKEN, 'minted-session')
  assert.equal(scrubbed.OPENROUTER_API_KEY, undefined)
  assert.equal(scrubbed.TELEGRAM_BOT_TOKEN, undefined)
  assert.equal(scrubbed.HERMES_DESKTOP_REMOTE_TOKEN, undefined)
})
