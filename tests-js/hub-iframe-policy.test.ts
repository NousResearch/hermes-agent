import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  HERMES_HUB_ORIGIN,
  isHermesHubClipboardWrite,
  isHermesHubExternalUrl,
  isHermesHubOrigin
} from '../apps/desktop/electron/hub-iframe-policy'

describe('Hub iframe trust policy', () => {
  test('matches only the exact Hub origin', () => {
    assert.equal(isHermesHubOrigin(HERMES_HUB_ORIGIN), true)
    assert.equal(isHermesHubOrigin(`${HERMES_HUB_ORIGIN}/docs/skills?embed=picker`), true)
    assert.equal(isHermesHubOrigin('https://attacker.test'), false)
    assert.equal(isHermesHubOrigin('null'), false)
  })

  test('allows only browser-safe external schemes from the Hub', () => {
    assert.equal(isHermesHubExternalUrl('https://github.com'), true)
    assert.equal(isHermesHubExternalUrl('mailto:hello@example.com'), true)
    assert.equal(isHermesHubExternalUrl('file:///etc/passwd'), false)
    assert.equal(isHermesHubExternalUrl('javascript:alert(1)'), false)
  })

  test('grants only sanitized clipboard write to the Hub origin', () => {
    assert.equal(isHermesHubClipboardWrite('clipboard-sanitized-write', HERMES_HUB_ORIGIN), true)
    assert.equal(isHermesHubClipboardWrite('clipboard-read', HERMES_HUB_ORIGIN), false)
    assert.equal(isHermesHubClipboardWrite('clipboard-sanitized-write', 'https://attacker.test'), false)
  })
})
