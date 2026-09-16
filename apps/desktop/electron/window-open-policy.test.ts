import assert from 'node:assert/strict'

import { test } from 'vitest'

import { decideHubWindowOpen } from './window-open-policy'

test('decideHubWindowOpen routes http popups to the OS browser', () => {
  const decision = decideHubWindowOpen('https://github.com/NousResearch/hermes-agent')

  assert.deepEqual(decision, {
    action: 'deny',
    openExternal: 'https://github.com/NousResearch/hermes-agent'
  })
})

test('decideHubWindowOpen routes https popups to the OS browser', () => {
  const decision = decideHubWindowOpen('https://discord.gg/NousResearch')

  assert.deepEqual(decision, {
    action: 'deny',
    openExternal: 'https://discord.gg/NousResearch'
  })
})

test('decideHubWindowOpen routes mailto popups to the OS browser', () => {
  const decision = decideHubWindowOpen('mailto:hello@example.com')

  assert.deepEqual(decision, {
    action: 'deny',
    openExternal: 'mailto:hello@example.com'
  })
})

test('decideHubWindowOpen denies javascript URLs', () => {
  const decision = decideHubWindowOpen('javascript:alert(1)')

  assert.deepEqual(decision, { action: 'deny' })
})

test('decideHubWindowOpen denies file URLs', () => {
  const decision = decideHubWindowOpen('file:///etc/passwd')

  assert.deepEqual(decision, { action: 'deny' })
})

test('decideHubWindowOpen denies about and data URLs', () => {
  assert.deepEqual(decideHubWindowOpen('about:blank'), { action: 'deny' })
  assert.deepEqual(decideHubWindowOpen('data:text/html,<b>hi</b>'), { action: 'deny' })
})

test('decideHubWindowOpen denies unparseable URLs', () => {
  const decision = decideHubWindowOpen('not a url at all')

  assert.deepEqual(decision, { action: 'deny' })
})
