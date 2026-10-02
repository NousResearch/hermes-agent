import { afterEach, expect, test } from 'vitest'

import { $billingBlock, clearBillingBlock, setBillingBlock } from './billing-block'
import { $notifications, clearNotifications, notify } from './notifications'

const block = {
  billing_url: null,
  is_nous: false,
  message: 'Out of credits',
  model: 'fixture',
  provider: 'fixture-provider',
  provider_label: 'Fixture'
}

function raiseWall(sessionId: string) {
  setBillingBlock(sessionId, block)
  notify({ id: `billing-block:${block.provider}`, message: block.message, durationMs: 0 })
}

afterEach(() => {
  $billingBlock.set(null)
  clearNotifications()
})

test('clearing a recovered session removes its sticky credit toast, not unrelated notices', () => {
  raiseWall('session-a')
  notify({ id: 'unrelated', message: 'Keep me', durationMs: 0 })

  clearBillingBlock('session-a')

  expect($billingBlock.get()).toBeNull()
  expect($notifications.get().map(item => item.id)).toEqual(['unrelated'])
})

test('recovery in another session leaves the active wall and toast intact', () => {
  raiseWall('session-a')
  const before = $notifications.get()

  clearBillingBlock('session-b')

  expect($billingBlock.get()?.sessionId).toBe('session-a')
  expect($notifications.get()).toBe(before)
})

test('global dismissal removes the matching toast and repeated dismissal is harmless', () => {
  raiseWall('session-a')
  clearBillingBlock()
  clearBillingBlock()

  expect($notifications.get()).toEqual([])
})
