import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $billingBlock } from '@/store/billing-block'
import { $notifications, clearNotifications } from '@/store/notifications'

import { renderMessageStream } from './test-harness'

const billing = {
  billing_url: null,
  is_nous: false,
  message: 'Out of credits',
  model: 'fixture',
  provider: 'fixture-provider',
  provider_label: 'Fixture'
}

afterEach(() => {
  cleanup()
  $billingBlock.set(null)
  clearNotifications()
  vi.restoreAllMocks()
})

it('a new accepted turn clears both surfaces of its previous billing failure', async () => {
  const stream = renderMessageStream('session-a')
  const start = () =>
    act(() =>
      stream.handleEvent({
        type: 'message.start',
        session_id: 'session-a',
        payload: {}
      })
    )
  const fail = () =>
    act(() =>
      stream.handleEvent({
        type: 'message.complete',
        session_id: 'session-a',
        payload: { status: 'error', error: billing.message, billing }
      })
    )

  await start()
  await fail()
  expect($billingBlock.get()?.sessionId).toBe('session-a')
  expect($notifications.get().some(item => item.id === `billing-block:${billing.provider}`)).toBe(true)

  await start()
  expect($billingBlock.get()).toBeNull()
  expect($notifications.get().some(item => item.id === `billing-block:${billing.provider}`)).toBe(false)

  // If the retry still cannot pay, both surfaces are raised again.
  await fail()
  expect($billingBlock.get()?.sessionId).toBe('session-a')
  expect($notifications.get().some(item => item.id === `billing-block:${billing.provider}`)).toBe(true)
})
