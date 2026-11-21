// The phone-presence row says what this window does while a conversation is open in it, plus the
// evidence it has now. It is read on the Settings route, where no conversation is ever in view, so
// every case below renders it there.
import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'
import { startPresenceReporter } from '@/store/presence'

import { NotificationsSettings } from './notifications-settings'

vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))

const ROW = '#setting-field-notifications\\.phone-presence'

let stop: (() => void) | null = null

interface RowOptions {
  clientActivity?: boolean
  connectionState?: string
  systemIdleSeconds?: number
}

async function renderRow({ clientActivity = true, connectionState = 'open', systemIdleSeconds }: RowOptions) {
  if (systemIdleSeconds !== undefined) {
    const snapshot = { system: true, idleSeconds: systemIdleSeconds, locked: false, suspended: false, seq: 1, sampledAt: Date.now() }
    window.hermesDesktop = { presence: { get: async () => snapshot, onChange: () => () => undefined } } as never
  }

  window.location.hash = '#/settings/notifications'
  $gateway.set({ clientActivity, connectionState, request: vi.fn(async () => ({})) } as never)
  stop = startPresenceReporter()
  await act(() => vi.advanceTimersByTimeAsync(0))

  const view = render(
    <MemoryRouter>
      <NotificationsSettings subpage="alerts" />
    </MemoryRouter>
  )

  return () => view.container.querySelector(ROW)?.textContent ?? ''
}

beforeEach(() => void vi.useFakeTimers())

afterEach(() => {
  cleanup()
  stop?.()
  stop = null
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  window.location.hash = ''
  $gateway.set(null)
  vi.useRealTimers()
})

it('describes what it tells Hermes, without promising what a phone will do', async () => {
  const row = await renderRow({ systemIdleSeconds: 4 })

  expect(row()).toContain('This computer for phone notifications')
  expect(row()).toContain(
    'What this window tells Hermes about you being here. A phone notification plugin decides what to do with it.'
  )
})

it('says Not connected while the main connection is not open', async () => {
  const row = await renderRow({ connectionState: 'connecting', systemIdleSeconds: 4 })

  expect(row()).toContain('Not connected.')
  expect(row()).not.toContain('While a conversation is open')
})

it('says activity does not matter on a backend that did not advertise the feature', async () => {
  const row = await renderRow({ clientActivity: false, systemIdleSeconds: 4 })

  expect(row()).toContain('Phone notifications are not affected by activity on this backend.')
  expect(row()).not.toContain('While a conversation is open')
})

it('with the OS idle clock, says Hermes is told when you last used this computer', async () => {
  const row = await renderRow({ systemIdleSeconds: 4 })

  expect(row()).toContain(
    'While a conversation is open in this window, Hermes is told when you last used this computer. Last input 4 sec ago.'
  )
})

it('with page input only, says Hermes is told when you last used this window', async () => {
  const row = await renderRow({})

  // No input seen yet: the row never invents one.
  act(() => void vi.advanceTimersByTime(10_000))
  expect(row()).toContain(
    'While a conversation is open in this window, Hermes is told when you last used this window. No input seen yet.'
  )
  expect(row()).not.toContain('Last input')

  window.document.dispatchEvent(new Event('pointerdown'))
  act(() => void vi.advanceTimersByTime(120_000))
  expect(row()).toContain('Hermes is told when you last used this window. Last input 2 min ago.')
})
