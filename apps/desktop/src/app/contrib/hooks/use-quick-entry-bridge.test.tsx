import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import type { QuickEntrySubmitPayload } from '@/store/quick-entry'

import { useQuickEntryBridge } from './use-quick-entry-bridge'

const route = vi.hoisted(() => ({ connectionId: 'local', profile: 'work' }))
const delegate = vi.hoisted(() => ({ resumeTile: vi.fn(), submitToSession: vi.fn() }))
vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: () => route.connectionId,
  activeGatewayProfileKey: () => route.profile
}))
vi.mock('@/store/session', () => ({ $gatewayState: atom('open'), $sessions: atom([]) }))
vi.mock('@/store/session-states', () => ({ sessionTileDelegate: () => delegate }))
vi.mock('@/store/windows', () => ({ isAuxiliaryWindow: () => false }))

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  route.profile = 'work'
})

it('preserves a trusted owner through the bridge and rejects a changed owner before sending', () => {
  let receive!: (payload: QuickEntrySubmitPayload) => void
  vi.stubGlobal('hermesDesktop', {
    quickEntry: {
      onSubmit: (fn: typeof receive) => {
        receive = fn

        return () => {}
      },
      pushState: vi.fn()
    }
  })
  const submitText = vi.fn()
  renderHook(() => useQuickEntryBridge({ submitText, startFreshSessionDraft: vi.fn() }))
  const payload = { target: 'current', text: 'saved thought', thoughtOwner: { connectionId: 'local', profile: 'work' } }
  act(() => receive(payload))
  expect(submitText).toHaveBeenCalledOnce()
  route.profile = 'other'
  act(() => receive(payload))
  expect(submitText).toHaveBeenCalledOnce()
  act(() => receive({ target: 'current', text: 'ordinary chat' }))
  expect(submitText).toHaveBeenCalledTimes(2)
})

it('does not send to a changed profile after an asynchronous resume or its failure fallback', async () => {
  let receive!: (payload: QuickEntrySubmitPayload) => void
  vi.stubGlobal('hermesDesktop', {
    quickEntry: {
      onSubmit: (fn: typeof receive) => {
        receive = fn

        return () => {}
      },
      pushState: vi.fn()
    }
  })
  const submitText = vi.fn()
  renderHook(() => useQuickEntryBridge({ submitText, startFreshSessionDraft: vi.fn() }))

  const payload = {
    target: 'saved-session',
    text: 'saved thought',
    thoughtOwner: { connectionId: 'local', profile: 'work' }
  }

  let finish!: (id: string) => void
  delegate.resumeTile.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  act(() => receive(payload))
  route.profile = 'other'
  await act(async () => finish('runtime'))
  expect(delegate.submitToSession).not.toHaveBeenCalled()
  expect(submitText).not.toHaveBeenCalled()
  route.profile = 'work'
  let fail!: (error: Error) => void
  delegate.resumeTile.mockImplementationOnce(
    () =>
      new Promise((_resolve, reject) => {
        fail = reject
      })
  )
  act(() => receive(payload))
  route.profile = 'other'
  await act(async () => fail(new Error('gone')))
  expect(submitText).not.toHaveBeenCalled()
})
