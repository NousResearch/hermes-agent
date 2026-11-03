import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $compactingSessions } from '@/store/compaction'
import { clearNotifications } from '@/store/notifications'
import { forgetSessionOwnerHintsForSession, setSessionOwnerHint } from '@/store/session'
import type * as Router from '@/store/session-request-router'
import { requestForSessionProfile } from '@/store/session-request-router'
import { $sessionStates } from '@/store/session-states'

import { useCompressConversationAction } from '../session/hooks/use-compress-conversation-action'

import { ContextUsagePanel } from './context-usage-panel'
import { _resetManualCompactingForTests, useCompactConversation } from './hooks/use-compact-conversation'

// Only the transport is simulated. The visible controls, controller action,
// compression implementation, ownership resolver and session cache are real.
vi.mock('@/store/session-request-router', async importOriginal => ({
  ...(await importOriginal<typeof Router>()),
  requestForSessionProfile: vi.fn()
}))

const owner = { connectionId: 'device-a', profile: 'alice' }
const storedIds = ['bot-root', 'bot-lineage-tip', 'ordinary-side-chat']

beforeEach(() => {
  vi.clearAllMocks()
  clearNotifications()
  _resetManualCompactingForTests()
  $compactingSessions.set({})
  $sessionStates.set({})
})

afterEach(() => {
  cleanup()

  for (const id of storedIds) {
    forgetSessionOwnerHintsForSession(id)
  }

  $sessionStates.set({})
  _resetManualCompactingForTests()
})

it.each(storedIds)('compacts %s through its owner without creating another conversation', async storedSessionId => {
  const sessionId = `runtime-${storedSessionId}`
  const original = { ...createClientSessionState(), storedSessionId }
  const ambient = vi.fn().mockRejectedValue(new Error('The ambient backend must not receive this action'))
  const activeSessionIdRef = { current: sessionId as null | string }

  const updateSessionState = vi.fn((id: string, update: (state: ClientSessionState) => ClientSessionState) => {
    const states = $sessionStates.get()
    const state = states[id]

    if (state) {
      $sessionStates.set({ ...states, [id]: update(state) })
    }
  })

  let finish: (value: unknown) => void = () => undefined

  $sessionStates.set({ [sessionId]: original })
  setSessionOwnerHint(storedSessionId, owner)
  vi.mocked(requestForSessionProfile).mockImplementation(
    () =>
      new Promise<never>(resolve => {
        finish = value => resolve(value as never)
      })
  )

  function Harness() {
    const compress = useCompressConversationAction({ activeSessionIdRef, requestGateway: ambient, updateSessionState })
    const compact = useCompactConversation({ busy: false, compress, sessionId, storedSessionId })

    return (
      <ContextUsagePanel
        breakdown={null}
        compact={compact}
        loading={false}
        usage={{ calls: 0, input: 0, output: 0, total: 0 }}
      />
    )
  }

  render(<Harness />)
  fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
  expect(requestForSessionProfile).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(requestForSessionProfile).not.toHaveBeenCalled()

  fireEvent.click(screen.getByRole('button', { name: 'Compact conversation' }))
  fireEvent.click(screen.getByRole('button', { name: 'Compact' }))

  await waitFor(() => expect(requestForSessionProfile).toHaveBeenCalledTimes(1))
  expect(requestForSessionProfile).toHaveBeenCalledWith(
    owner,
    ambient,
    'session.compress',
    { session_id: sessionId },
    660_000,
    undefined
  )

  // A late reply must update its captured conversation, not the new focus.
  activeSessionIdRef.current = 'runtime-other-bot'
  await act(async () =>
    finish({
      messages: [{ content: 'working-context summary', role: 'system' }],
      summary: { headline: 'Context compacted' },
      usage: { context_used: 5_000 }
    })
  )

  expect(ambient).not.toHaveBeenCalled()
  expect(Object.keys($sessionStates.get())).toEqual([sessionId])
  expect($sessionStates.get()[sessionId].storedSessionId).toBe(storedSessionId)
  expect(updateSessionState).toHaveBeenCalledWith(sessionId, expect.any(Function), storedSessionId)
  expect(screen.getByRole('button', { name: 'Compact conversation' })).toBeTruthy()
  expect(requestForSessionProfile).toHaveBeenCalledTimes(1)
})
