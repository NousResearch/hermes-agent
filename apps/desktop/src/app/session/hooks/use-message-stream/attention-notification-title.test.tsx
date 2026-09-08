import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { clearClarifyRequest } from '@/store/clarify'
import { dispatchNativeNotification } from '@/store/native-notifications'
import type * as NativeNotifications from '@/store/native-notifications'
import { clearAllPrompts } from '@/store/prompts'
import type * as Prompts from '@/store/prompts'
import { $sessions, setSessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

// The OS toast for a blocking prompt names the chat it belongs to. "Approval
// needed" on its own sends the user hunting through the sidebar for which of
// six sessions wants them; "Approval needed — Fix the build" does not.

vi.mock('@/store/native-notifications', async importOriginal => {
  const actual = await importOriginal<typeof NativeNotifications>()

  return { ...actual, dispatchNativeNotification: vi.fn(() => true) }
})

vi.mock('@/store/prompts', async importOriginal => {
  const actual = await importOriginal<typeof Prompts>()

  // The approval handler rounds-trips `approval.received` through the gateway;
  // there is none here, and the notification fires regardless of that call.
  return { ...actual, receiveApprovalRequest: vi.fn(async () => undefined) }
})

const RUNTIME = 'rt-blocked'
const STORED = 'stored-blocked'
const OTHER = 'rt-other'

const row = (id: string, title: string): SessionInfo =>
  ({ id, message_count: 3, profile: 'default', source: 'cli', started_at: 0, title }) as SessionInfo

let stream: MessageStreamHarness
const notify = vi.mocked(dispatchNativeNotification)

const lastTitle = () => notify.mock.calls.at(-1)?.[0].title

describe('blocking-prompt native notifications name their session', () => {
  beforeEach(() => {
    notify.mockClear()
    clearClarifyRequest()
    clearAllPrompts()
    setSessions([row(STORED, 'Fix the build')])

    const states = new Map([[RUNTIME, { ...createClientSessionState(STORED), busy: true }]])
    // Focus is elsewhere: the blocked session is a background one, which is the
    // case the notification exists for.
    stream = renderMessageStream(OTHER, { states })
  })

  afterEach(() => {
    cleanup()
    clearClarifyRequest()
    clearAllPrompts()
    $sessions.set([])
  })

  it('appends the session title to an approval toast', () => {
    act(() =>
      stream.handleEvent({
        payload: { command: 'rm -rf build', description: 'dangerous', request_id: 'req-1' },
        session_id: RUNTIME,
        type: 'approval.request'
      })
    )

    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'approval', sessionId: RUNTIME }))
    expect(lastTitle()).toBe('Approval needed — Fix the build')
  })

  it('appends the session title to a clarify toast', () => {
    act(() =>
      stream.handleEvent({
        payload: { choices: ['yes', 'no'], question: 'Ship it?', request_id: 'req-2' },
        session_id: RUNTIME,
        type: 'clarify.request'
      })
    )

    expect(lastTitle()).toBe('Input needed — Fix the build')
  })

  it('falls back to the bare title when the session has no row yet', () => {
    act(() =>
      stream.handleEvent({
        payload: { command: 'x', description: 'd', request_id: 'req-3' },
        session_id: 'rt-unknown',
        type: 'approval.request'
      })
    )

    expect(lastTitle()).toBe('Approval needed')
  })
})
