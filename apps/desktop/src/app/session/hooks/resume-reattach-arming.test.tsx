import { useStore } from '@nanostores/react'
import { cleanup, render, waitFor } from '@testing-library/react'
import { useEffect, useRef } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { getLatestSessionMessages } from '@/hermes'
import { recordFriction } from '@/store/desktop-metrics'
import { $notifications, clearNotifications, notify } from '@/store/notifications'
import {
  $activeSessionId,
  $resumeFailedSessionId,
  $resumeReattachSessionId,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setMessages,
  setResumeReattachSessionId,
  setSelectedStoredSessionId
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

import { useSessionActions } from './use-session-actions'
import { useSessionStateCache } from './use-session-state-cache'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getLatestSessionMessages: vi.fn(),
  getSession: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn()
}))

vi.mock('@/store/profile', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureGatewayAgent: vi.fn().mockResolvedValue(undefined),
  ensureGatewayProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(),
  requestGatewayForProfile: vi.fn(),
  retainGatewayForAgent: vi.fn(async () => () => undefined)
}))

vi.mock('@/store/desktop-metrics', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  recordFriction: vi.fn()
}))

type Resume = ReturnType<typeof useSessionActions>['resumeSession']

function Harness({ onReady }: { onReady: (resume: Resume) => void }) {
  const activeSessionId = useStore($activeSessionId)
  const busyRef = useRef(false)

  const cache = useSessionStateCache({
    activeSessionId,
    busyRef,
    selectedStoredSessionId: null,
    setAwaitingResponse,
    setBusy,
    setMessages
  })

  const actions = useSessionActions({
    ...cache,
    activeSessionId,
    busyRef,
    creatingSessionRef: useRef(false),
    getRouteToken: () => 'reattach-arming',
    getRoutedStoredSessionId: () => null,
    navigate: vi.fn() as never,
    requestGateway: async method => {
      throw new Error(`injected transient failure: ${method}`)
    },
    routedSessionId: null,
    selectedStoredSessionId: null
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

afterEach(() => {
  cleanup()
  clearNotifications()
  clearAllSessionStates()
  setResumeReattachSessionId(null)
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setMessages([])
  vi.clearAllMocks()
})

it('a resume that fails after painting the history arms the quiet re-attach, and its retry stays quiet', async () => {
  vi.mocked(getLatestSessionMessages).mockResolvedValue({
    messages: [
      { content: 'run the long job', role: 'user', timestamp: 1 },
      { content: 'started it', role: 'assistant', timestamp: 2 }
    ],
    session_id: 'stored-1'
  } as never)

  let resume: null | Resume = null
  render(<Harness onReady={r => (resume = r)} />)
  await waitFor(() => expect(resume).not.toBeNull())

  await resume!('stored-1', true)

  // Readable but detached: re-attach, never the stranded-window latch whose
  // exhaustion would cover the painted history with the error overlay.
  expect($resumeReattachSessionId.get()).toBe('stored-1')
  expect($resumeFailedSessionId.get()).toBeNull()

  // The re-attach retry fails the same way while another toast is up.
  notify({ id: 'unrelated', kind: 'warning', message: 'something else the user is reading' })
  await resume!('stored-1', true, undefined, { reattach: true })

  expect($resumeReattachSessionId.get()).toBe('stored-1')
  expect($notifications.get().map(n => n.id)).toEqual(['unrelated', 'resume-failed:stored-1'])
  expect(vi.mocked(recordFriction).mock.calls.filter(([kind]) => kind === 'error_toast')).toHaveLength(1)
})
