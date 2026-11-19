// Settle archives the OPEN chat and moves on: `leaveTo` must replace the fresh
// draft an archived selected session otherwise falls back to.
import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { type SessionInfo, setSessionArchived } from '@/hermes'
import { setSessions } from '@/store/session'

import { NEW_CHAT_ROUTE } from '../../../routes'
import type { ClientSessionState } from '../../../types'

import { useSessionActions } from './index'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  setSessionArchived: vi.fn().mockResolvedValue({ ok: true })
}))

type ArchiveSession = ReturnType<typeof useSessionActions>['archiveSession']

function Harness({ navigate, onReady }: { navigate: ReturnType<typeof vi.fn>; onReady: (a: ArchiveSession) => void }) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    getRoutedStoredSessionId: () => 'open',
    navigate: navigate as never,
    requestGateway: vi.fn().mockResolvedValue(undefined),
    resetViewSync: vi.fn(),
    routedSessionId: 'open',
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: 'open',
    selectedStoredSessionIdRef: ref<string | null>('open'),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  onReady(actions.archiveSession)

  return null
}

const row = (id: string) =>
  ({ id, message_count: 2, profile: 'work', source: 'desktop', started_at: 1, title: id }) as SessionInfo

beforeAll(() => {
  window.hermesDesktop = {} as typeof window.hermesDesktop
})

beforeEach(() => {
  vi.clearAllMocks()
  setSessions([row('open'), row('next')])
})

afterEach(() => {
  cleanup()
  setSessions([])
})

async function archiveOpen(options?: Parameters<ArchiveSession>[1]) {
  const navigate = vi.fn()
  let archive!: ArchiveSession
  render(<Harness navigate={navigate} onReady={value => (archive = value)} />)
  await act(async () => {
    await archive('open', options)
  })

  return navigate
}

it('falls back to a fresh draft when archiving the open chat', async () => {
  const navigate = await archiveOpen()

  expect(navigate).toHaveBeenCalledWith(NEW_CHAT_ROUTE, { replace: true })
})

it('hands off to leaveTo instead of the fresh draft', async () => {
  const leaveTo = vi.fn()
  const navigate = await archiveOpen({ leaveTo })

  expect(leaveTo).toHaveBeenCalledTimes(1)
  expect(navigate).not.toHaveBeenCalledWith(NEW_CHAT_ROUTE, expect.anything())
  expect(setSessionArchived).toHaveBeenCalledWith('open', true, 'work')
})
