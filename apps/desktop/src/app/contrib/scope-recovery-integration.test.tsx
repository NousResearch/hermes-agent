import { act, cleanup, renderHook, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type SidebarSessionsResponse } from '@/hermes'
import {
  $composerDraft,
  clearSessionDraft,
  freshDraftScope,
  mainComposerScope,
  rotateFreshDraftKey
} from '@/store/composer'
import {
  $activeGatewayProfile,
  ALL_PROFILES,
  captureNewChatSource,
  newSessionInAgent,
  newSessionInProfile,
  requestFreshSession,
  selectProfile
} from '@/store/profile'
import { $activeSessionId, $connection, $selectedStoredSessionId, $sessions, setSessions } from '@/store/session'
import { noteUserSessionOpen } from '@/store/session-open-intent'

import { markActiveComposer, onComposerDraftRequests } from '../chat/composer/focus'
import { useComposerDraft } from '../chat/composer/hooks/use-composer-draft'
import { useSessionListActions } from '../session/hooks/use-session-list-actions'

import { useBackgroundSync } from './hooks/use-background-sync'
import { mainComposerRecoveryIntent, recoverMainComposerSession } from './recover-main-composer-session'
import { useGatewayScopeRefresh } from './use-gateway-scope-refresh'

vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ composer: () => ({ setText: () => undefined }) }),
  useAuiState: (select: (state: { composer: { text: string } }) => unknown) => select({ composer: { text: '' } }),
  useComposerRuntime: () => ({ getState: () => ({ text: '' }), subscribe: () => () => undefined })
}))

function PrimaryComposer({ children }: { children: ReactNode }) {
  const draft = useComposerDraft({
    activeQueueSessionKey: null,
    focusKey: null,
    inputDisabled: false,
    queueEditRef: { current: null },
    sessionId: null
  })

  return (
    <>
      {children}
      <div contentEditable data-testid="main-editor" ref={draft.editorRef} suppressContentEditableWarning />
    </>
  )
}

// Only the network activation boundary is stubbed; no socket/backend is started.
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  activeGatewayConnectionId: () => 'vps',
  ensureGatewayForAgent: async () => true
}))

const api = vi.hoisted(() => ({ list: vi.fn() }))
vi.mock('@/hermes', async original => ({
  ...(await original<Record<string, unknown>>()),
  listSidebarSessions: (...args: unknown[]) => api.list(...args),
  getCronJobs: async () => []
}))

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

const page = (id: string, profile: string, connectionId = 'vps'): SidebarSessionsResponse => ({
  recents: {
    sessions: [
      {
        id,
        profile,
        connection_id: connectionId,
        message_count: 2,
        last_active: 2,
        started_at: 1,
        ended_at: null,
        input_tokens: 0,
        is_active: false,
        model: null,
        output_tokens: 0,
        preview: 'fixture',
        source: 'desktop',
        title: id,
        tool_call_count: 0
      }
    ]
  },
  cron: { sessions: [] },
  messaging: { sessions: [] }
})

const noop = () => undefined

const unexpectedSessionUpdate = (): never => {
  throw new Error('No active session should be updated by this fixture')
}

interface ScopeHarnessProps {
  connectionId?: string
  profile: string
}
const refresh = async () => undefined
// Generic RPC fixture boundary; all exercised methods here return an empty session list.
const request = async <T,>(): Promise<T> => ({ sessions: [] }) as T

function mount({ unified = false, editor = true } = {}) {
  const resume = vi.fn()
  const initialProps: ScopeHarnessProps = { profile: 'A' }

  const view = renderHook(
    ({ profile, connectionId = 'vps' }: ScopeHarnessProps) => {
      const actions = useSessionListActions({ profileScope: unified ? ALL_PROFILES : profile })
      useGatewayScopeRefresh(
        connectionId,
        profile,
        refresh,
        refresh,
        actions.refreshSessions,
        isCurrent =>
          recoverMainComposerSession({
            isCurrent,
            onResume: row => resume(row.id),
            scope: { connectionId, profile }
          }),
        actions.onSessionsRefreshed,
        () => mainComposerRecoveryIntent('/new')
      )
      // Same ordering as ContribWiring: the background hook issues the newer request.
      useBackgroundSync({
        activeConnectionId: connectionId,
        activeGatewayProfile: profile,
        activeIsMessaging: false,
        activeSessionId: null,
        activeStoredSessionId: null,
        freshDraftReady: false,
        gatewayState: 'open',
        refreshActiveTranscript: noop,
        refreshCronJobs: noop,
        refreshCurrentModel: refresh,
        refreshHermesConfig: refresh,
        refreshMessagingSessions: noop,
        refreshSessions: actions.refreshSessions,
        requestGateway: request,
        updateSessionState: unexpectedSessionUpdate
      })

      return actions
    },
    { initialProps, wrapper: editor ? PrimaryComposer : undefined }
  )

  return { ...view, resume }
}

beforeEach(() => {
  vi.useFakeTimers()
  setSessions([])
  captureNewChatSource('vps')
  $composerDraft.set('')
  mainComposerScope.clear()
  clearSessionDraft(freshDraftScope())
  $connection.set({
    connectionId: 'vps',
    profile: 'A',
    mode: 'remote',
    baseUrl: 'https://fixture.invalid',
    wsUrl: 'wss://fixture.invalid/ws',
    token: '',
    logs: [],
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null
  })
  $activeGatewayProfile.set('A')
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  api.list.mockReset()
  api.list.mockResolvedValue(page('old-A', 'A'))
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  setSessions([])
  mainComposerScope.clear()
  clearSessionDraft(freshDraftScope())
  $connection.set(null)
})

async function pendingSwitch(options?: Parameters<typeof mount>[0]) {
  const view = mount(options)
  await act(async () => undefined)
  const stale = deferred<SidebarSessionsResponse>()
  const latest = deferred<SidebarSessionsResponse>()
  api.list.mockImplementationOnce(() => stale.promise).mockImplementationOnce(() => latest.promise)
  $activeGatewayProfile.set('B')
  view.rerender({ profile: 'B' })

  return { ...view, stale, latest }
}

describe('scope recovery and the real competing background refresh', () => {
  it('does not navigate when the main editor has text typed before any runtime or draft-atom sync', async () => {
    const view = mount()
    await act(async () => undefined)
    const stale = deferred<SidebarSessionsResponse>()
    const latest = deferred<SidebarSessionsResponse>()
    api.list.mockImplementationOnce(() => stale.promise).mockImplementationOnce(() => latest.promise)
    $activeGatewayProfile.set('B')
    view.rerender({ profile: 'B' })
    const editor = screen.getByTestId('main-editor')
    editor.textContent = 'words typed during refresh'
    expect($composerDraft.get()).toBe('')
    await act(async () => {
      latest.resolve(page('latest-B', 'B'))
      stale.resolve(page('discarded-B', 'B'))
    })
    expect(view.resume).not.toHaveBeenCalled()
    expect(editor.textContent).toBe('words typed during refresh')
  })

  it('keeps an explicit New Chat in another profile fresh, even when the request precedes scope activation', async () => {
    const view = mount()
    await act(async () => undefined)
    const latest = deferred<SidebarSessionsResponse>()
    api.list.mockReturnValue(latest.promise)
    await act(async () => {
      newSessionInProfile('B')
    })
    view.rerender({ profile: 'B' })
    await act(async () => {
      latest.resolve(page('latest-B', 'B'))
    })
    expect(view.resume).not.toHaveBeenCalled()
  })

  it('preserves an attachment-only draft and does not retry after the user removes it', async () => {
    const view = await pendingSwitch()

    const attachment = {
      id: 'pending-file',
      kind: 'file' as const,
      label: 'fixture.txt',
      uploadState: 'uploading' as const
    }

    mainComposerScope.add(attachment)
    const attachments = mainComposerScope.$attachments.get()
    await act(async () => {
      view.latest.resolve(page('latest-B', 'B'))
      view.stale.resolve(page('discarded-B', 'B'))
    })
    expect(view.resume).not.toHaveBeenCalled()
    expect(mainComposerScope.$attachments.get()).toBe(attachments)
    mainComposerScope.clear()
    api.list.mockResolvedValue(page('later-B', 'B'))
    await act(async () => {
      await view.result.current.refreshSessions()
    })
    expect(view.resume).not.toHaveBeenCalled()
  })

  it.each(['fresh-key', 'fresh-request', 'selected-session', 'pending-session-open'] as const)(
    'does not recover after a newer %s intent during refresh',
    async intent => {
      const view = await pendingSwitch()
      act(() => {
        if (intent === 'fresh-key') {
          rotateFreshDraftKey()
        } else if (intent === 'fresh-request') {
          requestFreshSession()
        } else if (intent === 'selected-session') {
          $selectedStoredSessionId.set('user-selected')
        } else {
          noteUserSessionOpen()
        }
      })
      await act(async () => {
        view.latest.resolve(page('latest-B', 'B'))
        view.stale.resolve(page('discarded-B', 'B'))
      })
      expect(view.resume).not.toHaveBeenCalled()
    }
  )

  it('never substitutes the focused tile draft for the main composer', async () => {
    const view = await pendingSwitch()
    markActiveComposer('tile:fixture')

    const unsubscribe = onComposerDraftRequests(
      { getIds: () => ['tile-session'], isActive: () => true, target: 'tile:fixture' },
      { read: () => 'tile words, not the main draft', write: () => false }
    )

    try {
      await act(async () => {
        view.latest.resolve(page('latest-B', 'B'))
        view.stale.resolve(page('discarded-B', 'B'))
      })
      expect(view.resume).toHaveBeenCalledExactlyOnceWith('latest-B')
    } finally {
      unsubscribe()
      markActiveComposer('main')
    }
  })

  it('fails closed when there is no mounted main composer to read', async () => {
    const view = await pendingSwitch({ editor: false })
    await act(async () => {
      view.latest.resolve(page('latest-B', 'B'))
      view.stale.resolve(page('discarded-B', 'B'))
    })
    await act(async () => {
      vi.advanceTimersByTime(100)
    })
    expect(view.resume).not.toHaveBeenCalled()
  })

  it('recovers from a valid unified-list publication for the active profile', async () => {
    const view = await pendingSwitch({ unified: true })
    await act(async () => {
      view.latest.resolve(page('latest-B', 'B'))
      view.stale.resolve(page('discarded-B', 'B'))
    })
    expect(view.resume).toHaveBeenCalledExactlyOnceWith('latest-B')
  })

  it('does not resume twice when the latest refresh resolves before the discarded refresh', async () => {
    const view = await pendingSwitch()
    await act(async () => {
      view.latest.resolve(page('latest-B', 'B'))
    })
    expect(view.resume).toHaveBeenCalledExactlyOnceWith('latest-B')
    await act(async () => {
      view.stale.resolve(page('discarded-B', 'B'))
    })
    expect(view.resume).toHaveBeenCalledTimes(1)
    expect($sessions.get().some(row => row.id === 'discarded-B')).toBe(false)
  })

  it('cancels B in A B A and waits for the new A publication, not cached A history', async () => {
    const view = await pendingSwitch()
    const staleA = deferred<SidebarSessionsResponse>()
    const latestA = deferred<SidebarSessionsResponse>()
    api.list.mockImplementationOnce(() => staleA.promise).mockImplementationOnce(() => latestA.promise)
    $activeGatewayProfile.set('A')
    view.rerender({ profile: 'A' })
    await act(async () => {
      view.stale.resolve(page('discarded-B', 'B'))
      view.latest.resolve(page('latest-B', 'B'))
      staleA.resolve(page('discarded-A', 'A'))
    })
    expect(view.resume).not.toHaveBeenCalled()
    await act(async () => {
      latestA.resolve(page('new-A', 'A'))
    })
    expect(view.resume).toHaveBeenCalledExactlyOnceWith('new-A')
  })

  it('consumes an empty valid publication without later background navigation', async () => {
    const view = await pendingSwitch()
    await act(async () => {
      view.latest.resolve({ recents: { sessions: [] }, cron: { sessions: [] }, messaging: { sessions: [] } })
      view.stale.resolve(page('discarded-B', 'B'))
    })
    expect(view.resume).not.toHaveBeenCalled()
    api.list.mockResolvedValue(page('later-B', 'B'))
    await act(async () => {
      await view.result.current.refreshSessions()
    })
    expect(view.resume).not.toHaveBeenCalled()
  })

  it('keeps a registry-agent New Chat fresh but allows a subsequent plain profile pick to recover', async () => {
    const view = mount()
    await act(async () => undefined)
    api.list.mockResolvedValue(page('latest-B', 'B'))
    await act(async () => {
      newSessionInAgent({ connectionId: 'vps', profile: 'B' })
    })
    view.rerender({ profile: 'B' })
    await act(async () => undefined)
    expect(view.resume).not.toHaveBeenCalled()
    api.list.mockResolvedValue(page('latest-C', 'C'))
    await act(async () => {
      selectProfile('C')
    })
    view.rerender({ profile: 'C' })
    await act(async () => undefined)
    expect(view.resume).toHaveBeenCalledExactlyOnceWith('latest-C')
  })

  it('uses the connection-scoped publication when two sources expose the same profile', async () => {
    const view = mount()
    await act(async () => undefined)
    const stale = deferred<SidebarSessionsResponse>()
    const latest = deferred<SidebarSessionsResponse>()
    api.list.mockImplementationOnce(() => stale.promise).mockImplementationOnce(() => latest.promise)
    $connection.set({ ...$connection.get()!, connectionId: 'other' })
    view.rerender({ profile: 'A', connectionId: 'other' })
    await act(async () => {
      stale.resolve(page('discarded-other', 'A', 'other'))
    })
    expect(view.resume).not.toHaveBeenCalled()
    await act(async () => {
      latest.resolve(page('latest-other', 'A', 'other'))
    })
    expect(view.resume).toHaveBeenCalledExactlyOnceWith('latest-other')
  })

  it('rechecks user intent after the live-draft request replies', async () => {
    const view = await pendingSwitch()
    const userOpen = () => noteUserSessionOpen()
    window.addEventListener('hermes:composer-get-draft', userOpen)

    try {
      await act(async () => {
        view.latest.resolve(page('latest-B', 'B'))
        view.stale.resolve(page('discarded-B', 'B'))
      })
      expect(view.resume).not.toHaveBeenCalled()
    } finally {
      window.removeEventListener('hermes:composer-get-draft', userOpen)
    }
  })

  it('recovers only when the latest request publishes, not when the discarded one resolves', async () => {
    const view = mount()
    await act(async () => undefined)
    const stale = deferred<SidebarSessionsResponse>()
    const latest = deferred<SidebarSessionsResponse>()
    api.list.mockImplementationOnce(() => stale.promise).mockImplementationOnce(() => latest.promise)
    $activeGatewayProfile.set('B')
    view.rerender({ profile: 'B' })
    expect(api.list).toHaveBeenCalledTimes(3)
    await act(async () => {
      stale.resolve(page('discarded-B', 'B'))
    })
    expect(view.resume).not.toHaveBeenCalled()
    expect($sessions.get().some(row => row.profile === 'B')).toBe(false)
    await act(async () => {
      latest.resolve(page('latest-B', 'B'))
    })
    expect($sessions.get().some(row => row.id === 'latest-B')).toBe(true)
    expect(view.resume).toHaveBeenCalledExactlyOnceWith('latest-B')
    // A subsequent background publication must not repeat recovery.
    await act(async () => {
      await view.result.current.refreshSessions()
    })
    expect(view.resume).toHaveBeenCalledTimes(1)
  })
})
