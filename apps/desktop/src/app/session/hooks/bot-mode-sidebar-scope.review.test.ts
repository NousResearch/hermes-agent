/**
 * Bot Mode sidebar scope edges (#126732 review): the sidebar re-scopes to a
 * selected bot only when that bot lives on the source the window's list and
 * search reads ride, its search follows the same scope as its list, a
 * messaging row's persisted unread flag paints its dot, and an open bot tile
 * whose row is only an owner stub keeps re-reading its transcript.
 */

import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { reconcileTileTranscripts } from '@/app/contrib/hooks/use-background-sync'
import { $workspaceMode, $workspaceNewSessionTarget } from '@/components/pane-shell/workspace-scope'
import { sessionListFingerprint } from '@/lib/session-signatures'
import { $changeEventsAvailable, resetLiveSync } from '@/store/live-sync'
import {
  setConnection,
  setCronSessions,
  setMessagingSessions,
  setSessions,
  setUnlistedSessionOwnerRows
} from '@/store/session'
import { $sessionDotStateById } from '@/store/session-dot-state'
import { $unreadWriteGuard } from '@/store/session-unread-remote'
import { makeSessionInfo } from '@/test/session-info'

import { botModeFetchScope, useSidebarDisplayScope } from './bot-mode-fetch-scope'

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal()),
  getLatestSessionMessages: vi.fn()
}))

const { getLatestSessionMessages } = await import('@/hermes')

const selectBot = (connectionId: string, profile: string, targetProfile = profile) => {
  $workspaceMode.set('bots')
  $workspaceNewSessionTarget.set({
    kind: 'route',
    route: { connectionId, mode: connectionId === 'local' ? 'local' : 'remote', profile, targetProfile }
  })
}

beforeEach(() => {
  setConnection({ mode: 'local' } as never)
})

afterEach(() => {
  $workspaceMode.set('sessions')
  $workspaceNewSessionTarget.set(null)
  setConnection(null)
  setSessions([])
  setMessagingSessions([])
  setCronSessions([])
  setUnlistedSessionOwnerRows([])
  $unreadWriteGuard.set(new Map())
  resetLiveSync()
  vi.clearAllMocks()
})

describe('Bot Mode sidebar scope follows the window source', () => {
  it('does not re-scope local reads to a bot on another connection', () => {
    selectBot('remote-a', 'default')

    // The window's reads still land on the local backend: re-scoping them to
    // the remote bot's profile name would paint local rows under that bot.
    expect(botModeFetchScope('default')).toBeNull()
  })

  it('re-scopes once the window is on the bot source', () => {
    selectBot('remote-a', 'moxie', 'default')
    setConnection({ connectionId: 'remote-a', mode: 'remote' } as never)

    expect(botModeFetchScope('default')).toEqual({ displayProfile: 'default', fetchProfile: 'moxie' })
  })

  it('searches the profile the list shows', () => {
    selectBot('local', 'mybot')

    const { result } = renderHook(() => useSidebarDisplayScope('default'))

    expect(result.current[0]).toBe('mybot')
    expect(result.current[2]).toBe('mybot')

    act(() => selectBot('local', 'otherbot'))
    expect(result.current[2]).toBe('otherbot')
  })
})

describe('messaging rows carry their persisted unread dot', () => {
  it('paints a telegram row marked unread', () => {
    setMessagingSessions([makeSessionInfo({ id: 'tg-1', profile: 'mybot', source: 'telegram', unread: true })])

    expect($sessionDotStateById.get()['tg-1']).toBe('unread')
  })

  it('honours the write guard on a messaging row', () => {
    setMessagingSessions([makeSessionInfo({ id: 'tg-2', source: 'telegram', unread: false })])
    $unreadWriteGuard.set(new Map([['tg-2', { at: Date.now(), value: true }]]))

    expect($sessionDotStateById.get()['tg-2']).toBe('unread')
  })
})

describe('bot tile with only an owner stub', () => {
  it('keeps fetching: a stub fingerprint never gates the transcript read', async () => {
    $changeEventsAvailable.set(true)

    // A hidden Bot Chat row parked off-list after the tile opened it.
    const stub = makeSessionInfo({ hidden: true, id: 'bot-chat-1', last_active: 5, message_count: 2, preview: 'a' })
    setUnlistedSessionOwnerRows([stub])

    const signatureRef = { current: new Map<string, string>([['tile-meta:bot-chat-1', sessionListFingerprint(stub)]]) }

    vi.mocked(getLatestSessionMessages).mockResolvedValue({
      messages: [
        { content: 'q', role: 'user', timestamp: 1 },
        { content: 'new background reply', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'bot-chat-1'
    } as never)

    await act(async () => {
      await reconcileTileTranscripts({
        requestSequenceRef: { current: 0 },
        signatureRef,
        tiles: [{ runtimeId: 'rt-bot-1', storedSessionId: 'bot-chat-1' }],
        updateSessionState: vi.fn() as never
      })
    })

    expect(getLatestSessionMessages).toHaveBeenCalled()
  })
})
