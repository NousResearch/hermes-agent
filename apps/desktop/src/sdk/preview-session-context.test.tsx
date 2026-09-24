import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { useSessionContributionContext } from '@/contrib/session-context'
import { $previewTabs } from '@/store/preview'
import {
  $activeSessionId,
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  _resetSessionOwnerHintsForTests
} from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { host } from './index'

const slices = [
  { source: 'cron', store: $cronSessions },
  { source: 'telegram', store: $messagingSessions }
]

const disposers: Array<() => void> = []

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(dispose => dispose())
  Reflect.deleteProperty(window, 'hermesDesktop')
  $previewTabs.set([])
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $sessionTiles.set([])
  $sessionStates.set({})
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  _resetSessionOwnerHintsForTests()
})

it.each(slices)(
  'accepts a fresh qualified $source context in both viewer actions despite a recent-row ID collision',
  async ({ source, store }) => {
    const recent = { id: 'shared', profile: 'default', connection_id: 'local' } as SessionInfo
    const row = { id: 'shared', profile: 'worker', connection_id: 'remote', source } as SessionInfo
    $sessions.set([recent])
    store.set([row])
    const { result } = renderHook(() => useSessionContributionContext({ row }))
    const session = result.current!
    expect(session).toEqual({
      runtimeSessionId: null,
      storedSessionId: row.id,
      connectionId: row.connection_id,
      profile: row.profile
    })
    const openPluginViewer = vi.fn(async () => true)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { openPluginViewer, closePluginViewer: vi.fn(async () => true) }
    })
    const ctx = createPluginContext('split-list', dispose => disposers.push(dispose))
    const url = 'https://example.org/viewer?ticket=split-list'
    expect(await host.openPreview({ url, session })).toBe(true)
    expect($previewTabs.get().at(-1)?.target).toMatchObject({ url, transient: true, browserContext: 'isolated' })
    expect(await ctx.os.openViewer({ id: 'watch', title: 'Viewer', url, session })).toBe(true)
    expect(openPluginViewer).toHaveBeenCalledWith('split-list', { id: 'watch', title: 'Viewer', url })

    act(() => store.set([]))
    expect(await host.openPreview({ url, session })).toBe(false)
    expect(await ctx.os.openViewer({ id: 'watch', title: 'Viewer', url, session })).toBe(false)
    expect(openPluginViewer).toHaveBeenCalledOnce()
  }
)

it.each(slices)(
  'rechecks unqualified ambiguity when only the $source slice changes without borrowing the foreground runtime',
  ({ source, store }) => {
    const recent = { id: 'shared', profile: 'default', connection_id: 'local' } as SessionInfo
    const row = { id: 'shared', profile: 'worker', connection_id: 'remote', source } as SessionInfo
    $sessions.set([recent])
    $activeSessionId.set('local-runtime')
    $selectedStoredSessionId.set(recent.id)

    const { result } = renderHook(() => ({
      unqualified: useSessionContributionContext({ storedSessionId: recent.id }),
      qualified: useSessionContributionContext({ row })
    }))

    expect(result.current.unqualified).toMatchObject({ connectionId: 'local', runtimeSessionId: 'local-runtime' })

    act(() => store.set([row]))
    expect(result.current.unqualified).toBeNull()
    expect(result.current.qualified).toEqual({
      runtimeSessionId: null,
      storedSessionId: row.id,
      connectionId: row.connection_id,
      profile: row.profile
    })

    act(() => store.set([]))
    expect(result.current.unqualified).toMatchObject({ connectionId: 'local', runtimeSessionId: 'local-runtime' })
  }
)
