import { afterEach, beforeAll, beforeEach, describe, expect, it, type Mock, vi } from 'vitest'

import type * as previewAct from '@/app/chat/right-rail/preview-act'
import { registerPreviewNav } from '@/app/chat/right-rail/preview-nav'
import { registerPreviewPageReader } from '@/app/chat/right-rail/preview-reader'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $rightRailActiveTabId, type RightRailTabId, selectRightRailTab } from '@/store/layout'
import { $previewTabs, closePreviewMatching, openPreview } from '@/store/preview'
import { $sessionTiles } from '@/store/session-states'

import { handleServerRequest } from './server-requests'
import type { ServerRequestContext } from './server-requests'

/**
 * Bridge-level admission and execution binding for drive_preview (#95459,
 * #95475 re-review): the scoped request rail must admit a background session
 * whose OWN preview is the live on-screen tab, refuse everyone else, and bind
 * admission and effect to the SAME tab id across the lazy engine-load await —
 * including the deferred-load tab-switch and close schedules.
 *
 * The engine module is wrapped with a gate so a schedule can hold the load
 * open; everything around it (dispatch, admission, ownership revalidation,
 * nav-handle resolution) is the real production code, observed through real
 * registered handles.
 */

const hoisted = vi.hoisted(() => {
  const state = {
    release: () => undefined,
    gate: Promise.resolve()
  }

  return state
})

vi.mock('@/app/chat/right-rail/preview-act', async importOriginal => {
  const actual = await importOriginal<typeof previewAct>()

  return {
    ...actual,
    actOnActivePreview: async (
      action: Parameters<typeof previewAct.actOnActivePreview>[0],
      options: Parameters<typeof previewAct.actOnActivePreview>[1]
    ) => {
      await hoisted.gate

      return actual.actOnActivePreview(action, options)
    }
  }
})

/** Decode the `{ value: JSON.stringify(...) }` answer the handler gives the tool. */
const answerOf = (respond: Mock): Record<string, unknown> | undefined => {
  const first = respond.mock.calls[0]?.[0] as { value?: string } | undefined

  return first?.value ? (JSON.parse(first.value) as Record<string, unknown>) : undefined
}

/** Long enough to absorb the lazy engine import on a cold worker, short enough
 *  to fail loudly. Real assertions run the moment the promise settles. */
const WAIT = { interval: 50, timeout: 20_000 }

const deps = {
  activeSessionIdRef: { current: null },
  sessionInterrupted: () => false,
  sessionStateByRuntimeIdRef: { current: new Map() },
  updateSessionState: (sessionId: string, update: (state: ReturnType<typeof createClientSessionState>) => ReturnType<typeof createClientSessionState>) =>
    update(createClientSessionState('stored-x')),
  upsertToolCall: () => undefined
} as ServerRequestContext['deps']

function deliverPreviewAct(sessionId: string | undefined, activeSessionId: null | string) {
  const respond = vi.fn()
  const fail = vi.fn()
  const decline = vi.fn()

  const handled = handleServerRequest(
    {
      decline,
      fail,
      id: 'srq-preview',
      method: 'preview.act',
      params: {
        action: 'reload',
        request_id: 'req-1',
        session_id: sessionId
      },
      profile: 'default',
      replayed: false,
      respond
    },
    deps,
    activeSessionId
  )

  return { decline, fail, handled, respond }
}

function urlTarget(url: string) {
  return { kind: 'url' as const, label: url, source: url, url }
}

const cleanupFns: Array<() => void> = []

function registerReader(tabId: RightRailTabId, sessionId: string, storedSessionId: string) {
  const unregister = registerPreviewPageReader(
    tabId,
    async () => ({ text: 'page', title: 't', url: 'u' }),
    sessionId,
    storedSessionId
  )

  cleanupFns.push(unregister)

  return unregister
}

const reloadSpies = new Map<string, ReturnType<typeof vi.fn>>()

function registerNavSpy(tabId: RightRailTabId) {
  const reload = vi.fn()

  const unregister = registerPreviewNav(tabId, {
    back: vi.fn(),
    forward: vi.fn(),
    reload
  })

  cleanupFns.push(unregister)
  reloadSpies.set(tabId, reload)

  return reload
}

function openOwnedTab(url: string, sessionId?: string, storedSessionId?: string): RightRailTabId {
  openPreview(urlTarget(url), sessionId ? { ownerSessionId: sessionId, ownerStoredSessionId: storedSessionId } : undefined)

  // Resolve by the URL just opened, not by list order: several url tabs can be
  // open at once and `.find(id.startsWith('url:'))` would return the OLDEST one.
  const tabId = $previewTabs.get().filter(tab => tab.target.source === url).at(-1)!.id

  registerReader(tabId, sessionId ?? 'user', storedSessionId ?? 'user-stored')
  registerNavSpy(tabId)
  selectRightRailTab(tabId)

  return tabId
}

describe('preview.act scoped request admission and effect binding', () => {
  beforeAll(async () => {
    // Warm the lazy engine graph once: in a fresh worker the first dynamic
    // import of preview-act.ts pays full on-demand transform (seconds), while
    // every later import is a cache hit. Without this the first test's
    // assertion window expires before the loader ever settles.
    await import('@/app/chat/right-rail/preview-act')
  }, 30_000)

  beforeEach(() => {
    window.localStorage.clear()
    reloadSpies.clear()
    hoisted.gate = Promise.resolve()
  })

  afterEach(() => {
    for (const cleanup of cleanupFns.splice(0)) {
      cleanup()
    }

    closePreviewMatching('https://a.test')
    closePreviewMatching('https://b.test')
    $sessionTiles.set([])
  })

  it('admits an active session and runs its active tab', async () => {
    const tabId = openOwnedTab('https://a.test', 'rt-A', 'stored-A')

    const { handled, respond } = deliverPreviewAct('rt-A', 'rt-A')

    expect(handled).toBe(true)
    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), WAIT)

    expect(reloadSpies.get(tabId)).toHaveBeenCalledTimes(1)
    expect(answerOf(respond)).toMatchObject({ acted: 'reload', success: true })
  })

  it('admits a background session whose own preview is the on-screen tab', async () => {
    // The window hosts B as an open session tile (this is what makes the
    // scoped request reach THIS window's admission at all — otherwise the
    // multi-window rail declines it for the owner window to answer).
    $sessionTiles.set([{ runtimeId: 'rt-B', storedSessionId: 'stored-B' }])

    const tabB = openOwnedTab('https://b.test', 'rt-B', 'stored-B')

    const { handled, respond } = deliverPreviewAct('rt-B', null)

    expect(handled).toBe(true)
    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), WAIT)

    expect(reloadSpies.get(tabB)).toHaveBeenCalledTimes(1)
    expect(answerOf(respond)).toMatchObject({ acted: 'reload', success: true })
  })

  it('refuses a hosted background session that owns no live preview', async () => {
    const tabA = openOwnedTab('https://a.test', 'rt-A', 'stored-A')

    // C is hosted by this window (its tile is open) but owns no live preview,
    // so the request reaches the handler and is refused in words.
    $sessionTiles.set([{ runtimeId: 'rt-C', storedSessionId: 'stored-C' }])

    const { handled, respond } = deliverPreviewAct('rt-C', null)

    expect(handled).toBe(true)
    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), WAIT)

    expect(reloadSpies.get(tabA)).not.toHaveBeenCalled()
    expect(answerOf(respond)).toMatchObject({
      error: expect.stringContaining('only takes actions in the session the user is looking at'),
      success: false
    })
  })

  it('binds admission and effect to the captured tab across a deferred load (switch schedule)', async () => {
    $sessionTiles.set([{ runtimeId: 'rt-B', storedSessionId: 'stored-B' }])

    const tabA = openOwnedTab('https://a.test', 'rt-A', 'stored-A')
    const tabB = openOwnedTab('https://b.test', 'rt-B', 'stored-B')

    expect(tabB).not.toBe(tabA)

    // Hold the engine load; B's request is admitted while B's tab is active.
    let release!: () => void

    hoisted.gate = new Promise<void>(resolve => {
      release = resolve
    })

    const { handled, respond } = deliverPreviewAct('rt-B', null)
    expect(handled).toBe(true)

    // The user switches to A while the load is still pending.
    selectRightRailTab(tabA)
    expect($rightRailActiveTabId.get()).toBe(tabA)

    release()
    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), WAIT)

    // B's authorized action hit B's tab — never A's newly-active one.
    expect(reloadSpies.get(tabB)).toHaveBeenCalledTimes(1)
    expect(reloadSpies.get(tabA)).not.toHaveBeenCalled()
    expect(answerOf(respond)).toMatchObject({ acted: 'reload', success: true })
  })

  it('fails closed when the captured tab closes during the deferred load', async () => {
    $sessionTiles.set([{ runtimeId: 'rt-B', storedSessionId: 'stored-B' }])

    openOwnedTab('https://b.test', 'rt-B', 'stored-B')

    let release!: () => void

    hoisted.gate = new Promise<void>(resolve => {
      release = resolve
    })

    const { handled, respond } = deliverPreviewAct('rt-B', null)
    expect(handled).toBe(true)

    // Tab and handles vanish while the load is pending.
    cleanupFns.splice(0).forEach(cleanup => cleanup())
    reloadSpies.clear()
    closePreviewMatching('https://b.test')

    release()
    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), WAIT)

    expect(answerOf(respond)).toMatchObject({ error: expect.stringContaining('No live page'), success: false })
  })

  it('refuses a replacement tab opened under a different owner after close', async () => {
    $sessionTiles.set([{ runtimeId: 'rt-B', storedSessionId: 'stored-B' }])

    const tabB = openOwnedTab('https://b.test', 'rt-B', 'stored-B')

    let release!: () => void

    hoisted.gate = new Promise<void>(resolve => {
      release = resolve
    })

    const { handled, respond } = deliverPreviewAct('rt-B', null)
    expect(handled).toBe(true)

    // B's tab closes; A opens the same URL (a fresh vessel with a fresh id)
    // and becomes the on-screen tab before the load resolves.
    cleanupFns.splice(0).forEach(cleanup => cleanup())
    reloadSpies.clear()
    closePreviewMatching('https://b.test')

    const tabA = openOwnedTab('https://b.test', 'rt-A', 'stored-A')

    expect(tabA).not.toBe(tabB)

    release()
    await vi.waitFor(() => expect(respond).toHaveBeenCalledTimes(1), WAIT)

    // The captured identity resolved nothing: B's action died with its tab,
    // and A's replacement tab was never driven.
    expect(reloadSpies.get(tabA)).not.toHaveBeenCalled()
    expect(answerOf(respond)).toMatchObject({ error: expect.stringContaining('No live page'), success: false })
  })
})
