import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The consent dialog has its own test file and needs a QueryClientProvider;
// this test exercises the pop-out content handoff, not the prompt.
vi.mock('./right-rail/real-profile-consent-dialog', () => ({
  RealProfileConsentDialog: () => null
}))

import type { BrowserWorkspace, BrowserWorkspaceApi } from '../../../electron/browser-workspace-types'
// A popped-out Browser window is a FRESH renderer: no in-memory atoms cross
// the window boundary, and no session ever pushes a rail scope there. All it
// knows arrives via shared persisted tabs plus `?win=browser&tab=` in its own
// URL. Each test therefore resets the module registry (fresh atoms, cold
// caches — exactly what a new renderer process boots with), seeds storage the
// way the main window left it, sets the pop-out query string, and only then
// imports the shell. Regression tests for the black pop-out window (shell
// spawns, no URL bar, content never paints — #119850).
import { BrowserWorkspaces } from '../../../electron/browser-workspaces'

import type { PreviewInputEvent } from './right-rail/preview-input'

const TABS_KEY = 'hermes.desktop.previewTabs.v2'

// Cold module graph, not test logic: the shell pulls in the whole app.
vi.setConfig({ testTimeout: 90000 })

function tabRow(id: string, url: string) {
  return { id, target: { kind: 'url', label: 'kanban', source: url, url } }
}

function storedBuckets(): Record<string, { id: string }[]> {
  const raw = window.localStorage.getItem(TABS_KEY)

  return raw ? (JSON.parse(raw) as Record<string, { id: string }[]>) : {}
}

beforeEach(() => {
  vi.resetModules()
  window.localStorage.clear()
})

afterEach(() => {
  cleanup()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  window.localStorage.clear()
  window.history.replaceState({}, '', '/')
})

async function renderFreshPopout(tabId: string) {
  window.history.replaceState({}, '', `/?win=browser&tab=${encodeURIComponent(tabId)}#/`)
  const { BrowserPopoutShell } = await import('./browser-popout-shell')

  let rendered!: ReturnType<typeof render>
  await act(async () => {
    rendered = render(<BrowserPopoutShell />)
  })

  return rendered
}

async function expectGuestWithUrl(rendered: ReturnType<typeof render>, url: string) {
  // The tab arrives via a store update after mount; wait for the re-render.
  // Generous timeout: the first test in this file pays the cold module-graph
  // import and every poll must survive it.
  await waitFor(
    () => {
      expect(rendered.container.querySelector('webview')?.getAttribute('src')).toBe(url)
    },
    { timeout: 30000 }
  )

  // Identity: the pop-out bar offers pop-in for this tab (icon-only glyph with
  // an accessible name, so assert the aria-label, not text content).
  expect(rendered.container.querySelector('[aria-label="Pop in"]')).not.toBeNull()

  // Address bar tracks the handed-off URL.
  expect((rendered.container.querySelector('input') as HTMLInputElement | null)?.value).toContain('127.0.0.1')
}

describe('browser pop-out content handoff', () => {
  it('hands the persisted tab URL to a guest (pre-scoping single-array storage)', async () => {
    const id = 'url:browser-legacy-1'
    const url = 'http://127.0.0.1:9119/kanban'
    window.localStorage.setItem(TABS_KEY, JSON.stringify([tabRow(id, url)]))

    const rendered = await renderFreshPopout(id)
    await expectGuestWithUrl(rendered, url)
  })

  it('hands the persisted tab URL to a guest (profile-bucketed storage)', async () => {
    const id = 'url:browser-bucket-1'
    const url = 'http://127.0.0.1:9119/kanban'
    window.localStorage.setItem(TABS_KEY, JSON.stringify({ local: [tabRow(id, url)] }))

    const rendered = await renderFreshPopout(id)
    await expectGuestWithUrl(rendered, url)
  })

  it('re-homes onto the owning bucket without duplicating the tab elsewhere', async () => {
    // The popped tab belongs to a secondary profile. Adoption must move the
    // VIEW onto that profile's bucket — not splice the tab into the 'default'
    // bucket the renderer booted on, which would resurrect it in the primary
    // profile's rail.
    const id = 'url:browser-owned-1'
    const url = 'http://127.0.0.1:9119/kanban'
    window.localStorage.setItem(TABS_KEY, JSON.stringify({ local: [tabRow(id, url)] }))

    const rendered = await renderFreshPopout(id)
    await expectGuestWithUrl(rendered, url)

    expect(Object.keys(storedBuckets())).toEqual(['local'])
  })

  it('leaves an unknown tab blank rather than adopting a stranger', async () => {
    const url = 'http://127.0.0.1:9119/kanban'
    window.localStorage.setItem(TABS_KEY, JSON.stringify({ local: [tabRow('url:browser-known-1', url)] }))

    const rendered = await renderFreshPopout('url:browser-nope')

    expect(rendered.container.querySelector('webview')).toBeNull()
    expect(rendered.container.textContent).not.toContain('Pop in')
  })
})

async function renderFreshWorkspace() {
  const listeners = new Set<(state: BrowserWorkspace) => void>()
  const runtime = new BrowserWorkspaces((_recipients, state) => listeners.forEach(listener => listener(state)))
  const relayListeners = new Set<(message: unknown) => void>()
  const deliveredReplies: unknown[] = []

  const windowRelay = {
    onMessage: (listener: (message: unknown) => void) => {
      relayListeners.add(listener)

      return () => { relayListeners.delete(listener) }
    },
    send: (message: unknown) => {
      const isRequest = Boolean(message && typeof message === 'object' && 'payload' in message)
      const recipient = runtime.relay(isRequest ? 1 : 2, message)

      if (recipient === null) { return }

      if (!isRequest) { deliveredReplies.push(message) }

      for (const listener of relayListeners) { listener(message) }
    }
  }

  const seed = runtime.open(
    1,
    {
      tab: {
        id: 'url:seed',
        sessionId: 'stored-session',
        pinned: false,
        target: { kind: 'url', label: 'Seed', source: 'https://example.test/a', url: 'https://example.test/a' }
      },
      scope: 'alpha',
      destination: { kind: 'composer', surfaceId: 'chat', target: 'session', windowId: 'origin',
        conversation: { kind: 'session', id: 'stored-session', connectionId: 'connection', profile: 'alpha' } }
    },
    { connectionId: 'connection', profile: 'alpha' }
  )

  runtime.attach(seed.id, 2)
  const second = runtime.command(2, seed.id, { kind: 'new' })!.activeTabId!
  runtime.command(2, seed.id, { kind: 'page', tabId: second, url: 'https://example.test/b', title: 'Second' })

  const api: BrowserWorkspaceApi = {
    snapshots: async () => runtime.snapshots(2),
    command: async (id, command) => runtime.command(2, id, command),
    onChanged: callback => {
      listeners.add(callback)

      return () => {
        listeners.delete(callback)
      }
    },
    setShortcuts: vi.fn(),
    onShortcut: () => () => {}
  }

  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { browserWorkspace: api, windowRelay }
  window.history.replaceState({}, '', `/?win=browser&tab=url:seed&browserWindow=${seed.id}#/`)
  const { BrowserPopoutShell } = await import('./browser-popout-shell')
  let rendered!: ReturnType<typeof render>
  await act(async () => {
    rendered = render(<BrowserPopoutShell />)
  })

  return { rendered, runtime, id: seed.id, second, deliveredReplies }
}

describe('multi-tab detached workspace handoff', () => {
  it('renews the visible guest selection intent without letting hidden guests or metadata select', async () => {
    const { rendered, runtime, id, second } = await renderFreshWorkspace()
    const guests = [...rendered.container.querySelectorAll('webview')]
    const initial = runtime.snapshots(2)[0]!

    const interact = (guest: Element) => guest.dispatchEvent(
      Object.assign(new Event('ipc-message'), { channel: 'preview-guest-interaction', args: [] })
    )

    await act(async () => {
      guests[1]!.dispatchEvent(new Event('focus'))
      guests[1]!.dispatchEvent(new Event('page-title-updated'))
      guests[1]!.dispatchEvent(Object.assign(new Event('did-navigate'), { url: 'https://example.test/passive' }))
      interact(guests[0]!)
    })
    expect(runtime.snapshots(2)[0]!.selectionIntentVersion).toBe(initial.selectionIntentVersion)
    expect(runtime.snapshots(2)[0]!.activeTabId).toBe(second)

    await act(async () => { interact(guests[1]!) })
    const selected = runtime.snapshots(2)[0]!
    expect(selected.activeTabId).toBe(second)
    expect(selected.selectionIntentVersion).toBe(initial.selectionIntentVersion + 1)
    expect(selected.selectionVersion).toBe(initial.selectionVersion)
    await act(async () => { interact(guests[1]!) })
    expect(runtime.snapshots(2)[0]!.selectionIntentVersion).toBe(selected.selectionIntentVersion + 1)
    expect(runtime.snapshots(2)[0]!.selectionVersion).toBe(selected.selectionVersion)
    expect([...rendered.container.querySelectorAll('webview')]).toEqual(guests)

    await act(async () => { runtime.command(2, id, { kind: 'close', tabId: second }) })
    const closed = runtime.snapshots(2)[0]!
    await act(async () => { interact(guests[1]!) })
    expect(runtime.snapshots(2)[0]!.selectionIntentVersion).toBe(closed.selectionIntentVersion)
    expect(runtime.snapshots(2)[0]!.activeTabId).toBe('url:seed')
  })

  it.each(['click', 'type', 'drag'])('returns the %s effect through the relay when native input renews the active guest mid-request', async kind => {
    const { rendered, runtime, second, deliveredReplies } = await renderFreshWorkspace()
    const { requestPopoutPreviewAct } = await import('./right-rail/preview-popout-bridge')
    const { selectedPopoutTarget, validPopoutTarget } = await import('@/store/browser-workspaces')
    const initial = runtime.snapshots(2)[0]!
    const captured = selectedPopoutTarget(initial.owner.conversation)!
    const guest = rendered.container.querySelectorAll('webview')[1]!
    const effects = { downs: 0, ups: 0, value: '' }
    let renewals = 0

    // Electron is the boundary double, not the action/guard/relay. Each native
    // pointer/key down emits the preload notice *during* the real driven action.
    Object.assign(guest, {
      getWebContentsId: () => 71,
      sendInputEvent: (event: PreviewInputEvent) => {
        if (event.type === 'mouseDown' || event.type === 'keyDown') {
          renewals++
          guest.dispatchEvent(Object.assign(new Event('ipc-message'), { channel: 'preview-guest-interaction', args: [] }))
        }

        if (event.type === 'mouseDown') { effects.downs++ }

        if (event.type === 'mouseUp') { effects.ups++ }

        if (event.type === 'char') { effects.value += event.keyCode }
      },
      executeJavaScript: async (code: string) => {
        if (code.includes('"kind":"locate"')) {
          return JSON.stringify({ success: true, point: { x: 40, y: 20 }, acted: 'looking at fixture', typable: true })
        }

        if (code.includes('hermes-focus-probe')) {
          return JSON.stringify({ success: true, focused: effects.downs === 1, tag: 'INPUT' })
        }

        if (code.includes('hasStart')) {
          return JSON.stringify({ success: true, field: true, len: effects.value.length })
        }

        return JSON.stringify({
          success: true,
          hit: effects.downs ? { tag: 'INPUT', trusted: true } : null,
          title: `${effects.downs}/${effects.ups}:${effects.value}`,
          elements: []
        })
      }
    })

    vi.useFakeTimers()

    try {
      let pending!: ReturnType<typeof requestPopoutPreviewAct>
      await act(async () => {
        pending = requestPopoutPreviewAct(kind === 'drag'
          ? { kind, selector: '#fixture', dx: -40.5, dy: 20 }
          : { kind, selector: '#fixture', text: 'abc' }, initial.owner.conversation)
        await vi.advanceTimersByTimeAsync(1_000)
      })
      expect(effects).toEqual({ downs: 1, ups: 1, value: kind === 'type' ? 'abc' : '' })
      expect(renewals).toBe(kind === 'type' ? 5 : 1) // click, select-all and three key downs
      expect(validPopoutTarget(captured)).toBe(true)
      expect(runtime.snapshots(2)[0]!.selectionVersion).toBe(captured.selectionVersion)
      expect(runtime.snapshots(2)[0]!.selectionIntentVersion).toBe(initial.selectionIntentVersion + renewals)
      expect(deliveredReplies).toHaveLength(1)
      expect(deliveredReplies[0]).toMatchObject({ target: captured, kind: 'act', result: { success: true } })
      expect(await pending).toMatchObject({ success: true, title: `1/1:${effects.value}` })
      expect(runtime.snapshots(2)[0]!.activeTabId).toBe(second)
      expect(rendered.container.querySelectorAll('webview')[1]).toBe(guest)
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it.each(['switch-away-and-back', 'close', 'dock'] as const)('stops queued native typing when %s retires the captured target', async transition => {
    const { rendered, runtime, id, second, deliveredReplies } = await renderFreshWorkspace()
    const { requestPopoutPreviewAct } = await import('./right-rail/preview-popout-bridge')
    const { selectedPopoutTarget, validPopoutTarget } = await import('@/store/browser-workspaces')
    const owner = runtime.snapshots(2)[0]!.owner.conversation!
    const captured = selectedPopoutTarget(owner)!
    const guest = rendered.container.querySelectorAll('webview')[1]!
    let value = ''

    Object.assign(guest, {
      getWebContentsId: () => 71,
      sendInputEvent: (event: PreviewInputEvent) => {
        if (event.type !== 'char') { return }
        value += event.keyCode

        if (value.length !== 1) { return }

        if (transition === 'switch-away-and-back') {
          runtime.command(2, id, { kind: 'select', tabId: 'url:seed' })
          runtime.command(2, id, { kind: 'select', tabId: second })
        } else {
          runtime.command(2, id, { kind: transition, tabId: second })
        }
      },
      executeJavaScript: async (code: string) => {
        if (code.includes('"kind":"locate"')) {
          return JSON.stringify({ success: true, point: { x: 40, y: 20 }, typable: true })
        }

        if (code.includes('hermes-focus-probe')) {
          return JSON.stringify({ success: true, focused: true, tag: 'INPUT' })
        }

        return JSON.stringify({ success: true, field: true, len: value.length })
      }
    })

    vi.useFakeTimers()

    try {
      let pending!: ReturnType<typeof requestPopoutPreviewAct>
      await act(async () => {
        pending = requestPopoutPreviewAct({ kind: 'type', selector: '#fixture', text: 'abc' }, owner)
        await vi.advanceTimersByTimeAsync(1_000)
      })
      expect(value).toBe('a')
      expect(validPopoutTarget(captured)).toBe(false)
      expect(deliveredReplies).toEqual([])
      await act(async () => { await vi.advanceTimersByTimeAsync(20_000) })
      expect(await pending).toMatchObject({ success: false, error: expect.stringContaining('not redirected') })
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it('hydrates the exact ordered workspace without changing shared persisted preview buckets', async () => {
    const before = JSON.stringify({ other: [tabRow('url:stranger', 'https://stranger.test')] })
    window.localStorage.setItem(TABS_KEY, before)
    const { rendered } = await renderFreshWorkspace()
    expect([...rendered.container.querySelectorAll('webview')].map(guest => guest.getAttribute('src'))).toEqual([
      'https://example.test/a',
      'https://example.test/b'
    ])
    expect(window.localStorage.getItem(TABS_KEY)).toBe(before)
    expect(rendered.container.querySelectorAll('[role="tab"]')).toHaveLength(2)
    expect(rendered.container.querySelectorAll('[data-pane-hidden][inert]')).toHaveLength(1)
  })

  it('keeps both guest elements mounted across selection and retains the window after seed closure', async () => {
    const { rendered, runtime, id, second } = await renderFreshWorkspace()
    const guests = [...rendered.container.querySelectorAll('webview')]
    await act(async () => {
      runtime.command(2, id, { kind: 'select', tabId: 'url:seed' })
    })
    expect([...rendered.container.querySelectorAll('webview')]).toEqual(guests)
    await act(async () => {
      runtime.command(2, id, { kind: 'select', tabId: second })
    })
    expect([...rendered.container.querySelectorAll('webview')]).toEqual(guests)
    await act(async () => {
      runtime.command(2, id, { kind: 'close', tabId: 'url:seed' })
    })
    expect(rendered.container.querySelector('webview')).toBe(guests[1])
    expect(runtime.snapshots(2)[0]!.id).toBe(id)
    await act(async () => {
      fireEvent.click(rendered.getByRole('button', { name: 'New browser tab' }))
    })
    expect(runtime.snapshots(2)[0]!.tabs).toHaveLength(2)
    expect(rendered.container.querySelectorAll('webview')).toHaveLength(2)
    await waitFor(() => expect(rendered.container.ownerDocument.activeElement?.getAttribute('inputmode')).toBe('url'))
  })

  it('pop-in returns only its selected tab and stale hydration cannot overwrite a newer event', async () => {
    const { rendered, runtime, id, second } = await renderFreshWorkspace()
    const old = runtime.snapshots(2)[0]!
    const { receiveBrowserWorkspace } = await import('@/store/browser-workspaces')
    await act(async () => {
      fireEvent.click(rendered.container.querySelector('[data-browser-active] [aria-label="Pop in"]')!)
    })
    expect(runtime.snapshots(2)[0]!.docked.map(tab => tab.id)).toEqual([second])
    expect(rendered.container.querySelectorAll('webview')).toHaveLength(1)
    await act(async () => {
      receiveBrowserWorkspace(old)
    })
    expect(rendered.container.querySelectorAll('webview')).toHaveLength(1)
    expect(runtime.snapshots(2)[0]!.id).toBe(id)
  })
})

describe('a fresh renderer adopts stored tabs without clobbering them', () => {
  // Same bug family, store half: the scoped-tabs subscribe fires on module
  // init, and echoing the just-read (empty) view back over storage wiped the
  // record before any adoption could read it.
  it('seeds the default bucket into the view at boot', async () => {
    window.localStorage.setItem(
      TABS_KEY,
      JSON.stringify({ default: [tabRow('url:boot-1', 'http://127.0.0.1:9119/kanban')] })
    )

    const { $previewTabs } = await import('@/store/preview')

    expect($previewTabs.get().map(tab => tab.id)).toEqual(['url:boot-1'])
    expect(window.localStorage.getItem(TABS_KEY)).not.toBeNull()
  })

  it('does not wipe a legacy single-array store at boot', async () => {
    window.localStorage.setItem(TABS_KEY, JSON.stringify([tabRow('url:boot-legacy-1', 'http://127.0.0.1:9119/kanban')]))

    const { adoptPersistedBrowserTab, $previewTabs } = await import('@/store/preview')
    adoptPersistedBrowserTab('url:boot-legacy-1')

    expect($previewTabs.get().map(tab => tab.id)).toEqual(['url:boot-legacy-1'])
    expect(storedBuckets().default?.map(tab => tab.id)).toEqual(['url:boot-legacy-1'])
  })
})
