import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $browserEnabled,
  $browserTabs,
  appendBrowserActionEvent,
  appendBrowserConsoleEntry,
  appendBrowserNetworkEvent,
  appendBrowserScreenshotEntry,
  BROWSER_SCREENSHOT_HISTORY_LIMIT,
  BROWSER_TAB_LIMIT,
  BROWSER_TABS_STORAGE_KEY,
  type BrowserTabId,
  clearBrowserConsoleEntries,
  clearBrowserNetworkEvents,
  clearBrowserTabs,
  closeBrowserTab,
  createBrowserTab,
  getBrowserActionEvents,
  getBrowserConsoleEntries,
  getBrowserNetworkEvents,
  getBrowserScreenshotEntries,
  isBrowserTabId,
  moveBrowserTab,
  openBrowserTab,
  setBrowserEnabled,
  updateBrowserTab
} from './browser'
import { $rightRailActiveTabId, PREVIEW_PANE_ID, RIGHT_RAIL_PREVIEW_TAB_ID } from './layout'
import { $paneOpen } from './panes'

describe('browser store', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearBrowserTabs()
    setBrowserEnabled(false)
  })

  afterEach(() => {
    clearBrowserTabs()
    setBrowserEnabled(false)
    window.localStorage.clear()
  })

  it('creates an additive browser tab without replacing the preview tab id', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    expect(isBrowserTabId(tab.id)).toBe(true)
    expect(tab.id.startsWith('browser:')).toBe(true)
    expect(tab.sessionId).toBe('session-1')
    expect(tab.url).toBe('https://example.com')
    expect(tab.title).toBe('example.com')
    expect(tab.loading).toBe(false)
    expect(tab.controlMode).toBe('idle')
    expect($browserTabs.get()).toEqual([tab])
    expect($rightRailActiveTabId.get()).toBe(tab.id)
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
  })

  it('uses persistent browser partitions scoped to profile and session', () => {
    const first = createBrowserTab({ profile: 'default', sessionId: 'session-1', url: 'https://example.com/one' })
    const second = createBrowserTab({ profile: 'default', sessionId: 'session-1', url: 'https://example.com/two' })
    const otherSession = createBrowserTab({ profile: 'default', sessionId: 'session-2', url: 'https://example.com/three' })
    const otherProfile = createBrowserTab({ profile: 'coder', sessionId: 'session-1', url: 'https://example.com/four' })

    expect(first.partition).toMatch(/^persist:hermes-browser:/)
    expect(second.partition).toBe(first.partition)
    expect(otherSession.partition).not.toBe(first.partition)
    expect(otherProfile.partition).not.toBe(first.partition)
  })

  it('clamps unsafe, legacy, and stale persisted browser partitions to the current scope', async () => {
    vi.resetModules()
    window.localStorage.clear()
    window.localStorage.setItem(
      BROWSER_TABS_STORAGE_KEY,
      JSON.stringify([
        { id: 'browser:persist-legacy', profile: 'default', sessionId: 'session-1', url: 'https://example.com/legacy', partition: 'hermes-browser:old' },
        { id: 'browser:persist-stale', profile: 'default', sessionId: 'session-1', url: 'https://example.com/stale', partition: 'persist:hermes-browser:wrong-scope' },
        { id: 'browser:persist-evil', profile: 'default', sessionId: 'session-1', url: 'https://example.com/evil', partition: 'persist:evil' },
        { id: 'browser:persist-missing', profile: 'coder', sessionId: 'session-2', url: 'https://example.com/missing' }
      ])
    )

    const browserModule = await import('./browser')
    const tabs = browserModule.$browserTabs.get()

    expect(tabs.map(tab => tab.partition)).toEqual([
      tabs[0]?.partition,
      tabs[0]?.partition,
      tabs[0]?.partition,
      tabs[3]?.partition
    ])
    expect(tabs[0]?.partition).toMatch(/^persist:hermes-browser:/)
    expect(tabs[3]?.partition).toMatch(/^persist:hermes-browser:/)
    expect(tabs[3]?.partition).not.toBe(tabs[0]?.partition)
    expect(window.localStorage.getItem(BROWSER_TABS_STORAGE_KEY)).not.toContain('persist:evil')
    expect(window.localStorage.getItem(BROWSER_TABS_STORAGE_KEY)).not.toContain('hermes-browser:old')

    browserModule.clearBrowserTabs()
  })

  it('drops non-Hermes browser partitions while preserving Hermes-scoped partitions', () => {
    const unsafe = createBrowserTab({ partition: 'persist:evil', sessionId: 'session-1', url: 'https://example.com' })
    const safe = createBrowserTab({ partition: 'persist:hermes-browser:default:session-1', sessionId: 'session-1', url: 'https://example.com' })

    expect(unsafe.partition).toBe('persist:hermes-browser:default:session-1')
    expect(safe.partition).toBe('persist:hermes-browser:default:session-1')
  })

  it('opens a browser tab with safe defaults from global UI entrypoints once enabled', () => {
    setBrowserEnabled(true)
    const tab = openBrowserTab({ sessionId: null })

    expect(tab).not.toBeNull()
    expect(tab?.sessionId).toBe('desktop')
    expect(tab?.url).toBe('about:blank')
    expect(tab?.title).toBe('Browser')
    expect($rightRailActiveTabId.get()).toBe(tab?.id)
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
  })

  it('requires an explicit Browser enable switch before global UI entrypoints open tabs', () => {
    expect($browserEnabled.get()).toBe(false)
    expect(openBrowserTab({ sessionId: null })).toBeNull()
    expect($browserTabs.get()).toEqual([])

    setBrowserEnabled(true)

    expect($browserEnabled.get()).toBe(true)
    expect(openBrowserTab({ sessionId: null })?.url).toBe('about:blank')
  })

  it('resets persisted observe/control consent on decode while preserving paused tabs', async () => {
    vi.resetModules()
    window.localStorage.clear()
    window.localStorage.setItem(
      BROWSER_TABS_STORAGE_KEY,
      JSON.stringify([
        { id: 'browser:persist-observe', sessionId: 'session-1', url: 'https://example.com/observe', controlMode: 'observe' },
        { id: 'browser:persist-control', sessionId: 'session-1', url: 'https://example.com/control', controlMode: 'control' },
        { id: 'browser:persist-paused', sessionId: 'session-1', url: 'https://example.com/paused', controlMode: 'paused' },
        { id: 'browser:persist-idle', sessionId: 'session-1', url: 'https://example.com/idle', controlMode: 'idle' }
      ])
    )

    const browserModule = await import('./browser')

    expect(browserModule.$browserTabs.get().map(tab => ({ controlMode: tab.controlMode, url: tab.url }))).toEqual([
      { controlMode: 'idle', url: 'https://example.com/observe' },
      { controlMode: 'idle', url: 'https://example.com/control' },
      { controlMode: 'paused', url: 'https://example.com/paused' },
      { controlMode: 'idle', url: 'https://example.com/idle' }
    ])
    expect(window.localStorage.getItem(BROWSER_TABS_STORAGE_KEY)).not.toContain('"observe"')
    expect(window.localStorage.getItem(BROWSER_TABS_STORAGE_KEY)).not.toContain('"control"')

    browserModule.clearBrowserTabs()
  })

  it('updates mutable tab state while preserving stable identity fields', () => {
    const tab = createBrowserTab({ profile: 'default', sessionId: 'session-1', url: 'https://example.com' })

    updateBrowserTab(tab.id, {
      canGoBack: true,
      canGoForward: true,
      consoleCount: 3,
      controlMode: 'observe',
      id: 'browser:evil' as BrowserTabId,
      loading: true,
      networkCount: 2,
      profile: 'other',
      sessionId: 'other',
      title: 'Example',
      url: 'https://example.com/app'
    })

    expect($browserTabs.get()[0]).toMatchObject({
      id: tab.id,
      canGoBack: true,
      canGoForward: true,
      consoleCount: 3,
      controlMode: 'observe',
      loading: true,
      networkCount: 2,
      profile: 'default',
      sessionId: 'session-1',
      title: 'Example',
      url: 'https://example.com/app'
    })
  })

  it('caps persisted tabs and keeps the newest browser tab active', () => {
    for (let i = 0; i < BROWSER_TAB_LIMIT + 2; i += 1) {
      createBrowserTab({ sessionId: 'session-1', url: `https://example.com/${i}` })
    }

    const tabs = $browserTabs.get()

    expect(tabs).toHaveLength(BROWSER_TAB_LIMIT)
    expect(tabs.map(tab => tab.url)).not.toContain('https://example.com/0')
    expect(tabs.at(-1)?.url).toBe(`https://example.com/${BROWSER_TAB_LIMIT + 1}`)
    expect($rightRailActiveTabId.get()).toBe(tabs.at(-1)?.id)
    expect(window.localStorage.getItem('hermes.desktop.browserTabs.v1')).toContain(`https://example.com/${BROWSER_TAB_LIMIT + 1}`)
  })

  it('reorders browser tabs while preserving active tab selection', () => {
    const first = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/first' })
    const second = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/second' })
    const third = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/third' })

    moveBrowserTab(third.id, 'left')
    expect($browserTabs.get().map(tab => tab.id)).toEqual([first.id, third.id, second.id])
    expect($rightRailActiveTabId.get()).toBe(third.id)

    moveBrowserTab(first.id, 'left')
    expect($browserTabs.get().map(tab => tab.id)).toEqual([first.id, third.id, second.id])

    moveBrowserTab(third.id, 'right')
    expect($browserTabs.get().map(tab => tab.id)).toEqual([first.id, second.id, third.id])
  })

  it('closes browser tabs without clearing preview/file state ownership', () => {
    const first = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/first' })
    const second = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/second' })

    closeBrowserTab(second.id)

    expect($browserTabs.get().map(tab => tab.id)).toEqual([first.id])
    expect($rightRailActiveTabId.get()).toBe(first.id)
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)

    closeBrowserTab(first.id)

    expect($browserTabs.get()).toEqual([])
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_PREVIEW_TAB_ID)
  })

  it('records bounded console and network histories per browser tab', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    appendBrowserConsoleEntry(tab.id, { level: 'log', message: 'hello', source: 'console', url: 'https://example.com' })
    appendBrowserConsoleEntry(tab.id, { level: 'error', message: 'boom', source: 'exception', url: 'https://example.com/app.js' })
    appendBrowserNetworkEvent(tab.id, { method: 'GET', status: 200, type: 'navigation', url: 'https://example.com' })
    appendBrowserNetworkEvent(tab.id, { error: 'ERR_ABORTED', method: 'GET', type: 'load-error', url: 'https://example.com/app.js' })

    expect($browserTabs.get()[0]?.consoleCount).toBe(2)
    expect($browserTabs.get()[0]?.networkCount).toBe(2)
    expect(getBrowserConsoleEntries(tab.id).map(entry => ({ level: entry.level, message: entry.message, source: entry.source }))).toEqual([
      { level: 'log', message: 'hello', source: 'console' },
      { level: 'error', message: 'boom', source: 'exception' }
    ])
    expect(getBrowserNetworkEvents(tab.id).map(event => ({ error: event.error, status: event.status, type: event.type, url: event.url }))).toEqual([
      { error: undefined, status: 200, type: 'navigation', url: 'https://example.com' },
      { error: 'ERR_ABORTED', status: undefined, type: 'load-error', url: 'https://example.com/app.js' }
    ])

    clearBrowserConsoleEntries(tab.id)
    clearBrowserNetworkEvents(tab.id)

    expect(getBrowserConsoleEntries(tab.id)).toEqual([])
    expect(getBrowserNetworkEvents(tab.id)).toEqual([])
    expect($browserTabs.get()[0]?.consoleCount).toBe(0)
    expect($browserTabs.get()[0]?.networkCount).toBe(0)
  })

  it('caps browser observability field sizes before retaining history', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })
    const huge = 'x'.repeat(20_000)

    appendBrowserConsoleEntry(tab.id, {
      level: 'error',
      message: huge,
      source: 'exception',
      sourceId: `https://example.com/${huge}`,
      url: `https://example.com/${huge}`
    })
    appendBrowserNetworkEvent(tab.id, {
      error: huge,
      method: huge,
      type: 'load-error',
      url: `https://example.com/${huge}`
    })

    const [consoleEntry] = getBrowserConsoleEntries(tab.id)
    const [networkEvent] = getBrowserNetworkEvents(tab.id)

    expect(consoleEntry?.message.length).toBeLessThan(4_200)
    expect(consoleEntry?.sourceId?.length).toBeLessThan(4_200)
    expect(consoleEntry?.url?.length).toBeLessThan(4_200)
    expect(networkEvent?.error?.length).toBeLessThan(4_200)
    expect(networkEvent?.method?.length).toBeLessThan(200)
    expect(networkEvent?.url.length).toBeLessThan(4_200)
    expect(consoleEntry?.message).toContain('truncated')
    expect(networkEvent?.url).toContain('truncated')
  })

  it('keeps bounded screenshot history and action timeline per browser tab', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    appendBrowserActionEvent(tab.id, { command: 'navigate', status: 'success', target: 'https://example.com' })
    appendBrowserActionEvent(tab.id, { command: 'clickRef', error: 'missing ref', status: 'error', target: '@e404' })

    for (let i = 0; i < BROWSER_SCREENSHOT_HISTORY_LIMIT + 2; i += 1) {
      appendBrowserScreenshotEntry(tab.id, {
        dataUrl: `data:image/png;base64,${i}`,
        title: `Shot ${i}`,
        url: `https://example.com/${i}`
      })
    }

    expect(getBrowserActionEvents(tab.id).map(event => ({ command: event.command, status: event.status, target: event.target }))).toEqual([
      { command: 'navigate', status: 'success', target: 'https://example.com' },
      { command: 'clickRef', status: 'error', target: '@e404' }
    ])
    expect(getBrowserScreenshotEntries(tab.id)).toHaveLength(BROWSER_SCREENSHOT_HISTORY_LIMIT)
    expect(getBrowserScreenshotEntries(tab.id)[0]?.title).toBe('Shot 2')
    expect($browserTabs.get()[0]).toMatchObject({ actionCount: 2, screenshotCount: BROWSER_SCREENSHOT_HISTORY_LIMIT })
  })

  it('drops browser observability histories when the tab closes', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com' })

    appendBrowserConsoleEntry(tab.id, { level: 'warn', message: 'careful', source: 'console' })
    appendBrowserNetworkEvent(tab.id, { method: 'GET', type: 'navigation', url: 'https://example.com' })
    appendBrowserActionEvent(tab.id, { command: 'snapshot', status: 'success' })
    appendBrowserScreenshotEntry(tab.id, { dataUrl: 'data:image/png;base64,abc', title: 'Shot' })

    closeBrowserTab(tab.id)

    expect(getBrowserActionEvents(tab.id)).toEqual([])
    expect(getBrowserConsoleEntries(tab.id)).toEqual([])
    expect(getBrowserNetworkEvents(tab.id)).toEqual([])
    expect(getBrowserScreenshotEntries(tab.id)).toEqual([])
  })
})
