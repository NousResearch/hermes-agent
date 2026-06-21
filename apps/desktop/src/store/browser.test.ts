import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $browserNavRequest,
  $browserState,
  closeBrowser,
  DEFAULT_BROWSER_URL,
  getBrowserWebview,
  navigateBrowser,
  normalizeBrowserInput,
  openBrowser,
  openCurrentInSystemBrowser,
  registerBrowserWebview,
  reloadBrowser,
  setBrowserNavState
} from './browser'
import { $rightRailActiveTabId, RIGHT_RAIL_BROWSER_TAB_ID, RIGHT_RAIL_PREVIEW_TAB_ID } from './layout'

afterEach(() => {
  // Reset stores between tests.
  closeBrowser()
  $browserState.set({
    canGoBack: false,
    canGoForward: false,
    loading: false,
    loadError: null,
    open: false,
    title: '',
    url: ''
  })
  $browserNavRequest.set({ nonce: 0, url: '' })
  $rightRailActiveTabId.set(RIGHT_RAIL_PREVIEW_TAB_ID)
})

describe('normalizeBrowserInput', () => {
  it('passes through full http(s) URLs', () => {
    expect(normalizeBrowserInput('https://example.com/x')).toBe('https://example.com/x')
    expect(normalizeBrowserInput('http://localhost:3000')).toBe('http://localhost:3000')
  })

  it('passes through file/about/data URLs', () => {
    expect(normalizeBrowserInput('file:///tmp/a.html')).toBe('file:///tmp/a.html')
    expect(normalizeBrowserInput('about:blank')).toBe('about:blank')
  })

  it('prefixes https for a bare host', () => {
    expect(normalizeBrowserInput('example.com')).toBe('https://example.com')
    expect(normalizeBrowserInput('sub.example.com/path')).toBe('https://sub.example.com/path')
  })

  it('treats localhost[:port] as a host', () => {
    expect(normalizeBrowserInput('localhost:8080')).toBe('https://localhost:8080')
  })

  it('turns a search phrase into a DuckDuckGo query', () => {
    expect(normalizeBrowserInput('how to center a div')).toBe(
      'https://duckduckgo.com/?q=how%20to%20center%20a%20div'
    )
  })

  it('treats a single word (no dot) as a search', () => {
    expect(normalizeBrowserInput('hermes')).toBe('https://duckduckgo.com/?q=hermes')
  })

  it('returns empty for blank input', () => {
    expect(normalizeBrowserInput('   ')).toBe('')
  })
})

describe('openBrowser', () => {
  it('opens the tab, selects it, and emits a nav request to the default URL', () => {
    const url = openBrowser()

    expect(url).toBe(DEFAULT_BROWSER_URL)
    expect($browserState.get().open).toBe(true)
    expect($browserState.get().url).toBe(DEFAULT_BROWSER_URL)
    expect($browserNavRequest.get().url).toBe(DEFAULT_BROWSER_URL)
    expect($browserNavRequest.get().nonce).toBeGreaterThan(0)
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_BROWSER_TAB_ID)
  })

  it('normalizes an explicit input', () => {
    const url = openBrowser('example.com')

    expect(url).toBe('https://example.com')
    expect($browserState.get().url).toBe('https://example.com')
  })

  it('bumps the nonce on each open so the webview reloads', () => {
    openBrowser('https://a.com')
    const first = $browserNavRequest.get().nonce
    openBrowser('https://b.com')

    expect($browserNavRequest.get().nonce).toBeGreaterThan(first)
    expect($browserNavRequest.get().url).toBe('https://b.com')
  })
})

describe('navigateBrowser', () => {
  it('navigates the open browser to a new address', () => {
    openBrowser('https://a.com')
    navigateBrowser('https://b.com/page')

    expect($browserState.get().url).toBe('https://b.com/page')
    expect($browserNavRequest.get().url).toBe('https://b.com/page')
  })
})

describe('reloadBrowser', () => {
  it('bumps the nonce without changing the URL', () => {
    openBrowser('https://a.com')
    const nonce = $browserNavRequest.get().nonce
    reloadBrowser()

    expect($browserNavRequest.get().nonce).toBe(nonce + 1)
    expect($browserNavRequest.get().url).toBe('https://a.com')
  })

  it('no-ops when there is no URL', () => {
    reloadBrowser()
    expect($browserNavRequest.get().nonce).toBe(0)
  })
})

describe('setBrowserNavState', () => {
  it('patches live nav state without touching open', () => {
    openBrowser('https://a.com')
    setBrowserNavState({ canGoBack: true, loading: true, title: 'A' })

    const state = $browserState.get()
    expect(state.canGoBack).toBe(true)
    expect(state.loading).toBe(true)
    expect(state.title).toBe('A')
    expect(state.open).toBe(true)
  })
})

describe('closeBrowser', () => {
  it('closes the tab but retains the URL for reopen', () => {
    openBrowser('https://a.com')
    closeBrowser()

    expect($browserState.get().open).toBe(false)
    expect($browserState.get().url).toBe('https://a.com')
  })
})

describe('openCurrentInSystemBrowser', () => {
  afterEach(() => {
    delete (globalThis as { window?: unknown }).window
  })

  function installOpenExternal() {
    const openExternal = vi.fn()

    ;(globalThis as unknown as { window: { hermesDesktop: { openExternal: typeof openExternal } } }).window = {
      hermesDesktop: { openExternal }
    } as never

    return openExternal
  }

  it('hands the current http(s) page to the OS browser (the passkey escape hatch)', () => {
    const openExternal = installOpenExternal()
    openBrowser('https://accounts.google.com/signin')

    const handed = openCurrentInSystemBrowser()

    expect(handed).toBe('https://accounts.google.com/signin')
    expect(openExternal).toHaveBeenCalledWith('https://accounts.google.com/signin')
  })

  it('no-ops (and does not call openExternal) when there is no URL', () => {
    const openExternal = installOpenExternal()

    expect(openCurrentInSystemBrowser()).toBe('')
    expect(openExternal).not.toHaveBeenCalled()
  })

  it('refuses non-http(s) URLs so it never hands a file:/about: page to the OS', () => {
    const openExternal = installOpenExternal()
    $browserState.set({ ...$browserState.get(), url: 'about:blank' })

    expect(openCurrentInSystemBrowser()).toBe('')
    expect(openExternal).not.toHaveBeenCalled()
  })
})

describe('webview registry (getBrowserWebview / registerBrowserWebview)', () => {
  afterEach(() => {
    registerBrowserWebview(null)
  })

  it('returns null when no webview is registered', () => {
    expect(getBrowserWebview()).toBeNull()
  })

  it('returns the registered webview element', () => {
    const mockWebview = { executeJavaScript: vi.fn() }
    registerBrowserWebview(mockWebview)

    expect(getBrowserWebview()).toBe(mockWebview)
  })

  it('allows clearing the registry by registering null', () => {
    const mockWebview = { capturePage: vi.fn() }
    registerBrowserWebview(mockWebview)
    expect(getBrowserWebview()).toBe(mockWebview)

    registerBrowserWebview(null)
    expect(getBrowserWebview()).toBeNull()
  })
})
