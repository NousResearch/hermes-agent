import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $browserTabs,
  type BrowserTabId,
  clearBrowserTabs,
  createBrowserTab,
  getBrowserConsoleEntries,
  getBrowserNetworkEvents,
  getBrowserScreenshotEntries,
  setBrowserEnabled,
  updateBrowserTab
} from '@/store/browser'
import { runBrowserBridgeCommand } from '@/store/browser-bridge'

import { BrowserPane } from './browser-pane'

function BrowserPaneHarness({ tabId }: { tabId: BrowserTabId }) {
  const tab = useStore($browserTabs).find(candidate => candidate.id === tabId)

  if (!tab) {
    return null
  }

  return <BrowserPane tab={tab} />
}

describe('BrowserPane', () => {
  beforeEach(() => {
    window.localStorage.clear()
    clearBrowserTabs()
    setBrowserEnabled(true)
  })

  afterEach(() => {
    cleanup()
    clearBrowserTabs()
    setBrowserEnabled(false)
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it('renders browser chrome and an isolated sandboxed webview for the tab', () => {
    const tab = createBrowserTab({ profile: 'default', sessionId: 'session-1', url: 'https://example.com/app' })
    const rendered = render(<BrowserPane tab={tab} />)
    const webview = rendered.container.querySelector('webview')

    expect((screen.getByRole('textbox', { name: 'Browser URL' }) as HTMLInputElement).value).toBe(
      'https://example.com/app'
    )
    expect((screen.getByRole('button', { name: 'Back' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Forward' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByRole('button', { name: 'Bind agent to visible browser' })).toBeDefined()
    expect(webview).toBeInstanceOf(HTMLElement)
    expect(webview?.getAttribute('src')).toBe('https://example.com/app')
    expect(webview?.getAttribute('partition')).toMatch(/^persist:hermes-browser:/)
    expect(webview?.getAttribute('webpreferences')).toBe('contextIsolation=yes,nodeIntegration=no,sandbox=yes')
  })

  it('forces BrowserPane webviews onto Hermes browser partitions even when persisted state is stale', () => {
    const tab = createBrowserTab({ partition: 'persist:evil', sessionId: 'session-1', url: 'https://example.com/app' })
    const rendered = render(<BrowserPane tab={{ ...tab, partition: 'persist:evil' }} />)
    const webview = rendered.container.querySelector('webview')

    expect(tab.partition).toBe('persist:hermes-browser:default:session-1')
    expect(webview?.getAttribute('partition')).toBe('persist:hermes-browser:default:session-1')
    expect(webview?.getAttribute('partition')).not.toBe('persist:evil')
  })

  it('normalizes unsafe address-bar schemes to safe HTTPS search navigation', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })
    const rendered = render(<BrowserPaneHarness tabId={tab.id} />)
    const input = screen.getByRole('textbox', { name: 'Browser URL' }) as HTMLInputElement
    const form = input.closest('form') as HTMLFormElement
    const webview = rendered.container.querySelector('webview')

    fireEvent.change(input, { target: { value: 'file:///etc/passwd' } })
    fireEvent.submit(form)

    expect(webview?.getAttribute('src')).toMatch(/^https:\/\/www\.google\.com\/search\?q=file%3A%2F%2F%2Fetc%2Fpasswd$/)
  })

  it('preserves the live webview node when the same tab receives url prop updates', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })
    const rendered = render(<BrowserPane tab={tab} />)
    const webview = rendered.container.querySelector('webview')

    rendered.rerender(<BrowserPane tab={{ ...tab, updatedAt: tab.updatedAt + 1, url: 'https://example.com/next' }} />)

    expect(rendered.container.querySelector('webview')).toBe(webview)
    expect((screen.getByRole('textbox', { name: 'Browser URL' }) as HTMLInputElement).value).toBe(
      'https://example.com/next'
    )
  })

  it('preserves the live webview node when navigation updates tab url state', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })
    const rendered = render(<BrowserPaneHarness tabId={tab.id} />)
    const webview = rendered.container.querySelector('webview') as HTMLElement & { getURL?: () => string }
    const event = new Event('did-navigate') as Event & { url?: string }

    event.url = 'https://example.com/next'

    act(() => {
      webview.dispatchEvent(event)
    })

    expect($browserTabs.get()[0]?.url).toBe('https://example.com/next')
    expect(rendered.container.querySelector('webview')).toBe(webview)
    expect((screen.getByRole('textbox', { name: 'Browser URL' }) as HTMLInputElement).value).toBe(
      'https://example.com/next'
    )
  })

  it('lets the user grant observe then confirmed control consent from the browser chrome', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', url: 'https://example.com/app' })
    render(<BrowserPaneHarness tabId={tab.id} />)
    const bindButton = screen.getByRole('button', { name: 'Bind agent to visible browser' })

    fireEvent.click(bindButton)

    expect($browserTabs.get()[0]?.controlMode).toBe('observe')
    expect(bindButton.textContent).toBe('Observe')

    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)

    fireEvent.click(bindButton)

    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('control'))
    expect($browserTabs.get()[0]?.controlMode).toBe('observe')

    confirm.mockReturnValue(true)
    fireEvent.click(bindButton)

    expect($browserTabs.get()[0]?.controlMode).toBe('control')
    expect(bindButton.textContent).toBe('Control')

    fireEvent.click(bindButton)

    expect($browserTabs.get()[0]?.controlMode).toBe('paused')
    expect(bindButton.textContent).toBe('Paused')
  })

  it('registers the mounted webview for consent-gated bridge commands', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Example', url: 'https://example.com/app' })
    const rendered = render(<BrowserPaneHarness tabId={tab.id} />)
    const webview = rendered.container.querySelector('webview') as HTMLElement & { getTitle?: () => string; getURL?: () => string }

    webview.getTitle = () => 'Live Example'
    webview.getURL = () => 'https://example.com/live'
    updateBrowserTab(tab.id, { controlMode: 'observe' })

    await expect(runBrowserBridgeCommand(tab.id, 'getState')).resolves.toMatchObject({
      title: 'Live Example',
      url: 'https://example.com/live'
    })
  })

  it('guards webview methods until dom-ready so pre-ready events do not crash the pane', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Example', url: 'https://example.com/app' })
    const rendered = render(<BrowserPaneHarness tabId={tab.id} />)

    const webview = rendered.container.querySelector('webview') as HTMLElement & {
      canGoBack?: () => boolean
      canGoForward?: () => boolean
      getTitle?: () => string
      getURL?: () => string
    }

    // A real Electron <webview> throws on these until the guest emits `dom-ready`.
    let domReady = false

    const notReady = (): never => {
      throw new Error('The WebView must be attached to the DOM and the dom-ready event emitted before this method can be called.')
    }

    webview.canGoBack = () => (domReady ? false : notReady())
    webview.canGoForward = () => (domReady ? false : notReady())
    webview.getTitle = () => (domReady ? 'Example' : notReady())
    webview.getURL = () => (domReady ? 'https://example.com/app' : notReady())

    // Navigation events fire BEFORE dom-ready and read those accessors — must not throw.
    expect(() => {
      act(() => {
        webview.dispatchEvent(new Event('did-start-navigation'))
        webview.dispatchEvent(new Event('did-navigate'))
        webview.dispatchEvent(new Event('did-stop-loading'))
      })
    }).not.toThrow()

    // Re-running the URL-sync effect (a tab URL change) also reads getURL pre-ready — must not throw.
    expect(() => {
      act(() => {
        updateBrowserTab(tab.id, { url: 'https://example.com/other' })
      })
    }).not.toThrow()

    // After dom-ready the real navigation state syncs through the same guarded path.
    domReady = true
    act(() => {
      webview.dispatchEvent(new Event('dom-ready'))
    })
    expect($browserTabs.get().find(candidate => candidate.id === tab.id)?.title).toBe('Example')
  })

  it('opens the visible browser tab URL externally from the chrome', () => {
    const openExternal = vi.fn(async () => undefined)

    ;(window as unknown as { hermesDesktop?: { openExternal: (url: string) => Promise<void> } }).hermesDesktop = {
      openExternal
    }
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Example', url: 'https://example.com/app' })

    render(<BrowserPaneHarness tabId={tab.id} />)
    fireEvent.click(screen.getByRole('button', { name: 'Open browser tab externally' }))

    expect(openExternal).toHaveBeenCalledWith('https://example.com/app')
  })

  it('captures visible webview console and network events and exposes clearable panels', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Example', url: 'https://example.com/app' })
    const rendered = render(<BrowserPaneHarness tabId={tab.id} />)
    const webview = rendered.container.querySelector('webview') as HTMLElement & { getURL?: () => string }

    webview.getURL = () => 'https://example.com/live'

    const consoleEvent = new Event('console-message') as Event & {
      level?: number
      line?: number
      message?: string
      sourceId?: string
    }

    consoleEvent.level = 1
    consoleEvent.message = 'careful there'
    consoleEvent.line = 42
    consoleEvent.sourceId = 'https://example.com/app.js'

    const jsErrorEvent = new Event('console-message') as Event & {
      level?: number
      line?: number
      message?: string
      sourceId?: string
    }

    jsErrorEvent.level = 3
    jsErrorEvent.message = 'Uncaught TypeError: boom'
    jsErrorEvent.line = 99
    jsErrorEvent.sourceId = 'https://example.com/error.js'

    const navigationEvent = new Event('did-navigate') as Event & { url?: string }
    navigationEvent.url = 'https://example.com/live'

    const requestEvent = new Event('did-start-navigation') as Event & { method?: string; url?: string }
    requestEvent.method = 'POST'
    requestEvent.url = 'https://example.com/api'

    const responseEvent = new Event('did-get-response-details') as Event & { body?: string; httpResponseCode?: number; method?: string; newURL?: string }
    responseEvent.body = 'secret raw response body must not be retained'
    responseEvent.httpResponseCode = 201
    responseEvent.method = 'POST'
    responseEvent.newURL = 'https://example.com/api'

    const loadErrorEvent = new Event('did-fail-load') as Event & {
      errorCode?: number
      errorDescription?: string
      validatedURL?: string
    }

    loadErrorEvent.errorCode = -105
    loadErrorEvent.errorDescription = 'ERR_NAME_NOT_RESOLVED'
    loadErrorEvent.validatedURL = 'https://example.invalid/api'

    const provisionalErrorEvent = new Event('did-fail-provisional-load') as Event & {
      errorCode?: number
      errorDescription?: string
      validatedURL?: string
    }

    provisionalErrorEvent.errorCode = -7
    provisionalErrorEvent.errorDescription = 'ERR_TIMED_OUT'
    provisionalErrorEvent.validatedURL = 'http://localhost:5173/'

    act(() => {
      webview.dispatchEvent(consoleEvent)
      webview.dispatchEvent(jsErrorEvent)
      webview.dispatchEvent(requestEvent)
      webview.dispatchEvent(responseEvent)
      webview.dispatchEvent(navigationEvent)
      webview.dispatchEvent(loadErrorEvent)
      webview.dispatchEvent(provisionalErrorEvent)
    })

    expect(getBrowserConsoleEntries(tab.id)).toEqual([
      expect.objectContaining({ level: 'warn', line: 42, message: 'careful there', source: 'console' }),
      expect.objectContaining({ level: 'error', line: 99, message: 'Uncaught TypeError: boom', source: 'exception' })
    ])
    const networkEvents = getBrowserNetworkEvents(tab.id)
    expect(networkEvents).toEqual([
      expect.objectContaining({ method: 'POST', type: 'request', url: 'https://example.com/api' }),
      expect.objectContaining({ method: 'POST', status: 201, type: 'response', url: 'https://example.com/api' }),
      expect.objectContaining({ type: 'navigation', url: 'https://example.com/live' }),
      expect.objectContaining({ error: 'ERR_NAME_NOT_RESOLVED', status: -105, type: 'load-error', url: 'https://example.invalid/api' }),
      expect.objectContaining({ error: 'ERR_TIMED_OUT', status: -7, type: 'load-error', url: 'http://localhost:5173/' })
    ])
    expect(networkEvents.some(event => 'body' in (event as unknown as Record<string, unknown>))).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: /Console 2/i }))
    expect(screen.getByText('careful there')).toBeDefined()
    fireEvent.click(screen.getByRole('button', { name: /Clear console/i }))
    expect(getBrowserConsoleEntries(tab.id)).toEqual([])

    fireEvent.click(screen.getByRole('button', { name: /Network 5/i }))
    expect(screen.getAllByText(/ERR_NAME_NOT_RESOLVED/).length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('button', { name: /Clear network/i }))
    expect(getBrowserNetworkEvents(tab.id)).toEqual([])
  })

  it('shows screenshot history and action timeline panels from BrowserPane chrome and bridge commands', async () => {
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Example', url: 'https://example.com/app' })
    const rendered = render(<BrowserPaneHarness tabId={tab.id} />)

    const webview = rendered.container.querySelector('webview') as HTMLElement & {
      capturePage?: () => Promise<string>
      getTitle?: () => string
      getURL?: () => string
    }

    webview.capturePage = async () => 'data:image/png;base64,abc123'
    webview.getTitle = () => 'Captured App'
    webview.getURL = () => 'https://example.com/captured'

    fireEvent.click(screen.getByRole('button', { name: 'Capture browser screenshot' }))
    fireEvent.click(screen.getByRole('button', { name: /Screenshots/i }))
    await screen.findByText('Captured App')

    expect(getBrowserScreenshotEntries(tab.id)).toEqual([
      expect.objectContaining({ title: 'Captured App', url: 'https://example.com/captured' })
    ])

    updateBrowserTab(tab.id, { controlMode: 'control' })
    await runBrowserBridgeCommand(tab.id, 'press', { key: 'Enter' })

    fireEvent.click(screen.getByRole('button', { name: /Timeline/i }))
    expect(screen.getByText('press · success')).toBeDefined()
    expect(screen.getByText(/Enter/)).toBeDefined()
  })

  it('shows the inspector and accessibility audit affordance panels', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Example App', url: 'https://example.com/app' })

    render(<BrowserPaneHarness tabId={tab.id} />)

    fireEvent.click(screen.getByRole('button', { name: 'Inspect' }))
    expect(screen.getByText('No element selected')).toBeDefined()
    expect(screen.getByRole('button', { name: 'Components' })).toBeDefined()
    expect(screen.getByRole('button', { name: 'CSS' })).toBeDefined()

    fireEvent.click(screen.getByRole('button', { name: 'A11y' }))
    expect(screen.getByText('Accessibility audit')).toBeDefined()
    expect(screen.getByText(/browser_accessibility_audit/)).toBeDefined()
    expect(screen.getByText(/missing alt text/)).toBeDefined()
  })

  it('surfaces blocked schemes, failed loads, certificate failures, crashes, and agent errors', () => {
    const tab = createBrowserTab({ sessionId: 'session-1', title: 'Example', url: 'https://example.com/app' })
    const rendered = render(<BrowserPaneHarness tabId={tab.id} />)
    const webview = rendered.container.querySelector('webview') as HTMLElement

    const blocked = new Event('did-fail-load') as Event & {
      errorCode?: number
      errorDescription?: string
      validatedURL?: string
    }

    blocked.errorCode = -301
    blocked.errorDescription = 'ERR_DISALLOWED_URL_SCHEME'
    blocked.validatedURL = 'file:///etc/passwd'

    act(() => {
      webview.dispatchEvent(blocked)
    })

    expect(screen.getByText(/Blocked scheme/i)).toBeDefined()
    expect(screen.getByText(/file:\/\/\/etc\/passwd/)).toBeDefined()

    const timeout = new Event('did-fail-provisional-load') as Event & {
      errorCode?: number
      errorDescription?: string
      validatedURL?: string
    }

    timeout.errorCode = -7
    timeout.errorDescription = 'ERR_TIMED_OUT'
    timeout.validatedURL = 'http://localhost:5173/'

    act(() => {
      webview.dispatchEvent(timeout)
    })

    expect(screen.getByText(/Load timed out/i)).toBeDefined()
    expect(screen.getByText(/localhost:5173/)).toBeDefined()

    const cert = new Event('certificate-error') as Event & {
      error?: string
      url?: string
    }

    cert.error = 'ERR_CERT_AUTHORITY_INVALID'
    cert.url = 'https://self-signed.badssl.com/'

    act(() => {
      webview.dispatchEvent(cert)
    })

    expect(screen.getByText(/Certificate error/i)).toBeDefined()
    expect(screen.getByText(/ERR_CERT_AUTHORITY_INVALID/)).toBeDefined()

    act(() => {
      webview.dispatchEvent(new Event('render-process-gone'))
    })

    expect(screen.getByText(/Browser renderer crashed/i)).toBeDefined()

    act(() => {
      updateBrowserTab(tab.id, { agentError: 'Agent tried to click @e99 but no matching element exists.' })
    })

    expect(screen.getByText(/Agent browser error/i)).toBeDefined()
    expect(screen.getByText(/@e99/)).toBeDefined()
  })
})
