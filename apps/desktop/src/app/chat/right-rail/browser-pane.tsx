import { useStore } from '@nanostores/react'
import { type FormEvent, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { requestComposerAttachImage, requestComposerFocus } from '@/app/chat/composer/focus'
import { Codicon } from '@/components/ui/codicon'
import {
  appendBrowserConsoleEntry,
  appendBrowserNetworkEvent,
  appendBrowserScreenshotEntry,
  type BrowserActionEvent,
  type BrowserConsoleEntry,
  type BrowserConsoleLevel,
  type BrowserErrorState,
  type BrowserNetworkEvent,
  browserPartitionForTab,
  type BrowserScreenshotEntry,
  type BrowserTabState,
  clearBrowserActionEvents,
  clearBrowserConsoleEntries,
  clearBrowserNetworkEvents,
  clearBrowserScreenshotEntries,
  getBrowserActionEvents,
  getBrowserConsoleEntries,
  getBrowserNetworkEvents,
  getBrowserScreenshotEntries,
  updateBrowserTab
} from '@/store/browser'
import { registerBrowserWebview } from '@/store/browser-bridge'
import { dispatchBrowserGuestEvent, parseGuestSentinel } from '@/store/browser-guest-bus'
import { $browserPickerActive, dropBrowserGuestState, setBrowserPickerActive } from '@/store/browser-guest-state'

import { InspectorPanel } from './inspector-panel'

interface BrowserPaneProps {
  tab: BrowserTabState
}

type BrowserWebview = HTMLElement & {
  canGoBack?: () => boolean
  canGoForward?: () => boolean
  capturePage?: () => Promise<{ toDataURL?: () => string } | string>
  closeDevTools?: () => void
  executeJavaScript?: <T = unknown>(script: string) => Promise<T>
  getTitle?: () => string
  getURL?: () => string
  goBack?: () => void
  goForward?: () => void
  openDevTools?: () => void
  reload?: () => void
  stop?: () => void
}

// An Electron <webview> throws "must be attached + dom-ready" if these methods are
// called before the guest's `dom-ready` event (optional chaining does NOT help — the method
// exists, invoking it throws). Wrap every accessor/mutator so a not-ready (or torn-down) webview
// returns a fallback instead of crashing uncaught into the React root error boundary.
function safeWebviewCall<T, F>(fn: () => T, fallback: F): T | F {
  try {
    return fn()
  } catch {
    return fallback
  }
}

export function BrowserPane({ tab }: BrowserPaneProps) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const latestTabRef = useRef(tab)
  const webviewRef = useRef<BrowserWebview | null>(null)
  const [address, setAddress] = useState(tab.url)
  const [diagnosticsPane, setDiagnosticsPane] = useState<BrowserDiagnosticsPane | null>(null)

  latestTabRef.current = tab

  const tabId = tab.id
  const tabPartition = tab.partition
  const tabUrl = tab.url
  const pickerActive = useStore($browserPickerActive)[tabId] ?? false

  const toggleInspect = () => {
    if (pickerActive) {
      setBrowserPickerActive(tabId, false)

      return
    }

    if (tab.controlMode === 'idle') {
      updateBrowserTab(tabId, { controlMode: 'observe' })
    }

    setDiagnosticsPane('inspect')
    setBrowserPickerActive(tabId, true)
  }

  useEffect(() => {
    setAddress(tabUrl)

    const webview = webviewRef.current

    if (!webview) {
      return
    }

    const currentUrl = safeWebviewCall(() => webview.getURL?.(), undefined)
    const currentSrc = webview.getAttribute('src')

    if (currentUrl !== tabUrl && currentSrc !== tabUrl) {
      webview.setAttribute('src', tabUrl)
    }
  }, [tabUrl])

  useLayoutEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()

    const webview = document.createElement('webview') as BrowserWebview
    webview.className = 'flex h-full w-full flex-1 bg-transparent'
    webview.setAttribute('partition', safeBrowserPartition(tabPartition, latestTabRef.current))
    webview.setAttribute('src', latestTabRef.current.url)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const syncNavigationState = () => {
      const current = latestTabRef.current

      updateBrowserTab(current.id, {
        canGoBack: safeWebviewCall(() => Boolean(webview.canGoBack?.()), false),
        canGoForward: safeWebviewCall(() => Boolean(webview.canGoForward?.()), false),
        loading: false,
        title: safeWebviewCall(() => webview.getTitle?.(), undefined) || current.title,
        url: safeWebviewCall(() => webview.getURL?.(), undefined) || current.url
      })
    }

    const markLoading = () => updateBrowserTab(latestTabRef.current.id, { browserError: undefined, loading: true })
    const markStopped = () => syncNavigationState()

    const syncUrl = (event: Event) => {
      const current = latestTabRef.current
      const detail = event as Event & { url?: string }
      const url = detail.url || safeWebviewCall(() => webview.getURL?.(), undefined) || current.url

      appendBrowserNetworkEvent(current.id, { method: 'GET', type: 'navigation', url })
      updateBrowserTab(current.id, { url })
    }

    const captureConsole = (event: Event) => {
      const current = latestTabRef.current
      const detail = event as BrowserConsoleMessageEvent

      // Guest interactive tooling (picker/tree/design) reports discrete events
      // over a nonce-bound console sentinel. Route + suppress it so it never
      // shows in the console panel. A real page log that merely contains the
      // prefix has no valid nonce → null → falls through to normal capture.
      const guestEvent = parseGuestSentinel(current.id, detail.message)

      if (guestEvent) {
        dispatchBrowserGuestEvent(guestEvent)

        return
      }

      const level = consoleLevelFromEvent(detail.level)

      appendBrowserConsoleEntry(current.id, {
        level,
        line: typeof detail.line === 'number' ? detail.line : undefined,
        message: detail.message || '',
        source: level === 'error' ? 'exception' : 'console',
        sourceId: detail.sourceId,
        url: detail.sourceId || safeWebviewCall(() => webview.getURL?.(), undefined) || current.url
      })
    }

    const captureRequest = (event: Event) => {
      const current = latestTabRef.current
      const detail = event as BrowserRequestEvent
      const url = detail.url || safeWebviewCall(() => webview.getURL?.(), undefined) || current.url

      appendBrowserNetworkEvent(current.id, {
        method: detail.method || 'GET',
        type: 'request',
        url
      })
    }

    const captureResponse = (event: Event) => {
      const current = latestTabRef.current
      const detail = event as BrowserResponseEvent
      const url = detail.newURL || detail.url || detail.originalURL || safeWebviewCall(() => webview.getURL?.(), undefined) || current.url

      const status = typeof detail.statusCode === 'number'
        ? detail.statusCode
        : typeof detail.httpResponseCode === 'number'
          ? detail.httpResponseCode
          : undefined

      appendBrowserNetworkEvent(current.id, {
        method: detail.method || 'GET',
        status,
        type: 'response',
        url
      })
    }

    const captureLoadError = (event: Event) => {
      const current = latestTabRef.current
      const detail = event as BrowserLoadErrorEvent
      const url = detail.validatedURL || detail.url || safeWebviewCall(() => webview.getURL?.(), undefined) || current.url
      const error = detail.errorDescription || (typeof detail.errorCode === 'number' ? `ERR_${detail.errorCode}` : 'Load failed')

      appendBrowserNetworkEvent(current.id, {
        error,
        method: 'GET',
        status: typeof detail.errorCode === 'number' ? detail.errorCode : undefined,
        type: 'load-error',
        url
      })
      updateBrowserTab(current.id, { browserError: browserErrorFromLoadFailure(detail, url, error), loading: false, url })
    }

    const captureCertificateError = (event: Event) => {
      const current = latestTabRef.current
      const detail = event as BrowserCertificateErrorEvent
      const url = detail.url || safeWebviewCall(() => webview.getURL?.(), undefined) || current.url
      const message = detail.error || 'Certificate validation failed'

      updateBrowserTab(current.id, {
        browserError: {
          code: message,
          kind: 'certificate',
          message,
          title: 'Certificate error',
          url
        },
        loading: false,
        url
      })
    }

    const captureRendererGone = () => {
      const current = latestTabRef.current

      updateBrowserTab(current.id, {
        browserError: {
          kind: 'crash',
          message: 'The embedded browser renderer exited unexpectedly. Reload this tab to recover.',
          title: 'Browser renderer crashed',
          url: safeWebviewCall(() => webview.getURL?.(), undefined) || current.url
        },
        crashed: true,
        loading: false
      })

      // Selection / picker / design preview state from the dead page is now stale.
      dropBrowserGuestState(current.id)
    }

    webview.addEventListener('dom-ready', syncNavigationState)
    webview.addEventListener('did-start-loading', markLoading)
    webview.addEventListener('did-stop-loading', markStopped)
    webview.addEventListener('did-start-navigation', captureRequest)
    webview.addEventListener('did-get-response-details', captureResponse)
    webview.addEventListener('did-navigate', syncUrl)
    webview.addEventListener('did-navigate-in-page', syncUrl)
    webview.addEventListener('did-fail-load', captureLoadError)
    webview.addEventListener('did-fail-provisional-load', captureLoadError)
    webview.addEventListener('certificate-error', captureCertificateError)
    webview.addEventListener('render-process-gone', captureRendererGone)
    webview.addEventListener('page-title-updated', syncNavigationState)
    webview.addEventListener('console-message', captureConsole)

    host.appendChild(webview)
    webviewRef.current = webview
    const unregisterBrowserWebview = registerBrowserWebview(tabId, webview)

    return () => {
      webview.removeEventListener('dom-ready', syncNavigationState)
      webview.removeEventListener('did-start-loading', markLoading)
      webview.removeEventListener('did-stop-loading', markStopped)
      webview.removeEventListener('did-start-navigation', captureRequest)
      webview.removeEventListener('did-get-response-details', captureResponse)
      webview.removeEventListener('did-navigate', syncUrl)
      webview.removeEventListener('did-navigate-in-page', syncUrl)
      webview.removeEventListener('did-fail-load', captureLoadError)
      webview.removeEventListener('did-fail-provisional-load', captureLoadError)
      webview.removeEventListener('certificate-error', captureCertificateError)
      webview.removeEventListener('render-process-gone', captureRendererGone)
      webview.removeEventListener('page-title-updated', syncNavigationState)
      webview.removeEventListener('console-message', captureConsole)
      unregisterBrowserWebview()
      webview.remove()
      webviewRef.current = null
    }
  }, [tabId, tabPartition])

  const navigate = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const nextUrl = normalizeAddress(address)

    setAddress(nextUrl)
    updateBrowserTab(tab.id, { url: nextUrl })

    if (webviewRef.current) {
      webviewRef.current.setAttribute('src', nextUrl)
    }
  }

  const grantAgentBinding = () => {
    const current = latestTabRef.current

    if (current.controlMode === 'control') {
      updateBrowserTab(current.id, { controlMode: 'paused' })

      return
    }

    if (current.controlMode === 'observe') {
      const confirmed = globalThis.confirm?.(
        'Allow Hermes agent control of this visible browser tab? The agent may click, type, and navigate this page.'
      )

      if (confirmed) {
        updateBrowserTab(current.id, { controlMode: 'control' })
      }

      return
    }

    updateBrowserTab(current.id, { controlMode: 'observe' })
  }

  const captureScreenshot = async () => {
    const current = latestTabRef.current

    // Don't bake the picker/highlight overlay into the image the agent sees.
    try {
      await webviewRef.current?.executeJavaScript?.(
        "(() => { document.querySelectorAll('[data-hermes-highlight],[data-hermes-overlay]').forEach(el => { el.style.display = 'none' }) })()"
      )
    } catch {
      // Overlay may be absent or the webview not ready — ignore.
    }

    let image: { toDataURL?: () => string } | string | undefined

    try {
      image = await webviewRef.current?.capturePage?.()
    } catch {
      image = undefined
    }

    const dataUrl = typeof image === 'string' ? image : image?.toDataURL?.()

    if (!dataUrl) {
      updateBrowserTab(current.id, {
        browserError: {
          kind: 'agent',
          message: 'This BrowserPane webview does not support screenshot capture.',
          title: 'Screenshot unavailable',
          url: current.url
        }
      })

      return
    }

    appendBrowserScreenshotEntry(current.id, {
      dataUrl,
      title: safeWebviewCall(() => webviewRef.current?.getTitle?.(), undefined) || current.title,
      url: safeWebviewCall(() => webviewRef.current?.getURL?.(), undefined) || current.url
    })

    // Attach the shot straight into the main chat composer and focus it.
    requestComposerAttachImage(dataUrl, { target: 'main' })
    requestComposerFocus('main')
  }

  return (
    <section className="flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-editor-surface-background) text-(--ui-text-tertiary)">
      <form
        className="flex h-9 shrink-0 items-center gap-1 border-b border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background) px-2"
        onSubmit={navigate}
      >
        <button
          aria-label="Back"
          className="grid size-6 place-items-center rounded-md text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-40"
          disabled={!tab.canGoBack}
          onClick={() => safeWebviewCall(() => webviewRef.current?.goBack?.(), undefined)}
          type="button"
        >
          <Codicon name="arrow-left" size="0.8rem" />
        </button>
        <button
          aria-label="Forward"
          className="grid size-6 place-items-center rounded-md text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-40"
          disabled={!tab.canGoForward}
          onClick={() => safeWebviewCall(() => webviewRef.current?.goForward?.(), undefined)}
          type="button"
        >
          <Codicon name="arrow-right" size="0.8rem" />
        </button>
        <button
          aria-label={tab.loading ? 'Stop loading' : 'Reload'}
          className="grid size-6 place-items-center rounded-md text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => (tab.loading ? safeWebviewCall(() => webviewRef.current?.stop?.(), undefined) : safeWebviewCall(() => webviewRef.current?.reload?.(), undefined))}
          type="button"
        >
          <Codicon name={tab.loading ? 'debug-stop' : 'refresh'} size="0.8rem" />
        </button>
        <label className="sr-only" htmlFor={`${tab.id}-address`}>
          Browser URL
        </label>
        <input
          aria-label="Browser URL"
          className="h-6 min-w-0 flex-1 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-input-background) px-2 text-xs text-foreground outline-none focus:border-(--ui-stroke-primary)"
          id={`${tab.id}-address`}
          onChange={event => setAddress(event.currentTarget.value)}
          spellCheck={false}
          value={address}
        />
        <button
          aria-label="Inspect element"
          aria-pressed={pickerActive}
          className="grid size-6 place-items-center rounded-md text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground aria-pressed:bg-(--ui-control-active-background) aria-pressed:text-foreground"
          onClick={toggleInspect}
          title="Inspect element"
          type="button"
        >
          <Codicon name="inspect" />
        </button>
        <button
          aria-label="Capture browser screenshot"
          className="grid size-6 place-items-center rounded-md text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => { void captureScreenshot() }}
          type="button"
        >
          <Codicon name="device-camera" />
        </button>
        <button
          aria-label="Open browser tab externally"
          className="grid size-6 place-items-center rounded-md text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => openExternalBrowserUrl(tab.url)}
          type="button"
        >
          <Codicon name="link-external" size="0.8rem" />
        </button>
        <button
          aria-label="Bind agent to visible browser"
          aria-pressed={tab.controlMode === 'observe' || tab.controlMode === 'control'}
          className="rounded-md border border-(--ui-stroke-tertiary) px-2 py-1 text-[0.65rem] font-medium uppercase tracking-wide text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={grantAgentBinding}
          type="button"
        >
          {tab.controlMode === 'control'
            ? 'Control'
            : tab.controlMode === 'observe'
              ? 'Observe'
              : tab.controlMode === 'paused'
                ? 'Paused'
                : 'Bind agent'}
        </button>
      </form>
      <BrowserErrorBanner tab={tab} />
      <div className="relative min-h-0 flex-1 overflow-hidden bg-transparent" ref={hostRef} />
      <BrowserDiagnosticsPanel
        activePane={diagnosticsPane}
        onClearConsole={() => clearBrowserConsoleEntries(tab.id)}
        onClearNetwork={() => clearBrowserNetworkEvents(tab.id)}
        onTogglePane={pane => setDiagnosticsPane(current => (current === pane ? null : pane))}
        tab={tab}
      />
    </section>
  )
}

type BrowserDiagnosticsPane = 'accessibility' | 'console' | 'inspect' | 'network' | 'screenshots' | 'timeline'

type BrowserConsoleMessageEvent = Event & {
  level?: number | string
  line?: number
  message?: string
  sourceId?: string
}

type BrowserLoadErrorEvent = Event & {
  errorCode?: number
  errorDescription?: string
  url?: string
  validatedURL?: string
}

type BrowserRequestEvent = Event & {
  isMainFrame?: boolean
  method?: string
  url?: string
}

type BrowserResponseEvent = Event & {
  httpResponseCode?: number
  method?: string
  newURL?: string
  originalURL?: string
  statusCode?: number
  url?: string
}

type BrowserCertificateErrorEvent = Event & {
  error?: string
  url?: string
}

function BrowserErrorBanner({ tab }: { tab: BrowserTabState }) {
  if (!tab.browserError && !tab.agentError) {
    return null
  }

  return (
    <div className="shrink-0 border-b border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-100 dark:text-amber-100">
      {tab.browserError ? <BrowserErrorItem error={tab.browserError} /> : null}
      {tab.agentError ? (
        <div className="mt-1 first:mt-0">
          <div className="font-semibold text-amber-50">Agent browser error</div>
          <div className="break-words text-amber-100/85">{tab.agentError}</div>
        </div>
      ) : null}
    </div>
  )
}

function BrowserErrorItem({ error }: { error: BrowserErrorState }) {
  return (
    <div>
      <div className="font-semibold text-amber-50">{error.title}</div>
      <div className="break-words text-amber-100/85">
        {error.message}{error.url ? ` — ${error.url}` : ''}
      </div>
    </div>
  )
}

interface BrowserDiagnosticsPanelProps {
  activePane: BrowserDiagnosticsPane | null
  onClearConsole: () => void
  onClearNetwork: () => void
  onTogglePane: (pane: BrowserDiagnosticsPane) => void
  tab: BrowserTabState
}

function BrowserDiagnosticsPanel({
  activePane,
  onClearConsole,
  onClearNetwork,
  onTogglePane,
  tab
}: BrowserDiagnosticsPanelProps) {
  const consoleEntries = getBrowserConsoleEntries(tab.id)
  const networkEvents = getBrowserNetworkEvents(tab.id)
  const screenshotEntries = getBrowserScreenshotEntries(tab.id)
  const actionEvents = getBrowserActionEvents(tab.id)

  return (
    <div className="shrink-0 border-t border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background)">
      <div className="flex h-8 items-center gap-1 px-2 text-[0.65rem] text-(--ui-text-tertiary)">
        <button
          aria-pressed={activePane === 'console'}
          className="rounded-md border border-(--ui-stroke-tertiary) px-2 py-1 uppercase tracking-wide hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => onTogglePane('console')}
          type="button"
        >
          Console {tab.consoleCount}
        </button>
        <button
          aria-pressed={activePane === 'network'}
          className="rounded-md border border-(--ui-stroke-tertiary) px-2 py-1 uppercase tracking-wide hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => onTogglePane('network')}
          type="button"
        >
          Network {tab.networkCount}
        </button>
        <button
          aria-pressed={activePane === 'screenshots'}
          className="rounded-md border border-(--ui-stroke-tertiary) px-2 py-1 uppercase tracking-wide hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => onTogglePane('screenshots')}
          type="button"
        >
          Screenshots {tab.screenshotCount}
        </button>
        <button
          aria-pressed={activePane === 'timeline'}
          className="rounded-md border border-(--ui-stroke-tertiary) px-2 py-1 uppercase tracking-wide hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => onTogglePane('timeline')}
          type="button"
        >
          Timeline {tab.actionCount}
        </button>
        <button
          aria-pressed={activePane === 'inspect'}
          className="rounded-md border border-(--ui-stroke-tertiary) px-2 py-1 uppercase tracking-wide hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => onTogglePane('inspect')}
          type="button"
        >
          Inspect
        </button>
        <button
          aria-pressed={activePane === 'accessibility'}
          className="rounded-md border border-(--ui-stroke-tertiary) px-2 py-1 uppercase tracking-wide hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={() => onTogglePane('accessibility')}
          type="button"
        >
          A11y
        </button>
      </div>
      {activePane === 'console' ? (
        <DiagnosticsList
          emptyLabel="No console messages captured."
          items={consoleEntries}
          onClear={onClearConsole}
          renderItem={renderConsoleEntry}
          title="Console"
        />
      ) : null}
      {activePane === 'network' ? (
        <DiagnosticsList
          emptyLabel="No network events captured."
          items={networkEvents}
          onClear={onClearNetwork}
          renderItem={renderNetworkEvent}
          title="Network"
        />
      ) : null}
      {activePane === 'screenshots' ? (
        <DiagnosticsList
          emptyLabel="No screenshots captured."
          items={screenshotEntries}
          onClear={() => clearBrowserScreenshotEntries(tab.id)}
          renderItem={renderScreenshotEntry}
          title="Screenshots"
        />
      ) : null}
      {activePane === 'timeline' ? (
        <DiagnosticsList
          emptyLabel="No browser actions recorded."
          items={actionEvents}
          onClear={() => clearBrowserActionEvents(tab.id)}
          renderItem={renderActionEvent}
          title="Timeline"
        />
      ) : null}
      {activePane === 'inspect' ? <InspectorPanel tab={tab} /> : null}
      {activePane === 'accessibility' ? <BrowserAccessibilityPanel /> : null}
    </div>
  )
}

interface DiagnosticsListProps<T> {
  emptyLabel: string
  items: T[]
  onClear: () => void
  renderItem: (item: T) => React.ReactElement
  title: 'Console' | 'Network' | 'Screenshots' | 'Timeline'
}

function DiagnosticsList<T>({ emptyLabel, items, onClear, renderItem, title }: DiagnosticsListProps<T>) {
  return (
    <div className="max-h-44 overflow-auto border-t border-(--ui-stroke-tertiary) px-2 py-2 text-[0.68rem]">
      <div className="mb-2 flex items-center justify-between gap-2 text-(--ui-text-tertiary)">
        <span className="font-medium uppercase tracking-wide">{title}</span>
        <button
          className="rounded-md px-2 py-1 hover:bg-(--ui-control-hover-background) hover:text-foreground"
          onClick={onClear}
          type="button"
        >
          Clear {title.toLowerCase()}
        </button>
      </div>
      {items.length === 0 ? (
        <p className="text-(--ui-text-quaternary)">{emptyLabel}</p>
      ) : (
        <ul className="space-y-1 text-(--ui-text-secondary)">
          {items.map((item, index) => (
            <li className="rounded-md bg-(--ui-editor-surface-background) px-2 py-1" key={index}>
              {renderItem(item)}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function renderConsoleEntry(entry: BrowserConsoleEntry): React.ReactElement {
  return (
    <div className="min-w-0">
      <div className="truncate font-mono text-(--ui-text-primary)">{entry.message || '(empty message)'}</div>
      <div className="truncate text-[0.62rem] uppercase tracking-wide text-(--ui-text-tertiary)">
        {entry.level} · {entry.source}{entry.line ? `:${entry.line}` : ''}{entry.url ? ` · ${entry.url}` : ''}
      </div>
    </div>
  )
}

function renderNetworkEvent(event: BrowserNetworkEvent): React.ReactElement {
  return (
    <div className="min-w-0">
      <div className="truncate font-mono text-(--ui-text-primary)">{event.url || '(unknown url)'}</div>
      <div className="truncate text-[0.62rem] uppercase tracking-wide text-(--ui-text-tertiary)">
        {event.type}{event.method ? ` · ${event.method}` : ''}{typeof event.status === 'number' ? ` · ${event.status}` : ''}{event.error ? ` · ${event.error}` : ''}
      </div>
    </div>
  )
}

function renderScreenshotEntry(entry: BrowserScreenshotEntry): React.ReactElement {
  return (
    <div className="min-w-0">
      <div className="truncate text-(--ui-text-primary)">{entry.title || 'Screenshot'}</div>
      <div className="truncate font-mono text-[0.62rem] text-(--ui-text-tertiary)">{entry.url || 'current page'}</div>
    </div>
  )
}

function renderActionEvent(event: BrowserActionEvent): React.ReactElement {
  return (
    <div className="min-w-0">
      <div className="truncate font-mono text-(--ui-text-primary)">{event.command} · {event.status}</div>
      <div className="truncate text-[0.62rem] text-(--ui-text-tertiary)">
        {event.target ? `${event.target} · ` : ''}{event.error || event.result || 'ok'}
      </div>
    </div>
  )
}

function BrowserAccessibilityPanel(): React.ReactElement {
  return (
    <div className="max-h-44 overflow-auto border-t border-(--ui-stroke-tertiary) px-3 py-2 text-[0.68rem] text-(--ui-text-secondary)">
      <div className="font-medium uppercase tracking-wide text-(--ui-text-tertiary)">Accessibility audit</div>
      <p className="mt-1">
        Run <code>browser_accessibility_audit</code> to expose missing alt text, unnamed controls/links,
        unlabeled form fields, and aria-hidden focus traps to the agent.
      </p>
      <p className="mt-1 text-(--ui-text-tertiary)">
        Findings are returned through the visible BrowserPane command bridge for debugging and code-review workflows.
      </p>
    </div>
  )
}

function browserErrorFromLoadFailure(detail: BrowserLoadErrorEvent, url: string, message: string): BrowserErrorState {
  const blockedScheme = message.includes('ERR_DISALLOWED_URL_SCHEME') || /^(file|ftp|chrome|devtools):/i.test(url)

  if (blockedScheme) {
    return {
      code: detail.errorCode,
      kind: 'blocked-scheme',
      message,
      title: 'Blocked scheme',
      url
    }
  }

  if (/timeout|timed out|ERR_TIMED_OUT/i.test(message)) {
    return {
      code: detail.errorCode,
      kind: 'timeout',
      message,
      title: 'Load timed out',
      url
    }
  }

  return {
    code: detail.errorCode,
    kind: 'load',
    message,
    title: 'Load failed',
    url
  }
}

function consoleLevelFromEvent(level: number | string | undefined): BrowserConsoleLevel {
  if (typeof level === 'string') {
    const normalized = level.toLowerCase()

    if (normalized === 'debug' || normalized === 'error' || normalized === 'info' || normalized === 'warn') {
      return normalized
    }

    if (normalized === 'warning') {
      return 'warn'
    }

    return 'log'
  }

  if (level === 1) {
    return 'warn'
  }

  if (level === 2 || level === 3) {
    return 'error'
  }

  if (level === 4) {
    return 'debug'
  }

  return 'log'
}

function safeBrowserPartition(partition: string | undefined, tab: BrowserTabState): string {
  const canonical = browserPartitionForTab(tab)

  return partition === canonical ? partition : canonical
}

function openExternalBrowserUrl(url: string): void {
  const trimmed = url.trim()

  if (!trimmed) {
    return
  }

  if (window.hermesDesktop?.openExternal) {
    void window.hermesDesktop.openExternal(trimmed)

    return
  }

  window.open(trimmed, '_blank', 'noopener,noreferrer')
}

function normalizeAddress(value: string): string {
  const trimmed = value.trim()

  if (!trimmed) {
    return 'about:blank'
  }

  if (/^(localhost|127(?:\.\d{1,3}){3}|\[[^\]]+\]|[\w.-]+):\d+(?:\/|$)/i.test(trimmed)) {
    return `http://${trimmed}`
  }

  if (!/^[a-zA-Z][a-zA-Z\d+.-]*:/.test(trimmed)) {
    return `https://${trimmed}`
  }

  try {
    const parsed = new URL(trimmed)

    if (parsed.protocol === 'http:' || parsed.protocol === 'https:') {
      return parsed.href
    }

    if (parsed.protocol === 'about:' && parsed.href === 'about:blank') {
      return 'about:blank'
    }
  } catch {
    // Fall through to a safe HTTPS search URL.
  }

  return `https://www.google.com/search?q=${encodeURIComponent(trimmed)}`
}
