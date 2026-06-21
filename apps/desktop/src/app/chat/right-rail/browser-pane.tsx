import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $browserNavRequest,
  $browserState,
  navigateBrowser,
  openCurrentInSystemBrowser,
  reloadBrowser,
  registerBrowserWebview,
  setBrowserNavState
} from '@/store/browser'

// The Electron <webview> nav surface we rely on. Typed locally because the DOM
// lib doesn't ship webview typings and we only touch a small slice.
type BrowserWebview = HTMLElement & {
  canGoBack?: () => boolean
  canGoForward?: () => boolean
  getTitle?: () => string
  getURL?: () => string
  goBack?: () => void
  goForward?: () => void
  loadURL?: (url: string) => Promise<void>
  reload?: () => void
  stop?: () => void
  executeJavaScript?: (code: string) => Promise<unknown>
  capturePage?: () => Promise<{ toDataURL: () => string }>
  sendInputEvent?: (e: object) => void
}

interface NavButtonProps {
  disabled?: boolean
  icon: string
  label: string
  onClick: () => void
}

function NavButton({ disabled, icon, label, onClick }: NavButtonProps) {
  return (
    <Tip label={label}>
      <button
        aria-label={label}
        className={cn(
          'grid size-6 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors',
          disabled
            ? 'pointer-events-none opacity-35'
            : 'hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring'
        )}
        disabled={disabled}
        onClick={onClick}
        type="button"
      >
        <Codicon name={icon} size="0.85rem" />
      </button>
    </Tip>
  )
}

/**
 * Cursor-style in-app browser: a navigable Electron <webview> with a back /
 * forward / reload toolbar and an editable address bar. State lives in the
 * browser store so both the user and the agent (via the gateway browser bridge)
 * drive the same pane. The webview is created imperatively — React can't own a
 * <webview> child without losing its navigation methods across re-renders.
 */
export function BrowserPane() {
  const { t } = useI18n()
  const b = t.browser
  const state = useStore($browserState)
  const navRequest = useStore($browserNavRequest)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const webviewRef = useRef<BrowserWebview | null>(null)
  const lastNonceRef = useRef(navRequest.nonce)
  const [draftUrl, setDraftUrl] = useState(state.url)
  const [editing, setEditing] = useState(false)

  // Mirror the live URL into the address bar unless the user is mid-edit.
  useEffect(() => {
    if (!editing) {
      setDraftUrl(state.url)
    }
  }, [editing, state.url])

  const submitAddress = useCallback(
    (event: React.FormEvent) => {
      event.preventDefault()
      const value = draftUrl.trim()

      if (value) {
        navigateBrowser(value)
      }

      setEditing(false)
    },
    [draftUrl]
  )

  // Create the webview once and wire its navigation lifecycle into the store.
  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()

    const webview = document.createElement('webview') as BrowserWebview
    webview.className = 'flex h-full w-full flex-1 bg-white'
    webview.setAttribute('partition', 'persist:hermes-browser')
    webview.setAttribute('allowpopups', 'true')
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    if (state.url) {
      webview.setAttribute('src', state.url)
    }

    const syncNav = () => {
      setBrowserNavState({
        canGoBack: webview.canGoBack?.() ?? false,
        canGoForward: webview.canGoForward?.() ?? false,
        title: webview.getTitle?.() || '',
        url: webview.getURL?.() || state.url
      })
    }

    const onStart = () => setBrowserNavState({ loadError: null, loading: true })

    const onStop = () => {
      setBrowserNavState({ loading: false })
      syncNav()
    }

    const onNavigate = (event: Event) => {
      const detail = event as Event & { url?: string }

      setBrowserNavState({ loadError: null, url: detail.url || webview.getURL?.() || state.url })
      syncNav()
    }

    const onTitle = (event: Event) => {
      const detail = event as Event & { title?: string }
      setBrowserNavState({ title: detail.title || '' })
    }

    const onFail = (event: Event) => {
      const detail = event as Event & { errorCode?: number; errorDescription?: string; validatedURL?: string }

      // -3 is ABORTED (user navigated away mid-load); not a real failure.
      if (detail.errorCode === -3) {
        return
      }

      setBrowserNavState({
        loading: false,
        loadError: detail.errorDescription || b.unreachable
      })
    }

    webview.addEventListener('did-start-loading', onStart)
    webview.addEventListener('did-stop-loading', onStop)
    webview.addEventListener('did-navigate', onNavigate)
    webview.addEventListener('did-navigate-in-page', onNavigate)
    webview.addEventListener('page-title-updated', onTitle)
    webview.addEventListener('did-fail-load', onFail)

    host.appendChild(webview)
    webviewRef.current = webview
    registerBrowserWebview(webview)

    return () => {
      webview.removeEventListener('did-start-loading', onStart)
      webview.removeEventListener('did-stop-loading', onStop)
      webview.removeEventListener('did-navigate', onNavigate)
      webview.removeEventListener('did-navigate-in-page', onNavigate)
      webview.removeEventListener('page-title-updated', onTitle)
      webview.removeEventListener('did-fail-load', onFail)
      webview.remove()
      webviewRef.current = null
      registerBrowserWebview(null)
    }
    // Created once; navigation is driven by the nav-request effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Drive the webview from one-shot nav requests (address bar, agent, reload).
  useEffect(() => {
    if (navRequest.nonce === lastNonceRef.current) {
      return
    }

    lastNonceRef.current = navRequest.nonce
    const webview = webviewRef.current

    if (!webview || !navRequest.url) {
      return
    }

    // loadURL is the reliable cross-navigation primitive; setAttribute('src')
    // only takes effect on first attach.
    if (webview.loadURL) {
      void webview.loadURL(navRequest.url).catch(() => undefined)
    } else {
      webview.setAttribute('src', navRequest.url)
    }
  }, [navRequest])

  const goBack = useCallback(() => webviewRef.current?.goBack?.(), [])
  const goForward = useCallback(() => webviewRef.current?.goForward?.(), [])

  const reloadOrStop = useCallback(() => {
    if (state.loading) {
      webviewRef.current?.stop?.()
      setBrowserNavState({ loading: false })

      return
    }

    reloadBrowser()
  }, [state.loading])

  return (
    <div className="flex h-full w-full min-w-0 flex-col overflow-hidden bg-(--ui-editor-surface-background)">
      <div className="flex h-9 shrink-0 items-center gap-1 border-b border-(--ui-stroke-tertiary) px-1.5">
        <NavButton disabled={!state.canGoBack} icon="arrow-left" label={b.back} onClick={goBack} />
        <NavButton disabled={!state.canGoForward} icon="arrow-right" label={b.forward} onClick={goForward} />
        <NavButton
          icon={state.loading ? 'close' : 'refresh'}
          label={state.loading ? b.stop : b.reload}
          onClick={reloadOrStop}
        />
        {/* Grouped with the left nav buttons (not at the toolbar's right edge):
            the right edge abuts the file-browser pane's hover-reveal trigger
            strip, so a right-edge button makes hovering it slide the file list
            out instead of showing the button. */}
        <NavButton
          disabled={!/^https?:\/\//i.test(state.url)}
          icon="link-external"
          label={b.openInSystemBrowser}
          onClick={() => openCurrentInSystemBrowser()}
        />
        <form className="min-w-0 flex-1" onSubmit={submitAddress}>
          <input
            aria-label={b.addressBar}
            className="h-6 w-full min-w-0 rounded-md border border-(--ui-stroke-quaternary) bg-(--ui-bg-secondary) px-2 text-[0.7rem] text-foreground outline-none transition-colors focus:border-(--ui-stroke-primary) focus-visible:ring-1 focus-visible:ring-sidebar-ring"
            onBlur={() => setEditing(false)}
            onChange={event => setDraftUrl(event.target.value)}
            onFocus={event => {
              setEditing(true)
              event.target.select()
            }}
            placeholder={b.addressPlaceholder}
            spellCheck={false}
            value={draftUrl}
          />
        </form>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden">
        <div className="absolute inset-0 flex" ref={hostRef} />
        {state.loadError && (
          <div className="absolute inset-0 grid place-items-center bg-(--ui-editor-surface-background) px-6 text-center">
            <div className="max-w-xs">
              <div className="text-[0.78rem] font-semibold text-foreground">{b.failedToLoad}</div>
              <div className="mt-1 break-words text-[0.68rem] text-muted-foreground/75">{state.loadError}</div>
              <button
                className="mt-3 rounded-md border border-(--ui-stroke-quaternary) px-3 py-1 text-[0.7rem] font-medium text-foreground transition-colors hover:bg-(--ui-control-hover-background)"
                onClick={() => reloadBrowser()}
                type="button"
              >
                {b.tryAgain}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
