import { useStore } from '@nanostores/react'
import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react'

import type { SetTitlebarToolGroup, TitlebarTool } from '@/app/shell/titlebar-controls'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { Bug } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $browserCurrentState,
  $browserDriveCommand,
  $browserSessionId,
  normalizeBrowserUrl,
  setBrowserSessionState
} from '@/store/browser'

import { compactUrl } from './preview-console'
import { PreviewEmptyState } from './preview-file'

const AGENT_ACTION_LABELS = {
  goBack: 'Back',
  goForward: 'Forward',
  navigate: 'Navigate',
  open: 'Open',
  reload: 'Reload'
}

type BrowserWebview = HTMLElement & {
  canGoBack?: () => boolean
  canGoForward?: () => boolean
  closeDevTools?: () => void
  getTitle?: () => string
  getURL?: () => string
  goBack?: () => void
  goForward?: () => void
  isDevToolsOpened?: () => boolean
  loadURL?: (url: string) => void
  openDevTools?: () => void
  reload?: () => void
  stop?: () => void
}

interface BrowserPaneProps {
  setTitlebarToolGroup?: SetTitlebarToolGroup
}

interface BrowserLoadErrorState {
  code?: number
  description: string
  url: string
}

const TITLEBAR_GROUP_ID = 'browser'

function browserPartitionForSession(sessionId: string): string {
  return `persist:hermes-browser-${encodeURIComponent(sessionId).replaceAll('%', '_')}`
}

function BrowserLoadError({ error, onRetry }: { error: BrowserLoadErrorState; onRetry: () => void }) {
  return (
    <PreviewEmptyState
      body={
        <>
          <div className="font-mono text-[0.6875rem] text-muted-foreground/90">{compactUrl(error.url)}</div>
          <div className="mt-1 text-[0.6875rem] text-muted-foreground/70">
            {error.description}
            {error.code ? ` (${error.code})` : ''}
          </div>
        </>
      }
      primaryAction={{ label: 'Try again', onClick: onRetry }}
      title="Page failed to load"
    />
  )
}

function navigateWebview(webview: BrowserWebview | null, url: string) {
  if (!webview) {
    return
  }

  if (webview.loadURL) {
    webview.loadURL(url)
  } else {
    webview.setAttribute('src', url)
  }
}

export function BrowserPane({ setTitlebarToolGroup }: BrowserPaneProps) {
  const sessionId = useStore($browserSessionId)
  const browserState = useStore($browserCurrentState)
  const driveCommand = useStore($browserDriveCommand)
  const webviewRef = useRef<BrowserWebview | null>(null)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const lastHandledDriveCommandIdRef = useRef(0)
  const lastRequestedUrlRef = useRef(browserState.url)
  const [addressValue, setAddressValue] = useState(browserState.url)
  const [canGoBack, setCanGoBack] = useState(false)
  const [canGoForward, setCanGoForward] = useState(false)
  const [currentUrl, setCurrentUrl] = useState(browserState.url)
  const [devtoolsOpen, setDevtoolsOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<BrowserLoadErrorState | null>(null)

  const publishBrowserState = useCallback(
    (patch: {
      canGoBack?: boolean
      canGoForward?: boolean
      loading?: boolean
      title?: string
      url?: string
    }) => {
      void window.hermesDesktop?.browser?.updateState?.({ ...patch, sessionId })
    },
    [sessionId]
  )

  const updateNavigationState = useCallback(() => {
    const webview = webviewRef.current
    const nextCanGoBack = Boolean(webview?.canGoBack?.())
    const nextCanGoForward = Boolean(webview?.canGoForward?.())

    setCanGoBack(nextCanGoBack)
    setCanGoForward(nextCanGoForward)
    publishBrowserState({ canGoBack: nextCanGoBack, canGoForward: nextCanGoForward })

    return { canGoBack: nextCanGoBack, canGoForward: nextCanGoForward }
  }, [publishBrowserState])

  const navigateTo = useCallback(
    (rawUrl: string) => {
      const url = normalizeBrowserUrl(rawUrl)

      lastRequestedUrlRef.current = url
      setAddressValue(url)
      setCurrentUrl(url)
      setLoadError(null)
      setBrowserSessionState(sessionId, { url })
      publishBrowserState({ loading: true, url })
      navigateWebview(webviewRef.current, url)
    },
    [publishBrowserState, sessionId]
  )

  const reload = useCallback(() => {
    setLoadError(null)
    publishBrowserState({ loading: true })
    webviewRef.current?.reload?.()
  }, [publishBrowserState])

  const toggleDevTools = useCallback(() => {
    const webview = webviewRef.current

    if (!webview?.openDevTools) {
      return
    }

    if (webview.isDevToolsOpened?.()) {
      webview.closeDevTools?.()
      setDevtoolsOpen(false)

      return
    }

    webview.openDevTools()
    setDevtoolsOpen(true)
  }, [])

  useEffect(() => {
    if (!setTitlebarToolGroup) {
      return
    }

    const tools: TitlebarTool[] = [
      {
        active: devtoolsOpen,
        icon: <Bug />,
        id: `${TITLEBAR_GROUP_ID}-devtools`,
        label: devtoolsOpen ? 'Hide browser DevTools' : 'Open browser DevTools',
        onSelect: toggleDevTools
      }
    ]

    setTitlebarToolGroup(TITLEBAR_GROUP_ID, tools)

    return () => setTitlebarToolGroup(TITLEBAR_GROUP_ID, [])
  }, [devtoolsOpen, setTitlebarToolGroup, toggleDevTools])

  useEffect(() => {
    setAddressValue(browserState.url)
    setCurrentUrl(browserState.url)
    setLoadError(null)

    const webviewUrl = webviewRef.current?.getURL?.()

    if (webviewUrl !== browserState.url && lastRequestedUrlRef.current !== browserState.url) {
      lastRequestedUrlRef.current = browserState.url
      navigateWebview(webviewRef.current, browserState.url)
    }
  }, [browserState.url, sessionId])

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()
    const initialUrl = $browserCurrentState.get().url
    const webview = document.createElement('webview') as BrowserWebview
    webview.className = 'flex h-full w-full flex-1 bg-white dark:bg-black'
    webview.setAttribute('partition', browserPartitionForSession(sessionId))
    webview.setAttribute('src', initialUrl)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const onNavigate = (event: Event) => {
      const detail = event as Event & { url?: string }
      const url = normalizeBrowserUrl(detail.url || webview.getURL?.() || initialUrl)

      lastRequestedUrlRef.current = url
      setLoadError(null)
      setCurrentUrl(url)
      setAddressValue(url)
      setBrowserSessionState(sessionId, { title: webview.getTitle?.() || undefined, url })
      const navState = updateNavigationState()
      publishBrowserState({ ...navState, loading: false, title: webview.getTitle?.() || undefined, url })
    }

    const onTitle = (event: Event) => {
      const detail = event as Event & { title?: string }
      const title = detail.title || webview.getTitle?.() || ''

      if (title) {
        setBrowserSessionState(sessionId, { title })
        publishBrowserState({ title })
      }
    }

    const onFail = (event: Event) => {
      const detail = event as Event & {
        errorCode?: number
        errorDescription?: string
        validatedURL?: string
      }

      if (detail.errorCode === -3) {
        return
      }

      const url = detail.validatedURL || webview.getURL?.() || lastRequestedUrlRef.current

      setLoadError({
        code: detail.errorCode,
        description: detail.errorDescription || 'The embedded browser could not load this page.',
        url
      })
      setLoading(false)
      const navState = updateNavigationState()
      publishBrowserState({ ...navState, loading: false, url })
    }

    const onStart = () => {
      setLoading(true)
      setLoadError(null)
      publishBrowserState({ loading: true, url: webview.getURL?.() || lastRequestedUrlRef.current })
    }

    const onStop = () => {
      setLoading(false)
      const navState = updateNavigationState()
      publishBrowserState({ ...navState, loading: false, title: webview.getTitle?.() || undefined, url: webview.getURL?.() })
    }

    webview.addEventListener('did-fail-load', onFail)
    webview.addEventListener('did-navigate', onNavigate)
    webview.addEventListener('did-navigate-in-page', onNavigate)
    webview.addEventListener('did-start-loading', onStart)
    webview.addEventListener('did-stop-loading', onStop)
    webview.addEventListener('page-title-updated', onTitle)
    host.appendChild(webview)
    webviewRef.current = webview

    return () => {
      webview.removeEventListener('did-fail-load', onFail)
      webview.removeEventListener('did-navigate', onNavigate)
      webview.removeEventListener('did-navigate-in-page', onNavigate)
      webview.removeEventListener('did-start-loading', onStart)
      webview.removeEventListener('did-stop-loading', onStop)
      webview.removeEventListener('page-title-updated', onTitle)
      webview.remove()
      webviewRef.current = null
    }
  }, [publishBrowserState, sessionId, updateNavigationState])

  // Listen for drive commands from the store. open/navigate are primarily
  // handled by the persisted URL state; reload/back/forward must touch the
  // live <webview> instance directly.
  useEffect(() => {
    if (
      !driveCommand ||
      driveCommand.sessionId !== sessionId ||
      driveCommand.id === lastHandledDriveCommandIdRef.current
    ) {
      return
    }

    lastHandledDriveCommandIdRef.current = driveCommand.id

    const webview = webviewRef.current

    if (!webview) {
      return
    }

    if (driveCommand.action === 'reload') {
      reload()
    } else if (driveCommand.action === 'goBack') {
      webview.goBack?.()
    } else if (driveCommand.action === 'goForward') {
      webview.goForward?.()
    }
  }, [driveCommand, reload, sessionId])

  const activeDriveCommand = driveCommand?.sessionId === sessionId ? driveCommand : null
  const agentActionLabel = activeDriveCommand ? AGENT_ACTION_LABELS[activeDriveCommand.action] : null

  const submitAddress = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    navigateTo(addressValue)
  }

  const openExternal = () => {
    if (currentUrl && currentUrl !== 'about:blank') {
      void window.hermesDesktop?.openExternal(currentUrl)
    }
  }

  return (
    <aside className="relative flex h-full w-full min-w-0 flex-col overflow-hidden bg-(--ui-editor-surface-background) text-(--ui-text-tertiary)">
      <div className="flex h-(--titlebar-height) shrink-0 items-center gap-1 border-b border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background) px-2 [-webkit-app-region:no-drag]">
        <Tip label="Back">
          <button
            aria-label="Back"
            className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            disabled={!canGoBack}
            onClick={() => webviewRef.current?.goBack?.()}
            type="button"
          >
            <Codicon name="arrow-left" size="1rem" />
          </button>
        </Tip>
        <Tip label="Forward">
          <button
            aria-label="Forward"
            className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            disabled={!canGoForward}
            onClick={() => webviewRef.current?.goForward?.()}
            type="button"
          >
            <Codicon name="arrow-right" size="1rem" />
          </button>
        </Tip>
        <Tip label={loading ? 'Stop loading' : 'Reload'}>
          <button
            aria-label={loading ? 'Stop loading' : 'Reload'}
            className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
            onClick={() => (loading ? webviewRef.current?.stop?.() : reload())}
            type="button"
          >
            <Codicon name={loading ? 'debug-stop' : 'refresh'} size="1rem" />
          </button>
        </Tip>
        <form className="min-w-0 flex-1" onSubmit={submitAddress}>
          <input
            aria-label="Browser address"
            className={cn(
              'h-8 w-full rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-editor-surface-background) px-2.5 font-mono text-xs text-foreground outline-none transition-colors',
              'focus:border-(--ui-stroke-primary) focus:ring-2 focus:ring-sidebar-ring/35'
            )}
            onChange={event => setAddressValue(event.target.value)}
            spellCheck={false}
            value={addressValue}
          />
        </form>
        {agentActionLabel && (
          <div
            aria-label="Browser agent status"
            className="hidden shrink-0 items-center rounded-full border border-(--ui-stroke-tertiary) bg-(--ui-editor-surface-background) px-2 py-0.5 text-[0.625rem] font-medium text-(--ui-text-secondary) sm:flex"
          >
            Agent: {agentActionLabel}
          </div>
        )}
        <Tip label="Open in system browser">
          <button
            aria-label="Open in system browser"
            className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
            disabled={!currentUrl || currentUrl === 'about:blank'}
            onClick={openExternal}
            type="button"
          >
            <Codicon name="link-external" size="1rem" />
          </button>
        </Tip>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden bg-white dark:bg-black">
        <div className={cn('absolute inset-0 flex', loadError && 'pointer-events-none opacity-0')} ref={hostRef} />
        {loadError && <BrowserLoadError error={loadError} onRetry={reload} />}
      </div>
    </aside>
  )
}
