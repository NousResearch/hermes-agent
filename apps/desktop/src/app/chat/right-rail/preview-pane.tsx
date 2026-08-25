// Side-effect import: watches the turn edge so the overlay keeps a pulse while
// the model reasons. Lives here because the pane is what makes it reachable.
import './preview-mind'

import { useStore } from '@nanostores/react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { openGuestContextMenu } from '@/app/context-menu/store'
import { PanelEmpty } from '@/app/overlays/panel'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { guardGuestPointers } from '@/lib/guest-pointer-guard'
import { openPreviewTargetInBrowser, remoteHtmlPreviewDocument } from '@/lib/local-preview'
import { isRemoteGateway } from '@/lib/media'
import { reachablePreviewUrl } from '@/lib/preview-reach'
import { rafCoalesce } from '@/lib/raf-coalesce'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $previewServerRestart, failPreviewServerRestart, noteBrowserPage, type PreviewTarget } from '@/store/preview'

import { ArtifactPreview } from './preview-artifact'
import { PreviewBrowserBar } from './preview-browser-bar'
import {
  clampConsoleHeight,
  compactUrl,
  formatLogLine,
  isNearConsoleBottom,
  PreviewConsolePanel
} from './preview-console'
import { type ConsoleEntry } from './preview-console-state'
import { previewConsoleState } from './preview-console-store'
import { LocalFilePreview, PreviewEmptyState } from './preview-file'
import { type PreviewInputEvent, registerPreviewInput } from './preview-input'
import { PREVIEW_BROWSER_ATTR, registerPreviewNav } from './preview-nav'
import { registerPreviewPageReader } from './preview-reader'
import { registerPreviewScriptRunner } from './preview-script-runner'

type PreviewWebview = HTMLElement & {
  canGoBack?: () => boolean
  canGoForward?: () => boolean
  closeDevTools?: () => void
  copy?: () => void
  cut?: () => void
  executeJavaScript?: (code: string) => Promise<unknown>
  getTitle?: () => string
  getURL?: () => string
  getWebContentsId?: () => number
  goBack?: () => void
  goForward?: () => void
  inspectElement?: (x: number, y: number) => void
  isDevToolsOpened?: () => boolean
  loadURL?: (url: string) => Promise<void>
  openDevTools?: () => void
  paste?: () => void
  reload?: () => void
  reloadIgnoringCache?: () => void
  replaceMisspelling?: (word: string) => void
  selectAll?: () => void
  sendInputEvent?: (event: PreviewInputEvent) => void
}

/** The raw Chromium params riding the webview tag's `context-menu` event. */
interface GuestContextMenuParams {
  dictionarySuggestions?: string[]
  editFlags?: {
    canCopy?: boolean
    canCut?: boolean
    canPaste?: boolean
    canSelectAll?: boolean
  }
  hasImageContents?: boolean
  isEditable?: boolean
  linkURL?: string
  misspelledWord?: string
  selectionText?: string
  srcURL?: string
  x: number
  y: number
}

interface PreviewPaneProps {
  embedded?: boolean
  onRestartServer?: (url: string, context?: string) => Promise<string>
  reloadRequest?: number
  /** The preview tab this pane renders. Keys the per-tab console store the
   *  browser bar's console toggle and the console panel both read. */
  tabId?: string
  target: PreviewTarget
}

interface PreviewLoadErrorState {
  code?: number
  description: string
  url: string
}

const FILE_RELOAD_DEBOUNCE_MS = 200
const SERVER_RESTART_TIMEOUT_MS = 45_000

function loadErrorTitle(error: PreviewLoadErrorState, copy: Translations['preview']['web']): string {
  const description = error.description.toLowerCase()

  if (description.includes('module script') || description.includes('mime type')) {
    return copy.appFailedToBoot
  }

  if (description.includes('connection') || description.includes('refused') || description.includes('not found')) {
    return copy.serverNotFound
  }

  return copy.failedToLoad
}

/** Loopback hosts — the address family that means "this machine", and so the
 *  one family whose meaning changes with WHICH machine is running the page. */
const LOOPBACK_HOST_RE = /^(localhost|127(?:\.\d{1,3}){3}|0\.0\.0\.0|\[?::1\]?)$/i

/**
 * True when this address can't mean what the agent meant.
 *
 * The `<webview>` always runs on the user's own machine, never on the gateway
 * host. So against a remote gateway, an agent's `localhost:5173` resolves here
 * — where that port is usually nothing, or worse, somebody else's service. The
 * URL isn't wrong, it's just addressed to a different computer.
 */
function isRemoteLoopbackUrl(url: string): boolean {
  if (!isRemoteGateway()) {
    return false
  }

  try {
    return LOOPBACK_HOST_RE.test(new URL(url).hostname)
  } catch {
    return false
  }
}

function isModuleMimeError(message: string): boolean {
  const lower = message.toLowerCase()

  return lower.includes('failed to load module script') && lower.includes('mime type')
}

function PreviewLoadError({
  consoleHeight = 0,
  error,
  onRestartServer,
  onRetry,
  restarting
}: {
  consoleHeight?: number
  error: PreviewLoadErrorState
  onRestartServer?: () => void
  onRetry: () => void
  restarting?: boolean
}) {
  const { t } = useI18n()
  const copy = t.preview.web

  return (
    <PreviewEmptyState
      body={
        <>
          <a
            className="pointer-events-auto block font-mono text-muted-foreground/90 underline decoration-current/20 underline-offset-4 transition-colors hover:text-foreground"
            href={error.url}
            onClick={event => {
              event.preventDefault()
              void window.hermesDesktop?.openExternal(error.url)
            }}
          >
            {compactUrl(error.url)}
            {error.code ? ` (${error.code})` : ''}
          </a>
          <div className="mt-1 text-[0.6875rem] text-muted-foreground/70">{error.description}</div>
          {isRemoteLoopbackUrl(error.url) && (
            <div className="mt-2 text-[0.6875rem] leading-relaxed text-muted-foreground/70">{copy.remoteLoopback}</div>
          )}
        </>
      }
      consoleHeight={consoleHeight}
      primaryAction={{ label: copy.tryAgain, onClick: onRetry }}
      secondaryAction={
        onRestartServer
          ? {
              disabled: restarting,
              label: restarting ? copy.restarting : copy.askRestart,
              onClick: onRestartServer
            }
          : undefined
      }
      title={loadErrorTitle(error, copy)}
    />
  )
}

export function PreviewPane({ embedded = false, onRestartServer, reloadRequest = 0, tabId, target }: PreviewPaneProps) {
  const { t } = useI18n()
  const copy = t.preview.web
  // The console store belongs to the TAB, not this render: the toggles live on
  // the tab and must read the same logs this pane appends to.
  const consoleState = previewConsoleState(tabId ?? target.url)
  const consoleBodyRef = useRef<HTMLDivElement | null>(null)
  const consoleShouldStickRef = useRef(true)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const lastReloadRequestRef = useRef(reloadRequest)
  const lastRestartEventRef = useRef('')
  const previewContentRef = useRef<HTMLDivElement | null>(null)
  const webviewRef = useRef<PreviewWebview | null>(null)
  const previewServerRestart = useStore($previewServerRestart)
  const consoleHeight = useStore(consoleState.$height)
  const consoleOpen = useStore(consoleState.$open)
  const [currentUrl, setCurrentUrl] = useState(target.url)
  const [devtoolsOpen, setDevtoolsOpen] = useState(false)
  const [history, setHistory] = useState({ back: false, forward: false })
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<PreviewLoadErrorState | null>(null)
  const [localReloadKey, setLocalReloadKey] = useState(0)

  // Artifacts have no URL to load — they render from the registry, never in a
  // webview.
  const isWebPreview =
    target.kind !== 'artifact' &&
    (target.kind === 'url' || (target.previewKind === 'html' && target.renderMode !== 'source'))

  const isRemoteHtmlTarget =
    target.kind === 'file' && target.previewKind === 'html' && Boolean(target.dataUrl || target.transient)

  const isRemoteHtml = isRemoteHtmlTarget && target.renderMode !== 'source' && Boolean(target.dataUrl)

  const remoteHtmlDocument = useMemo(
    () => (isRemoteHtml ? remoteHtmlPreviewDocument(target.dataUrl!) : null),
    [isRemoteHtml, target.dataUrl]
  )

  const currentLabel = compactUrl(currentUrl)

  // Nothing loaded: no address yet, or the blank page itself. A webview on
  // `about:blank` paints a white void that reads as broken next to the app's
  // dark chrome, so the pane says what it is instead.
  const isBlankPage = isWebPreview && !isRemoteHtml && (!currentUrl || /^about:blank\/?$/i.test(currentUrl))

  const previewLabel =
    target.label && target.label.replace(/\/$/, '') !== currentLabel.replace(/\/$/, '') ? target.label : currentLabel

  const restartingServer =
    previewServerRestart?.status === 'running' &&
    (previewServerRestart.url === target.url || previewServerRestart.url === currentUrl)

  const startConsoleResize = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault()

      const handle = event.currentTarget
      const pointerId = event.pointerId
      const startY = event.clientY
      const startHeight = consoleHeight
      const previousCursor = document.body.style.cursor
      const previousUserSelect = document.body.style.userSelect
      let active = true

      handle.setPointerCapture?.(pointerId)

      document.body.style.cursor = 'row-resize'
      document.body.style.userSelect = 'none'
      // The webview above the console must not swallow the gesture.
      const releaseGuests = guardGuestPointers()

      // pointermove outpaces 60fps and each setHeight reflows the webview +
      // console split, so coalesce to one apply per frame (commits on cleanup).
      const resize = rafCoalesce((height: number) => consoleState.setHeight(height))

      const handleMove = (moveEvent: PointerEvent) => {
        if (!active) {
          return
        }

        resize.push(clampConsoleHeight(startHeight + startY - moveEvent.clientY))
      }

      const cleanup = () => {
        if (!active) {
          return
        }

        active = false
        resize.finish()
        releaseGuests()
        document.body.style.cursor = previousCursor
        document.body.style.userSelect = previousUserSelect
        handle.releasePointerCapture?.(pointerId)
        window.removeEventListener('pointermove', handleMove, true)
        window.removeEventListener('pointerup', cleanup, true)
        window.removeEventListener('pointercancel', cleanup, true)
        window.removeEventListener('blur', cleanup)
        handle.removeEventListener('lostpointercapture', cleanup)
      }

      window.addEventListener('pointermove', handleMove, true)
      window.addEventListener('pointerup', cleanup, true)
      window.addEventListener('pointercancel', cleanup, true)
      window.addEventListener('blur', cleanup)
      handle.addEventListener('lostpointercapture', cleanup)
    },
    [consoleHeight, consoleState]
  )

  const reloadPreview = useCallback(() => {
    setLoadError(null)

    if (!isWebPreview) {
      setLocalReloadKey(key => key + 1)

      return
    }

    if (webviewRef.current?.reloadIgnoringCache) {
      webviewRef.current.reloadIgnoringCache()
    } else {
      webviewRef.current?.reload?.()
    }
  }, [isWebPreview])

  const appendConsoleEntry = useCallback(
    (entry: Omit<ConsoleEntry, 'id'>) => {
      consoleShouldStickRef.current = isNearConsoleBottom(consoleBodyRef.current)
      consoleState.append(entry)
    },
    [consoleState]
  )

  const restartServer = useCallback(async () => {
    if (!onRestartServer) {
      return
    }

    // Auto-open the preview console so the user can see progress events
    // streaming back from the background agent. Without this, clicking
    // "Ask Hermes to restart the server" looked like it did nothing —
    // the work was happening, but in a collapsed pane.
    consoleState.setOpen(true)

    try {
      const context = consoleState.$logs.get().slice(-12).map(formatLogLine).join('\n')
      const taskId = await onRestartServer(currentUrl, context || undefined)

      appendConsoleEntry({
        level: 1,
        message: copy.lookingRestart(taskId)
      })

      notify({
        kind: 'info',
        title: copy.restartingTitle,
        message: copy.restartingMessage,
        durationMs: 4000
      })
    } catch (error) {
      appendConsoleEntry({
        level: 2,
        message: copy.startRestartFailed(error instanceof Error ? error.message : String(error))
      })
      notifyError(error, copy.restartFailed)
    }
  }, [appendConsoleEntry, consoleState, copy, currentUrl, onRestartServer])

  const toggleDevTools = useCallback(() => {
    const webview = webviewRef.current

    if (!webview?.openDevTools) {
      return
    }

    if (webview.isDevToolsOpened?.()) {
      webview.closeDevTools?.()

      return
    }

    webview.openDevTools()
  }, [])

  const navigateTo = useCallback(
    (url: string) => {
      setLoadError(null)
      // The reach probe below is a round-trip of its own, and `did-start-loading`
      // can't fire until it resolves — so own the loading state from the moment
      // we accept the address, or the bar sits idle over a request in flight.
      setLoading(true)
      // Typed addresses get the same loopback reach as agent-opened ones — on a
      // remote gateway `localhost:5173` is usually the dev server the user is
      // there to look at, not something on their own laptop.
      void reachablePreviewUrl(url)
        .then(reached =>
          // loadURL, not a `src` swap: `src` only reloads when the value CHANGES,
          // so re-entering the address you're already on would do nothing. A
          // rejected load is a real navigation failure the user has to see —
          // `did-fail-load` doesn't fire for every rejection (a bad scheme
          // rejects outright).
          webviewRef.current?.loadURL?.(reached)
        )
        .catch((error: unknown) => {
          setLoadError({
            description: error instanceof Error ? error.message : copy.unreachableDescription,
            url
          })
          setLoading(false)
        })
    },
    [copy.unreachableDescription]
  )

  const goBack = useCallback(() => {
    const webview = webviewRef.current

    if (webview?.canGoBack?.()) {
      webview.goBack?.()
    }
  }, [])

  const goForward = useCallback(() => {
    const webview = webviewRef.current

    if (webview?.canGoForward?.()) {
      webview.goForward?.()
    }
  }, [])

  // Gestures that land on the app's chrome (⌘R from the address bar, a mouse
  // button over the frame). A gesture made INSIDE the page is answered by main
  // against the focused guest — this renderer can't see into a webview.
  useEffect(() => {
    if (!isWebPreview || isRemoteHtml || !tabId) {
      return
    }

    return registerPreviewNav(tabId, { back: goBack, forward: goForward, reload: reloadPreview })
  }, [goBack, goForward, isRemoteHtml, isWebPreview, reloadPreview, tabId])

  // Publish the PAGE reader for this tab (the read_preview tool): extract the
  // rendered page's title + visible text from the webview. innerText (not
  // textContent) so hidden nodes and script/style bodies stay out, matching
  // what the user actually sees.
  useEffect(() => {
    if (!isWebPreview || !tabId) {
      return
    }

    return registerPreviewPageReader(tabId, async () => {
      const webview = webviewRef.current

      if (!webview?.executeJavaScript) {
        throw new Error('preview webview is not ready')
      }

      const text = await webview.executeJavaScript('document.body ? document.body.innerText : ""')

      return {
        text: typeof text === 'string' ? text : '',
        title: webview.getTitle?.() ?? '',
        url: webview.getURL?.() ?? ''
      }
    })
  }, [isWebPreview, tabId])

  // Publish the SCRIPT runner for this tab: the one channel into the guest
  // page, shared by the tour tool (injected driver.js walkthroughs) and the
  // drive_preview tool (clicking, typing, scrolling the page the user sees).
  useEffect(() => {
    if (!isWebPreview || !tabId) {
      return
    }

    return registerPreviewScriptRunner(tabId, async code => {
      const webview = webviewRef.current

      if (!webview?.executeJavaScript) {
        throw new Error('preview webview is not ready')
      }

      return webview.executeJavaScript(code)
    })
  }, [isWebPreview, tabId])

  // Publish the INPUT channel for this tab. Same idea as the script runner, but
  // it carries real Chromium input rather than script — the agent's clicks and
  // keystrokes arrive as trusted events, so the page hovers, focuses and reacts
  // exactly as it would under a human hand.
  useEffect(() => {
    if (!isWebPreview || isRemoteHtml || !tabId) {
      return
    }

    return registerPreviewInput(tabId, {
      focus: () => webviewRef.current?.focus?.(),
      send: event => {
        const webview = webviewRef.current

        // Never optional-chain this call away: a missing method would make every
        // agent click a silent no-op that still reports success, because the
        // overlay and the read-back both run on the separate script channel.
        if (typeof webview?.sendInputEvent !== 'function') {
          throw new Error('preview webview cannot take input events')
        }

        webview.sendInputEvent(event)
      }
    })
  }, [isRemoteHtml, isWebPreview, tabId])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!consoleOpen) {
      return
    }

    consoleShouldStickRef.current = true

    const handle = window.requestAnimationFrame(() => {
      const consoleBody = consoleBodyRef.current
      consoleBody?.scrollTo({ top: consoleBody.scrollHeight })
    })

    return () => window.cancelAnimationFrame(handle)
  }, [consoleOpen])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (
      !previewServerRestart ||
      !previewServerRestart.message ||
      (previewServerRestart.url !== target.url && previewServerRestart.url !== currentUrl)
    ) {
      return
    }

    const eventKey = `${previewServerRestart.taskId}:${previewServerRestart.status}:${previewServerRestart.message || ''}`

    if (eventKey === lastRestartEventRef.current) {
      return
    }

    lastRestartEventRef.current = eventKey
    appendConsoleEntry({
      level: previewServerRestart.status === 'error' ? 2 : 1,
      message:
        previewServerRestart.status === 'running'
          ? previewServerRestart.message
          : previewServerRestart.status === 'complete'
            ? copy.finishedRestarting(previewServerRestart.message)
            : copy.failedRestarting(previewServerRestart.message || copy.unknownError)
    })

    if (previewServerRestart.status === 'complete') {
      reloadPreview()
      notify({
        kind: 'success',
        title: copy.restartedTitle,
        message: previewServerRestart.message?.slice(0, 160) || copy.reloadingNow,
        durationMs: 3500
      })
    } else if (previewServerRestart.status === 'error') {
      notify({
        kind: 'warning',
        title: copy.restartFailedTitle,
        message: previewServerRestart.message?.slice(0, 200) || copy.restartFailedMessage,
        durationMs: 6000
      })
    }
  }, [appendConsoleEntry, copy, currentUrl, previewServerRestart, reloadPreview, target.url])

  useEffect(() => {
    if (!restartingServer || !previewServerRestart) {
      return
    }

    const taskId = previewServerRestart.taskId

    const timer = window.setTimeout(() => {
      failPreviewServerRestart(taskId, copy.stillWorking)
    }, SERVER_RESTART_TIMEOUT_MS)

    return () => window.clearTimeout(timer)
  }, [copy.stillWorking, previewServerRestart, restartingServer])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (reloadRequest === lastReloadRequestRef.current) {
      return
    }

    lastReloadRequestRef.current = reloadRequest

    if (target.kind !== 'url') {
      return
    }

    appendConsoleEntry({
      level: 1,
      message: copy.workspaceReloading
    })
    reloadPreview()
  }, [appendConsoleEntry, copy.workspaceReloading, reloadPreview, reloadRequest, target.kind])

  useEffect(() => {
    if (
      target.kind !== 'file' ||
      isDesktopFsRemoteMode() ||
      !window.hermesDesktop?.watchPreviewFile ||
      !window.hermesDesktop?.onPreviewFileChanged
    ) {
      return
    }

    let active = true
    let pendingReloadCount = 0
    let pendingReloadUrl = ''
    let reloadTimer: ReturnType<typeof setTimeout> | null = null
    let watchId = ''

    const flushReload = () => {
      if (!active || pendingReloadCount === 0) {
        return
      }

      const changedCount = pendingReloadCount
      const changedUrl = pendingReloadUrl

      pendingReloadCount = 0
      pendingReloadUrl = ''

      appendConsoleEntry({
        level: 1,
        message:
          changedCount === 1
            ? copy.fileChanged(compactUrl(changedUrl))
            : copy.filesChanged(changedCount, compactUrl(changedUrl))
      })

      reloadPreview()
    }

    const unsubscribe = window.hermesDesktop.onPreviewFileChanged(payload => {
      if (!active || payload.id !== watchId) {
        return
      }

      pendingReloadCount += 1
      pendingReloadUrl = payload.url

      if (reloadTimer) {
        clearTimeout(reloadTimer)
      }

      reloadTimer = setTimeout(() => {
        reloadTimer = null
        flushReload()
      }, FILE_RELOAD_DEBOUNCE_MS)
    })

    void window.hermesDesktop
      .watchPreviewFile(target.url)
      .then(watch => {
        if (!active) {
          void window.hermesDesktop?.stopPreviewFileWatch?.(watch.id)

          return
        }

        watchId = watch.id
      })
      .catch(error => {
        appendConsoleEntry({
          level: 2,
          message: copy.watchFailed(error instanceof Error ? error.message : String(error))
        })
      })

    return () => {
      active = false
      unsubscribe()

      if (reloadTimer) {
        clearTimeout(reloadTimer)
      }

      if (watchId) {
        void window.hermesDesktop?.stopPreviewFileWatch?.(watchId)
      }
    }
  }, [appendConsoleEntry, copy, reloadPreview, target.kind, target.url])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()
    webviewRef.current = null
    setCurrentUrl(target.url)
    setDevtoolsOpen(false)
    setHistory({ back: false, forward: false })
    setLoadError(null)
    consoleState.reset()
    setLoading(true)

    if (!isWebPreview || isRemoteHtml) {
      setLoading(false)

      return
    }

    const webview = document.createElement('webview') as PreviewWebview
    webview.className = 'flex h-full w-full flex-1 bg-transparent'
    webview.setAttribute('partition', 'persist:hermes-preview')
    webview.setAttribute('src', target.url)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const onConsole = (event: Event) => {
      const detail = event as Event & {
        level?: number
        line?: number
        message?: string
        sourceId?: string
      }

      const message = detail.message || ''

      appendConsoleEntry({
        level: detail.level ?? 0,
        line: detail.line,
        message,
        source: detail.sourceId
      })

      if ((detail.level ?? 0) >= 3 && isModuleMimeError(message)) {
        setLoadError({
          description: copy.moduleMimeDescription,
          url: webview.getURL?.() || target.url
        })
        setLoading(false)
      }
    }

    const syncHistory = () =>
      setHistory({ back: webview.canGoBack?.() ?? false, forward: webview.canGoForward?.() ?? false })

    // Tell the strip what this Browser is showing, so its tab renames itself
    // like a tab anywhere else. Deliberately NOT written back into the tab's
    // target: the guest is built from `target.url`, so that would rebuild the
    // webview mid-navigation and throw away the history.
    const notePage = () => {
      if (target.kind !== 'url' || !tabId) {
        return
      }

      noteBrowserPage(tabId, { title: webview.getTitle?.() ?? '', url: webview.getURL?.() || target.url })
    }

    const onNavigate = (event: Event) => {
      const detail = event as Event & { url?: string }

      if (detail.url) {
        setLoadError(null)
        setCurrentUrl(detail.url)
      }

      notePage()

      // Ask the webview rather than counting navigations: the guest page can
      // move itself (redirects, history.pushState, a link into a new document),
      // so it is the only thing that knows what its history holds. Wired to
      // `did-navigate-in-page` too, or SPA route changes never update it.
      syncHistory()
    }

    const onFail = (event: Event) => {
      const detail = event as Event & {
        errorCode?: number
        errorDescription?: string
        validatedURL?: string
      }

      const errorCode = detail.errorCode

      if (errorCode === -3) {
        return
      }

      appendConsoleEntry({
        level: 3,
        message: copy.loadFailedConsole(errorCode, detail.errorDescription || detail.validatedURL || copy.unknownError)
      })
      setLoadError({
        code: errorCode,
        description: detail.errorDescription || copy.unreachableDescription,
        url: detail.validatedURL || webview.getURL?.() || target.url
      })
      setLoading(false)
    }

    const onStart = () => setLoading(true)

    const onStop = () => {
      setLoading(false)
      // A load that ends without a `did-navigate` (an in-place reload, a
      // cancelled navigation) still settles the history — resync so the
      // buttons can't be left stale.
      syncHistory()
      notePage()
    }

    // The WEBVIEW is the source of truth for DevTools, not our click handler:
    // closing the DevTools window itself fires devtools-closed with no click,
    // and the glyph was left stuck "on" when we tracked it locally.
    const onDevToolsOpened = () => setDevtoolsOpen(true)
    const onDevToolsClosed = () => setDevtoolsOpen(false)

    // Right-clicks INSIDE the guest page. The tag surfaces Chromium's full
    // context-menu params (link, image, editable, selection, spellcheck), so
    // the app coordinator renders the same translated menu it shows
    // everywhere else.
    //
    // Coordinates: params.x/y are WINDOW-relative device-independent pixels
    // — the guest offset is already included, and CSS values are multiplied
    // by the window zoom factor. Measured live (zoom 0.9): a click whose
    // true window CSS point was (901, 272) arrived as params (811, 246) =
    // (901*0.9, 272*0.9). Dividing by the zoom factor recovers CSS
    // coordinates; adding the webview rect on top double-counted the offset
    // and dropped the menu far right+below the click.
    const onGuestContextMenu = (event: Event) => {
      const detail = event as Event & { params?: GuestContextMenuParams }
      const params = detail.params

      if (!params) {
        return
      }

      const zoom = window.hermesDesktop?.zoom?.factor?.() || 1
      // Window CSS point of the click (the menu anchors here).
      const windowX = params.x / zoom
      const windowY = params.y / zoom
      // Guest CSS point (inspectElement wants coordinates INSIDE the page):
      // subtract the webview's own offset from the window point.
      const rect = webview.getBoundingClientRect()
      const guestX = Math.max(0, Math.round(windowX - rect.left))
      const guestY = Math.max(0, Math.round(windowY - rect.top))

      openGuestContextMenu(
        windowX,
        windowY,
        {
          dictionarySuggestions: Array.isArray(params.dictionarySuggestions) ? params.dictionarySuggestions : [],
          // Chromium's availability verdict for the edit verbs. Absent only
          // if a future Electron drops it — then everything stays enabled,
          // which is the pre-editFlags behavior, not a lockout.
          editFlags: {
            canCopy: params.editFlags?.canCopy ?? true,
            canCut: params.editFlags?.canCut ?? true,
            canPaste: params.editFlags?.canPaste ?? true,
            canSelectAll: params.editFlags?.canSelectAll ?? true
          },
          hasImageContents: Boolean(params.hasImageContents),
          isEditable: Boolean(params.isEditable),
          linkURL: params.linkURL || '',
          misspelledWord: params.misspelledWord || '',
          selectionText: params.selectionText || '',
          srcURL: params.srcURL || ''
        },
        {
          addToDictionary: (word: string) => {
            const webContentsId = webview.getWebContentsId?.()

            if (typeof webContentsId === 'number') {
              void window.hermesDesktop?.contextMenuGuestAddWord?.({ webContentsId, word })
            }
          },
          copyImage: () => void window.hermesDesktop?.contextMenuCopyImage?.(),
          // The tag's edit commands act on the focused webContents, and the
          // menu click just parked focus on the HOST body — measured live:
          // selectAll() with host focus selected the address bar + chat
          // instead of the page. Focus the webview first, every verb.
          editCommand: (command: 'copy' | 'cut' | 'paste' | 'selectAll') => {
            webview.focus()
            webview[command]?.()
          },
          inspectElement: () => webview.inspectElement?.(guestX, guestY),
          replaceMisspelling: (word: string) => webview.replaceMisspelling?.(word)
        }
      )
    }

    webview.addEventListener('console-message', onConsole)
    webview.addEventListener('context-menu', onGuestContextMenu)
    webview.addEventListener('devtools-closed', onDevToolsClosed)
    webview.addEventListener('devtools-opened', onDevToolsOpened)
    webview.addEventListener('did-fail-load', onFail)
    webview.addEventListener('did-navigate', onNavigate)
    webview.addEventListener('did-navigate-in-page', onNavigate)
    webview.addEventListener('did-start-loading', onStart)
    webview.addEventListener('did-stop-loading', onStop)
    // SPAs title themselves long after the load settles, and a route change
    // renames the page without navigating at all.
    webview.addEventListener('page-title-updated', notePage)
    host.appendChild(webview)
    webviewRef.current = webview

    return () => {
      webview.removeEventListener('console-message', onConsole)
      webview.removeEventListener('context-menu', onGuestContextMenu)
      webview.removeEventListener('devtools-closed', onDevToolsClosed)
      webview.removeEventListener('devtools-opened', onDevToolsOpened)
      webview.removeEventListener('did-fail-load', onFail)
      webview.removeEventListener('did-navigate', onNavigate)
      webview.removeEventListener('did-navigate-in-page', onNavigate)
      webview.removeEventListener('did-start-loading', onStart)
      webview.removeEventListener('did-stop-loading', onStop)
      webview.removeEventListener('page-title-updated', notePage)
      webview.remove()
    }
  }, [appendConsoleEntry, consoleState, copy, isRemoteHtml, isWebPreview, tabId, target.kind, target.url])

  return (
    <aside
      className="relative flex h-full w-full min-w-0 flex-col overflow-hidden bg-transparent text-muted-foreground"
      // Buttons 3/4 are a mouse's back/forward. Chromium delivers them to the
      // renderer as a normal mouse event inside the app's own chrome (the
      // guest page gets its own via `app-command` in main), and unhandled they
      // walk the HOST document's history.
      onMouseDown={event => {
        if (event.button !== 3 && event.button !== 4) {
          return
        }

        event.preventDefault()

        if (event.button === 3) {
          goBack()
        } else {
          goForward()
        }
      }}
      {...(isWebPreview && !isRemoteHtml && tabId ? { [PREVIEW_BROWSER_ATTR]: tabId } : {})}
    >
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {!embedded && (
          <div className="pointer-events-none flex min-h-(--titlebar-height) items-center gap-1.5 border-b border-border/60 bg-background px-2 py-1">
            <div className="min-w-0 flex-1">
              <Tip label={copy.openTarget(currentUrl)}>
                <a
                  className="pointer-events-auto inline max-w-full truncate text-left text-xs font-medium text-foreground underline-offset-4 decoration-current/20 transition-colors hover:text-primary hover:underline"
                  href={isRemoteHtmlTarget ? undefined : currentUrl}
                  onClick={event => {
                    if (isRemoteHtmlTarget) {
                      event.preventDefault()
                      void openPreviewTargetInBrowser(target).catch(error => notifyError(error, t.preview.unavailable))
                    }
                  }}
                  rel="noreferrer"
                  target={isRemoteHtmlTarget ? undefined : '_blank'}
                >
                  {previewLabel || copy.fallbackTitle}
                </a>
              </Tip>
            </div>
          </div>
        )}

        {isWebPreview && !isRemoteHtml && (
          <PreviewBrowserBar
            canGoBack={history.back}
            canGoForward={history.forward}
            consoleOpen={consoleOpen}
            devToolsOpen={devtoolsOpen}
            loading={loading}
            onBack={goBack}
            onForward={goForward}
            onNavigate={navigateTo}
            onOpenExternal={() => void window.hermesDesktop?.openExternal(currentUrl)}
            onReload={reloadPreview}
            onToggleConsole={() => consoleState.setOpen(open => !open)}
            onToggleDevTools={toggleDevTools}
            url={currentUrl}
          />
        )}

        <div
          className="pointer-events-auto relative min-h-0 flex-1 overflow-hidden bg-transparent"
          ref={previewContentRef}
        >
          <div
            className={cn(
              'absolute inset-0 flex bg-transparent',
              (isRemoteHtml || !isWebPreview || loadError) && 'pointer-events-none opacity-0'
            )}
            ref={hostRef}
          />
          {isRemoteHtml && (
            <iframe
              className="absolute inset-0 size-full border-0 bg-white"
              referrerPolicy="no-referrer"
              sandbox=""
              srcDoc={remoteHtmlDocument || ''}
              title={target.label || copy.fallbackTitle}
            />
          )}
          {!isWebPreview &&
            (target.kind === 'artifact' ? (
              <ArtifactPreview target={target} />
            ) : (
              <LocalFilePreview reloadKey={localReloadKey} target={target} />
            ))}
          {isBlankPage && (
            <div className="absolute inset-0 grid bg-background">
              <PanelEmpty description={copy.blankPageBody} icon="globe" />
            </div>
          )}
          {loadError && (
            <PreviewLoadError
              consoleHeight={consoleOpen ? consoleHeight : 0}
              error={loadError}
              onRestartServer={target.kind === 'url' && onRestartServer ? () => void restartServer() : undefined}
              onRetry={reloadPreview}
              restarting={restartingServer}
            />
          )}

          {isWebPreview && !isRemoteHtml && consoleOpen && (
            <PreviewConsolePanel
              consoleBodyRef={consoleBodyRef}
              consoleShouldStickRef={consoleShouldStickRef}
              consoleState={consoleState}
              startConsoleResize={startConsoleResize}
            />
          )}
        </div>
      </div>
    </aside>
  )
}
