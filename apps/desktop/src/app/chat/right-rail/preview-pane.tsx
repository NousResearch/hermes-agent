import { useStore } from '@nanostores/react'
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { SetTitlebarToolGroup, TitlebarTool } from '@/app/shell/titlebar-controls'
import { Tip } from '@/components/ui/tooltip'
import { type Translations, useI18n } from '@/i18n'
import { isDesktopFsRemoteMode } from '@/lib/desktop-fs'
import { Bug, Pencil } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $previewServerRestart, failPreviewServerRestart, type PreviewTarget } from '@/store/preview'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import {
  clampConsoleHeight,
  compactUrl,
  formatLogLine,
  isNearConsoleBottom,
  PreviewConsolePanel,
  PreviewConsoleTitlebarIcon
} from './preview-console'
import { type ConsoleEntry, createPreviewConsoleState } from './preview-console-state'
import { LocalFilePreview, PreviewEmptyState } from './preview-file'
import { PREVIEW_RATIO_PRESETS, type PreviewRatioPreset, previewWidthForRatio } from './preview-sizing'

type PreviewWebview = HTMLElement & {
  closeDevTools?: () => void
  getURL?: () => string
  isDevToolsOpened?: () => boolean
  openDevTools?: () => void
  reload?: () => void
  reloadIgnoringCache?: () => void
}

type PreviewEngine = 'detecting' | 'electron-webview' | 'iframe-fallback' | 'local-file'

interface PreviewPaneProps {
  embedded?: boolean
  onRestartServer?: (url: string, context?: string) => Promise<string>
  reloadRequest?: number
  setTitlebarToolGroup?: SetTitlebarToolGroup
  target: PreviewTarget
}

interface PreviewLoadErrorState {
  code?: number
  description: string
  url: string
}

const FILE_RELOAD_DEBOUNCE_MS = 200
const SERVER_RESTART_TIMEOUT_MS = 45_000
const PREVIEW_WEBVIEW_PARTITION_PREFIX = 'persist:hermes-preview'

function previewWebviewPartition(sessionId: string | null | undefined): string {
  const safeSessionId = sessionId
    ?.trim()
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)

  return safeSessionId ? `${PREVIEW_WEBVIEW_PARTITION_PREFIX}-${safeSessionId}` : PREVIEW_WEBVIEW_PARTITION_PREFIX
}

function normalizePreviewAddress(value: string): string {
  const trimmed = value.trim()

  if (!trimmed) {
    return ''
  }

  if (/^(https?:|file:|data:|blob:)/i.test(trimmed)) {
    return trimmed
  }

  return `https://${trimmed}`
}

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

const TITLEBAR_GROUP_ID = 'preview'

export function PreviewPane({
  embedded = false,
  onRestartServer,
  reloadRequest = 0,
  setTitlebarToolGroup,
  target
}: PreviewPaneProps) {
  const { t } = useI18n()
  const copy = t.preview.web
  const [consoleState] = useState(() => createPreviewConsoleState())
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
  const activeSessionId = useStore($activeSessionId)
  const selectedStoredSessionId = useStore($selectedStoredSessionId)
  const [currentUrl, setCurrentUrl] = useState(target.url)
  const [addressDraft, setAddressDraft] = useState(target.url)
  const [annotationOverlayOpen, setAnnotationOverlayOpen] = useState(false)
  const [debugOverlayOpen, setDebugOverlayOpen] = useState(false)
  const [devtoolsAvailable, setDevtoolsAvailable] = useState(false)
  const [devtoolsOpen, setDevtoolsOpen] = useState(false)
  const [previewEngine, setPreviewEngine] = useState<PreviewEngine>('detecting')
  const [previewFrameWidth, setPreviewFrameWidth] = useState<number | null>(null)
  const [activeRatioLabel, setActiveRatioLabel] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<PreviewLoadErrorState | null>(null)
  const [localReloadKey, setLocalReloadKey] = useState(0)
  const isWebPreview = target.kind === 'url' || (target.previewKind === 'html' && target.renderMode !== 'source')
  const currentLabel = compactUrl(currentUrl)
  const webviewPartition = previewWebviewPartition(selectedStoredSessionId || activeSessionId)

  const previewLabel =
    target.label && target.label.replace(/\/$/, '') !== currentLabel.replace(/\/$/, '') ? target.label : currentLabel

  const previewFrameStyle = useMemo<CSSProperties>(
    () => ({ width: previewFrameWidth === null ? '100%' : `${previewFrameWidth}px` }),
    [previewFrameWidth]
  )

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

      const handleMove = (moveEvent: PointerEvent) => {
        if (!active) {
          return
        }

        consoleState.setHeight(clampConsoleHeight(startHeight + startY - moveEvent.clientY))
      }

      const cleanup = () => {
        if (!active) {
          return
        }

        active = false
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

  const applyPreviewRatio = useCallback((preset: PreviewRatioPreset) => {
    const height = previewContentRef.current?.getBoundingClientRect().height ?? 0

    // A responsive preset should size the page viewport inside the preview pane,
    // not mutate the PaneShell track. Mutating the outer pane was what made the
    // preview spill across the chat surface after choosing desktop/ultrawide.
    setPreviewFrameWidth(previewWidthForRatio({ height, ratio: preset.ratio }))
    setActiveRatioLabel(`${preset.label} ${preset.ratioLabel}`)
  }, [])

  const fitPreviewToPane = useCallback(() => {
    setPreviewFrameWidth(null)
    setActiveRatioLabel(null)
  }, [])


  const navigatePreview = useCallback((rawAddress: string) => {
    const nextUrl = normalizePreviewAddress(rawAddress)

    if (!nextUrl) {
      setAddressDraft(currentUrl)

      return
    }

    setLoadError(null)
    setLoading(true)
    setCurrentUrl(nextUrl)
    setAddressDraft(nextUrl)

    const webview = webviewRef.current

    if (webview) {
      webview.setAttribute('src', nextUrl)

      if (webview instanceof HTMLIFrameElement) {
        webview.src = nextUrl
      }
    }
  }, [currentUrl])

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

  const toggleAnnotations = useCallback(() => {
    const next = !annotationOverlayOpen

    setAnnotationOverlayOpen(next)
    appendConsoleEntry({
      level: 1,
      message: next ? 'Preview annotation overlay enabled.' : 'Preview annotation overlay hidden.'
    })
  }, [annotationOverlayOpen, appendConsoleEntry])

  const toggleDebugOverlay = useCallback(() => {
    const webview = webviewRef.current
    const nativeDevToolsOpen = Boolean(devtoolsOpen || webview?.isDevToolsOpened?.())
    const next = !(debugOverlayOpen || nativeDevToolsOpen)

    setDebugOverlayOpen(next)

    if (webview?.openDevTools) {
      if (next) {
        webview.openDevTools()
        setDevtoolsOpen(true)
      } else {
        webview.closeDevTools?.()
        setDevtoolsOpen(false)
      }
    } else {
      setDevtoolsOpen(false)
    }

    if (next) {
      consoleState.setOpen(true)
    }

    appendConsoleEntry({
      level: 1,
      message: next
        ? devtoolsAvailable
          ? 'Preview debug opened. Native DevTools requested and the in-pane debug overlay is visible.'
          : 'Preview debug overlay opened. Native DevTools are unavailable for this preview engine.'
        : 'Preview debug hidden.'
    })
  }, [appendConsoleEntry, consoleState, debugOverlayOpen, devtoolsAvailable, devtoolsOpen])

  useEffect(() => {
    if (!setTitlebarToolGroup) {
      return
    }

    const tools: TitlebarTool[] = [
      ...(isWebPreview
        ? [
            {
              active: annotationOverlayOpen,
              icon: <Pencil className="size-3.5" />,
              id: `${TITLEBAR_GROUP_ID}-annotate`,
              label: annotationOverlayOpen ? 'Hide preview annotations' : 'Show preview annotations',
              onSelect: toggleAnnotations
            },
            {
              active: consoleOpen,
              icon: <PreviewConsoleTitlebarIcon consoleState={consoleState} />,
              id: `${TITLEBAR_GROUP_ID}-console`,
              label: consoleOpen ? copy.hideConsole : copy.showConsole,
              onSelect: () => consoleState.setOpen(open => !open)
            },
            {
              active: debugOverlayOpen || devtoolsOpen,
              icon: <Bug />,
              id: `${TITLEBAR_GROUP_ID}-devtools`,
              label: debugOverlayOpen || devtoolsOpen ? 'Hide preview debug' : 'Show preview debug',
              onSelect: toggleDebugOverlay
            }
          ]
        : [])
    ]

    setTitlebarToolGroup(TITLEBAR_GROUP_ID, tools)

    return () => setTitlebarToolGroup(TITLEBAR_GROUP_ID, [])
  }, [
    annotationOverlayOpen,
    consoleOpen,
    consoleState,
    copy,
    debugOverlayOpen,
    devtoolsOpen,
    isWebPreview,
    setTitlebarToolGroup,
    toggleAnnotations,
    toggleDebugOverlay
  ])

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

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()
    webviewRef.current = null
    setCurrentUrl(target.url)
    setAddressDraft(target.url)
    setAnnotationOverlayOpen(false)
    setDebugOverlayOpen(false)
    setDevtoolsAvailable(false)
    setDevtoolsOpen(false)
    setPreviewEngine(isWebPreview ? 'detecting' : 'local-file')
    setLoadError(null)
    consoleState.reset()
    setLoading(true)

    if (!isWebPreview) {
      setLoading(false)

      return
    }

    const webview = document.createElement('webview') as PreviewWebview
    webview.className = 'h-full w-full bg-transparent'
    webview.setAttribute('partition', webviewPartition)
    webview.setAttribute('src', target.url)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const supportsElectronWebview =
      typeof webview.reload === 'function' ||
      typeof webview.reloadIgnoringCache === 'function' ||
      typeof webview.openDevTools === 'function'

    if (!supportsElectronWebview) {
      const iframe = document.createElement('iframe') as HTMLIFrameElement & PreviewWebview

      iframe.className = 'h-full w-full border-0 bg-background'
      iframe.setAttribute('allow', 'clipboard-read; clipboard-write; fullscreen; microphone; camera')
      iframe.setAttribute('referrerpolicy', 'no-referrer')
      iframe.src = target.url

      const reloadIframe = (cacheBust = false) => {
        const nextUrl = new URL(iframe.src || target.url, window.location.href)

        if (cacheBust) {
          nextUrl.searchParams.set('__hermes_preview_reload', String(Date.now()))
        }

        iframe.src = nextUrl.toString()
        setLoading(true)
      }

      iframe.reload = () => reloadIframe(false)
      iframe.reloadIgnoringCache = () => reloadIframe(true)
      iframe.getURL = () => iframe.src

      const onIframeLoad = () => {
        setLoading(false)
        setLoadError(null)
        setCurrentUrl(iframe.src || target.url)
        setAddressDraft(iframe.src || target.url)
      }

      const onIframeError = () => {
        setLoading(false)
        setLoadError({ description: 'Browser iframe preview failed to load.', url: target.url })
      }

      iframe.addEventListener('load', onIframeLoad)
      iframe.addEventListener('error', onIframeError)
      host.appendChild(iframe)
      webviewRef.current = iframe
      setPreviewEngine('iframe-fallback')
      appendConsoleEntry({
        level: 1,
        message: 'Browser preview fallback active; Electron webview console and DevTools are unavailable in browser mode.'
      })

      return () => {
        iframe.removeEventListener('load', onIframeLoad)
        iframe.removeEventListener('error', onIframeError)
        host.replaceChildren()

        if (webviewRef.current === iframe) {
          webviewRef.current = null
        }
      }
    }

    setPreviewEngine('electron-webview')
    setDevtoolsAvailable(typeof webview.openDevTools === 'function')

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

    const onNavigate = (event: Event) => {
      const detail = event as Event & { url?: string }

      if (detail.url) {
        setLoadError(null)
        setCurrentUrl(detail.url)
        setAddressDraft(detail.url)
      }
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
    const onStop = () => setLoading(false)

    webview.addEventListener('console-message', onConsole)
    webview.addEventListener('did-fail-load', onFail)
    webview.addEventListener('did-navigate', onNavigate)
    webview.addEventListener('did-navigate-in-page', onNavigate)
    webview.addEventListener('did-start-loading', onStart)
    webview.addEventListener('did-stop-loading', onStop)
    host.appendChild(webview)
    webviewRef.current = webview

    return () => {
      webview.removeEventListener('console-message', onConsole)
      webview.removeEventListener('did-fail-load', onFail)
      webview.removeEventListener('did-navigate', onNavigate)
      webview.removeEventListener('did-navigate-in-page', onNavigate)
      webview.removeEventListener('did-start-loading', onStart)
      webview.removeEventListener('did-stop-loading', onStop)
      webview.remove()
    }
  }, [appendConsoleEntry, consoleState, copy, isWebPreview, target.url, webviewPartition])

  return (
    <aside className="relative flex h-full w-full min-w-0 flex-col overflow-hidden bg-transparent text-muted-foreground">
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {(!embedded || isWebPreview) && (
          <div className="pointer-events-none flex min-h-(--titlebar-height) items-center gap-1.5 border-b border-border/60 bg-background px-2 py-1">
            <form
              className="pointer-events-auto min-w-0 flex-1"
              onSubmit={event => {
                event.preventDefault()
                navigatePreview(addressDraft)
              }}
            >
              <label className="sr-only" htmlFor="preview-address">
                Preview address
              </label>
              <Tip label={copy.openTarget(currentUrl)}>
                <input
                  aria-label="Preview address"
                  className="h-6 w-full min-w-0 rounded-sm border border-transparent bg-(--ui-editor-surface-background)/55 px-2 text-xs font-medium text-foreground outline-none transition-colors placeholder:text-muted-foreground/55 hover:border-(--ui-stroke-quaternary) focus:border-(--ui-stroke-secondary) focus:bg-background"
                  id="preview-address"
                  onBlur={() => setAddressDraft(currentUrl)}
                  onChange={event => setAddressDraft(event.currentTarget.value)}
                  title={previewLabel || copy.fallbackTitle}
                  value={addressDraft}
                />
              </Tip>
            </form>
            {isWebPreview && (
              <div aria-label="Preview responsive sizes" className="pointer-events-auto flex shrink-0 items-center gap-1">
                <Tip label="Fit preview to pane">
                  <button
                    aria-label="Fit preview to pane"
                    className={cn(
                      'h-5 rounded-sm border px-1.5 text-[0.625rem] font-medium leading-none transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring',
                      previewFrameWidth === null
                        ? 'border-(--ui-stroke-secondary) bg-(--chrome-action-hover) text-foreground'
                        : 'border-(--ui-stroke-quaternary) text-muted-foreground/85 hover:border-(--ui-stroke-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
                    )}
                    onClick={fitPreviewToPane}
                    type="button"
                  >
                    Fit
                  </button>
                </Tip>
                {PREVIEW_RATIO_PRESETS.map(preset => {
                  const label = `${preset.label} ${preset.ratioLabel}`
                  const active = activeRatioLabel === label

                  return (
                    <Tip key={preset.id} label={`Set preview viewport to ${label}`}>
                      <button
                        aria-label={`Set preview viewport to ${label}`}
                        aria-pressed={active}
                        className={cn(
                          'h-5 rounded-sm border px-1.5 text-[0.625rem] font-medium leading-none transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring',
                          active
                            ? 'border-(--ui-stroke-secondary) bg-(--chrome-action-hover) text-foreground'
                            : 'border-(--ui-stroke-quaternary) text-muted-foreground/85 hover:border-(--ui-stroke-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
                        )}
                        onClick={() => applyPreviewRatio(preset)}
                        type="button"
                      >
                        {preset.ratioLabel}
                      </button>
                    </Tip>
                  )
                })}
              </div>
            )}
            {isWebPreview && (
              <div aria-label="Preview tools" className="pointer-events-auto flex shrink-0 items-center gap-1">
                <Tip label={annotationOverlayOpen ? 'Hide preview annotations' : 'Show preview annotations'}>
                  <button
                    aria-label={annotationOverlayOpen ? 'Hide preview annotations' : 'Show preview annotations'}
                    aria-pressed={annotationOverlayOpen}
                    className={cn(
                      'grid size-6 place-items-center rounded-sm border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring',
                      annotationOverlayOpen
                        ? 'border-sky-400/70 bg-sky-400/15 text-sky-200'
                        : 'border-(--ui-stroke-quaternary) text-muted-foreground/85 hover:border-(--ui-stroke-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
                    )}
                    onClick={toggleAnnotations}
                    type="button"
                  >
                    <Pencil className="size-3" />
                  </button>
                </Tip>
                <Tip label={debugOverlayOpen || devtoolsOpen ? 'Hide preview debug' : 'Show preview debug'}>
                  <button
                    aria-label={debugOverlayOpen || devtoolsOpen ? 'Hide preview debug' : 'Show preview debug'}
                    aria-pressed={debugOverlayOpen || devtoolsOpen}
                    className={cn(
                      'grid size-6 place-items-center rounded-sm border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring',
                      debugOverlayOpen || devtoolsOpen
                        ? 'border-amber-400/70 bg-amber-400/15 text-amber-200'
                        : 'border-(--ui-stroke-quaternary) text-muted-foreground/85 hover:border-(--ui-stroke-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
                    )}
                    onClick={toggleDebugOverlay}
                    type="button"
                  >
                    <Bug className="size-3" />
                  </button>
                </Tip>
              </div>
            )}
          </div>
        )}

        <div
          className="pointer-events-auto relative min-h-0 flex-1 overflow-auto bg-transparent overscroll-contain [contain:paint]"
          data-preview-viewport=""
          ref={previewContentRef}
        >
          <div
            className="relative h-full min-h-full max-w-none shrink-0 overflow-hidden bg-transparent"
            data-preview-frame=""
            style={previewFrameStyle}
          >
            <div
              className={cn(
                'absolute inset-0 flex bg-transparent',
                (!isWebPreview || loadError) && 'pointer-events-none opacity-0'
              )}
              ref={hostRef}
            />
            {!isWebPreview && <LocalFilePreview reloadKey={localReloadKey} target={target} />}
            {loadError && (
              <PreviewLoadError
                consoleHeight={consoleOpen ? consoleHeight : 0}
                error={loadError}
                onRestartServer={target.kind === 'url' && onRestartServer ? () => void restartServer() : undefined}
                onRetry={reloadPreview}
                restarting={restartingServer}
              />
            )}

            {isWebPreview && annotationOverlayOpen && (
              <div
                aria-label="Preview annotation overlay active"
                className="pointer-events-none absolute inset-0 z-10 overflow-hidden border border-sky-400/60 bg-sky-400/5 text-[0.625rem] font-semibold uppercase tracking-[0.12em] text-sky-100 shadow-[inset_0_0_0_1px_rgba(125,211,252,0.18)]"
                data-preview-annotations=""
              >
                <div className="absolute left-3 top-3 rounded-sm border border-sky-300/50 bg-black/65 px-2 py-1 text-sky-100 shadow-lg">
                  Annotation overlay active
                </div>
                <div className="absolute left-[18%] top-[18%] rounded-sm border border-sky-300/70 bg-black/70 px-1.5 py-0.5 text-sky-100">[1]</div>
                <div className="absolute right-[18%] top-[26%] rounded-sm border border-sky-300/70 bg-black/70 px-1.5 py-0.5 text-sky-100">[2]</div>
                <div className="absolute bottom-[24%] left-[46%] rounded-sm border border-sky-300/70 bg-black/70 px-1.5 py-0.5 text-sky-100">[3]</div>
              </div>
            )}

            {isWebPreview && debugOverlayOpen && (
              <div
                aria-label="Preview debug overlay"
                className="pointer-events-auto absolute right-3 top-3 z-30 w-72 max-w-[calc(100%-1.5rem)] rounded-md border border-amber-400/40 bg-black/80 p-3 text-[0.6875rem] text-amber-50 shadow-2xl backdrop-blur"
                data-preview-debug-overlay=""
              >
                <div className="mb-2 flex items-center gap-2 text-[0.65rem] font-semibold uppercase tracking-[0.14em] text-amber-200">
                  <Bug className="size-3" />
                  Preview debug
                </div>
                <dl className="grid grid-cols-[4.75rem_1fr] gap-x-2 gap-y-1 font-mono text-[0.65rem] normal-case tracking-normal text-amber-50/85">
                  <dt className="text-amber-200/70">Engine</dt>
                  <dd>{previewEngine}</dd>
                  <dt className="text-amber-200/70">Native</dt>
                  <dd>{devtoolsAvailable ? 'DevTools available' : 'DevTools unavailable'}</dd>
                  <dt className="text-amber-200/70">Frame</dt>
                  <dd>{previewFrameWidth === null ? 'fit pane' : `${previewFrameWidth}px${activeRatioLabel ? ` (${activeRatioLabel})` : ''}`}</dd>
                  <dt className="text-amber-200/70">URL</dt>
                  <dd className="truncate" title={currentUrl}>{compactUrl(currentUrl)}</dd>
                </dl>
              </div>
            )}

            {isWebPreview && consoleOpen && (
              <PreviewConsolePanel
                consoleBodyRef={consoleBodyRef}
                consoleShouldStickRef={consoleShouldStickRef}
                consoleState={consoleState}
                startConsoleResize={startConsoleResize}
              />
            )}
          </div>
        </div>
      </div>
    </aside>
  )
}
