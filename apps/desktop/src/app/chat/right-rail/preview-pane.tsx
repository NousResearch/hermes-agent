import { useStore } from '@nanostores/react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'

import type { SetTitlebarToolGroup, TitlebarTool } from '@/app/shell/titlebar-controls'
import { type Translate, useTranslation } from '@/i18n'
import { Bug } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { $previewServerRestart, failPreviewServerRestart, type PreviewTarget } from '@/store/preview'

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

type PreviewWebview = HTMLElement & {
  closeDevTools?: () => void
  getURL?: () => string
  isDevToolsOpened?: () => boolean
  openDevTools?: () => void
  reload?: () => void
  reloadIgnoringCache?: () => void
}

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

function loadErrorTitle(error: PreviewLoadErrorState, t: Translate): string {
  const description = error.description.toLowerCase()

  if (description.includes('module script') || description.includes('mime type')) {
    return t('preview.errors.bootFailed')
  }

  if (description.includes('connection') || description.includes('refused') || description.includes('not found')) {
    return t('preview.errors.serverNotFound')
  }

  return t('preview.errors.loadFailed')
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
  restarting,
  t
}: {
  consoleHeight?: number
  error: PreviewLoadErrorState
  onRestartServer?: () => void
  onRetry: () => void
  restarting?: boolean
  t: Translate
}) {
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
      primaryAction={{ label: t('common.tryAgain'), onClick: onRetry }}
      secondaryAction={
        onRestartServer
          ? {
              disabled: restarting,
              label: restarting ? t('preview.restartingHermes') : t('preview.askRestartServer'),
              onClick: onRestartServer
            }
          : undefined
      }
      title={loadErrorTitle(error, t)}
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
  const t = useTranslation()
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
  const [currentUrl, setCurrentUrl] = useState(target.url)
  const [devtoolsOpen, setDevtoolsOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<PreviewLoadErrorState | null>(null)
  const [localReloadKey, setLocalReloadKey] = useState(0)
  const isWebPreview = target.kind === 'url' || (target.previewKind === 'html' && target.renderMode !== 'source')
  const currentLabel = compactUrl(currentUrl)

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
        message: t('preview.console.restartLooking', { taskId })
      })

      notify({
        kind: 'info',
        title: t('preview.notifications.restartStarted'),
        message: t('preview.notifications.watchConsole'),
        durationMs: 4000
      })
    } catch (error) {
      appendConsoleEntry({
        level: 2,
        message: t('preview.console.restartStartFailed', { error: error instanceof Error ? error.message : String(error) })
      })
      notifyError(error, t('preview.notifications.restartFailed'))
    }
  }, [appendConsoleEntry, consoleState, currentUrl, onRestartServer, t])

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
      ...(isWebPreview
        ? [
            {
              active: consoleOpen,
              icon: <PreviewConsoleTitlebarIcon consoleState={consoleState} />,
              id: `${TITLEBAR_GROUP_ID}-console`,
              label: consoleOpen ? t('preview.hideConsole') : t('preview.showConsole'),
              onSelect: () => consoleState.setOpen(open => !open)
            },
            {
              active: devtoolsOpen,
              icon: <Bug />,
              id: `${TITLEBAR_GROUP_ID}-devtools`,
              label: devtoolsOpen ? t('preview.hideDevTools') : t('preview.openDevTools'),
              onSelect: toggleDevTools
            }
          ]
        : [])
    ]

    setTitlebarToolGroup(TITLEBAR_GROUP_ID, tools)

    return () => setTitlebarToolGroup(TITLEBAR_GROUP_ID, [])
  }, [consoleOpen, consoleState, devtoolsOpen, isWebPreview, setTitlebarToolGroup, t, toggleDevTools])

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
            ? t('preview.console.restartComplete', { message: previewServerRestart.message ? `: ${previewServerRestart.message}` : '' })
            : t('preview.console.restartFailed', { error: previewServerRestart.message || t('common.unknownError') })
    })

    if (previewServerRestart.status === 'complete') {
      reloadPreview()
      notify({
        kind: 'success',
        title: t('preview.notifications.restartComplete'),
        message: previewServerRestart.message?.slice(0, 160) || t('preview.notifications.reloadingNow'),
        durationMs: 3500
      })
    } else if (previewServerRestart.status === 'error') {
      notify({
        kind: 'warning',
        title: t('preview.notifications.restartFailed'),
        message: previewServerRestart.message?.slice(0, 200) || t('preview.notifications.restartCouldNotComplete'),
        durationMs: 6000
      })
    }
  }, [appendConsoleEntry, currentUrl, previewServerRestart, reloadPreview, t, target.url])

  useEffect(() => {
    if (!restartingServer || !previewServerRestart) {
      return
    }

    const taskId = previewServerRestart.taskId

    const timer = window.setTimeout(() => {
      failPreviewServerRestart(
        taskId,
        t('preview.notifications.restartStillRunning')
      )
    }, SERVER_RESTART_TIMEOUT_MS)

    return () => window.clearTimeout(timer)
  }, [previewServerRestart, restartingServer, t])

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
      message: t('preview.console.workspaceChanged')
    })
    reloadPreview()
  }, [appendConsoleEntry, reloadPreview, reloadRequest, t, target.kind])

  useEffect(() => {
    if (
      target.kind !== 'file' ||
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
            ? t('preview.console.fileChanged', { path: compactUrl(changedUrl) })
            : t('preview.console.filesChanged', { count: changedCount, path: compactUrl(changedUrl) })
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
          message: t('preview.console.watchFailed', { error: error instanceof Error ? error.message : String(error) })
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
  }, [appendConsoleEntry, reloadPreview, t, target.kind, target.url])

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()
    webviewRef.current = null
    setCurrentUrl(target.url)
    setDevtoolsOpen(false)
    setLoadError(null)
    consoleState.reset()
    setLoading(true)

    if (!isWebPreview) {
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
          description:
            t('preview.errors.moduleMimeDescription'),
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
        message: `Load failed${errorCode ? ` (${errorCode})` : ''}: ${
          detail.errorDescription || detail.validatedURL || t('common.unknownError')
        }`
      })
      setLoadError({
        code: errorCode,
        description: detail.errorDescription || t('preview.errors.pageUnreachable'),
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
  }, [appendConsoleEntry, consoleState, isWebPreview, t, target.url])

  return (
    <aside className="relative flex h-full w-full min-w-0 flex-col overflow-hidden bg-transparent text-muted-foreground">
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {!embedded && (
          <div className="pointer-events-none flex min-h-(--titlebar-height) items-center gap-1.5 border-b border-border/60 bg-background px-2 py-1">
            <div className="min-w-0 flex-1">
              <a
                className="pointer-events-auto inline max-w-full truncate text-left text-xs font-medium text-foreground underline-offset-4 decoration-current/20 transition-colors hover:text-primary hover:underline"
                href={currentUrl}
                rel="noreferrer"
                target="_blank"
                title={t('preview.openUrl', { url: currentUrl })}
              >
                {previewLabel || t('preview.title')}
              </a>
            </div>
          </div>
        )}

        <div
          className="pointer-events-auto relative min-h-0 flex-1 overflow-hidden bg-transparent"
          ref={previewContentRef}
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
              t={t}
            />
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
    </aside>
  )
}
