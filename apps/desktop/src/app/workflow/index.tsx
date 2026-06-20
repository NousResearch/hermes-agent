import { useStore } from '@nanostores/react'
import { type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { DesktopWorkflowBackendStatus } from '@/global'
import { useI18n } from '@/i18n'
import { $workflowCopilotExpanded, $workflowCopilotOpen, toggleWorkflowCopilotExpanded } from '@/store/workflow'

const LANGFLOW_URL = 'http://127.0.0.1:7860'
const BACKEND_STATUS_POLL_MS = 1000
const COPILOT_COMPACT_WIDTH = 'clamp(24rem, 32vw, 30rem)'
const COPILOT_EXPANDED_WIDTH = 'min(46rem, max(28rem, 42vw))'
const KNOWLEDGE_PATH = '/assets/knowledge-bases'
// 注入到知识库视图的 webview:藏掉 Langflow 顶栏与左侧项目栏,只留知识库主体,让它
// 在爱马仕里像原生页面(统一走左侧「知识库」入口,不暴露 Langflow 自带导航)。
const HIDE_LANGFLOW_CHROME = `(function(){var i='kari-kb-chrome-hide';if(document.getElementById(i))return;var s=document.createElement('style');s.id=i;s.textContent='[data-testid="app-header"],[data-testid="project-sidebar"]{display:none !important;}';document.head.appendChild(s);})();`

type WorkflowWebview = HTMLElement & {
  executeJavaScript?: (code: string, userGesture?: boolean) => Promise<unknown>
  getURL?: () => string
  reloadIgnoringCache?: () => void
}

interface LoadError {
  code?: number
  description?: string
  url?: string
}

function safelyExecuteWebviewScript(webview: WorkflowWebview, code: string) {
  try {
    void webview.executeJavaScript?.(code, false)?.catch(() => {
      /* best-effort: failed loads / old webview mocks may reject injection */
    })
  } catch {
    /* Electron throws synchronously if the webview has not emitted dom-ready yet. */
  }
}

// 同一套本地 Langflow + backend 生命周期,两个左侧入口共用一个组件:
//   view='canvas'(默认,/workflow)   → 画布
//   view='knowledge'(/knowledge)     → 知识库(/assets/knowledge-bases,藏掉 Langflow 外壳)
// 两者 partition 一致 = 同一登录用户 = 同一份知识库;copilot 仅画布需要。
export function WorkflowView({ view = 'canvas' }: { view?: 'canvas' | 'knowledge' } = {}) {
  const knowledge = view === 'knowledge'
  const { t } = useI18n()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const webviewRef = useRef<WorkflowWebview | null>(null)
  // 右栏:爱马仕 copilot(本机 dashboard 的 /copilot 气泡聊天,与桌面原生聊天同一后端网关)。仅画布。
  const chatHostRef = useRef<HTMLDivElement | null>(null)
  const [chatUrl, setChatUrl] = useState<string | null>(null)
  // 打开工作流时右侧 copilot 默认收起;首次展开才真正挂载(之后保持,折叠不丢会话)。
  const chatOpen = useStore($workflowCopilotOpen)
  const chatExpanded = useStore($workflowCopilotExpanded)
  const [chatMounted, setChatMounted] = useState(false)
  const [backendAttempt, setBackendAttempt] = useState(0)
  const [backendStatus, setBackendStatus] = useState<DesktopWorkflowBackendStatus | null>(null)
  // Account status is checked only to let the backend inject Kari credentials
  // when present. Missing/broken account state must not block the local canvas.
  const [authChecked, setAuthChecked] = useState<boolean | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<LoadError | null>(null)
  const [mountKey, setMountKey] = useState(0)
  // 引擎(langflow)改成**手动启动**:工作流/知识库都靠它,弱机器(<8GB)不强行开,用完可关闭释放内存。
  const [started, setStarted] = useState(false)
  const [totalGb, setTotalGb] = useState<number | null>(null)

  useEffect(() => {
    const wf = window.hermesDesktop?.workflow

    wf?.totalMemoryGb?.()
      .then(setTotalGb)
      .catch(() => setTotalGb(null))

    // langflow 可能已被别处(如知识库入库)拉起;已在跑就直接进入"已启动"态,不再显示启动入口。
    wf?.status?.()
      .then(status => {
        if (status && (status.state === 'ready' || status.state === 'starting')) {
          setStarted(true)
        }
      })
      .catch(() => {
        /* 拿不到状态就按未启动处理 */
      })
  }, [])

  const stopEngine = useCallback(async () => {
    setStarted(false)
    setBackendStatus(null)
    setLoadError(null)

    try {
      await window.hermesDesktop?.workflow?.stop?.()
    } catch {
      /* 关不掉也回到未启动态;下次启动会幂等处理 */
    }
  }, [])

  const targetUrl = backendStatus?.url || LANGFLOW_URL
  // 知识库视图加载同一个 Langflow 的 /assets/knowledge-bases;画布视图加载根路径。
  const mainUrl = knowledge ? `${targetUrl.replace(/\/+$/, '')}${KNOWLEDGE_PATH}` : targetUrl
  const backendReady = backendStatus?.state === 'ready'
  const bridgeUnavailable = t.workflow.bridgeUnavailable
  const chatPanelWidth = chatExpanded ? COPILOT_EXPANDED_WIDTH : COPILOT_COMPACT_WIDTH

  const chatPanelStyle = useMemo(
    () =>
      ({
        '--workflow-copilot-width': chatPanelWidth,
        width: chatOpen ? 'var(--workflow-copilot-width)' : '0px'
      }) as CSSProperties,
    [chatOpen, chatPanelWidth]
  )

  const backendError =
    backendStatus?.state === 'error' || backendStatus?.state === 'exited'
      ? {
          description: backendStatus.error || backendStatus.state,
          url: backendStatus.url
        }
      : null

  const displayError = backendError || loadError

  const reload = useCallback(() => {
    setLoadError(null)
    setLoading(true)

    if (!backendReady) {
      setBackendAttempt(attempt => attempt + 1)

      return
    }

    if (webviewRef.current?.reloadIgnoringCache) {
      webviewRef.current.reloadIgnoringCache()

      return
    }

    setMountKey(key => key + 1)
  }, [backendReady])

  // Account login/logout restarts the Langflow backend in the main process. The
  // already-loaded webview points at the now-restarted backend, so reload it to
  // pick up the new Kari token state (nodes appear on login / disappear on
  // logout) without forcing the user to hit refresh.
  useEffect(() => {
    const workflow = window.hermesDesktop?.workflow

    if (!workflow?.onRestarted) {
      return
    }

    return workflow.onRestarted(() => {
      reload()
    })
  }, [reload])

  // Check account status before the initial start so any available Kari token
  // is on disk before Langflow inherits env. No account is valid: the canvas
  // still boots, only Kari media/billing nodes are unavailable.
  useEffect(() => {
    let cancelled = false
    const workflow = window.hermesDesktop?.workflow

    if (!workflow?.authStatus) {
      setAuthChecked(true)

      return
    }

    void workflow
      .authStatus()
      .then(status => {
        if (cancelled) {
          return
        }

        setAuthChecked(true)
      })
      .catch(() => {
        if (!cancelled) {
          setAuthChecked(true)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let pollTimer: ReturnType<typeof setTimeout> | null = null
    const workflow = window.hermesDesktop?.workflow

    const applyStatus = (status: DesktopWorkflowBackendStatus) => {
      if (cancelled) {
        return
      }

      setBackendStatus(status)
    }

    const schedulePoll = () => {
      if (pollTimer) {
        clearTimeout(pollTimer)
      }

      pollTimer = setTimeout(async () => {
        if (!workflow || cancelled) {
          return
        }

        try {
          const status = await workflow.status()
          applyStatus(status)

          if (status.state === 'starting') {
            schedulePoll()
          }
        } catch (error) {
          applyStatus({
            error: error instanceof Error ? error.message : String(error),
            external: false,
            pid: null,
            root: '',
            state: 'error',
            url: LANGFLOW_URL
          })
        }
      }, BACKEND_STATUS_POLL_MS)
    }

    const startBackend = async () => {
      setLoadError(null)
      setBackendStatus(null)

      if (!workflow) {
        applyStatus({
          error: bridgeUnavailable,
          external: false,
          pid: null,
          root: '',
          state: 'error',
          url: LANGFLOW_URL
        })

        return
      }

      try {
        const status = await workflow.start()
        applyStatus(status)

        if (status.state === 'starting') {
          schedulePoll()
        }
      } catch (error) {
        applyStatus({
          error: error instanceof Error ? error.message : String(error),
          external: false,
          pid: null,
          root: '',
          state: 'error',
          url: LANGFLOW_URL
        })
      }
    }

    if (authChecked && started) {
      void startBackend()
    }

    return () => {
      cancelled = true

      if (pollTimer) {
        clearTimeout(pollTimer)
      }
    }
  }, [backendAttempt, bridgeUnavailable, authChecked, started])

  // 主 webview:画布或知识库(由 view 决定 src)。partition 一致 → 同一份数据。
  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()
    webviewRef.current = null
    setLoadError(null)

    if (!backendReady) {
      setLoading(false)

      return
    }

    setLoading(true)

    const webview = document.createElement('webview') as WorkflowWebview
    webview.className = 'flex h-full w-full flex-1 bg-background'
    webview.setAttribute('partition', 'persist:hermes-workflow')
    webview.setAttribute('src', mainUrl)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const onFail = (event: Event) => {
      const detail = event as Event & {
        errorCode?: number
        errorDescription?: string
        isMainFrame?: boolean
        validatedURL?: string
      }

      if (detail.errorCode === -3 || detail.isMainFrame === false) {
        return
      }

      setLoadError({
        code: detail.errorCode,
        description: detail.errorDescription,
        url: detail.validatedURL || webview.getURL?.() || mainUrl
      })
      setLoading(false)
    }

    const onStart = () => {
      setLoadError(null)
      setLoading(true)
    }

    const injectKnowledgeChrome = () => {
      if (knowledge) {
        safelyExecuteWebviewScript(webview, HIDE_LANGFLOW_CHROME)
      }
    }

    const onDomReady = () => {
      injectKnowledgeChrome()
    }

    const onStop = () => {
      setLoading(false)
      // 知识库视图:加载后藏掉 Langflow 自带外壳,只留知识库主体。
      injectKnowledgeChrome()
    }

    webview.addEventListener('did-fail-load', onFail)
    webview.addEventListener('dom-ready', onDomReady)
    webview.addEventListener('did-start-loading', onStart)
    webview.addEventListener('did-stop-loading', onStop)
    host.appendChild(webview)
    webviewRef.current = webview

    return () => {
      webview.removeEventListener('did-fail-load', onFail)
      webview.removeEventListener('dom-ready', onDomReady)
      webview.removeEventListener('did-start-loading', onStart)
      webview.removeEventListener('did-stop-loading', onStop)
      webview.remove()
    }
  }, [backendReady, mountKey, mainUrl, knowledge])

  // copilot 面板宽度注入到画布(仅画布;知识库视图没有右栏)。
  useEffect(() => {
    if (knowledge) {
      return
    }

    const webview = webviewRef.current

    if (!webview?.executeJavaScript) {
      return
    }

    const width = chatOpen ? chatPanelWidth : '0px'
    const code = `document.documentElement.style.setProperty('--kari-panel-width', ${JSON.stringify(width)})`
    void webview.executeJavaScript(code, false).catch(() => {
      /* best-effort: old webview mocks and failed loads may not accept injection */
    })
  }, [chatOpen, chatPanelWidth, knowledge])

  // Resolve the local dashboard base URL → right-panel chat = `${baseUrl}/copilot`
  // (the desktop already spawns this dashboard; same gateway as the native chat,
  // but a chrome-less chat-bubble page). Canvas only.
  useEffect(() => {
    if (!authChecked || knowledge) {
      return
    }

    let cancelled = false
    const desktop = window.hermesDesktop

    if (!desktop?.getConnection) {
      return
    }

    void desktop
      .getConnection()
      .then(conn => {
        if (!cancelled && conn?.baseUrl) {
          setChatUrl(`${conn.baseUrl.replace(/\/$/, '')}/copilot`)
        }
      })
      .catch(() => {
        /* chat panel is best-effort; canvas still works without it */
      })

    return () => {
      cancelled = true
    }
  }, [authChecked, knowledge])

  // Mount the EasyHermes copilot webview (right panel), independent of the
  // langflow canvas backend lifecycle. Canvas only.
  useEffect(() => {
    if (!knowledge && chatOpen) {
      setChatMounted(true)
    }
  }, [chatOpen, knowledge])

  useEffect(() => {
    const host = chatHostRef.current

    if (!host || !chatUrl || !chatMounted) {
      return
    }

    host.replaceChildren()

    const webview = document.createElement('webview') as WorkflowWebview
    webview.className = 'h-full w-full bg-background'
    webview.setAttribute('partition', 'persist:hermes-workflow-chat')
    webview.setAttribute('src', chatUrl)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')
    host.appendChild(webview)

    return () => {
      webview.remove()
    }
  }, [chatUrl, chatMounted])

  if (authChecked === null) {
    return (
      <section className="grid h-full min-h-0 flex-1 place-items-center bg-background">
        <Codicon className="text-sm text-muted-foreground" name="loading" spinning />
      </section>
    )
  }

  // 引擎(langflow)未启动:手动启动入口。弱机器(<8GB)给内存提醒。
  if (!started) {
    const lowMem = typeof totalGb === 'number' && totalGb > 0 && totalGb < 8

    return (
      <section
        aria-label={knowledge ? t.workflow.knowledge : t.workflow.title}
        className="grid h-full min-h-0 flex-1 place-items-center bg-background px-6"
      >
        <div className="flex w-full max-w-sm flex-col items-center gap-4 text-center">
          <div className="grid size-14 place-items-center rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) text-(--ui-text-secondary)">
            <svg
              aria-hidden="true"
              className="size-7"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path d="M12 3v9" />
              <path d="M6.8 7.2a7.5 7.5 0 1 0 10.4 0" />
            </svg>
          </div>
          <div className="space-y-1.5">
            <h1 className="text-sm font-semibold text-foreground">{t.workflow.startEngineTitle}</h1>
            <p className="text-xs leading-5 text-muted-foreground">{t.workflow.startEngineHint}</p>
          </div>
          {lowMem && (
            <p className="rounded-md bg-amber-500/10 px-3 py-2 text-[0.6875rem] leading-5 text-amber-600 dark:text-amber-400">
              {t.workflow.lowMemWarn(Math.round(totalGb ?? 0))}
            </p>
          )}
          <button
            className="inline-flex items-center gap-2 rounded-lg bg-foreground px-4 py-2 text-xs font-medium text-background transition hover:opacity-90"
            onClick={() => setStarted(true)}
            type="button"
          >
            <svg aria-hidden="true" className="size-3.5" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" fill="currentColor" />
            </svg>
            {t.workflow.startBtn}
          </button>
          <p className="text-[0.625rem] leading-4 text-muted-foreground/80">{t.workflow.firstStartNote}</p>
        </div>
      </section>
    )
  }

  return (
    <section
      aria-label={knowledge ? t.workflow.knowledge : t.workflow.title}
      className="relative flex h-full min-h-0 flex-1 overflow-hidden bg-background"
    >
      <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col">
        {/* 引擎工具条:运行中才显示。放在 webview **上方**(不重叠)—— Electron 的原生 webview 层会盖住
            普通 DOM,绝对定位的按钮会被遮,所以关闭按钮必须在画布外的独立条里。 */}
        {backendReady && (
          <div className="flex h-9 shrink-0 items-center justify-between border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) px-3">
            <span className="flex items-center gap-1.5 text-[0.6875rem] font-medium text-(--ui-text-secondary)">
              <span className="size-1.5 rounded-full bg-emerald-500" />
              {t.workflow.engineRunning}
            </span>
            <button
              className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[0.6875rem] font-medium text-(--ui-text-secondary) transition hover:bg-(--ui-control-hover-background) hover:text-foreground"
              onClick={() => void stopEngine()}
              type="button"
            >
              <svg aria-hidden="true" className="size-3" viewBox="0 0 24 24">
                <rect fill="currentColor" height="12" rx="2" width="12" x="6" y="6" />
              </svg>
              {t.workflow.stopBtn}
            </button>
          </div>
        )}

        <div className="relative min-h-0 flex-1">
          <div className="h-full w-full" ref={hostRef} />

          {!backendReady && !displayError && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-background/80">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Codicon className="text-sm" name="loading" spinning />
              <span>{t.workflow.starting}</span>
            </div>
          </div>
        )}

        {backendReady && loading && !displayError && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-background/80">
            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
              <Codicon className="text-sm" name="loading" spinning />
              <span>{t.workflow.loading}</span>
            </div>
          </div>
        )}

        {displayError && (
          <div className="absolute inset-0 grid place-items-center bg-background px-6" role="alert">
            <div className="flex w-full max-w-md flex-col items-center gap-4 text-center">
              <div className="grid size-10 place-items-center rounded-md border border-border/70 bg-muted/40 text-muted-foreground">
                <Codicon className="text-lg" name="debug-disconnect" />
              </div>
              <div className="space-y-1.5">
                <h1 className="text-sm font-semibold text-foreground">{t.workflow.backendUnavailable}</h1>
                <p className="text-xs leading-5 text-muted-foreground">
                  {t.workflow.backendUnavailableDetail(displayError.url || targetUrl)}
                </p>
                {displayError.description && (
                  <p className="text-[0.6875rem] leading-4 text-muted-foreground/80">{displayError.description}</p>
                )}
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                <Button onClick={reload} size="sm" type="button" variant="secondary">
                  <Codicon name="refresh" />
                  {t.common.refresh}
                </Button>
              </div>
            </div>
          </div>
        )}
        </div>
      </div>

      {!knowledge && chatUrl && (
        <>
          <div
            aria-hidden={!chatOpen}
            className="h-full shrink-0 overflow-hidden transition-[width] duration-200 ease-out"
            data-expanded={String(chatExpanded)}
            data-testid="workflow-copilot-panel"
            style={chatPanelStyle}
          >
            <div
              className={`flex h-full w-full flex-col border-l border-(--ui-stroke-secondary) bg-[color:var(--ui-bg-elevated)] shadow-[var(--shadow-elevated,_0_18px_50px_rgba(0,0,0,0.12))] ${
                chatOpen ? 'opacity-100' : 'pointer-events-none opacity-0'
              }`}
            >
              <header className="flex h-10 shrink-0 items-center justify-between border-b border-(--ui-stroke-tertiary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_96%,transparent)] px-2.5">
                <div className="flex min-w-0 items-center gap-2 text-[0.75rem] font-medium text-(--ui-text-secondary)">
                  <span className="grid size-5 shrink-0 place-items-center rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-control-hover-background)">
                    <Codicon name="comment-discussion" size="0.8125rem" />
                  </span>
                  <span className="truncate">爱马仕 Copilot</span>
                </div>
                <div className="flex shrink-0 items-center gap-1">
                  <Button
                    aria-label={chatExpanded ? '还原爱马仕 Copilot' : '展开爱马仕 Copilot'}
                    className="size-7 p-0 text-(--ui-text-secondary) hover:text-foreground"
                    onClick={toggleWorkflowCopilotExpanded}
                    title={chatExpanded ? '还原爱马仕 Copilot' : '展开爱马仕 Copilot'}
                    type="button"
                    variant="ghost"
                  >
                    <Codicon name={chatExpanded ? 'screen-normal' : 'screen-full'} size="0.875rem" />
                  </Button>
                </div>
              </header>
              <div className="min-h-0 flex-1 bg-background">
                {chatMounted && <div className="h-full w-full" data-testid="workflow-copilot-host" ref={chatHostRef} />}
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  )
}
