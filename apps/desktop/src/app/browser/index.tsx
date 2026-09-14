import {
  type ChangeEvent,
  type FormEvent,
  type MouseEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'
import { $rightRailActiveTabId, selectRightRailTab } from '@/store/layout'
import {
  $previewTabs,
  $sessionPreviewTabs,
  activeSessionKey,
  closeRightRail,
  openWorkstationBrowserPreview
} from '@/store/preview'

import { TaskJournalDrawer } from './task-journal-drawer'
import { TaskRail } from './task-rail'
import type {
  BrowserTask,
  WorkstationBrowserBounds,
  WorkstationBrowserState,
  WorkstationBrowserTabState,
  WorkstationResourceSnapshot
} from './types'

function useOptionalNavigate(): null | ReturnType<typeof useNavigate> {
  try {
    return useNavigate()
  } catch {
    return null
  }
}

const EMPTY_STATE: WorkstationBrowserState = {
  runtime: 'electron-chromium',
  ready: false,
  attached: false,
  viewportHost: null,
  backgroundCapable: true,
  paused: false,
  controlOwner: 'agent',
  controlReady: false,
  profilePath: '',
  cacheBytes: null,
  activeTabId: null,
  tabs: [],
  tasks: [],
  downloads: [],
  lastError: null
}

function cacheLabel(bytes: number | null): string {
  if (bytes == null) {
    return '—'
  }
  const mb = bytes / (1024 * 1024)

  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`
}

function shortTitle(tab: WorkstationBrowserTabState): string {
  if (tab.title && tab.title !== 'New Tab') {
    return tab.title
  }

  if (!tab.url || tab.url === 'about:blank') {
    return 'New Tab'
  }

  try {
    return new URL(tab.url).hostname || tab.url
  } catch {
    return tab.url
  }
}

function rectToBounds(rect: DOMRect): WorkstationBrowserBounds {
  return {
    x: Math.max(0, Math.round(rect.left)),
    y: Math.max(0, Math.round(rect.top)),
    width: Math.max(1, Math.round(rect.width)),
    height: Math.max(1, Math.round(rect.height))
  }
}

function formatBrowserError(error: string): string {
  if (error === 'stale_or_unknown_ref') {
    return 'Element reference expired or was detached by page (stale_or_unknown_ref)'
  }

  if (error === 'element_not_visible') {
    return 'Element is not visible in current viewport (element_not_visible)'
  }

  if (error === 'element_unavailable') {
    return 'Element is unavailable for interaction (element_unavailable)'
  }

  return error
}

export function BrowserView() {
  const bridge = window.hermesDesktop?.workstationBrowser
  const hostRef = useRef<HTMLDivElement | null>(null)
  const [state, setState] = useState<WorkstationBrowserState>(EMPTY_STATE)
  const [address, setAddress] = useState('')
  const [busy, setBusy] = useState(false)
  const [showDownloads, setShowDownloads] = useState(false)
  const [auditTaskId, setAuditTaskId] = useState<string | null>(null)
  const [resourceSnapshot, setResourceSnapshot] = useState<WorkstationResourceSnapshot | null>(null)

  const activeTab = useMemo(
    () => state.tabs.find(tab => tab.id === state.activeTabId) ?? state.tabs[0] ?? null,
    [state.activeTabId, state.tabs]
  )

  useEffect(() => {
    if (activeTab?.url && document.activeElement?.getAttribute('data-browser-address') !== 'true') {
      setAddress(activeTab.url === 'about:blank' ? '' : activeTab.url)
    }
  }, [activeTab?.url])

  useLayoutEffect(() => {
    // When viewing Browser Hub (/browser), the full-screen browser is active.
    // The Right Rail must automatically close so it doesn't duplicate the browser.
    const activeKey = activeSessionKey()
    const stashedTabs = $previewTabs.get()
    const stashedActiveId = $rightRailActiveTabId.get()

    if (stashedTabs.length > 0) {
      if (activeKey) {
        const map = { ...$sessionPreviewTabs.get(), [activeKey]: stashedTabs }
        $sessionPreviewTabs.set(map)
      }

      closeRightRail()
    }

    return () => {
      // When leaving Browser Hub to return to Chat, restore the Right Rail if it was previously open
      const restoreKey = activeSessionKey()

      if (restoreKey) {
        const saved = $sessionPreviewTabs.get()[restoreKey] ?? stashedTabs

        if (saved && saved.length > 0) {
          $previewTabs.set(saved)
          selectRightRailTab(stashedActiveId || saved[0].id)
        }
      }
    }
  }, [])

  const run = useCallback(
    async (fn: () => Promise<WorkstationBrowserState>) => {
      if (!bridge) {
        return
      }

      try {
        setBusy(true)
        setState(await fn())
      } finally {
        setBusy(false)
      }
    },
    [bridge]
  )

  const navigate = useOptionalNavigate()

  const publishBounds = useCallback(
    async (attach = false) => {
      if (!bridge || !hostRef.current) {
        return
      }
      const rect = hostRef.current.getBoundingClientRect()

      if (rect.width < 1 || rect.height < 1) {
        return
      }
      const bounds = rectToBounds(rect)
      setState(await (attach ? bridge.attach(bounds, 'hub') : bridge.setBounds(bounds, 'hub')))
    },
    [bridge]
  )

  const transferToHub = useCallback(async () => {
    if (!bridge || !hostRef.current) {
      return
    }
    const rect = hostRef.current.getBoundingClientRect()

    if (rect.width < 1 || rect.height < 1) {
      return
    }
    const bounds = rectToBounds(rect)
    setState(await bridge.transferViewport('hub', bounds))
  }, [bridge])

  const transferToChat = useCallback(() => {
    openWorkstationBrowserPreview()

    if (navigate) {
      navigate('/')
    }
  }, [navigate])

  const handleSelectTask = useCallback(
    async (task: BrowserTask) => {
      if (!bridge || !hostRef.current) {
        return
      }
      const rect = hostRef.current.getBoundingClientRect()
      const bounds = rectToBounds(rect)
      await bridge.showTask(task.taskId, bounds, 'hub')
    },
    [bridge]
  )

  const handleParkTask = useCallback(
    async (taskId: string) => {
      if (!bridge) {
        return
      }
      await bridge.parkTask(taskId)
    },
    [bridge]
  )

  const handleHideTask = useCallback(
    async (taskId: string) => {
      if (!bridge) {
        return
      }
      await bridge.hideTask(taskId)
    },
    [bridge]
  )

  const handleDestroyTask = useCallback(
    async (taskId: string) => {
      if (!bridge) {
        return
      }
      await bridge.destroyTask(taskId)
    },
    [bridge]
  )

  const handleClearParked = useCallback(async () => {
    if (!bridge) {
      return
    }
    await bridge.clearParkedTasks()
  }, [bridge])

  useEffect(() => {
    if (!bridge) {
      return
    }

    let disposed = false

    const off = bridge.onState(next => {
      if (!disposed) {
        setState(next)
      }
    })

    void bridge
      .ensure()
      .then(next => {
        if (disposed) {
          return
        }
        setState(next)
        requestAnimationFrame(() => void publishBounds(true))
      })
      .catch(error => {
        if (!disposed) {
          setState(current => ({ ...current, lastError: String(error) }))
        }
      })

    return () => {
      disposed = true
      off()
      // Detaching removes the native view from the React page while preserving
      // its WebContents and persistent profile. Background tasks can continue.
      void bridge.detach()
    }
  }, [bridge, publishBounds])

  useEffect(() => {
    if (!bridge?.resources) {
      return
    }

    let disposed = false
    const refresh = () => {
      void bridge
        .resources()
        .then(next => {
          if (!disposed) {
            setResourceSnapshot(next)
          }
        })
        .catch(() => {
          if (!disposed) {
            setResourceSnapshot(null)
          }
        })
    }

    refresh()
    const timer = window.setInterval(refresh, 5_000)

    return () => {
      disposed = true
      window.clearInterval(timer)
    }
  }, [bridge])

  useEffect(() => {
    if (!bridge || !hostRef.current) {
      return
    }

    let frame = 0

    const schedule = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => void publishBounds(false))
    }

    const observer = new ResizeObserver(schedule)
    observer.observe(hostRef.current)
    window.addEventListener('resize', schedule)
    window.addEventListener('scroll', schedule, true)
    window.addEventListener('transitionend', schedule)

    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      window.removeEventListener('resize', schedule)
      window.removeEventListener('scroll', schedule, true)
      window.removeEventListener('transitionend', schedule)
    }
  }, [bridge, publishBounds])

  useEffect(() => {
    if (!bridge?.setVisible) {
      return
    }

    let isOverlayPresent = false

    const checkOverlay = () => {
      const active = Boolean(
        document.querySelector(
          '[data-radix-menu-content], [data-slot="context-menu-content"], [data-slot="dropdown-menu-content"], [role="menu"], [data-radix-popper-content-wrapper], [data-radix-dialog-content], [data-radix-select-content]'
        )
      )

      if (active !== isOverlayPresent) {
        isOverlayPresent = active
        void bridge.setVisible(!active)
      }
    }

    const observer = new MutationObserver(checkOverlay)
    observer.observe(document.body, { childList: true, subtree: true })

    return () => {
      observer.disconnect()

      if (isOverlayPresent) {
        void bridge.setVisible(true)
      }
    }
  }, [bridge])

  const submitAddress = (event: FormEvent) => {
    event.preventDefault()

    if (!bridge) {
      return
    }
    void run(() => bridge.navigate(address))
  }

  if (!bridge) {
    return (
      <section className="grid h-full place-items-center bg-(--ui-main-surface-background) p-8">
        <div className="max-w-xl rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-card-background) p-6">
          <h1 className="text-base font-semibold">Hermes Browser</h1>
          <p className="mt-2 text-sm text-(--ui-text-secondary)">
            The Workstation browser bridge is not installed in this Desktop build. Run
            <code className="mx-1 rounded bg-black/10 px-1 py-0.5 font-mono text-xs">workstation\install.ps1</code>
            from the repository root.
          </p>
        </div>
      </section>
    )
  }

  return (
    <section className="flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-main-surface-background)">
      <div className="shrink-0 border-b border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background)">
        <div className="flex h-9 min-w-0 items-center gap-1 border-b border-(--ui-stroke-tertiary) px-2">
          <div className="flex min-w-0 flex-1 items-end gap-1 overflow-x-auto">
            {state.tabs.map(tab => (
              <button
                className={cn(
                  'group flex h-8 max-w-52 min-w-28 items-center gap-1.5 rounded-t-md border border-transparent px-2 text-left text-xs',
                  tab.active
                    ? 'border-(--ui-stroke-secondary) bg-(--ui-main-surface-background) text-foreground'
                    : 'text-(--ui-text-secondary) hover:bg-(--ui-control-hover-background)'
                )}
                key={tab.id}
                onClick={() => void run(() => bridge.activateTab(tab.id))}
                type="button"
              >
                {tab.loading ? (
                  <Codicon className="shrink-0 animate-spin" name="loading" size="0.75rem" />
                ) : tab.crashed ? (
                  <Codicon className="shrink-0 text-red-400" name="error" size="0.75rem" />
                ) : (
                  <Codicon className="shrink-0" name="globe" size="0.75rem" />
                )}
                <span className="min-w-0 flex-1 truncate">{shortTitle(tab)}</span>
                <span
                  className="rounded p-0.5 text-(--ui-text-quaternary) hover:bg-(--ui-control-hover-background) hover:text-foreground opacity-60 group-hover:opacity-100 transition-opacity"
                  onClick={(event: MouseEvent<HTMLSpanElement>) => {
                    event.stopPropagation()
                    void run(async () => {
                      if (tab.ownerTaskId) {
                        try {
                          await bridge.destroyTask(tab.ownerTaskId)
                        } catch {
                          // Best effort task cleanup.
                        }
                      }

                      return bridge.closeTab(tab.id)
                    })
                  }}
                  role="button"
                  tabIndex={0}
                  title="Close tab"
                >
                  <Codicon name="close" size="0.7rem" />
                </span>
              </button>
            ))}
          </div>
          <Button
            aria-label="New browser tab"
            onClick={() => void run(() => bridge.newTab())}
            size="icon-sm"
            variant="ghost"
          >
            <Codicon name="add" />
          </Button>
        </div>

        <div className="flex h-10 items-center gap-1.5 px-2">
          <Button
            aria-label="Back"
            disabled={!activeTab?.canGoBack}
            onClick={() => void run(() => bridge.back())}
            size="icon-sm"
            variant="ghost"
          >
            <Codicon name="arrow-left" />
          </Button>
          <Button
            aria-label="Forward"
            disabled={!activeTab?.canGoForward}
            onClick={() => void run(() => bridge.forward())}
            size="icon-sm"
            variant="ghost"
          >
            <Codicon name="arrow-right" />
          </Button>
          <Button
            aria-label={activeTab?.loading ? 'Stop loading' : 'Reload'}
            onClick={() => void run(() => (activeTab?.loading ? bridge.stop() : bridge.reload()))}
            size="icon-sm"
            variant="ghost"
          >
            <Codicon name={activeTab?.loading ? 'debug-stop' : 'refresh'} />
          </Button>

          <form className="min-w-0 flex-1" onSubmit={submitAddress}>
            <input
              className="h-7 w-full rounded-md border border-(--ui-stroke-secondary) bg-(--ui-input-background) px-2.5 font-mono text-[12px] outline-none focus:border-(--ui-stroke-primary)"
              data-browser-address="true"
              onChange={(event: ChangeEvent<HTMLInputElement>) => setAddress(event.target.value)}
              placeholder="Search or enter address"
              value={address}
            />
          </form>

          <div
            className={cn(
              'hidden items-center gap-1 rounded-full border px-2 py-1 text-[10px] font-medium md:flex',
              state.paused
                ? 'border-amber-500/40 text-amber-400'
                : state.controlOwner === 'human'
                  ? 'border-sky-500/40 text-sky-400'
                  : 'border-emerald-500/30 text-emerald-400'
            )}
            title={`Controller: ${state.controlReady ? 'ready' : 'offline'} · Profile: ${state.profilePath || 'initializing'}`}
          >
            <span className="size-1.5 rounded-full bg-current" />
            {state.paused
              ? 'Paused'
              : state.controlOwner === 'human'
                ? 'Human control'
                : state.controlReady
                  ? 'Agent ready'
                  : 'Controller offline'}
          </div>
        </div>

        <div className="flex min-h-8 flex-wrap items-center gap-1 border-t border-(--ui-stroke-tertiary) px-2 py-1">
          {state.paused ? (
            <Button onClick={() => void run(() => bridge.resume())} size="sm" variant="ghost">
              <Codicon name="debug-continue" />
              Resume
            </Button>
          ) : (
            <Button onClick={() => void run(() => bridge.pause())} size="sm" variant="ghost">
              <Codicon name="debug-pause" />
              Pause
            </Button>
          )}
          <Button onClick={() => void run(() => bridge.stop())} size="sm" variant="ghost">
            <Codicon name="debug-stop" />
            Stop
          </Button>
          <Button onClick={() => void run(() => bridge.focus())} size="sm" variant="ghost">
            <Codicon name="target" />
            Focus Browser
          </Button>
          {state.controlOwner === 'human' ? (
            <Button onClick={() => void run(() => bridge.releaseControl())} size="sm" variant="ghost">
              <Codicon name="robot" />
              Release Control
            </Button>
          ) : (
            <Button onClick={() => void run(() => bridge.takeControl())} size="sm" variant="ghost">
              <Codicon name="person" />
              Take Control
            </Button>
          )}
          <Button onClick={transferToChat} size="sm" title="Transfer Viewport to Chat Right Rail" variant="ghost">
            <Codicon name="comment-discussion" />
            Move to Chat
          </Button>
          {state.downloads && state.downloads.length > 0 && (
            <Button
              onClick={() => setShowDownloads(prev => !prev)}
              size="sm"
              title="Toggle Downloads Panel"
              variant={showDownloads ? 'secondary' : 'ghost'}
            >
              <Codicon name="cloud-download" />
              Downloads ({state.downloads.length})
            </Button>
          )}
          <span className="ml-auto hidden font-mono text-[10px] text-(--ui-text-quaternary) lg:block">
            Electron Chromium · persistent profile · resources {resourceSnapshot?.resources.length ?? '—'} · cache{' '}
            {cacheLabel(state.cacheBytes)}
          </span>
        </div>
      </div>

      {showDownloads && state.downloads && state.downloads.length > 0 && (
        <div className="shrink-0 border-b border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) p-2 text-xs">
          <div className="flex items-center justify-between mb-1.5 font-medium text-(--ui-text-primary)">
            <span className="flex items-center gap-1.5">
              <Codicon name="cloud-download" size="0.8rem" />
              Downloads ({state.downloads.length})
            </span>
            <Button onClick={() => setShowDownloads(false)} size="icon-xs" variant="ghost">
              <Codicon name="close" size="0.75rem" />
            </Button>
          </div>
          <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
            {state.downloads.map(dl => {
              const pct = dl.totalBytes > 0 ? Math.round((dl.receivedBytes / dl.totalBytes) * 100) : 0

              return (
                <div
                  className="rounded border border-(--ui-stroke-tertiary) p-1.5 bg-(--ui-card-background)"
                  key={dl.id}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium text-(--ui-text-primary)" title={dl.filename}>
                      {dl.filename}
                    </span>
                    <span className="shrink-0 text-[10px] font-mono text-(--ui-text-tertiary)">
                      {dl.state === 'completed' ? 'Done' : `${pct}%`}
                    </span>
                  </div>
                  {dl.state === 'progressing' && (
                    <div className="mt-1 h-1 w-full rounded-full bg-(--ui-stroke-tertiary) overflow-hidden">
                      <div className="h-full bg-emerald-500 transition-all duration-200" style={{ width: `${pct}%` }} />
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {state.lastError && (
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-red-500/30 bg-red-500/10 px-3 py-1.5 text-xs text-red-300">
          <span className="truncate" title={state.lastError}>
            {formatBrowserError(state.lastError)}
          </span>
          <button
            aria-label="Dismiss error"
            className="ml-auto shrink-0 rounded px-1.5 py-0.5 text-[10px] text-red-300/80 hover:bg-red-500/20 hover:text-white"
            onClick={() => void bridge?.clearError?.()}
            type="button"
          >
            ✕
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <TaskRail
          activeTaskId={activeTab?.ownerTaskId}
          onAuditTask={taskId => setAuditTaskId(prev => (prev === taskId ? null : taskId))}
          onClearParked={() => void handleClearParked()}
          onDestroyTask={taskId => void handleDestroyTask(taskId)}
          onHideTask={taskId => void handleHideTask(taskId)}
          onParkTask={taskId => void handleParkTask(taskId)}
          onSelectTask={task => void handleSelectTask(task)}
          tabs={state.tabs}
          tasks={state.tasks}
        />

        {auditTaskId && <TaskJournalDrawer onClose={() => setAuditTaskId(null)} taskId={auditTaskId} />}

        <div className="relative min-h-0 flex-1 bg-black">
          {/* The Electron main process places the active WebContentsView exactly
              over this rectangle. The native view is deliberately outside the
              React tree so it survives route changes and can keep running in the
              background. */}
          <div className="absolute inset-0" ref={hostRef} />
          {state.attached && state.viewportHost && state.viewportHost !== 'hub' && (
            <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between gap-2 border-b border-amber-500/30 bg-amber-500/90 px-3 py-1.5 text-xs font-medium text-black">
              <span>Live viewport is currently active in Chat ({state.viewportHost})</span>
              <Button onClick={() => void transferToHub()} size="xs" variant="secondary">
                Bring Viewport to Hub
              </Button>
            </div>
          )}
          {!state.ready && (
            <div className="absolute inset-0 grid place-items-center text-xs text-(--ui-text-tertiary)">
              Starting Hermes Browser…
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
