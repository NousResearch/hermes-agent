import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { WorkstationBrowserBounds, WorkstationBrowserState, WorkstationBrowserTabState } from '@/app/browser/types'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

export interface WorkstationBrowserPaneProps {
  onPopOut?: () => void
  className?: string
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

function rectToBounds(rect: DOMRect): WorkstationBrowserBounds {
  return {
    x: Math.max(0, Math.round(rect.left)),
    y: Math.max(0, Math.round(rect.top)),
    width: Math.max(1, Math.round(rect.width)),
    height: Math.max(1, Math.round(rect.height))
  }
}

function shortTitle(tab: WorkstationBrowserTabState | null): string {
  if (!tab) {
    return 'Workstation Browser'
  }

  if (tab.title && tab.title !== 'New Tab') {
    return tab.title
  }

  if (!tab.url || tab.url === 'about:blank') {
    return 'Blank Page'
  }

  try {
    return new URL(tab.url).hostname || tab.url
  } catch {
    return tab.url
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

export function WorkstationBrowserPane({ onPopOut, className }: WorkstationBrowserPaneProps) {
  const bridge = window.hermesDesktop?.workstationBrowser
  const [state, setState] = useState<WorkstationBrowserState>(EMPTY_STATE)
  const [busy, setBusy] = useState(false)
  const hostRef = useRef<HTMLDivElement | null>(null)

  const activeTab = useMemo(
    () => state.tabs.find(tab => tab.id === state.activeTabId) ?? state.tabs[0] ?? null,
    [state.activeTabId, state.tabs]
  )

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

      if (attach) {
        setState(await bridge.attach(bounds, 'chat'))
      } else {
        setState(await bridge.setBounds(bounds, 'chat'))
      }
    },
    [bridge]
  )

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
      void bridge.detach()
    }
  }, [bridge, publishBounds])

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
    window.addEventListener('transitionend', schedule)

    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      window.removeEventListener('resize', schedule)
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

  const transferToChat = useCallback(async () => {
    if (!bridge || !hostRef.current) {
      return
    }
    const rect = hostRef.current.getBoundingClientRect()

    if (rect.width < 1 || rect.height < 1) {
      return
    }
    const bounds = rectToBounds(rect)
    setState(await bridge.transferViewport('chat', bounds))
  }, [bridge])

  if (!bridge) {
    return (
      <div
        className={cn('grid h-full place-items-center p-4 text-center text-xs text-(--ui-text-secondary)', className)}
      >
        Workstation Browser bridge is not available.
      </div>
    )
  }

  const isAttachedHere = state.attached && state.viewportHost === 'chat'
  const isAttachedElsewhere = state.attached && state.viewportHost !== 'chat'

  return (
    <div className={cn('flex h-full flex-col overflow-hidden bg-(--ui-main-surface-background)', className)}>
      <div className="flex h-9 shrink-0 items-center justify-between gap-1.5 border-b border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) px-2">
        <div className="flex items-center gap-1">
          <Button
            aria-label="Back"
            disabled={!activeTab?.canGoBack || busy}
            onClick={() => void run(() => bridge.back())}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="arrow-left" size="0.75rem" />
          </Button>
          <Button
            aria-label="Forward"
            disabled={!activeTab?.canGoForward || busy}
            onClick={() => void run(() => bridge.forward())}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="arrow-right" size="0.75rem" />
          </Button>
          <Button
            aria-label={activeTab?.loading ? 'Stop' : 'Reload'}
            onClick={() => void run(() => (activeTab?.loading ? bridge.stop() : bridge.reload()))}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name={activeTab?.loading ? 'debug-stop' : 'refresh'} size="0.75rem" />
          </Button>
        </div>

        <div className="min-w-0 flex-1 truncate px-1 text-center font-mono text-[11px] text-(--ui-text-secondary)">
          {shortTitle(activeTab)}
        </div>

        <div className="flex items-center gap-1">
          <span
            className={cn(
              'rounded-full px-1.5 py-0.5 text-[9px] font-medium uppercase',
              state.paused
                ? 'bg-amber-500/20 text-amber-400'
                : state.controlOwner === 'human'
                  ? 'bg-sky-500/20 text-sky-400'
                  : 'bg-emerald-500/20 text-emerald-400'
            )}
            title={`Control: ${state.controlOwner}`}
          >
            {state.paused ? 'Paused' : state.controlOwner}
          </span>

          {onPopOut && (
            <Button
              aria-label="Pop out to Browser Hub"
              onClick={onPopOut}
              size="icon-xs"
              title="Pop out to Browser Hub"
              variant="ghost"
            >
              <Codicon name="link-external" size="0.75rem" />
            </Button>
          )}
        </div>
      </div>

      {isAttachedElsewhere && (
        <div className="flex items-center justify-between gap-2 border-b border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-300">
          <span>Viewport active in Browser Hub ({state.viewportHost})</span>
          <Button onClick={() => void transferToChat()} size="xs" variant="secondary">
            Bring to Chat
          </Button>
        </div>
      )}

      {state.lastError && (
        <div className="flex items-center justify-between gap-2 border-b border-red-500/30 bg-red-500/10 px-3 py-1 text-xs text-red-300">
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

      <div className="relative min-h-0 flex-1 bg-black">
        <div className="absolute inset-0" ref={hostRef} />
        {!state.ready && (
          <div className="absolute inset-0 grid place-items-center text-xs text-(--ui-text-tertiary)">
            Starting Workstation Browser…
          </div>
        )}
      </div>
    </div>
  )
}
