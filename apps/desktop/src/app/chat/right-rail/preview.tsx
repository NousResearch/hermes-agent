import { useStore } from '@nanostores/react'
import type { CSSProperties, ReactNode, PointerEvent as ReactPointerEvent } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { SetTitlebarToolGroup } from '@/app/shell/titlebar-controls'
import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { Tip } from '@/components/ui/tooltip'
import { translateNow, useI18n } from '@/i18n'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import {
  $panesFlipped,
  $rightRailActiveTabId,
  RIGHT_RAIL_PREVIEW_TAB_ID,
  type RightRailTabId,
  selectRightRailTab
} from '@/store/layout'
import {
  $filePreviewTabs,
  $previewReloadRequest,
  $previewSurfaceLayouts,
  $previewTarget,
  $webPreviewTabs,
  closeOtherRightRailTabs,
  closeRightRail,
  closeRightRailTab,
  closeRightRailTabsToRight,
  detachRightRailTab,
  dockRightRailTab,
  maximizeRightRailTab,
  minimizeRightRailTab,
  type PreviewSurfaceLayout,
  type PreviewSurfacePlacement,
  type PreviewTarget,
  restoreRightRailTab,
  setRightRailTabFloatingGeometry,
  snapRightRailTab
} from '@/store/preview'
import { $dirtyPreviewUrls } from '@/store/preview-edit'
import {
  clampFloatingGeometry,
  edgeSnapPlacement,
  type FloatingGeometry,
  type PreviewSnapSlot,
  type PreviewViewport,
  surfaceGeometryForPlacement,
  WIN11_SNAP_LAYOUTS
} from '@/store/preview-surface-layout'

import { PreviewPane } from './preview-pane'

export const PREVIEW_RAIL_MIN_WIDTH = '18rem'
export const PREVIEW_RAIL_MAX_WIDTH = '38rem'

const INTRINSIC = `clamp(${PREVIEW_RAIL_MIN_WIDTH}, 36vw, 32rem)`
const TITLEBAR_INSET = 40
const TASKBAR_INSET = 44

export const PREVIEW_RAIL_PANE_WIDTH = `min(${INTRINSIC}, max(0rem, calc(100vw - var(--pane-chat-sidebar-width) - var(--pane-file-browser-width, 0rem) - var(--chat-min-width))))`

interface ChatPreviewRailProps {
  onRestartServer?: (url: string, context?: string) => Promise<string>
  setTitlebarToolGroup?: SetTitlebarToolGroup
}

interface RailTab {
  id: RightRailTabId
  label: string
  target: PreviewTarget
}

function tabLabelFor(target: PreviewTarget): string {
  const value = target.label || target.path || target.source || target.url
  const tail = value.split(/[\\/]/).filter(Boolean).at(-1)

  return tail || value || translateNow('preview.tab')
}

function viewportMetrics(): PreviewViewport {
  if (typeof window === 'undefined') {
    return { bottomInset: TASKBAR_INSET, height: 900, topInset: TITLEBAR_INSET, width: 1440 }
  }

  return {
    bottomInset: TASKBAR_INSET,
    height: window.innerHeight,
    topInset: TITLEBAR_INSET,
    width: window.innerWidth
  }
}

function defaultFloatingGeometry(tabId: string): FloatingGeometry {
  const viewport = viewportMetrics()
  const offset = Math.abs([...tabId].reduce((hash, character) => (hash * 31 + character.charCodeAt(0)) | 0, 0)) % 84

  return clampFloatingGeometry(
    {
      height: Math.max(420, viewport.height * 0.62),
      width: Math.max(520, viewport.width * 0.54),
      x: 64 + offset,
      y: TITLEBAR_INSET + 24 + offset
    },
    viewport
  )
}

function layoutGeometry(tabId: RightRailTabId, layout: PreviewSurfaceLayout): FloatingGeometry {
  if (layout.placement === 'floating') {
    return clampFloatingGeometry(layout.geometry ?? defaultFloatingGeometry(tabId), viewportMetrics())
  }

  if (layout.placement === 'minimized' || layout.placement === 'docked') {
    return layout.geometry ?? defaultFloatingGeometry(tabId)
  }

  return surfaceGeometryForPlacement(layout.placement, viewportMetrics())
}

function slotLabel(
  slot: PreviewSnapSlot,
  slots: ReturnType<typeof useI18n>['t']['preview']['surface']['slots']
): string {
  const labels: Record<PreviewSnapSlot, string> = {
    'bottom-half': slots.bottomHalf,
    'bottom-left-quarter': slots.bottomLeftQuarter,
    'bottom-right-quarter': slots.bottomRightQuarter,
    'center-third': slots.centerThird,
    'left-half': slots.leftHalf,
    'left-third': slots.leftThird,
    'left-two-thirds': slots.leftTwoThirds,
    'right-half': slots.rightHalf,
    'right-third': slots.rightThird,
    'right-two-thirds': slots.rightTwoThirds,
    'top-half': slots.topHalf,
    'top-left-quarter': slots.topLeftQuarter,
    'top-right-quarter': slots.topRightQuarter
  }

  return labels[slot]
}

function SurfaceActionButton({
  children,
  label,
  onClick
}: {
  children: ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <button
      aria-label={label}
      className="grid size-7 shrink-0 place-items-center rounded-md text-(--ui-text-tertiary) transition-colors hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
      onClick={onClick}
      type="button"
    >
      {children}
    </button>
  )
}

function SnapLayoutPicker({ label, tabId }: { label: string; tabId: RightRailTabId }) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const firstButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (open) {
      firstButtonRef.current?.focus()
    }
  }, [open])

  return (
    <div className="relative">
      <SurfaceActionButton label={t.preview.surface.snapLayouts(label)} onClick={() => setOpen(value => !value)}>
        <Codicon name="layout" size="0.8rem" />
      </SurfaceActionButton>
      {open && (
        <div
          className="absolute right-0 top-8 z-[130] w-72 space-y-2 rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-editor-surface-background) p-2 shadow-2xl shadow-black/40"
          data-testid="preview-snap-layout-picker"
          onKeyDown={event => {
            if (event.key === 'Escape') {
              event.stopPropagation()
              setOpen(false)
            }
          }}
        >
          {WIN11_SNAP_LAYOUTS.map((snapLayout, layoutIndex) => (
            <div className="flex gap-1" key={snapLayout.id}>
              {snapLayout.slots.map((slot, slotIndex) => {
                const localizedSlot = slotLabel(slot, t.preview.surface.slots)

                return (
                  <button
                    aria-label={t.preview.surface.snapTo(label, localizedSlot)}
                    className="h-9 min-w-0 flex-1 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background) transition-colors hover:border-(--theme-primary) hover:bg-(--ui-control-hover-background) focus-visible:border-(--theme-primary) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-(--theme-primary)/60"
                    key={slot}
                    onClick={() => {
                      snapRightRailTab(tabId, slot)
                      setOpen(false)
                    }}
                    ref={layoutIndex === 0 && slotIndex === 0 ? firstButtonRef : undefined}
                    title={localizedSlot}
                    type="button"
                  >
                    <span
                      aria-hidden="true"
                      className="mx-auto block h-4 w-7 rounded-sm border border-current opacity-70"
                    />
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function useSurfaceInteractions(tabId: RightRailTabId, layout: PreviewSurfaceLayout) {
  const [geometry, setGeometry] = useState(() => layoutGeometry(tabId, layout))
  const cleanupRef = useRef<null | (() => void)>(null)

  useEffect(() => {
    setGeometry(layoutGeometry(tabId, layout))
  }, [layout, tabId])

  useEffect(() => () => cleanupRef.current?.(), [])

  const beginInteraction = useCallback(
    (event: ReactPointerEvent<HTMLElement>, mode: 'drag' | 'resize') => {
      if (
        event.button !== 0 ||
        (mode === 'drag' &&
          event.target instanceof HTMLElement &&
          event.target.closest('button,input,select,textarea,a'))
      ) {
        return
      }

      event.preventDefault()

      if (mode === 'resize') {
        event.stopPropagation()
      }

      cleanupRef.current?.()

      const start =
        layout.placement === 'floating' ? geometry : (layout.restore?.geometry ?? defaultFloatingGeometry(tabId))

      const startX = event.clientX
      const startY = event.clientY
      const pointerId = event.pointerId
      const captureTarget = event.currentTarget
      const previousCursor = document.body.style.cursor
      const previousSelect = document.body.style.userSelect
      let latest = start

      detachRightRailTab(tabId)
      setGeometry(start)
      captureTarget.setPointerCapture?.(pointerId)
      document.body.style.cursor = mode === 'drag' ? 'move' : 'nwse-resize'
      document.body.style.userSelect = 'none'

      const onMove = (moveEvent: PointerEvent) => {
        const deltaX = moveEvent.clientX - startX
        const deltaY = moveEvent.clientY - startY
        latest = clampFloatingGeometry(
          mode === 'drag'
            ? { ...start, x: start.x + deltaX, y: start.y + deltaY }
            : { ...start, height: start.height + deltaY, width: start.width + deltaX },
          viewportMetrics()
        )
        setGeometry(latest)
      }

      const removeListeners = () => {
        document.body.style.cursor = previousCursor
        document.body.style.userSelect = previousSelect
        captureTarget.releasePointerCapture?.(pointerId)
        window.removeEventListener('pointermove', onMove, true)
        window.removeEventListener('pointerup', onEnd, true)
        window.removeEventListener('pointercancel', onCancel, true)
        window.removeEventListener('blur', onBlur)
        cleanupRef.current = null
      }

      const saveFloating = () => setRightRailTabFloatingGeometry(tabId, latest, viewportMetrics())

      const onEnd = (endEvent: PointerEvent) => {
        saveFloating()

        if (mode === 'drag') {
          const snap = edgeSnapPlacement({ x: endEvent.clientX, y: endEvent.clientY }, viewportMetrics())

          if (snap === 'maximized') {
            maximizeRightRailTab(tabId)
          } else if (snap) {
            snapRightRailTab(tabId, snap)
          }
        }

        removeListeners()
      }

      const onCancel = () => {
        saveFloating()
        removeListeners()
      }

      const onBlur = () => {
        saveFloating()
        removeListeners()
      }

      cleanupRef.current = removeListeners
      window.addEventListener('pointermove', onMove, true)
      window.addEventListener('pointerup', onEnd, true)
      window.addEventListener('pointercancel', onCancel, true)
      window.addEventListener('blur', onBlur)
    },
    [geometry, layout, tabId]
  )

  const style = useMemo<CSSProperties>(() => {
    const current = layout.placement === 'floating' ? geometry : layoutGeometry(tabId, layout)

    return { height: current.height, left: current.x, top: current.y, width: current.width }
  }, [geometry, layout, tabId])

  return {
    startDrag: (event: ReactPointerEvent<HTMLElement>) => beginInteraction(event, 'drag'),
    startResize: (event: ReactPointerEvent<HTMLElement>) => beginInteraction(event, 'resize'),
    style
  }
}

function FloatingPreviewSurface({
  active,
  layout,
  onRestartServer,
  reloadRequest,
  setTitlebarToolGroup,
  tab
}: {
  active: boolean
  layout: PreviewSurfaceLayout
  onRestartServer?: (url: string, context?: string) => Promise<string>
  reloadRequest: number
  setTitlebarToolGroup?: SetTitlebarToolGroup
  tab: RailTab
}) {
  const { t } = useI18n()
  const { startDrag, startResize, style } = useSurfaceInteractions(tab.id, layout)
  const maximized = layout.placement === 'maximized'

  return (
    <section
      aria-label={t.preview.surface.activate(tab.label)}
      className={cn(
        'fixed z-[80] flex min-h-0 min-w-0 flex-col overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-editor-surface-background) text-(--ui-text-tertiary) shadow-2xl shadow-black/35',
        active && 'z-[90] ring-1 ring-(--theme-primary)/50'
      )}
      data-surface-placement={layout.placement}
      data-testid="floating-preview-surface"
      onPointerDown={() => selectRightRailTab(tab.id)}
      style={style}
    >
      <header
        className="flex min-h-10 cursor-move select-none items-center gap-0.5 border-b border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background) px-2"
        onPointerDown={startDrag}
      >
        <button
          aria-label={t.preview.surface.activate(tab.label)}
          className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-1 py-1 text-left text-[0.6875rem] font-semibold text-foreground hover:bg-(--ui-control-hover-background)"
          onClick={() => selectRightRailTab(tab.id)}
          type="button"
        >
          <Codicon name={tab.target.kind === 'file' ? 'file' : 'globe'} size="0.75rem" />
          <span className="min-w-0 truncate">{tab.label}</span>
        </button>
        <SurfaceActionButton label={t.preview.surface.dock(tab.label)} onClick={() => dockRightRailTab(tab.id)}>
          <Codicon name="dock-right" size="0.78rem" />
        </SurfaceActionButton>
        <SurfaceActionButton label={t.preview.surface.minimize(tab.label)} onClick={() => minimizeRightRailTab(tab.id)}>
          <Codicon name="chrome-minimize" size="0.78rem" />
        </SurfaceActionButton>
        <SurfaceActionButton
          label={maximized ? t.preview.surface.restore(tab.label) : t.preview.surface.maximize(tab.label)}
          onClick={() => (maximized ? restoreRightRailTab(tab.id) : maximizeRightRailTab(tab.id))}
        >
          <Codicon name={maximized ? 'screen-normal' : 'screen-full'} size="0.78rem" />
        </SurfaceActionButton>
        <SnapLayoutPicker label={tab.label} tabId={tab.id} />
        <SurfaceActionButton label={t.preview.closeTab(tab.label)} onClick={() => closeRightRailTab(tab.id)}>
          <Codicon name="close" size="0.78rem" />
        </SurfaceActionButton>
      </header>
      <div className="min-h-0 flex-1 overflow-hidden">
        <PreviewPane
          embedded
          onRestartServer={tab.target.kind === 'url' ? onRestartServer : undefined}
          reloadRequest={reloadRequest}
          setTitlebarToolGroup={active ? setTitlebarToolGroup : undefined}
          target={tab.target}
        />
      </div>
      {layout.placement === 'floating' && (
        <button
          aria-label={t.preview.surface.resize(tab.label)}
          className="absolute bottom-0 right-0 size-5 cursor-nwse-resize rounded-tl-md text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
          onPointerDown={startResize}
          type="button"
        >
          <Codicon name="gripper" size="0.75rem" />
        </button>
      )}
    </section>
  )
}

function DetachedPlaceholder({ tab }: { tab: RailTab }) {
  const { t } = useI18n()

  return (
    <div className="grid h-full place-items-center px-6 text-center text-xs text-(--ui-text-tertiary)">
      <div className="max-w-xs rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background)/70 p-4 shadow-lg">
        <div className="mb-1 text-sm font-semibold text-foreground">{t.preview.surface.detachedTitle(tab.label)}</div>
        <div className="mb-3 leading-relaxed">{t.preview.surface.detachedBody}</div>
        <button
          className="rounded-md border border-(--ui-stroke-secondary) px-3 py-1.5 font-semibold text-foreground hover:border-(--theme-primary) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
          onClick={() => dockRightRailTab(tab.id)}
          type="button"
        >
          {t.preview.surface.dock(tab.label)}
        </button>
      </div>
    </div>
  )
}

function MinimizedTaskbar({ tabs }: { tabs: readonly RailTab[] }) {
  const { t } = useI18n()

  if (tabs.length === 0) {
    return null
  }

  return (
    <div className="fixed bottom-2 left-1/2 z-[100] flex max-w-[calc(100vw-1rem)] -translate-x-1/2 gap-1 overflow-x-auto rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background)/95 p-1 shadow-2xl shadow-black/35 backdrop-blur">
      {tabs.map(tab => (
        <button
          aria-label={t.preview.surface.restore(tab.label)}
          className="flex h-8 max-w-48 items-center gap-2 rounded-lg px-3 text-xs font-medium text-foreground hover:bg-(--ui-control-hover-background) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
          key={tab.id}
          onClick={() => restoreRightRailTab(tab.id)}
          type="button"
        >
          <Codicon name={tab.target.kind === 'file' ? 'file' : 'globe'} size="0.75rem" />
          <span className="truncate">{tab.label}</span>
        </button>
      ))}
    </div>
  )
}

export function ChatPreviewRail({ onRestartServer, setTitlebarToolGroup }: ChatPreviewRailProps) {
  const { t } = useI18n()
  const previewReloadRequest = useStore($previewReloadRequest)
  const activeTabId = useStore($rightRailActiveTabId)
  const panesFlipped = useStore($panesFlipped)
  const filePreviewTabs = useStore($filePreviewTabs)
  const webPreviewTabs = useStore($webPreviewTabs)
  const previewTarget = useStore($previewTarget)
  const surfaceLayouts = useStore($previewSurfaceLayouts)
  const dirtyPreviewUrls = useStore($dirtyPreviewUrls)

  const tabs = useMemo<readonly RailTab[]>(
    () => [
      ...(previewTarget
        ? [{ id: RIGHT_RAIL_PREVIEW_TAB_ID, label: t.preview.tab, target: previewTarget } as RailTab]
        : []),
      ...webPreviewTabs.map(({ id, target }) => ({ id, label: tabLabelFor(target), target }) as RailTab),
      ...filePreviewTabs.map(({ id, target }) => ({ id, label: tabLabelFor(target), target }) as RailTab)
    ],
    [filePreviewTabs, previewTarget, t.preview.tab, webPreviewTabs]
  )

  const placementFor = useCallback(
    (id: RightRailTabId): PreviewSurfacePlacement => surfaceLayouts[id]?.placement ?? 'docked',
    [surfaceLayouts]
  )

  const activeTab =
    tabs.find(tab => tab.id === activeTabId) ?? tabs.find(tab => placementFor(tab.id) !== 'minimized') ?? tabs[0]

  const detachedTabs = tabs.filter(tab => {
    const placement = placementFor(tab.id)

    return placement !== 'docked' && placement !== 'minimized'
  })

  const minimizedTabs = tabs.filter(tab => placementFor(tab.id) === 'minimized')

  useEffect(() => {
    if (activeTab && activeTab.id !== activeTabId && placementFor(activeTab.id) !== 'minimized') {
      selectRightRailTab(activeTab.id)
    }
  }, [activeTab, activeTabId, placementFor])

  if (!activeTab) {
    return null
  }

  const activePlacement = placementFor(activeTab.id)

  return (
    <>
      <aside
        className={cn(
          'relative flex h-full w-full min-w-0 flex-col overflow-hidden border-(--ui-stroke-tertiary) bg-(--ui-editor-surface-background) text-(--ui-text-tertiary)',
          panesFlipped ? 'border-r' : 'border-l'
        )}
        style={{ paddingTop: 'var(--right-rail-top-inset, 0px)' }}
      >
        <div className="group/rail-tabs flex h-(--titlebar-height) shrink-0 border-b border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background)">
          <div
            className="flex min-w-0 flex-1 overflow-x-auto overflow-y-hidden overscroll-x-contain [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
            role="tablist"
          >
            {tabs.map((tab, index) => {
              const active = tab.id === activeTab.id
              const placement = placementFor(tab.id)
              const hasOthers = tabs.length > 1
              const hasTabsToRight = index < tabs.length - 1
              const dirty = Boolean(dirtyPreviewUrls[tab.target.url])

              return (
                <ContextMenu key={tab.id}>
                  <ContextMenuTrigger asChild>
                    <div
                      className={cn(
                        'group/tab relative flex h-full min-w-0 max-w-48 shrink-0 items-center text-[0.6875rem] font-medium [-webkit-app-region:no-drag] last:border-r last:border-(--ui-stroke-quaternary)',
                        active
                          ? 'bg-(--ui-editor-surface-background) text-foreground [--tab-bg:var(--ui-editor-surface-background)]'
                          : 'border-r border-(--ui-stroke-quaternary) text-(--ui-text-tertiary) [--tab-bg:var(--ui-sidebar-surface-background)] hover:bg-(--chrome-action-hover) hover:text-foreground'
                      )}
                      onAuxClick={event => {
                        if (event.button === 1) {
                          event.preventDefault()
                          closeRightRailTab(tab.id)
                        }
                      }}
                      onMouseDown={event => {
                        if (event.button === 1) {
                          event.preventDefault()
                        }
                      }}
                    >
                      {active && (
                        <span aria-hidden="true" className="absolute inset-x-0 top-0 h-px bg-(--ui-stroke-primary)" />
                      )}
                      <Tip label={tab.target.path || tab.target.url || tab.label}>
                        <button
                          aria-selected={active}
                          className="flex h-full min-w-0 max-w-full items-center gap-1 overflow-hidden pl-3 pr-12 text-left outline-none"
                          onClick={() =>
                            placement === 'minimized' ? restoreRightRailTab(tab.id) : selectRightRailTab(tab.id)
                          }
                          role="tab"
                          type="button"
                        >
                          <span className="block min-w-0 truncate">{tab.label}</span>
                          {placement !== 'docked' && (
                            <span aria-hidden="true" className="size-1.5 shrink-0 rounded-full bg-(--theme-primary)" />
                          )}
                        </button>
                      </Tip>
                      {dirty && (
                        <span
                          aria-hidden="true"
                          className="pointer-events-none absolute right-1.5 top-1/2 size-2 -translate-y-1/2 rounded-full bg-amber-500"
                        />
                      )}
                      <button
                        aria-label={
                          placement === 'docked'
                            ? t.preview.surface.detach(tab.label)
                            : t.preview.surface.dock(tab.label)
                        }
                        className="pointer-events-none absolute right-6 top-1/2 grid size-4 -translate-y-1/2 place-items-center rounded-sm opacity-0 hover:bg-(--ui-bg-secondary) focus-visible:pointer-events-auto focus-visible:opacity-100 group-hover/tab:pointer-events-auto group-hover/tab:opacity-100"
                        onClick={() => (placement === 'docked' ? detachRightRailTab(tab.id) : dockRightRailTab(tab.id))}
                        type="button"
                      >
                        <Codicon name={placement === 'docked' ? 'window' : 'dock-right'} size="0.72rem" />
                      </button>
                      <button
                        aria-label={t.preview.closeTab(tab.label)}
                        className="pointer-events-none absolute right-1.5 top-1/2 grid size-4 -translate-y-1/2 place-items-center rounded-sm opacity-0 hover:bg-(--ui-bg-secondary) focus-visible:pointer-events-auto focus-visible:opacity-100 group-hover/tab:pointer-events-auto group-hover/tab:opacity-100"
                        onClick={() => closeRightRailTab(tab.id)}
                        type="button"
                      >
                        <Codicon name="close" size="0.72rem" />
                      </button>
                    </div>
                  </ContextMenuTrigger>
                  <ContextMenuContent>
                    <ContextMenuItem onSelect={() => closeRightRailTab(tab.id)}>
                      {t.common.close}
                      <span className="ml-auto pl-4 text-(--ui-text-tertiary)">{formatCombo('mod+w')}</span>
                    </ContextMenuItem>
                    <ContextMenuItem disabled={!hasOthers} onSelect={() => closeOtherRightRailTabs(tab.id)}>
                      {t.preview.closeOthers}
                    </ContextMenuItem>
                    <ContextMenuItem disabled={!hasTabsToRight} onSelect={() => closeRightRailTabsToRight(tab.id)}>
                      {t.preview.closeToRight}
                    </ContextMenuItem>
                    <ContextMenuSeparator />
                    <ContextMenuItem onSelect={closeRightRail}>{t.preview.closeAll}</ContextMenuItem>
                  </ContextMenuContent>
                </ContextMenu>
              )
            })}
          </div>
          <button
            aria-label={t.preview.closePane}
            className="mr-1.5 grid size-6 shrink-0 self-center place-items-center rounded-md text-(--ui-text-tertiary) opacity-0 hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring group-hover/rail-tabs:opacity-100 [-webkit-app-region:no-drag]"
            onClick={closeRightRail}
            type="button"
          >
            <Codicon name="close" size="0.75rem" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          {activePlacement === 'docked' ? (
            <PreviewPane
              embedded
              onRestartServer={activeTab.target.kind === 'url' ? onRestartServer : undefined}
              reloadRequest={previewReloadRequest}
              setTitlebarToolGroup={setTitlebarToolGroup}
              target={activeTab.target}
            />
          ) : activePlacement === 'minimized' ? null : (
            <DetachedPlaceholder tab={activeTab} />
          )}
        </div>
      </aside>

      {detachedTabs.map(tab => (
        <FloatingPreviewSurface
          active={tab.id === activeTab.id}
          key={tab.id}
          layout={surfaceLayouts[tab.id] ?? { placement: 'floating' }}
          onRestartServer={onRestartServer}
          reloadRequest={previewReloadRequest}
          setTitlebarToolGroup={tab.id === activeTab.id ? setTitlebarToolGroup : undefined}
          tab={tab}
        />
      ))}
      <MinimizedTaskbar tabs={minimizedTabs} />
    </>
  )
}
