/** Design surfaces as layout-tree panes. Pen registers as the provider.
 *  One pane per provider — the live .pen can change without remounting.
 *  A provider's pane can also live in its own OS window (`?win=canvas`);
 *  while it is out, the docked tree shows nothing for it and every open /
 *  close for that provider is forwarded to the window. */

import { useStore } from '@nanostores/react'
import { atom, computed } from 'nanostores'
import type { ReactNode } from 'react'

import { isPaneVisible, revealTreePane } from '@/components/pane-shell/tree/store'
import { type MenuKit, renderActionItem } from '@/components/ui/actions-menu'
import { ContribRender } from '@/contrib/react/boundary'
import { translateNow } from '@/i18n'
import { canOpenCanvasWindow, isCanvasWindow, openCanvasInNewWindow, windowCanvasTab } from '@/store/windows'

import { paneMirror } from './pane-mirror'

export interface CanvasTab {
  provider: string
  docId: string
  title: string
  url: string
}

export interface CanvasProvider {
  id: string
  untitled: string
  tabLead: () => ReactNode
  render: () => ReactNode
  close: () => void
  /** The pane is leaving this window for its own (or coming back). A provider
   *  with a singleton guest tears it down here so the other window's can
   *  take over. */
  popOut?: () => void
  /** Provider rows for the tile's tab menu, above the canvas-wide ones. */
  tabMenu?: (kit: MenuKit) => ReactNode
}

const providers = new Map<string, CanvasProvider>()

export function registerCanvasProvider(provider: CanvasProvider): void {
  if (!providers.has(provider.id)) {
    providers.set(provider.id, provider)
  }
}

export const $canvasTabs = atom<CanvasTab[]>([])

/** Providers whose pane is in its own window, with the tab it was handed.
 *  Memory-only: a relaunch with no window up seats the tile again. */
export const $poppedCanvasTabs = atom<ReadonlyMap<string, CanvasTab>>(new Map())

export function canvasPopped(provider: string): boolean {
  return $poppedCanvasTabs.get().has(provider)
}

function markCanvasPopped(tab: CanvasTab | null, provider: string): void {
  const next = new Map($poppedCanvasTabs.get())

  if (tab) {
    next.set(provider, tab)
  } else {
    next.delete(provider)
  }

  $poppedCanvasTabs.set(next)
}

/** Tabs that belong in this window's layout tree. */
export const $dockedCanvasTabs = computed([$canvasTabs, $poppedCanvasTabs], (tabs, popped) =>
  popped.size === 0 ? tabs : tabs.filter(tab => !popped.has(tab.provider))
)

const CANVAS_TILE_PREFIX = 'canvas-tile'

const tileKey = (tab: Pick<CanvasTab, 'provider'>) => tab.provider

export function openCanvasTile(tab: CanvasTab): void {
  if (canvasPopped(tab.provider)) {
    markCanvasPopped(tab, tab.provider)
    void openCanvasInNewWindow(tab)

    return
  }

  $canvasTabs.set([...$canvasTabs.get().filter(t => t.provider !== tab.provider), tab])
  revealTreePane(`${CANVAS_TILE_PREFIX}:${tileKey(tab)}`)
}

export function closeCanvasTile(provider: string): void {
  $canvasTabs.set($canvasTabs.get().filter(t => t.provider !== provider))
}

/** The provider's canvas is gone (its document closed): take down the tile
 *  here and its window if it has one. */
export function dismissCanvasTile(provider: string): void {
  if (canvasPopped(provider)) {
    markCanvasPopped(null, provider)
    void window.hermesDesktop?.closeCanvasWindow?.(provider)
  }

  closeCanvasTile(provider)
}

/** The provider has a canvas up — here or in its own window. */
export function canvasTileOpen(provider?: string): boolean {
  const tabs = $canvasTabs.get()
  const popped = $poppedCanvasTabs.get()

  return provider ? tabs.some(t => t.provider === provider) || popped.has(provider) : tabs.length + popped.size > 0
}

/** The provider's tile is the active, un-minimized tab of its pane, or is
 *  out in its own window. */
export function canvasTileVisible(provider: string): boolean {
  return canvasPopped(provider) || isPaneVisible(`${CANVAS_TILE_PREFIX}:${provider}`)
}

/** Bring an open tile to the front; false when there is none to show. */
export function revealCanvasTile(provider: string): boolean {
  const popped = $poppedCanvasTabs.get().get(provider)

  if (popped) {
    void openCanvasInNewWindow(popped)

    return true
  }

  const tab = $canvasTabs.get().find(t => t.provider === provider)

  if (!tab) {
    return false
  }

  revealTreePane(`${CANVAS_TILE_PREFIX}:${tileKey(tab)}`)

  return true
}

function providerForKey(key: string): CanvasProvider | null {
  return providers.get(key) ?? null
}

/** Move a docked tile into its own OS window. The tab leaves this tree; the
 *  window seats it. Failure to open seats it back here. */
export function popOutCanvasTile(provider: string): void {
  const tab = $canvasTabs.get().find(t => t.provider === provider)

  if (!tab || !canOpenCanvasWindow()) {
    return
  }

  markCanvasPopped(tab, provider)
  providerForKey(provider)?.popOut?.()
  closeCanvasTile(provider)

  void openCanvasInNewWindow(tab).then(ok => {
    if (!ok) {
      markCanvasPopped(null, provider)
      openCanvasTile(tab)
    }
  })
}

/** The provider's window went away: seat its tab in this tree again. */
function dockCanvasTile(provider: string): void {
  const tab = $poppedCanvasTabs.get().get(provider)

  markCanvasPopped(null, provider)

  if (tab) {
    openCanvasTile(tab)
  }
}

function canvasTabMenuPrefix(provider: string) {
  const own = providerForKey(provider)?.tabMenu

  if (!own && !canOpenCanvasWindow()) {
    return undefined
  }

  return (kit: MenuKit) => (
    <>
      {own?.(kit)}
      {canOpenCanvasWindow()
        ? renderActionItem(kit, {
            icon: 'empty-window',
            key: 'pop-out',
            label: translateNow('preview.popOut'),
            onSelect: () => popOutCanvasTile(provider)
          })
        : null}
    </>
  )
}

function CanvasTabTitle({ provider }: { provider: string }) {
  const tabs = useStore($canvasTabs)

  return tabs.find(t => t.provider === provider)?.title || providerForKey(provider)?.untitled || 'Canvas'
}

const watchCanvasTileMirror = paneMirror<CanvasTab>({
  source: $dockedCanvasTabs,
  key: tileKey,
  prefix: CANVAS_TILE_PREFIX,
  dir: () => 'right',
  minWidth: '24rem',
  title: key => providerForKey(key)?.untitled || 'Canvas',
  tabLead: key => providerForKey(key)?.tabLead() ?? null,
  tabTitle: key => <CanvasTabTitle provider={key} />,
  tabMenuPrefix: canvasTabMenuPrefix,
  render: key => providerForKey(key)?.render() ?? null,
  close: key => {
    providerForKey(key)?.close()
    closeCanvasTile(key)
  }
})

/** Keep pane contributions mirroring the docked tabs and seat a popped tile
 *  again when its window closes. Call once from the root of a tree window. */
export function watchCanvasTiles(): void {
  watchCanvasTileMirror()
  window.hermesDesktop?.onCanvasPopoutClosed?.(dockCanvasTile)
}

/** The `?win=canvas` window's whole content: the seated provider's pane. */
export function CanvasPopoutHost() {
  const tabs = useStore($canvasTabs)
  const tab = tabs[0]
  const provider = tab ? providerForKey(tab.provider) : null

  return provider ? <ContribRender render={provider.render} /> : null
}

/** A `?win=canvas` window: seat the tab it was opened with, then follow
 *  whatever the docked side hands it. No-op anywhere else. */
export function seatCanvasWindow(): void {
  if (!isCanvasWindow()) {
    return
  }

  const tab = windowCanvasTab()

  if (tab) {
    $canvasTabs.set([tab])
  }

  window.hermesDesktop?.onCanvasPopoutTab?.(next => {
    $canvasTabs.set([next])
  })
}

/** Title for the popped window's document. */
export function useCanvasWindowTitle(): string {
  const tabs = useStore($canvasTabs)
  const tab = tabs[0]

  return tab?.title || (tab ? providerForKey(tab.provider)?.untitled : null) || 'Canvas'
}
