/** Design surfaces as layout-tree panes. Pen registers as the provider.
 *  One pane per provider — the live .pen can change without remounting. */

import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { atom } from 'nanostores'

import { revealTreePane } from '@/components/pane-shell/tree/store'

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
}

const providers = new Map<string, CanvasProvider>()

export function registerCanvasProvider(provider: CanvasProvider): void {
  if (!providers.has(provider.id)) {
    providers.set(provider.id, provider)
  }
}

export const $canvasTabs = atom<CanvasTab[]>([])

const CANVAS_TILE_PREFIX = 'canvas-tile'

const tileKey = (tab: Pick<CanvasTab, 'provider'>) => tab.provider

export function openCanvasTile(tab: CanvasTab): void {
  $canvasTabs.set([...$canvasTabs.get().filter(t => t.provider !== tab.provider), tab])
  revealTreePane(`${CANVAS_TILE_PREFIX}:${tileKey(tab)}`)
}

export function closeCanvasTile(provider: string): void {
  $canvasTabs.set($canvasTabs.get().filter(t => t.provider !== provider))
}

export function canvasTileOpen(provider?: string): boolean {
  const tabs = $canvasTabs.get()

  return provider ? tabs.some(t => t.provider === provider) : tabs.length > 0
}

function providerForKey(key: string): CanvasProvider | null {
  return providers.get(key) ?? null
}

function CanvasTabTitle({ provider }: { provider: string }) {
  const tabs = useStore($canvasTabs)

  return tabs.find(t => t.provider === provider)?.title || providerForKey(provider)?.untitled || 'Canvas'
}

export const watchCanvasTiles = paneMirror<CanvasTab>({
  source: $canvasTabs,
  key: tileKey,
  prefix: CANVAS_TILE_PREFIX,
  dir: () => 'right',
  minWidth: '24rem',
  title: key => providerForKey(key)?.untitled || 'Canvas',
  tabLead: key => providerForKey(key)?.tabLead() ?? null,
  tabTitle: key => <CanvasTabTitle provider={key} />,
  render: key => providerForKey(key)?.render() ?? null,
  close: key => {
    providerForKey(key)?.close()
    closeCanvasTile(key)
  }
})
