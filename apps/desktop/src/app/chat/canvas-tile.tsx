/**
 * CANVAS TILES — design surfaces as layout-tree panes, provider-shaped.
 *
 * The pane machinery (single-tile invariant, tree docking, tab, close
 * semantics) is design-tool-agnostic: a canvas is "an embedded editor surface
 * tied to the active session". Each TOOL plugs in as a CanvasProvider —
 * pen.dev is the first and currently only one, but the seam is the point:
 * a future provider registers itself here and inherits the whole surface
 * (pane, session swap, reveal, close) without touching the tree.
 *
 * Main owns each provider's documents; this file owns their presentation.
 */

import type { ReactNode } from 'react'
import { atom } from 'nanostores'

import { revealTreePane } from '@/components/pane-shell/tree/store'

import { paneMirror } from './pane-mirror'

export interface CanvasTab {
  /** Which provider owns this surface (e.g. 'pen'). */
  provider: string
  docId: string
  title: string
  url: string
}

export interface CanvasProvider {
  /** Stable id, also the tab's pane-id namespace segment. */
  id: string
  /** Fallback tab title when the document hasn't named itself. */
  untitled: string
  /** The tab's lead glyph (the tool's mark). */
  tabLead: () => ReactNode
  /** The pane body — the embedded editor surface for one document. */
  render: (docId: string) => ReactNode
  /** Put the document away (host-side close; autosave is the provider's
   *  responsibility). The tile list is pruned by the caller either way. */
  close: (docId: string) => void
}

const providers = new Map<string, CanvasProvider>()

/** Register a design-tool provider. First registration wins per id —
 *  providers are module-level singletons, not hot-swappable state. */
export function registerCanvasProvider(provider: CanvasProvider): void {
  if (!providers.has(provider.id)) {
    providers.set(provider.id, provider)
  }
}

/** The open canvas tabs (usually 0 or 1 — one surface per session). */
export const $canvasTabs = atom<CanvasTab[]>([])

const CANVAS_TILE_PREFIX = 'canvas-tile'

const tileKey = (tab: Pick<CanvasTab, 'docId' | 'provider'>) => `${tab.provider}:${tab.docId}`

export function openCanvasTile(tab: CanvasTab): void {
  // ONE canvas pane, mirroring the hosts' single-document invariant: a new
  // doc REPLACES the pane's content rather than adding a second pane. Keeping
  // the list single-entry here means even a missed close event can't strand a
  // ghost pane.
  $canvasTabs.set([tab])

  revealTreePane(`${CANVAS_TILE_PREFIX}:${tileKey(tab)}`)
}

export function closeCanvasTile(provider: string, docId: string): void {
  $canvasTabs.set($canvasTabs.get().filter(t => t.provider !== provider || t.docId !== docId))
}

export function canvasTileOpen(provider?: string): boolean {
  const tabs = $canvasTabs.get()

  return provider ? tabs.some(t => t.provider === provider) : tabs.length > 0
}

function tabForKey(key: string): CanvasTab | null {
  return $canvasTabs.get().find(t => tileKey(t) === key) ?? null
}

function providerForKey(key: string): CanvasProvider | null {
  return providers.get(key.split(':', 1)[0]) ?? null
}

const docIdOf = (key: string) => key.slice(key.indexOf(':') + 1)

/** Mirror `$canvasTabs` into tree panes. Call once from the contrib root. */
export const watchCanvasTiles = paneMirror<CanvasTab>({
  source: $canvasTabs,
  key: tileKey,
  prefix: CANVAS_TILE_PREFIX,
  // Docked right of the workspace, like preview tiles — its own zone, its own
  // sash, and the canvas participates in the ONE window's layout: no second
  // shadow, no seam, theme cascades like any other pane.
  dir: () => 'right',
  minWidth: '24rem',
  // The tab row STAYS. It was hidden once (headerVeto, "the editor is the
  // chrome") and that deleted the only visible close — the user sat with
  // three canvases and no exit. One slim row carrying the tool's mark, the
  // doc name, and ✕ is the honest price of an always-visible way out.
  title: key => tabForKey(key)?.title || providerForKey(key)?.untitled || 'Canvas',
  tabLead: key => providerForKey(key)?.tabLead() ?? null,
  render: key => providerForKey(key)?.render(docIdOf(key)) ?? null,
  close: key => {
    // Tab close = put the surface away for this session. The provider's host
    // saves and broadcasts its close event; the tab list is pruned by the
    // event watcher, but prune here too so the pane never outlives an
    // unreachable surface.
    providerForKey(key)?.close(docIdOf(key))
    closeCanvasTile(key.split(':', 1)[0], docIdOf(key))
  }
})
