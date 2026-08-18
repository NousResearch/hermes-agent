/**
 * PEN PROVIDER — pen.dev as a canvas-tile provider.
 *
 * The generic pane surface (single-tile invariant, docking, tab, close)
 * lives in canvas-tile.tsx; this file contributes only what is pen-shaped:
 * the mark, the webview body on hermes-pen://, and the host close door.
 * Thin re-exports keep existing call sites (store/pen.ts, controller) on
 * pen-named verbs while the machinery underneath is provider-generic.
 */

import penMark from '@/assets/pen-mark.png'

import {
  type CanvasTab,
  canvasTileOpen,
  closeCanvasTile,
  openCanvasTile,
  registerCanvasProvider
} from './canvas-tile'
import { PenTilePane } from './pen-tile-pane'

const PEN_PROVIDER = 'pen'

registerCanvasProvider({
  id: PEN_PROVIDER,
  untitled: 'Canvas',
  /** pen.dev's pencil-tip mark, extracted from their own logo SVG (the mark
   *  is a raster pattern inside it), alpha-trimmed and re-centered — the raw
   *  asset carried ~27% transparent padding that made it render as a dot at
   *  tab size. User-directed bundling of the processed mark. */
  tabLead: () => <img alt="" className="size-[0.8125rem] shrink-0" src={penMark} />,
  render: docId => <PenTilePane docId={docId} />,
  close: () => {
    // Main saves the dirty document (closeDocument autosaves) and broadcasts
    // close-document; the generic tile prunes the pane list.
    void window.hermesDesktop?.pen?.close()
  }
})

export function openPenCanvasTile(tab: Omit<CanvasTab, 'provider'>): void {
  openCanvasTile({ ...tab, provider: PEN_PROVIDER })
}

export function closePenCanvasTile(docId: string): void {
  closeCanvasTile(PEN_PROVIDER, docId)
}

export function penCanvasTileOpen(): boolean {
  return canvasTileOpen(PEN_PROVIDER)
}
