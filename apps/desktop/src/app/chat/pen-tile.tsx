import penMark from '@/assets/pen-mark.png'
import { renderActionItem } from '@/components/ui/actions-menu'
import { translateNow } from '@/i18n'
import { openBrowserForPenImport } from '@/store/pen-import'

import {
  canvasPopped,
  type CanvasTab,
  canvasTileOpen,
  canvasTileVisible,
  closeCanvasTile,
  dismissCanvasTile,
  openCanvasTile,
  registerCanvasProvider,
  revealCanvasTile
} from './canvas-tile'
import { PenTilePane } from './pen-tile-pane'
import { destroyPenWebview } from './pen-webview'

const PEN_PROVIDER = 'pen'

registerCanvasProvider({
  id: PEN_PROVIDER,
  untitled: 'Canvas',
  tabLead: () => <img alt="" className="size-[0.8125rem] shrink-0" src={penMark} />,
  render: () => <PenTilePane />,
  close: () => {
    destroyPenWebview()
    void window.hermesDesktop?.pen?.close()
  },
  // One editor guest at a time: the window that hosts the pane owns it, and
  // main's bridge binds whichever guest attaches next.
  popOut: destroyPenWebview,
  // The reverse door of the browser bar's Import glyph: from the canvas, go
  // find a page to bring in. Nobody guesses that a browser can feed a canvas.
  tabMenu: kit =>
    renderActionItem(kit, {
      icon: 'globe',
      key: 'import-from-web',
      label: translateNow('pen.importFromWeb'),
      onSelect: openBrowserForPenImport
    })
})

export function openPenCanvasTile(tab: Omit<CanvasTab, 'provider'>): void {
  openCanvasTile({ ...tab, provider: PEN_PROVIDER })
}

/** Put the pane away. The editor guest stays up so a background agent can work. */
export function hidePenCanvasTile(): void {
  closeCanvasTile(PEN_PROVIDER)
}

export function closePenCanvasTile(): void {
  destroyPenWebview()
  dismissCanvasTile(PEN_PROVIDER)
}

export function penCanvasTileOpen(): boolean {
  return canvasTileOpen(PEN_PROVIDER)
}

export function penCanvasTileVisible(): boolean {
  return canvasTileVisible(PEN_PROVIDER)
}

/** The pane is in its own window, which hosts the guest; this side must not. */
export function penCanvasPopped(): boolean {
  return canvasPopped(PEN_PROVIDER)
}

export function revealPenCanvasTile(): boolean {
  return revealCanvasTile(PEN_PROVIDER)
}
