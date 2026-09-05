import penMark from '@/assets/pen-mark.png'

import {
  type CanvasTab,
  canvasTileOpen,
  closeCanvasTile,
  openCanvasTile,
  registerCanvasProvider
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
  }
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
  closeCanvasTile(PEN_PROVIDER)
}

export function penCanvasTileOpen(): boolean {
  return canvasTileOpen(PEN_PROVIDER)
}
