/**
 * PREVIEW INPUT REGISTRY — real input into the preview pane's guest page, the
 * difference between the agent DRIVING the browser and merely poking its DOM.
 *
 * `executeJavaScript` can only ever dispatch synthetic events: `isTrusted` is
 * false, the browser's own hover target never moves, `:hover` rules never
 * match, and hover-gated menus never open — so a click lands on a dropdown item
 * that was never rendered. `sendInputEvent` goes in through Chromium's input
 * pipeline instead, producing the same events a hand on the mouse would.
 *
 * It has to be called on the `<webview>` ELEMENT. Sending to the embedder's
 * webContents does not reach a guest (electron/electron#20333), which is why
 * this is a per-pane registry rather than something main could do.
 *
 * Coordinates are relative to the webview element, in the host's
 * device-independent pixels. The act engine measures inside the guest in CSS
 * pixels, and Chromium places guest positions at css × zoom (the context-menu
 * handler in preview-pane.tsx measured it live), so a rect measured in the page
 * must be scaled by the guest's zoom factor on the way back out — the shipped
 * default zoom is 90 %, and at that zoom an unscaled click lands 11 % too far
 * from the origin and silently misses its target (#116281).
 */

import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'

/** The subset of Electron's input events the agent needs to drive a page. */
export type PreviewInputEvent =
  | { button: 'left'; clickCount: number; type: 'mouseDown' | 'mouseUp'; x: number; y: number }
  | { deltaX: number; deltaY: number; type: 'mouseWheel'; x: number; y: number }
  | { keyCode: string; modifiers?: string[]; type: 'char' | 'keyDown' | 'keyUp' }
  | { type: 'mouseMove'; x: number; y: number }

export interface PreviewInputHandle {
  /** Give the guest keyboard focus, so key events reach its active element. */
  focus: () => void
  send: (event: PreviewInputEvent) => void
}

/** Convert a point the act engine measured in guest CSS pixels into the
 *  webview's input space. Key events carry no point and pass through. */
export function toWebviewInputSpace(event: PreviewInputEvent, zoomFactor: number | undefined): PreviewInputEvent {
  if (!('x' in event) || !zoomFactor || !Number.isFinite(zoomFactor) || zoomFactor <= 0 || zoomFactor === 1) {
    return event
  }

  return { ...event, x: Math.round(event.x * zoomFactor), y: Math.round(event.y * zoomFactor) }
}

const handles = new Map<string, PreviewInputHandle>()

/** Register a live pane's input channel; returns an idempotent unregister. */
export function registerPreviewInput(tabId: string, handle: PreviewInputHandle): () => void {
  handles.set(tabId, handle)

  return () => {
    if (handles.get(tabId) === handle) {
      handles.delete(tabId)
    }
  }
}

/** The ACTIVE preview tab's input channel. Null = nothing real to drive, and
 *  the caller falls back to synthesizing events inside the page. */
export function activePreviewInput(): PreviewInputHandle | null {
  const tabs = $previewTabs.get()
  const tab = tabs.find(t => t.id === $rightRailActiveTabId.get()) ?? tabs[0]

  return (tab && handles.get(tab.id)) || null
}
