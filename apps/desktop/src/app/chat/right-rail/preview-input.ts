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
 * Coordinates are relative to the webview, and the webview IS the guest
 * viewport — so a rect the act engine measured inside the page needs no
 * conversion on the way back out.
 */

import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs } from '@/store/preview'

/** The subset of Electron's input events the agent needs to drive a page. */
export type PreviewInputEvent =
  | { button: 'left'; clickCount: number; type: 'mouseDown' | 'mouseUp'; x: number; y: number }
  | { deltaX: number; deltaY: number; type: 'mouseWheel'; x: number; y: number }
  | { keyCode: string; modifiers?: string[]; type: 'char' | 'keyDown' | 'keyUp' }
  | { type: 'mouseMove'; x: number; y: number }

/**
 * Convert a pointer event's CSS-pixel coordinates into the physical widget
 * space Chromium's input router expects for a scaled guest.
 *
 * On displays where the guest's `devicePixelRatio` is ≠ 1 (fractional scaling
 * on Wayland/Windows, or any webview frame Chromium composites at a scale
 * ≠ 1), `webview.sendInputEvent` coordinates are DIVIDED by that scale factor
 * before the event is dispatched into the page, so a click aimed at a
 * CSS-space point lands at `point / DPR` — short by `1 − 1/DPR`. Multiplying
 * by the guest's DPR before sending cancels the division and puts the click
 * exactly where the act engine measured it (issue #91130).
 *
 * Identity at DPR 1, and keyboard events (which carry no x/y) pass through
 * untouched.
 */
export function scalePreviewInputForDpr(event: PreviewInputEvent, dpr: number): PreviewInputEvent {
  if (dpr === 1 || !('x' in event)) {
    return event
  }

  return { ...event, x: Math.round(event.x * dpr), y: Math.round(event.y * dpr) }
}

export interface PreviewInputHandle {
  /** Give the guest keyboard focus, so key events reach its active element. */
  focus: () => void
  send: (event: PreviewInputEvent) => void
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
