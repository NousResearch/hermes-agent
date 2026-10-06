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
 * must be scaled by the guest's zoom factor on the way back out (#116281).
 */

import type { PreviewOwner } from '@/store/preview-ownership'

import { activePreviewTabFor } from './preview-active-tab'

/** The subset of Electron's input events the agent needs to drive a page. */
export type PreviewInputEvent =
  | { button: 'left'; clickCount: number; type: 'mouseDown' | 'mouseUp'; x: number; y: number }
  | { deltaX: number; deltaY: number; type: 'mouseWheel'; x: number; y: number }
  | { keyCode: string; modifiers?: string[]; type: 'char' | 'keyDown' | 'keyUp' }
  | { modifiers?: string[]; type: 'mouseMove'; x: number; y: number }

export interface PreviewInputHandle {
  /** Give the guest keyboard focus, so key events reach its active element. */
  focus: () => void
  send: (event: PreviewInputEvent) => void | Promise<void>
  /** Captured guest identity, shared across captures of this guest. */
  identity?: object
  run?: (code: string) => Promise<unknown>
  assertCurrent?: () => void
  /** Cleanup may reach the original surviving guest, never a replacement. */
  release?: (event: PreviewInputEvent) => void | Promise<void>
}

/** Convert a point the act engine measured in guest CSS pixels into the
 *  webview's input space. Key events carry no point and pass through. */
export function toWebviewInputSpace(event: PreviewInputEvent, zoomFactor: number | undefined): PreviewInputEvent {
  if (!('x' in event) || !zoomFactor || !Number.isFinite(zoomFactor) || zoomFactor <= 0 || zoomFactor === 1) {
    return event
  }

  return { ...event, x: Math.round(event.x * zoomFactor), y: Math.round(event.y * zoomFactor) }
}

type InputSource = PreviewInputHandle | (() => PreviewInputHandle | null)
const handles = new Map<string, InputSource>()
const leased = new WeakSet<object>()

export function leasePreviewInput(input: PreviewInputHandle): () => void {
  const identity = input.identity ?? input

  if (leased.has(identity)) {
    throw new Error("Another interaction is using this preview guest.")
  }

  leased.add(identity)

  return () => leased.delete(identity)
}

/** Register a live pane's input channel; returns an idempotent unregister. */
export function registerPreviewInput(tabId: string, handle: InputSource): () => void {
  handles.set(tabId, handle)

  return () => {
    if (handles.get(tabId) === handle) {
      handles.delete(tabId)
    }
  }
}

/** The ACTIVE preview tab's input channel among those `owner` (omitted = the
 *  focused session) may see. Null = nothing real to drive, and the caller
 *  falls back to synthesizing events inside the page. */
export function activePreviewInput(owner?: PreviewOwner, tabId?: string): PreviewInputHandle | null {
  const tab = activePreviewTabFor(owner, tabId)

  const source = tab && handles.get(tab.id)

  return typeof source === 'function' ? source() : source || null
}
