import * as React from 'react'

// Layering system for portalled popovers (Select / Popover / DropdownMenu) that
// open INSIDE a Dialog.
//
// By default every Radix popover portals to `document.body`, i.e. as a sibling
// of the Dialog — OUTSIDE the dialog's DOM subtree. That breaks two things:
//   1. Dismissing the popover (clicking elsewhere in the dialog) moves focus and
//      fires pointer/focus events the Dialog's DismissableLayer reads as
//      "outside", so the whole dialog closes.
//   2. z-index across separate body-level portals is fragile — the popover and
//      dialog live in different stacking siblings.
//
// The fix: a Dialog publishes its own content node here; any popover rendered
// inside that Dialog portals into that node instead of `document.body`. Now the
// popover is a real DOM descendant of the dialog — focus never leaves, the
// DismissableLayer sees it as inside, and it shares the dialog's stacking
// context so z-index is deterministic. Outside a dialog the value is null and
// popovers fall back to the default body portal.
export const DialogPortalContainerContext = React.createContext<HTMLElement | null>(null)

export type BrowserPortalLayer = 'menu' | 'select' | 'tooltip'

// On the browser-hosted Webapp, UI scale lives as CSS `zoom` on `<html>`.
// Floating UI skips `getScale()` when a `position: fixed` popper's
// `offsetParent` is `window` (direct child of `document.body`), double-scaling
// coordinates, available sizes, anchor widths, and arrows (#148414, #150502,
// #150729). Portaling into a full-viewport fixed host with `contain: layout`
// gives Floating UI a real element containing block whose `getScale()` equals
// the active CSS zoom so every Radix primitive scales natively. Native Desktop
// leaves `data-hermes-desktop-host` unset and keeps Radix's default body portal.
export function getBrowserPortalContainer(layer: BrowserPortalLayer): HTMLElement | undefined {
  if (typeof document === 'undefined' || document.documentElement.dataset.hermesDesktopHost !== 'browser') {
    return undefined
  }

  const { body } = document

  if (!body) {
    return undefined
  }

  const existing = document.querySelector<HTMLElement>(`body > [data-browser-portal-host="${layer}"]`)

  if (existing) {
    return existing
  }

  const host = document.createElement('div')
  host.dataset.browserPortalHost = layer
  body.appendChild(host)

  return host
}

// The container a popover should portal into: an explicit `container` prop wins,
// then the enclosing dialog's content node, then the Webapp's containing-block
// portal host (or undefined on native Desktop, which uses Radix's body portal).
export function usePopoverPortalContainer(
  explicit?: HTMLElement | null,
  layer: BrowserPortalLayer = 'menu'
): HTMLElement | undefined {
  const fromDialog = React.useContext(DialogPortalContainerContext)

  return explicit ?? fromDialog ?? getBrowserPortalContainer(layer)
}
