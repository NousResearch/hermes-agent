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

// The container a popover should portal into: an explicit `container` prop wins,
// then the enclosing dialog's content node, then undefined (Radix default:
// document.body).
export function usePopoverPortalContainer(explicit?: HTMLElement | null): HTMLElement | undefined {
  const fromDialog = React.useContext(DialogPortalContainerContext)

  return explicit ?? fromDialog ?? undefined
}
