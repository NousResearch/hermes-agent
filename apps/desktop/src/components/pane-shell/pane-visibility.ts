/**
 * Keep-alive visibility — the one policy every document-wide lookup must obey.
 *
 * A tab group keeps each ever-active pane MOUNTED and hides the inactive ones
 * with `visibility: hidden` (see `tree/renderer/tree-group.tsx`), deliberately
 * preserving their layout box so scroll positions survive a tab round-trip. The
 * cost is that an inactive tab's rect is IDENTICAL to the visible tab's, so
 * neither selector order nor a rect hit-test can tell them apart. A lookup that
 * resolves "the chat surface / composer / viewport" from the document therefore
 * has to skip hidden panes, or it silently answers with the wrong tab.
 */

import { createContext, useContext } from 'react'

/** Marks a mounted-but-hidden pane layer (an inactive tab in a stack). */
export const PANE_HIDDEN_ATTR = 'data-pane-hidden'

const HIDDEN_PANE = `[${PANE_HIDDEN_ATTR}]`

/** Spread onto a kept pane layer so the lookups below can skip it. */
export const hiddenPaneProps = (hidden: boolean): Record<string, string> => (hidden ? { [PANE_HIDDEN_ATTR]: '' } : {})

/** React face of the same policy: the pane layer provides its visibility so a
 *  kept-alive surface can gate hot subscriptions (streaming re-renders) off
 *  while it's an inactive tab. Default TRUE — surfaces outside a tab stack
 *  (secondary windows, plain routes) are always visible. */
export const PaneVisibleContext = createContext(true)

export const usePaneVisible = (): boolean => useContext(PaneVisibleContext)

/** Fallback group key for a surface rendered outside the layout tree (secondary
 *  windows, plain routes) — one bucket, since there are no sibling zones there
 *  to tell apart. */
export const NO_PANE_GROUP = 'window'

/** The layout-tree GROUP (zone) a pane is rendered in — the identity of "this
 *  set of tabs". Panes stacked as tabs share one group; each split zone is its
 *  own. State that should be per-zone rather than per-window or per-tab keys off
 *  this (see the composer pop-out). Follows a pane dragged between zones,
 *  because the provider is the zone that renders it. */
export const PaneGroupContext = createContext(NO_PANE_GROUP)

export const usePaneGroup = (): string => useContext(PaneGroupContext)

/** `querySelectorAll` minus anything inside an inactive tab. */
export const queryAllVisible = <T extends HTMLElement>(selector: string, root: ParentNode = document): T[] =>
  [...root.querySelectorAll<T>(selector)].filter(el => !el.closest(HIDDEN_PANE))

/** `querySelector` minus anything inside an inactive tab. */
export const queryVisible = <T extends HTMLElement>(selector: string, root: ParentNode = document): null | T =>
  queryAllVisible<T>(selector, root)[0] ?? null
