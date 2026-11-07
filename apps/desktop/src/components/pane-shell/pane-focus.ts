/** Explicit pane selection is distinct from a background layout activation.
 * Consumers may claim typing focus after the requested pane becomes visible. */
const PANE_FOCUS_EVENT = 'hermes:pane-focus'

export function requestPaneFocus(paneId: string) {
  window.dispatchEvent(new CustomEvent<string>(PANE_FOCUS_EVENT, { detail: paneId }))
}

export function onPaneFocusRequest(handler: (paneId: string) => void): () => void {
  const listener = (event: Event) => handler((event as CustomEvent<string>).detail)
  window.addEventListener(PANE_FOCUS_EVENT, listener)

  return () => window.removeEventListener(PANE_FOCUS_EVENT, listener)
}

export const treeGroupOfEvent = (event: Event): null | string => {
  const el = event.target instanceof HTMLElement ? event.target : null

  return el?.closest<HTMLElement>('[data-tree-group]')?.dataset.treeGroup ?? null
}
