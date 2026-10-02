/**
 * PARKED PANES — registered contributions that are deliberately OUT of the
 * layout tree. A parked pane keeps its contribution, so a keep-alive body (a
 * live Browser guest) stays mounted — hidden and inert in its keep-alive host
 * — while adoption must not dock it back. The contribution's owner parks and
 * unparks it: a preview tab whose session is not on screen.
 *
 * Memory-only: after a relaunch there is no live body left to keep.
 */

const parked = new Set<string>()

export function isTreePaneParked(paneId: string): boolean {
  return parked.has(paneId)
}

export function setTreePaneParked(paneId: string, isParked: boolean): void {
  if (isParked) {
    parked.add(paneId)
  } else {
    parked.delete(paneId)
  }
}
