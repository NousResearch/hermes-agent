/**
 * When a lone pane must keep its tab strip (name card + close).
 *
 * Default: a single pane isn't a "tab", so the header auto-hides. Exceptions
 * force it on so a closeable surface never becomes an unclosable dead zone:
 *  - a closeable `placement: 'main'` pane — every mirrored TILE (a session, a
 *    page, a preview) is one, so dragging a tile into a zone of its own keeps
 *    its tab and its ✕
 *  - a collapse tool panel dragged into its own zone
 */

export interface LoneHeaderChrome {
  placement?: string
  uncloseable?: boolean
}

/** True when any shown pane is a closeable main-strip tile (preview / session / page). */
export function hasCloseableMainTile(
  shown: readonly string[],
  chromeOf: (id: string) => LoneHeaderChrome
): boolean {
  return shown.some(id => {
    const chrome = chromeOf(id)

    return !chrome.uncloseable && chrome.placement === 'main'
  })
}

export function forceLoneHeaderForPanes(
  shown: readonly string[],
  chromeOf: (id: string) => LoneHeaderChrome,
  isCollapsePane: (id: string) => boolean
): boolean {
  // "This pane can be closed, so it must expose the ✕." Only the uncloseable
  // workspace is exempt; standing side chrome (files / sessions) isn't 'main'.
  if (hasCloseableMainTile(shown, chromeOf)) {
    return true
  }

  return shown.length === 1 && isCollapsePane(shown[0])
}

/**
 * Whether the zone tab strip should stay hidden.
 *
 * `headerHiddenFlag` is the user's sticky "Hide tab bar" choice. That preference
 * still applies to tool-only zones, but it must not win over a closeable main
 * tile (in-app Browser / preview / session): otherwise the strip and close control
 * disappear with no recovery surface on the body.
 */
export function resolveZoneHeaderHidden(options: {
  headerVeto?: boolean
  /** `node.headerHidden` — explicit sticky hide, or undefined for auto. */
  headerHiddenFlag?: boolean
  shownLength: number
  forceLoneHeader: boolean
  hasCloseableMainTile: boolean
}): boolean {
  if (options.headerVeto) {
    return true
  }

  if (options.hasCloseableMainTile) {
    return false
  }

  return options.headerHiddenFlag ?? (options.shownLength <= 1 && !options.forceLoneHeader)
}
