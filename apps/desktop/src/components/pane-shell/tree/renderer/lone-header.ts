/**
 * When a lone pane must keep its tab strip (name card + close).
 *
 * Default: a single pane isn't a "tab", so the header auto-hides. Exceptions
 * force it on so a closeable surface never becomes an unclosable dead zone:
 *  - session tiles (`session-tile:*`) — even before chrome registers
 *  - any closeable `placement: 'main'` contribution
 *  - a collapse tool panel dragged into its own zone
 */

export interface LoneHeaderChrome {
  placement?: string
  uncloseable?: boolean
}

export function forceLoneHeaderForPanes(
  shown: readonly string[],
  chromeOf: (id: string) => LoneHeaderChrome,
  isCollapsePane: (id: string) => boolean
): boolean {
  if (shown.some(id => id.startsWith('session-tile:'))) {
    return true
  }

  if (
    shown.some(id => {
      const chrome = chromeOf(id)

      return !chrome.uncloseable && chrome.placement === 'main'
    })
  ) {
    return true
  }

  return shown.length === 1 && isCollapsePane(shown[0])
}
