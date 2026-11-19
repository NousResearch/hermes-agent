import { registry } from '@/contrib/registry'

import { findGroupOfPane, groupLeafIds, type LayoutNode } from './model'

/**
 * Where preset adoption puts a pane the new tree doesn't place: the group of
 * a former zone-mate that made the trip. With none left, a chat surface (a
 * session tile, any main-placed pane) rejoins the chat zone. Falling back to
 * the first group dropped a lone session tile into the sessions sidebar,
 * which then took the main's width.
 */
export function adoptionTargetGroup(
  source: LayoutNode,
  next: LayoutNode,
  paneId: string,
  placed: ReadonlySet<string>
): string | undefined {
  const sibling = findGroupOfPane(source, paneId)?.panes.find(p => placed.has(p))

  if (sibling) {
    return findGroupOfPane(next, sibling)?.id ?? groupLeafIds(next)[0]
  }

  const data = registry.getArea('panes').find(c => c.id === paneId)?.data as { placement?: string } | undefined
  const chat = paneId.startsWith('session-tile:') || data?.placement === 'main'

  return (chat ? findGroupOfPane(next, 'workspace')?.id : undefined) ?? groupLeafIds(next)[0]
}
