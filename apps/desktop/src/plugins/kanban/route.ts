/**
 * The board page's address. Dependency-free so the notification path
 * (completion-notify, loaded by api.ts) can build links without a cycle.
 *
 * `/kanban?board=<slug>&task=<id>` opens the page on `board` with `task`'s
 * drawer open; either part may be left out. `deep-link.ts` consumes it.
 */

export const KANBAN_ROUTE = '/kanban'

/** The board page on `board` with `task`'s drawer open; the plain page when
 *  neither is given. */
export function kanbanHref(link: { board?: string; task?: string } = {}): string {
  const params = new URLSearchParams()

  if (link.board) {
    params.set('board', link.board)
  }

  if (link.task) {
    params.set('task', link.task)
  }

  const query = params.toString()

  return query ? `${KANBAN_ROUTE}?${query}` : KANBAN_ROUTE
}
