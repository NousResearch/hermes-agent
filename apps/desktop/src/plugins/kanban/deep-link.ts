/**
 * Deep links into the board page: `/kanban?board=<slug>&task=<id>` (build it
 * with `kanbanHref`, ./route) opens the page on `board` with `task`'s drawer
 * open. Either part may be left out: `?task=` alone opens the card on the
 * selected board, `?board=` alone just switches boards. Any caller can use it: `host.navigate(...)` from another
 * plugin, a toast action, or `hermes://open/kanban?board=…&task=…`.
 *
 * The board is checked against the active connection's board list before the
 * selection moves (an unknown slug keeps the current board and says so), and
 * the card opens only once that board's snapshot has it. Picking a board this
 * way is a real selection, persisted like a switcher click.
 */

import { host, useQueryClient, useRouteQuery } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { $boardSlug, boardsKey, fetchBoards, kanbanConnectionScope } from './api'
import type { KanbanText } from './i18n'
import { KANBAN_ROUTE } from './route'
import type { KanbanBoard } from './types'
import { errText } from './ui'

interface PendingLink {
  /** Connection the link arrived on — a switch mid-resolve drops it. */
  scope: string
  /** Requested board slug ('' = stay on the selected board). */
  board: string
  /** Task id to open ('' = none). */
  task: string
  /** The `$boardSlug` the task is looked up on; null until the board resolves. */
  slug: null | string
}

interface DeepLinkOptions {
  /** The page's current board snapshot (undefined while loading). */
  board: KanbanBoard | undefined
  /** True while that snapshot is (re)fetching — a miss only counts once settled. */
  boardFetching: boolean
  k: KanbanText
  onOpenTask: (id: string) => void
  scope: string
  slug: string
}

const hasTask = (board: KanbanBoard, id: string) => board.columns.some(col => col.tasks.some(task => task.id === id))

/** Consume a `/kanban?board=…&task=…` link once and drive the page to it. */
export function useKanbanDeepLink({ board, boardFetching, k, onOpenTask, scope, slug }: DeepLinkOptions): void {
  const qc = useQueryClient()
  const query = useRouteQuery(KANBAN_ROUTE)
  const linkBoard = query?.params.get('board')?.trim() ?? ''
  const linkTask = query?.params.get('task')?.trim() ?? ''
  const [pending, setPending] = useState<null | PendingLink>(null)

  useEffect(() => {
    if (!query || (!linkBoard && !linkTask) || !query.take()) {
      return
    }

    setPending({ board: linkBoard, scope, slug: linkBoard ? null : slug, task: linkTask })
  }, [linkBoard, linkTask, query, scope, slug])

  // Resolve the board against this connection's list — fetched fresh, so a
  // board created moments before the link was sent still counts.
  useEffect(() => {
    if (!pending || pending.slug !== null) {
      return
    }

    let cancelled = false

    qc.fetchQuery({ queryFn: fetchBoards, queryKey: boardsKey(pending.scope) })
      .then(list => {
        if (cancelled) {
          return
        }

        // A connection switch landed mid-fetch: this list (and slug) is the
        // old gateway's, and the new one keeps its own selection.
        if (kanbanConnectionScope() !== pending.scope) {
          setPending(null)

          return
        }

        if (!list.boards.some(meta => meta.slug === pending.board)) {
          host.notify({ kind: 'info', message: k.linkBoardMissing(pending.board) })
          setPending(null)

          return
        }

        // The server's current board is spelled '' — the switcher's convention.
        const next = pending.board === list.current ? '' : pending.board

        $boardSlug.set(next)
        setPending(pending.task ? { ...pending, slug: next } : null)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          host.notify({ kind: 'error', message: errText(err) })
          setPending(null)
        }
      })

    return () => {
      cancelled = true
    }
  }, [k, pending, qc])

  // Open the card once the target board's snapshot has settled.
  useEffect(() => {
    if (!pending || pending.slug === null) {
      return
    }

    // The user (or a connection switch) moved on before the board arrived.
    if (pending.scope !== scope || pending.slug !== slug) {
      setPending(null)

      return
    }

    if (!board) {
      return
    }

    if (hasTask(board, pending.task)) {
      onOpenTask(pending.task)
    } else if (boardFetching) {
      return
    } else {
      host.notify({ kind: 'info', message: k.linkTaskMissing(pending.task) })
    }

    setPending(null)
  }, [board, boardFetching, k, onOpenTask, pending, scope, slug])
}
