/**
 * `/kanban?board=<slug>&task=<id>` deep-link contract — the gap this module
 * fills: board selection lived only in the persisted `$boardSlug` atom and
 * nothing read `?task=` anywhere, not even the native completion-notify
 * click-through (`completion-notify.ts`) or the statusbar `KanbanCount`
 * click (`plugin.tsx`) — both keep navigating to plain `/kanban` and are
 * unaffected by this file. This is a NEW entry point other code/URLs can use:
 * `host.navigate('/kanban?board=default&task=t_c3626291')`.
 */

import { useSearchParams } from '@hermes/plugin-sdk'
import { useEffect, useRef } from 'react'

import { $boardSlug } from './api'

/** Pure parse of the deep-link params — no React, no store reads, so the
 *  parsing rules (trim, empty-string-means-absent) are unit-testable without
 *  mounting anything. */
export function parseKanbanDeepLink(search: string): { board: string; task: string } {
  const params = new URLSearchParams(search)

  return { board: params.get('board')?.trim() ?? '', task: params.get('task')?.trim() ?? '' }
}

/**
 * Applies a `board`/`task` deep link exactly once per mount: points
 * `$boardSlug` at the requested board first (only when it differs from the
 * current selection, so a link back to the already-open board doesn't thrash
 * the events socket — see `api.ts`'s `bindApi`), THEN hands the task id to
 * `onOpenTask` — in that order, so by the time the drawer's own `fetchTask`
 * query key reads `$boardSlug`, it already resolves against the right board.
 *
 * Applied once: after the initial navigation, the board switcher and the
 * drawer's own close button own that state, not the URL — re-running this on
 * every incidental `searchParams` identity change would reopen a drawer the
 * user just closed.
 */
export function useKanbanDeepLink(onOpenTask: (id: string) => void): void {
  const [searchParams] = useSearchParams()
  const applied = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write: a one-shot mount guard, not an atom mirror
  useEffect(() => {
    if (applied.current) {
      return
    }

    applied.current = true

    const { board, task } = parseKanbanDeepLink(searchParams.toString())

    if (board && board !== $boardSlug.get()) {
      $boardSlug.set(board)
    }

    if (task) {
      onOpenTask(task)
    }
  }, [searchParams, onOpenTask])
}
