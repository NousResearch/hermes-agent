/**
 * Board pause (#141451) — the operator's "hold everything, I'm changing the
 * provider" control.
 *
 * Three surfaces, one state: the header toggle, the warning banner across the
 * board, and the red PAUSED chip in the left nav. All read `board.pause` from
 * the shared board query (the backend puts it on every payload), so a pause
 * engaged from the CLI, the web dashboard or another window shows up on the
 * next poll/event with no extra endpoint to keep in sync.
 *
 * Semantics the copy promises and the backend enforces: nothing NEW starts
 * (dispatch, auto-decompose, review spawns); running workers finish.
 */

import {
  Button,
  cn,
  Codicon,
  host,
  queryClient as qc,
  Tip,
  useMutation,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'

import { $boardSlug, boardKeyPrefix, boardsKey, fetchBoards, pauseBoard, resumeBoard, useKanbanScope } from './api'
import { useKanban } from './i18n'
import type { KanbanPause } from './types'
import { errText } from './ui'

/** Pause (or resume) the selected board; the board query is what refreshes. */
export function usePauseToggle() {
  const scope = useKanbanScope()

  return useMutation({
    mutationFn: (resume: boolean) => (resume ? resumeBoard() : pauseBoard()),
    onError: err => host.notify({ kind: 'error', message: errText(err) }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: boardKeyPrefix(scope) })
      // The switcher and the nav chip read the boards list — it carries each
      // board's pause state, so refresh it with the board itself.
      void qc.invalidateQueries({ queryKey: boardsKey(scope) })
    }
  })
}

/** Header toggle: pause when running, resume when paused. */
export function PauseButton({ paused }: { paused: boolean }) {
  const k = useKanban()
  const toggle = usePauseToggle()
  const label = paused ? k.resumeBoard : k.pauseBoard

  return (
    <Tip label={paused ? k.resumeTip : k.pauseTip}>
      <Button
        aria-label={label}
        aria-pressed={paused}
        className={cn(paused && 'text-red-600 dark:text-red-300')}
        disabled={toggle.isPending}
        onClick={() => toggle.mutate(paused)}
        size="sm"
        variant="outline"
      >
        <Codicon name={paused ? 'debug-continue' : 'debug-pause'} size="0.8rem" />
        {label}
      </Button>
    </Tip>
  )
}

/** Full-width warning strip pinned above the columns while the board is paused. */
export function PauseBanner({ pause }: { pause: KanbanPause }) {
  const k = useKanban()
  const toggle = usePauseToggle()

  return (
    <div
      className="flex shrink-0 items-center gap-2 border-b border-amber-400/40 bg-amber-50 px-4 py-2 text-xs text-amber-900 dark:border-amber-300/30 dark:bg-amber-300/10 dark:text-amber-100"
      data-testid="kanban-pause-banner"
      role="status"
    >
      <Codicon className="shrink-0 text-amber-600 dark:text-amber-300" name="debug-pause" size="0.9rem" />
      <span className="font-semibold">{k.pauseBannerTitle}</span>
      <span className="min-w-0 truncate opacity-80">{k.pauseBannerBody}</span>
      {pause.reason && (
        <span className="min-w-0 truncate rounded-sm bg-amber-500/20 px-1.5 py-px">{pause.reason}</span>
      )}
      <Button
        className="ml-auto shrink-0"
        disabled={toggle.isPending}
        onClick={() => toggle.mutate(true)}
        size="xs"
        variant="outline"
      >
        <Codicon name="debug-continue" size="0.75rem" />
        {k.resumeBoard}
      </Button>
    </div>
  )
}

/** Red PAUSED chip on the Kanban row in the left nav (sidebar.nav.trailing).
 *
 * Pause is PER BOARD, so the chip must not read as "the kanban is paused":
 * with one board held it names that board, with several it shows how many —
 * the tooltip lists them. A board-only signal here would be wrong as soon as
 * a second board exists (#141451). */
export function KanbanPauseNavBadge() {
  const k = useKanban()
  const scope = useKanbanScope()
  const slug = useValue($boardSlug)

  // Same list the switcher reads (`/boards` now carries each board's pause),
  // refreshed on the board's own heartbeat.
  const { data: boards } = useQuery({
    queryFn: fetchBoards,
    queryKey: boardsKey(scope),
    refetchInterval: 60_000,
    staleTime: 30_000
  })

  if (!boards) {
    return null
  }

  const held = boards.boards.filter(b => b.pause)

  if (held.length === 0) {
    return null
  }

  const currentSlug = slug || boards.current
  const current = boards.boards.find(b => b.slug === currentSlug)
  const one = held.length === 1 ? held[0] : null
  const name = (b: (typeof held)[number]) => b.name || b.slug

  // One held board: say WHICH one unless it is the board on screen — a bare
  // "PAUSED" beside the Kanban row reads as the board being viewed.
  const label = one
    ? current?.pause
      ? k.pausedBadge
      : k.pausedOtherBoard(k.pausedBadge, name(one))
    : k.pausedBadgeCount(held.length)

  const title = one
    ? [name(one), one.pause?.reason].filter(Boolean).join(' — ')
    : k.pausedListTip(held.map(name).join(', '))

  return (
    <span
      className="ml-auto inline-flex min-w-0 shrink items-center gap-1 rounded-full bg-red-500 px-1.5 py-px text-[0.625rem] font-semibold tracking-wide text-white"
      data-testid="kanban-nav-paused"
      title={title}
    >
      <Codicon name="debug-pause" size="0.6rem" />
      <span className="truncate">{label}</span>
    </span>
  )
}
