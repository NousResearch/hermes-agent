/**
 * Conversation-side Kanban surfaces: the sidebar row badge and the composer
 * strip. Both read ONE view (`useOriginView`) and paint only what it can back.
 *
 *  - Badge: a small glyph beside the row's status dot (never a circle, never a
 *    replacement for it — the dot keeps owning foreground/subagent/process
 *    state). Shown only for a confirmed, not-yet-finished link.
 *  - Strip: above the composer's status stack, collapsed by default. It lists
 *    every linked task — completed ones too, for history — each with an
 *    explicit "open task" and "worker log" action that go through the board's
 *    own drawer. Nothing here ever opens or focuses anything by itself.
 *
 * Empty and unavailable are different: with no linked tasks there is nothing to
 * paint, but a failed or unroutable lookup says so out loud — it is never idle.
 */

import { atom, Button, cn, Codicon, type SessionRouteContext, Tip, useValue } from '@hermes/plugin-sdk'

import {
  isTerminalState,
  type OriginState,
  type OriginView,
  refState,
  summarizeOrigin,
  useOriginView
} from './origin-links'
import { openLinkedTask, shortId, useKanban } from './ui'

/** Glyph + token tone per state. A glyph, because the sidebar dot's circle vocabulary is core's. */
const STATE_META: Record<OriginState, { icon: string; tone: string }> = {
  'needs-input': { icon: 'comment-discussion', tone: 'text-amber-500' },
  background: { icon: 'server-process', tone: 'text-(--ui-accent)' },
  stale: { icon: 'warning', tone: 'text-(--ui-text-secondary)' },
  unavailable: { icon: 'warning', tone: 'text-(--ui-text-secondary)' },
  unknown: { icon: 'question', tone: 'text-(--ui-text-tertiary)' },
  reserved: { icon: 'lock', tone: 'text-(--ui-text-tertiary)' },
  blocked: { icon: 'error', tone: 'text-(--ui-text-tertiary)' },
  review: { icon: 'eye', tone: 'text-(--ui-text-tertiary)' },
  waiting: { icon: 'watch', tone: 'text-(--ui-text-tertiary)' },
  queued: { icon: 'circle-large-outline', tone: 'text-(--ui-text-tertiary)' },
  done: { icon: 'check', tone: 'text-(--ui-text-quaternary)' },
  archived: { icon: 'archive', tone: 'text-(--ui-text-quaternary)' }
}

export function OriginRowBadge({ context }: { context: SessionRouteContext }) {
  const k = useKanban()
  const view = useOriginView(context)

  // Loading, unroutable and failed lookups paint nothing on a row: a row-level
  // mark would have to claim something, and the strip is where a gap is spelled out.
  if (view.kind !== 'ready') {
    return null
  }

  const { live, state } = summarizeOrigin(view.refs)

  if (!state || isTerminalState(state)) {
    return null
  }

  const label = k.origin.state[state]
  const tip = live > 1 ? `${label} · ${k.origin.summary(live)}` : label
  const meta = STATE_META[state]

  return (
    <Tip label={tip}>
      <span
        aria-label={tip}
        className={cn('inline-flex shrink-0 items-center gap-0.5 text-[0.625rem] tabular-nums', meta.tone)}
        data-kanban-origin={state}
        role="status"
      >
        <Codicon name={meta.icon} size="0.7rem" />
        {live > 1 && <span>{live}</span>}
      </span>
    </Tip>
  )
}

// Disclosure is per conversation and per owner; ephemeral by design.
const $expanded = atom<Record<string, boolean>>({})

type ReadyView = Extract<OriginView, { kind: 'ready' }>

function OriginList({ view }: { view: ReadyView }) {
  const k = useKanban()

  return (
    <ul className="flex max-h-40 flex-col gap-0.5 overflow-y-auto" data-slot="kanban-origin-list">
      {view.refs.map(ref => {
        const state = refState(ref)
        const meta = STATE_META[state]
        const task = ref.evidence === 'ok' ? ref.task : null

        return (
          <li
            className="flex min-w-0 items-center gap-2 text-[0.6875rem]"
            data-kanban-origin-ref={ref.task_id}
            data-kanban-origin-state={state}
            key={`${ref.board}\0${ref.task_id}`}
          >
            <Codicon className={cn('shrink-0', meta.tone)} name={meta.icon} size="0.75rem" />
            {task ? (
              <Button
                aria-label={k.origin.openTask(task.title)}
                className="min-w-0 truncate"
                onClick={() => openLinkedTask({ board: ref.board, id: ref.task_id })}
                size="inline"
                type="button"
                variant="text"
              >
                {task.title}
              </Button>
            ) : (
              <span className="min-w-0 truncate text-(--ui-text-tertiary)">{shortId(ref.task_id)}</span>
            )}
            <span className="shrink-0 text-(--ui-text-quaternary)">{k.origin.board(ref.board)}</span>
            <span className="ml-auto shrink-0 text-(--ui-text-tertiary)">
              {task
                ? k.origin.state[state]
                : (k.origin.evidence[ref.evidence as keyof typeof k.origin.evidence] ?? k.origin.state.unavailable)}
            </span>
            {task && (
              <Button
                aria-label={k.origin.logs(task.title)}
                onClick={() => openLinkedTask({ board: ref.board, id: ref.task_id, section: 'log' })}
                size="xs"
                type="button"
                variant="ghost"
              >
                {k.workerLog}
              </Button>
            )}
          </li>
        )
      })}
      {view.truncated && (
        <li className="text-[0.625rem] text-(--ui-text-quaternary)">
          {k.origin.truncated(view.truncated.shown, view.truncated.total)}
        </li>
      )}
    </ul>
  )
}

export function OriginStrip({ context }: { context: SessionRouteContext }) {
  const k = useKanban()
  const view = useOriginView(context)
  const expanded = useValue($expanded)
  const key = `${context.connectionId}\0${context.profile}\0${context.sessionId}`

  if (view.kind === 'loading' || view.kind === 'out-of-scope') {
    return null
  }

  if (view.kind === 'unavailable') {
    return (
      <p
        className="flex items-center gap-1.5 text-[0.6875rem] text-(--ui-text-tertiary)"
        data-kanban-origin-unavailable={view.reason}
        role="status"
      >
        <Codicon name="warning" size="0.7rem" />
        {view.reason === 'request' ? k.origin.unreadable : k.origin.unknownHere}
      </p>
    )
  }

  if (view.refs.length === 0) {
    return null
  }

  const open = expanded[key] === true
  const { state } = summarizeOrigin(view.refs)
  const meta = STATE_META[state ?? 'unknown']

  return (
    <section className="flex min-w-0 flex-col gap-1" data-slot="kanban-origin-strip">
      <Button
        aria-expanded={open}
        aria-label={open ? k.origin.collapse : k.origin.expand}
        className="w-full justify-start gap-1.5"
        onClick={() => $expanded.set({ ...$expanded.get(), [key]: !open })}
        size="xs"
        type="button"
        variant="ghost"
      >
        <Codicon className={meta.tone} name={meta.icon} size="0.75rem" />
        <span>{k.origin.title}</span>
        <span className="text-(--ui-text-tertiary)">{k.origin.summary(view.refs.length)}</span>
        {state && <span className="truncate text-(--ui-text-tertiary)">· {k.origin.state[state]}</span>}
        <Codicon className="ml-auto" name={open ? 'chevron-up' : 'chevron-down'} size="0.7rem" />
      </Button>
      {open && <OriginList view={view} />}
    </section>
  )
}
