/**
 * The ONE status glyph a roster row shows (roster-status.ts picks which), and
 * the Needs you band that lifts waiting rows above every section.
 *
 * Dots keep core's session-dot vocabulary (amber = act now, accent = working,
 * success = unread, faint grey = idle); the two states core has no dot for get
 * a codicon: a routine run (history) and a failure (warning). An unread @user
 * mention reads as a mention, never as the amber prompt dot, so a question you
 * must answer and a line you have not read stay visibly different.
 */

import { cn, Codicon, Tip } from '@hermes/plugin-sdk'
import type { ReactNode } from 'react'

import type { useBots } from './i18n'
import type { BotStatus } from './roster-status'

type BotsText = ReturnType<typeof useBots>

const QUOTE_MAX = 160

function clip(text: string): string {
  const flat = text.replace(/\s+/g, ' ').trim()

  return flat.length > QUOTE_MAX ? `${flat.slice(0, QUOTE_MAX - 1)}…` : flat
}

/** The tooltip: the state, then what it is waiting on, verbatim. */
export function rosterStatusLabel(status: BotStatus, b: BotsText): string {
  const s = b.status
  const quote = status.quote ? clip(status.quote) : ''

  const head =
    status.kind === 'needs-you'
      ? {
          approval: quote ? `${s.wantsToRun}: ${quote}` : s.needsYou,
          blocked: quote ? `${s.blocked} · ${quote}` : s.blocked,
          clarify: quote ? `${s.asks}: “${quote}”` : s.needsYou,
          mention: quote ? `${s.mentioned}: “${quote}”` : s.mentioned
        }[status.reason || 'clarify']
      : status.kind === 'failed' && quote
        ? `${s.failed} · ${quote}`
        : { failed: s.failed, idle: s.idle, routine: s.routine, unread: s.unread, working: s.working }[status.kind]

  return status.stale ? `${head} · ${s.stale}` : head
}

const DOT = 'block size-1.5 rounded-full'

const GLYPHS: Record<string, () => ReactNode> = {
  'needs-you': () => <span className={cn(DOT, 'bg-amber-500')} />,
  mention: () => <Codicon className="text-[0.6875rem] text-amber-600 dark:text-amber-300" name="mention" />,
  routine: () => <Codicon className="text-[0.6875rem] text-(--ui-accent)" name="history" />,
  working: () => <span className={cn(DOT, 'bg-(--ui-accent)')} />,
  failed: () => <Codicon className="text-[0.6875rem] text-destructive" name="warning" />,
  unread: () => <span className={cn(DOT, 'bg-(--ui-success)')} />,
  idle: () => <span className="block size-1 rounded-full bg-(--ui-text-quaternary)" />
}

export function RosterStatusGlyph({ b, status }: { b: BotsText; status: BotStatus }) {
  const label = rosterStatusLabel(status, b)
  const glyph = status.kind === 'needs-you' && status.reason === 'mention' ? 'mention' : status.kind

  return (
    <Tip label={label}>
      <span
        aria-label={label}
        className={cn('flex shrink-0 items-center justify-center', status.stale && 'opacity-50')}
        data-bot-status={status.kind}
        data-bot-status-reason={status.reason}
        role={status.kind === 'idle' ? undefined : 'status'}
      >
        {GLYPHS[glyph]()}
      </span>
    </Tip>
  )
}

/** The pinned band above every section. Rows inside keep their click and
 *  context menu but never drag: the band is a view of who is waiting, not a
 *  place to file anything. Absent when nobody waits. */
export function NeedsYouBand({ b, children, count }: { b: BotsText; children: ReactNode; count: number }) {
  if (!count) {
    return null
  }

  return (
    <section
      aria-label={b.status.band}
      className="mb-1 rounded-md border border-amber-500/30 bg-amber-500/5 pb-0.5"
      data-slot="bots-needs-you"
    >
      <div className="flex items-center gap-1.5 px-2 pt-1.5 pb-0.5 text-[0.6875rem] font-medium text-amber-700 dark:text-amber-300">
        <Codicon name="bell-dot" />
        <span>{b.status.band}</span>
        <span className="text-(--ui-text-quaternary)">{count}</span>
      </div>
      {children}
    </section>
  )
}
