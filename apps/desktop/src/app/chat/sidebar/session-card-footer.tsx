import { cn } from '@/lib/utils'

import { SIDEBAR_TRUNCATED_LEADING } from './chrome'

interface SessionCardFooterProps {
  /** Only a card row has this footer line; other densities render nothing. */
  card: boolean
  /** Owning profile, for a list that merges profiles (#89888) — the card has
   *  no metadata line to carry it, so it leads this one. */
  owner: null | string
  model: string
  size: string
  todoProgress: string | undefined
  todoTitle: string
}

/**
 * Card footer line: which model worked on it and how big it got. Rendered as
 * separate spans with a flex gap — a joined string can't put real space
 * between them (HTML collapses runs of whitespace to one).
 */
export function SessionCardFooter({ card, owner, model, size, todoProgress, todoTitle }: SessionCardFooterProps) {
  const owningProfile = card ? owner : null

  if (!owningProfile && !model && !size && !todoProgress) {
    return null
  }

  return (
    <span
      className={cn(
        'flex min-w-0 items-baseline gap-2 text-[0.625rem] text-(--ui-text-tertiary)',
        SIDEBAR_TRUNCATED_LEADING
      )}
    >
      {owningProfile ? <span className="min-w-0 truncate text-(--ui-text-quaternary)">{owningProfile}</span> : null}
      {model ? <span className="min-w-0 truncate">{model}</span> : null}
      {size ? <span className="shrink-0 tabular-nums">{size}</span> : null}
      {todoProgress ? (
        <span className="ml-auto shrink-0 tabular-nums" title={todoTitle}>
          {todoProgress}
        </span>
      ) : null}
    </span>
  )
}
