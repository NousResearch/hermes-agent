import { type KeyboardEvent, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { findBarKeyAction, formatMatchLabel } from '@/lib/find-in-page'
import { ChevronDown, ChevronUp } from '@/lib/icons'
import { findLogSearchHits, type LogSearchHit } from '@/lib/log-search'
import { cn } from '@/lib/utils'

export interface LogSearch {
  /** Index into `hits` of the match the view is parked on; -1 with no hits. */
  active: number
  hits: LogSearchHit[]
  next: () => void
  previous: () => void
  query: string
}

/**
 * Highlight-and-step search over a log's lines, the same model as ⌘F in chat.
 * The cursor is stored with the query it belongs to, so typing a new query
 * lands on its first hit without an effect resetting state.
 */
export function useLogSearch(lines: readonly string[] | null, query: string): LogSearch {
  const needle = query.trim()
  const hits = useMemo(() => findLogSearchHits(lines ?? [], needle), [lines, needle])
  const [cursor, setCursor] = useState({ index: 0, needle: '' })

  const active = hits.length === 0 ? -1 : cursor.needle === needle ? Math.min(cursor.index, hits.length - 1) : 0

  const step = (delta: number) => {
    if (hits.length > 0) {
      setCursor({ index: (active + delta + hits.length) % hits.length, needle })
    }
  }

  return { active, hits, next: () => step(1), previous: () => step(-1), query: needle }
}

interface LogSearchFieldProps {
  containerClassName?: string
  onChange: (query: string) => void
  placeholder: string
  search: LogSearch
  value: string
}

export function LogSearchField({ containerClassName, onChange, placeholder, search, value }: LogSearchFieldProps) {
  const { t } = useI18n()
  const label = formatMatchLabel(search.query, search.active + 1, search.hits.length)

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    const action =
      event.key === 'Enter' ? (event.shiftKey ? 'previous' : 'next') : findBarKeyAction(event, { inInput: true })

    if (!action) {
      return
    }

    event.preventDefault()

    if (action === 'close') {
      onChange('')
    } else {
      search[action]()
    }
  }

  return (
    <SearchField
      containerClassName={containerClassName}
      onChange={onChange}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      trailingAction={
        search.query ? (
          <span className="flex items-center gap-0.5">
            <span
              aria-live="polite"
              className={cn(
                'mr-0.5 text-[0.6875rem] leading-none tabular-nums',
                search.hits.length ? 'text-(--ui-text-secondary)' : 'text-destructive'
              )}
            >
              {label}
            </span>
            {(['previous', 'next'] as const).map(direction => {
              const Icon = direction === 'next' ? ChevronDown : ChevronUp

              return (
                <Tip key={direction} label={t.findInPage[direction]}>
                  <Button
                    aria-label={t.findInPage[direction]}
                    className="size-5 shrink-0 text-muted-foreground/85 hover:bg-accent/60 hover:text-foreground"
                    disabled={search.hits.length === 0}
                    onClick={search[direction]}
                    size="icon-xs"
                    variant="ghost"
                  >
                    <Icon className="size-3.5" />
                  </Button>
                </Tip>
              )
            })}
          </span>
        ) : null
      }
      value={value}
    />
  )
}
