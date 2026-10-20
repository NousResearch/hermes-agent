import { useEffect, useMemo, useRef } from 'react'

import { CodeCardBody } from '@/components/chat/code-card'
import type { LogSearch } from '@/components/chat/log-search'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { ArrowBarToDown, ArrowBarToUp } from '@/lib/icons'
import { logLineSegments, logLineSeverity, type LogSearchHit, type LogSeverity } from '@/lib/log-search'
import { cn } from '@/lib/utils'

interface LogTailProps {
  /** null = still loading (shows the loading glyph); [] = loaded-but-empty
   *  (shows `emptyLabel`); non-empty renders as a tailing terminal log. */
  lines: null | string[]
  emptyLabel: string
  className?: string
  /** From useLogSearch: highlights every hit and parks the view on the active one. */
  search?: LogSearch
}

// A hairline in the gutter, not a tint on the text: the level word is already
// in the line, this only makes warnings and errors findable at a glance.
const SEVERITY_RULE: Record<LogSeverity, string> = {
  critical: 'before:bg-destructive',
  error: 'before:bg-destructive',
  warning: 'before:bg-(--ui-yellow)'
}

const HOVER_CONTROL =
  'h-5 rounded-md px-1 opacity-5 transition-opacity group-hover/logs:opacity-100 hover:opacity-100 focus-visible:opacity-100'

/** The shared terminal-log surface: CodeCardBody typography, hover-reveal copy
 *  and jump controls, follow-the-tail scrolling (releases when the user scrolls
 *  up), and optional in-place search. One component behind every log pane —
 *  MCP stdio/agent, Command Center, hub action logs — so they all read, copy,
 *  search, and scroll identically. */
export function LogTail({ className, emptyLabel, lines, search }: LogTailProps) {
  const { t } = useI18n()
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const stickRef = useRef(true)
  const searching = Boolean(search?.query)
  const active = search?.active ?? -1

  const hitsByLine = useMemo(() => {
    const byLine = new Map<number, { hit: LogSearchHit; index: number }[]>()

    search?.hits.forEach((hit, index) => {
      byLine.set(hit.line, [...(byLine.get(hit.line) ?? []), { hit, index }])
    })

    return byLine
  }, [search?.hits])

  // Jumping to the end is also re-engaging the follow: a programmatic scroll's
  // event can land after the next lines render, so don't wait for onScroll.
  const followTail = () => {
    const el = scrollRef.current

    stickRef.current = true

    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }

  // Follow the tail while nobody is searching; a search parks the view on its
  // hit instead, and polling must not yank it back down.
  useEffect(() => {
    if (!searching && stickRef.current) {
      followTail()
    }
  }, [lines, searching])

  // Clearing a search returns to the live tail. Scrolling to a hit released the
  // follow, so without this the pane would stay parked forever.
  useEffect(() => {
    if (!searching) {
      followTail()
    }
  }, [searching])

  useEffect(() => {
    if (active >= 0) {
      scrollRef.current?.querySelector('[data-log-hit="active"]')?.scrollIntoView({ block: 'center' })
    }
  }, [active, search?.query])

  const jump = (edge: 'bottom' | 'top') => {
    const el = scrollRef.current

    el?.scrollTo({ behavior: 'smooth', top: edge === 'top' ? 0 : el.scrollHeight })
  }

  return (
    <div className={cn('group/logs relative h-full min-h-0', className)}>
      <div className="absolute right-2.5 top-1.5 z-10 flex items-center gap-0.5">
        {(['top', 'bottom'] as const).map(edge => {
          const Icon = edge === 'top' ? ArrowBarToUp : ArrowBarToDown

          return (
            <Tip key={edge} label={t.ui.logs[edge]}>
              <Button
                aria-label={t.ui.logs[edge]}
                className={cn(HOVER_CONTROL, 'text-muted-foreground hover:text-foreground')}
                onClick={() => jump(edge)}
                size="icon-xs"
                variant="ghost"
              >
                <Icon className="size-3" />
              </Button>
            </Tip>
          )
        })}
        <CopyButton
          appearance="inline"
          className={cn(HOVER_CONTROL, 'gap-0')}
          iconClassName="size-3"
          showLabel={false}
          text={() => (lines ?? []).join('\n')}
        />
      </div>
      <div
        className="h-full min-h-0 overflow-y-auto [scrollbar-gutter:stable]"
        data-selectable-text="true"
        onScroll={event => {
          const el = event.currentTarget
          stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
        }}
        ref={scrollRef}
      >
        {lines === null || lines.length === 0 ? (
          <p className="px-2 py-1.5 font-mono text-[0.7rem] leading-relaxed text-muted-foreground/50">
            {lines === null ? '…' : emptyLabel}
          </p>
        ) : (
          <CodeCardBody>
            <pre className="whitespace-pre-wrap break-words">
              {lines.map((line, index) => {
                const severity = logLineSeverity(line)
                const hits = hitsByLine.get(index)

                return (
                  <span
                    className={cn(
                      'block',
                      line.startsWith('=====') && 'mt-1 text-(--ui-text-tertiary)',
                      severity &&
                        cn(
                          'relative before:absolute before:-left-1.5 before:inset-y-0 before:w-0.5 before:rounded-full',
                          SEVERITY_RULE[severity]
                        )
                    )}
                    data-log-severity={severity ?? undefined}
                    key={index}
                  >
                    {hits
                      ? logLineSegments(line, hits).map((segment, part) =>
                          segment.hit === null ? (
                            segment.text
                          ) : (
                            <mark
                              className={cn(
                                'rounded-[2px] text-inherit',
                                segment.hit === active ? 'bg-(--ui-yellow)/55 text-foreground' : 'bg-(--ui-yellow)/30'
                              )}
                              data-log-hit={segment.hit === active ? 'active' : 'match'}
                              key={part}
                            >
                              {segment.text}
                            </mark>
                          )
                        )
                      : line}
                  </span>
                )
              })}
            </pre>
          </CodeCardBody>
        )}
      </div>
    </div>
  )
}
